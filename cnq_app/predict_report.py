"""Figures, HTML report and results zip for a prediction run.

Consumes the dict returned by :func:`predict.run_prediction` (which carries the
log/time prediction arrays and, when time/event were supplied, an external
validation block). Writes ``report.html``, ``results.zip`` and
``predictions.csv`` into the job directory and returns a JSON-safe summary for
the UI. Rendering uses the headless Agg backend so it runs in the job thread.
"""
from __future__ import annotations

import base64
import html
import io
import json
import zipfile
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import model_io  # noqa: E402

_COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]

CAVEATS = [
    "The quantile spread for each subject describes between-subject variation in "
    "survival time at those covariate values (how outcomes differ across similar "
    "subjects). It is not the model's own (epistemic) uncertainty about its "
    "parameters, which is not quantified here.",
    "Predictions assume the new subjects are drawn from a population similar to the "
    "training data. Subjects with features outside the training range (flagged "
    "'out_of_range') are extrapolations and less reliable; broad population shift "
    "degrades all predictions, even in-range ones.",
    "External validation metrics are computed on the uploaded data with a censoring "
    "distribution estimated from those same rows. They describe fit on this sample, "
    "not a guarantee of performance on future data.",
]


# --------------------------------------------------------------------------- #
def _fig_to_png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def _color(i: int) -> str:
    return _COLORS[i % len(_COLORS)]


def _time_label(unit) -> str:
    return f"Predicted survival time ({unit})" if unit else "Predicted survival time"


def _select_subjects(median_time: np.ndarray, k: int) -> np.ndarray:
    """Indices spread across the median-time distribution (min..max), so the
    profile plot shows a representative range rather than random subjects."""
    n = len(median_time)
    k = min(k, n)
    if k <= 0:
        return np.array([], dtype=int)
    order = np.argsort(median_time)
    picks = np.linspace(0, n - 1, k).round().astype(int)
    return order[np.unique(picks)]


def quantile_profiles_plot(pred_time, quantiles, median_time, ids, unit, k=8) -> bytes:
    idx = _select_subjects(np.asarray(median_time, float), k)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for c, i in enumerate(idx):
        ax.plot(quantiles, pred_time[i], marker="o", color=_color(c), alpha=0.85,
                label=f"subj {ids[i]}")
    ax.set_xlabel("Quantile level τ")
    ax.set_ylabel(_time_label(unit))
    ax.set_title("Predicted quantile profiles (selected subjects)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    return _fig_to_png(fig)


def survival_curves_plot(pred_time, quantiles, median_time, ids, unit, k=8) -> bytes:
    taus = np.asarray(quantiles, float)
    surv = 1.0 - taus
    idx = _select_subjects(np.asarray(median_time, float), k)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for c, i in enumerate(idx):
        ax.step(pred_time[i], surv, where="post", color=_color(c), alpha=0.85,
                label=f"subj {ids[i]}")
    ax.set_xlabel(f"Time ({unit})" if unit else "Time")
    ax.set_ylabel("Predicted survival  S(t) = 1 − τ")
    ax.set_ylim(0, 1.02)
    ax.set_title("Predicted survival curves (selected subjects)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    return _fig_to_png(fig)


def distribution_plot(median_time, width80, unit) -> bytes:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(np.asarray(median_time, float), bins=30, color=_color(0), alpha=0.8)
    axes[0].set_xlabel(f"Median survival time ({unit})" if unit else "Median survival time")
    axes[0].set_ylabel("Subjects")
    axes[0].set_title("Distribution of predicted medians")
    axes[0].grid(alpha=0.3, axis="y")
    axes[1].hist(np.asarray(width80, float), bins=30, color=_color(1), alpha=0.8)
    axes[1].set_xlabel(f"80% interval width ({unit})" if unit else "80% interval width")
    axes[1].set_ylabel("Subjects")
    axes[1].set_title("Distribution of 80% interval widths")
    axes[1].grid(alpha=0.3, axis="y")
    return _fig_to_png(fig)


def external_km_plot(external: dict, unit) -> bytes:
    km = external["km"]
    ps = external["predicted_survival"]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.step(km["time"], km["survival"], where="post", color=_color(0), lw=2,
            label=f"Kaplan–Meier (observed, n={km['n']})")
    ax.plot(ps["time"], ps["survival"], color=_color(1), lw=2, ls="--",
            label="Mean predicted survival")
    ax.set_xlabel(f"Time ({unit})" if unit else "Time")
    ax.set_ylabel("Survival probability S(t)")
    ax.set_ylim(0, 1.02)
    ax.set_title("External validation: observed vs predicted survival")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return _fig_to_png(fig)


def external_calibration_plot(external: dict) -> bytes:
    calib = external["calibration"]
    taus = sorted(float(k) for k in calib)
    emp = [calib[str(t)] if str(t) in calib else calib[f"{t}"] for t in taus]
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], color="grey", ls=":", label="ideal")
    ax.plot(taus, emp, marker="o", color=_color(2), label="observed")
    ax.set_xlabel("Nominal quantile level τ")
    ax.set_ylabel("Empirical coverage P(Y ≤ q_τ)")
    ax.set_title("External validation: per-τ calibration")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return _fig_to_png(fig)


def generate_plots(bundle, result: dict) -> dict[str, bytes]:
    meta = bundle.meta
    unit = meta.get("time_unit")
    quantiles = result["quantiles"]
    pred_time = np.asarray(result["pred_time"], float)
    median_time = np.asarray(result["median_time"], float)
    width80 = np.asarray(result["width80"], float)
    ids = result["subject_ids"]

    figs = {
        "quantile_profiles": quantile_profiles_plot(pred_time, quantiles, median_time, ids, unit),
        "distributions": distribution_plot(median_time, width80, unit),
    }
    if len(quantiles) > 5:
        figs["survival_curves"] = survival_curves_plot(pred_time, quantiles, median_time, ids, unit)

    external = result.get("external")
    if external and external.get("available"):
        figs["external_km"] = external_km_plot(external, unit)
        figs["external_calibration"] = external_calibration_plot(external)
    return figs


# --------------------------------------------------------------------------- #
# HTML report
# --------------------------------------------------------------------------- #
def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


_PLOT_TITLES = {
    "quantile_profiles": "Predicted quantile profiles",
    "survival_curves": "Predicted survival curves",
    "distributions": "Predicted medians & interval widths",
    "external_km": "Observed vs predicted survival (external)",
    "external_calibration": "Per-τ calibration (external)",
}
_PLOT_ORDER = ["quantile_profiles", "survival_curves", "distributions",
               "external_km", "external_calibration"]


def _fmt(v, digits=4, pct=False):
    if v is None:
        return "—"
    scale = 100.0 if pct else 1.0
    return f"{float(v) * scale:.{digits}f}{'%' if pct else ''}"


def _external_table(external: dict) -> str:
    rows = [
        ("Rows validated", str(external.get("n", "—"))),
        ("Events", str(external.get("n_events", "—"))),
        ("IPCW pinball (mean over τ)", _fmt(external.get("pinball_mean"))),
        ("50% interval coverage", _fmt(external.get("coverage_50"), digits=1, pct=True)),
        ("80% interval coverage", _fmt(external.get("coverage_80"), digits=1, pct=True)),
        ("Uno C-index", _fmt(external.get("uno_c"))),
    ]
    body = "".join(f"<tr><td class='lbl'>{html.escape(l)}</td><td>{v}</td></tr>" for l, v in rows)
    return f"<table class='metrics'>{body}</table>"


def build_html(bundle, result: dict, images: dict[str, bytes]) -> str:
    meta = bundle.meta
    summary = model_io.summarize(meta)
    unit = meta.get("time_unit")
    plots_html = []
    for name in _PLOT_ORDER:
        if name in images:
            title = html.escape(_PLOT_TITLES.get(name, name))
            plots_html.append(
                f"<figure><figcaption>{title}</figcaption>"
                f"<img src='data:image/png;base64,{_b64(images[name])}' alt='{title}'></figure>")

    model_rows = [
        ("Model", html.escape(str(summary.get("model")))),
        ("Bundle type", html.escape(str(summary.get("bundle_type")))
         + (f" ({summary.get('n_members')} members)" if summary.get("n_members") else "")),
        ("Trained on", f"{summary.get('n', '—')} subjects, {summary.get('n_events', '—')} events, "
                       f"{summary.get('censoring_pct', '—')}% censored"),
        ("Features", html.escape(", ".join(map(str, summary.get("features") or [])))),
        ("Quantile levels", html.escape(", ".join(f"{q:g}" for q in summary.get("quantiles") or []))),
        ("Time unit", html.escape(str(unit)) if unit else "—"),
        ("Saved", html.escape(str(summary.get("created_at") or "—"))),
        ("deepcnq version", html.escape(str(summary.get("deepcnq") or "unknown"))),
    ]
    model_table = "".join(
        f"<tr><td class='lbl'>{html.escape(l)}</td><td>{v}</td></tr>" for l, v in model_rows)

    pred_rows = [
        ("Subjects predicted", str(result.get("n", "—"))),
        ("Rows excluded (missing features)", str(result.get("rows_excluded", 0))),
        ("Out-of-range subjects (flagged)", str(result.get("out_of_range_count", 0))),
    ]
    pred_table = "".join(
        f"<tr><td class='lbl'>{html.escape(l)}</td><td>{v}</td></tr>" for l, v in pred_rows)

    fmap = result.get("feature_map") or {}
    mapping_rows = "".join(
        f"<tr><td class='lbl'>{html.escape(str(f))}</td><td>{html.escape(str(fmap.get(f, '—')))}</td></tr>"
        for f in (meta.get("feature_names") or []))
    mapping_table = (f"<table class='metrics'><tr><th>Model feature</th>"
                     f"<th>File column</th></tr>{mapping_rows}</table>")

    preview = _preview_table(result)
    external = result.get("external")
    external_html = ""
    if external and external.get("available"):
        external_html = (f"<h2>External validation</h2>{_external_table(external)}"
                         f"<p class='meta'>{html.escape(external.get('note', ''))}</p>")
    elif external and not external.get("available"):
        external_html = (f"<h2>External validation</h2><p class='meta'>"
                         f"{html.escape(external.get('note', 'Not available.'))}</p>")

    caveats_html = "".join(f"<li>{html.escape(c)}</li>" for c in CAVEATS)

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>CNQ prediction report</title>
<style>
  body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 2rem auto;
         max-width: 1000px; color: #1c2530; padding: 0 1rem; }}
  h1 {{ font-size: 1.5rem; }} h2 {{ margin-top: 2rem; border-bottom: 2px solid #e3e8ef;
        padding-bottom: .3rem; }}
  table.metrics {{ border-collapse: collapse; width: 100%; font-size: .9rem; }}
  table.metrics td, table.metrics th {{ border: 1px solid #dce1e8; padding: .35rem .6rem;
        text-align: right; }}
  table.metrics td.lbl {{ text-align: left; font-weight: 600; background: #fafbfd; }}
  .grid {{ display: grid; grid-template-columns: 1fr; gap: 1.2rem; }}
  figure {{ margin: 0; border: 1px solid #e3e8ef; border-radius: 8px; padding: .6rem; }}
  figcaption {{ font-weight: 600; margin-bottom: .4rem; }}
  figure img {{ max-width: 100%; }}
  .meta {{ color: #556; font-size: .9rem; }}
  .scroll {{ overflow-x: auto; }}
  ul.caveats li {{ margin-bottom: .5rem; }}
</style></head><body>
<h1>CNQ — prediction report</h1>

<h2>Model</h2>
<table class="metrics">{model_table}</table>

<h2>Prediction summary</h2>
<table class="metrics">{pred_table}</table>

<h2>Column mapping used</h2>
<table class="metrics">{mapping_table}</table>

<h2>Plots</h2>
<div class="grid">{''.join(plots_html)}</div>

{external_html}

<h2>Predictions (first rows)</h2>
<div class="scroll">{preview}</div>

<h2>How to read these predictions</h2>
<ul class="caveats">{caveats_html}</ul>
</body></html>"""


def _preview_table(result: dict) -> str:
    cols = result.get("columns") or []
    rows = result.get("preview_rows") or []
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in cols)
    body = []
    for row in rows:
        cells = "".join(f"<td>{'' if v is None else html.escape(_short(v))}</td>" for v in row)
        body.append(f"<tr>{cells}</tr>")
    return (f"<table class='metrics'><tr>{head}</tr>{''.join(body)}</table>")


def _short(v) -> str:
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


# --------------------------------------------------------------------------- #
# assemble: write files, return JSON-safe summary
# --------------------------------------------------------------------------- #
def build(job_dir: Path, bundle, result: dict, cfg: dict) -> dict:
    job_dir = Path(job_dir)
    images = generate_plots(bundle, result)
    html_report = build_html(bundle, result, images)

    (job_dir / "predictions.csv").write_text(result["predictions_csv"], encoding="utf-8")
    (job_dir / "report.html").write_text(html_report, encoding="utf-8")

    img_dir = job_dir / "plots"
    img_dir.mkdir(exist_ok=True)
    for name, data in images.items():
        (img_dir / f"{name}.png").write_bytes(data)

    external = result.get("external")
    payload = {
        "kind": "predict",
        "n": result["n"],
        "rows_excluded": result["rows_excluded"],
        "out_of_range_count": result["out_of_range_count"],
        "per_feature_oor": result["per_feature_oor"],
        "quantiles": result["quantiles"],
        "columns": result["columns"],
        "preview_rows": result["preview_rows"],
        "feature_map": dict(result.get("feature_map") or {}),
        "model_summary": model_io.summarize(bundle.meta),
        "bundle_warnings": list(getattr(bundle, "warnings", []) or []),
        "external": _external_summary(external),
    }
    (job_dir / "results.json").write_text(json.dumps(payload, indent=2, default=str),
                                          encoding="utf-8")

    with zipfile.ZipFile(job_dir / "results.zip", "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("report.html", html_report)
        zf.writestr("predictions.csv", result["predictions_csv"])
        zf.writestr("results.json", json.dumps(payload, indent=2, default=str))
        zf.writestr("model.json", json.dumps(bundle.meta, indent=2, default=str))
        for name, data in images.items():
            zf.writestr(f"plots/{name}.png", data)

    return {
        **payload,
        "_report_path": str(job_dir / "report.html"),
        "_zip_path": str(job_dir / "results.zip"),
        "_predictions_path": str(job_dir / "predictions.csv"),
    }


def _external_summary(external: Any) -> Any:
    """Drop the big curve arrays from the API payload; keep the scalar metrics."""
    if not external:
        return None
    if not external.get("available"):
        return {"available": False, "note": external.get("note")}
    return {
        "available": True,
        "n": external.get("n"),
        "n_events": external.get("n_events"),
        "pinball_mean": external.get("pinball_mean"),
        "pinball_per_tau": external.get("pinball_per_tau"),
        "coverage_50": external.get("coverage_50"),
        "coverage_80": external.get("coverage_80"),
        "uno_c": external.get("uno_c"),
        "calibration": external.get("calibration"),
        "note": external.get("note"),
    }
