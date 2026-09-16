"""Self-contained HTML report and results zip archive."""
from __future__ import annotations

import base64
import html
import io
import json
import zipfile
from pathlib import Path
from typing import Any

import grids


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _fmt(ms: dict, pct=False, digits=4) -> str:
    if ms is None or ms.get("mean") is None:
        return "—"
    mean, std = ms["mean"], ms["std"]
    scale = 100.0 if pct else 1.0
    suffix = "%" if pct else ""
    if std is None:
        return f"{mean * scale:.{digits}f}{suffix}"
    return f"{mean * scale:.{digits}f} ± {std * scale:.{digits}f}{suffix}"


def _metrics_table(results: dict) -> str:
    models = results["config"]["models"]
    quantiles = results["config"]["quantiles"]
    rows = []
    header = ["Metric"] + models
    rows.append("<tr>" + "".join(f"<th>{html.escape(str(h))}</th>" for h in header) + "</tr>")

    def row(label, fn):
        cells = "".join(f"<td>{fn(m)}</td>" for m in models)
        return f"<tr><td class='lbl'>{html.escape(label)}</td>{cells}</tr>"

    s = results["summaries"]
    # pinball_mean is already the average over every level in the (possibly dense) grid.
    rows.append(row("Pinball (mean over full grid)", lambda m: _fmt(s[m]["pinball_mean"])))
    # Per-τ rows: only the standard five levels, even when the grid is denser.
    tau_index = {round(float(d["tau"]), 6): j for j, d in enumerate(s[models[0]]["pinball_per_tau"])}
    for q in grids.standard_present(quantiles):
        j = tau_index.get(round(float(q), 6))
        if j is None:
            continue
        rows.append(row(f"  pinball τ={q:g}",
                        lambda m, j=j: _fmt(s[m]["pinball_per_tau"][j])))
    rows.append(row("ICP 80% (0.1–0.9)", lambda m: _fmt(s[m]["icp_80"], pct=True, digits=1)))
    rows.append(row("ICP 50% (0.25–0.75)", lambda m: _fmt(s[m]["icp_50"], pct=True, digits=1)))
    rows.append(row("Uno C-index", lambda m: _fmt(s[m]["uno_c"], digits=4)))
    rows.append(row("Adjacent crossing rate",
                    lambda m: _fmt(s[m]["adjacent_crossing"], pct=True, digits=2)))
    rows.append(row("Outer crossing rate",
                    lambda m: _fmt(s[m]["outer_crossing"], pct=True, digits=2)))
    rows.append(row("Best epoch", lambda m: _fmt(s[m]["best_epoch"], digits=1)))
    return "<table class='metrics'>" + "".join(rows) + "</table>"


_PLOT_TITLES = {
    "km_curve": "Kaplan–Meier outcome curve",
    "training_curves": "Training curves",
    "calibration": "Per-τ calibration",
    "quantile_profiles": "Individual quantile profiles",
    "survival_curves": "Individual survival curves",
    "coverage_by_width": "Coverage by interval-width tercile",
    "importance_heatmap": "Permutation importance",
    "pinball_boxplots": "Pinball across splits",
}
_PLOT_ORDER = ["km_curve", "training_curves", "calibration", "quantile_profiles",
               "survival_curves", "coverage_by_width", "importance_heatmap",
               "pinball_boxplots"]


def _data_section(results: dict) -> str:
    data = results.get("data") or {}
    if not data:
        return ""
    rows = []

    def row(label, value):
        rows.append(f"<tr><td class='lbl'>{html.escape(label)}</td>"
                    f"<td>{value}</td></tr>")

    unit = data.get("time_unit")
    unit_suffix = f" {html.escape(unit)}" if unit else ""
    rng = data.get("duration_range") or [None, None]
    excluded = data.get("rows_excluded", 0)
    reasons = data.get("exclusion_reasons") or []
    excl_txt = str(excluded)
    if excluded and reasons:
        excl_txt += " (" + html.escape("; ".join(reasons)) + ")"
    row("File", html.escape(str(data.get("file_name", "—"))))
    row("Rows used", str(data.get("rows_used", "—")))
    row("Rows excluded", excl_txt)
    row("Events", str(data.get("n_events", "—")))
    row("Censoring", f"{data.get('censoring_pct', '—')}%")
    if rng[0] is not None:
        row("Duration range", f"{rng[0]:g} – {rng[1]:g}{unit_suffix}")
    row("Time unit", html.escape(unit) if unit else "—")
    row("Duration column", html.escape(str(data.get("duration_col", "—"))))
    row("Event column", html.escape(str(data.get("event_col", "—"))))
    if data.get("id_col"):
        row("Subject ID column", html.escape(str(data["id_col"])))
    mapping = data.get("event_mapping")
    if mapping:
        row("Event mapping", html.escape(str(mapping.get("meaning", ""))))
    feats = data.get("features") or []
    row("Features", html.escape(", ".join(map(str, feats))) or "—")
    return ("<h2>Data</h2>\n<table class=\"metrics\">" + "".join(rows) + "</table>")


def build_html_report(results: dict, images: dict[str, bytes]) -> str:
    cfg = results["config"]
    ds = results["dataset"]
    repo = results.get("repo") or {}
    plots_html = []
    for name in _PLOT_ORDER:
        if name in images:
            title = html.escape(_PLOT_TITLES.get(name, name))
            src = f"data:image/png;base64,{_b64(images[name])}"
            plots_html.append(
                f"<figure><figcaption>{title}</figcaption>"
                f"<img src='{src}' alt='{title}'></figure>")

    grid = cfg.get("quantile_grid") or {}
    grid_kind = grids.GRID_LABELS.get(grid.get("kind", ""), grid.get("kind", "custom"))
    quantiles = cfg["quantiles"]
    grid_note = (f"Quantile grid: {len(quantiles)} levels ({grid_kind}). "
                 + ("Metric tables show the five standard levels; "
                    "the pinball mean is over the full grid." if len(quantiles) > 5 else ""))
    config_json = html.escape(json.dumps({
        "models": cfg["models"], "quantiles": cfg["quantiles"],
        "quantile_grid": cfg.get("quantile_grid"), "ratio": cfg["ratio"],
        "n_splits": cfg["n_splits"], "seed": cfg["seed"], "preset": cfg["preset"],
        "custom": cfg["custom"], "duration_col": cfg["duration_col"],
        "event_col": cfg["event_col"], "feature_cols": cfg["feature_cols"],
    }, indent=2))

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>CNQ report — {html.escape(results['job_id'])}</title>
<style>
  body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 2rem auto;
         max-width: 1000px; color: #1c2530; padding: 0 1rem; }}
  h1 {{ font-size: 1.5rem; }} h2 {{ margin-top: 2rem; border-bottom: 2px solid #e3e8ef;
        padding-bottom: .3rem; }}
  table.metrics {{ border-collapse: collapse; width: 100%; font-size: .9rem; }}
  table.metrics td, table.metrics th {{ border: 1px solid #dce1e8; padding: .35rem .6rem;
        text-align: right; }}
  table.metrics th {{ background: #f1f5fb; }}
  table.metrics td.lbl {{ text-align: left; font-weight: 600; background: #fafbfd; }}
  .grid {{ display: grid; grid-template-columns: 1fr; gap: 1.2rem; }}
  figure {{ margin: 0; border: 1px solid #e3e8ef; border-radius: 8px; padding: .6rem; }}
  figcaption {{ font-weight: 600; margin-bottom: .4rem; }}
  figure img {{ max-width: 100%; }}
  .meta {{ color: #556; font-size: .9rem; }}
  pre {{ background: #f6f8fb; padding: .8rem; border-radius: 6px; overflow: auto; font-size: .8rem; }}
</style></head><body>
<h1>Censored Non-crossing Quantile Regression — results</h1>
<p class="meta">Job <code>{html.escape(results['job_id'])}</code> ·
  {ds['n_rows']} rows · {ds['n_features']} features ·
  {ds['n_events']} events · censoring {ds['censoring_rate'] * 100:.1f}% ·
  {cfg['n_splits']} repeated split(s) · seed {cfg['seed']}</p>

{_data_section(results)}

<h2>Metrics summary</h2>
{_metrics_table(results)}
<p class="meta">Cells show mean ± standard deviation across repeated splits. Pinball and
Uno C are on the test split; ICP is interval coverage. {html.escape(grid_note)}</p>

<h2>Plots</h2>
<div class="grid">{''.join(plots_html)}</div>

<h2>Run information</h2>
<table class="metrics">
  <tr><td class="lbl">Job</td><td>{html.escape(str(results['job_id']))}</td></tr>
  <tr><td class="lbl">deepcnq repository</td><td>{html.escape(str(repo.get('path') or 'unknown'))}</td></tr>
  <tr><td class="lbl">Repository version</td><td>{html.escape(str(repo.get('version_str') or repo.get('version') or 'unknown'))}{' (dirty)' if repo.get('dirty') else ''}</td></tr>
  <tr><td class="lbl">Version source</td><td>{html.escape(str(repo.get('source') or 'unknown'))}</td></tr>
</table>

<h2>Run configuration</h2>
<pre>{config_json}</pre>
</body></html>"""


def build_zip(zip_path: Path, results: dict, images: dict[str, bytes], html_report: str,
              csv_path: str | None = None, predictions: dict[str, str] | None = None) -> None:
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("report.html", html_report)
        zf.writestr("results.json", json.dumps(_strip_private(results), indent=2, default=str))
        for name, data in images.items():
            zf.writestr(f"plots/{name}.png", data)
        for name, csv_text in (predictions or {}).items():
            zf.writestr(f"predictions/{name}", csv_text)  # full-grid predictions per model
        if csv_path and Path(csv_path).exists():
            zf.write(csv_path, "input_data.csv")


def _strip_private(results: dict) -> dict:
    return {k: v for k, v in results.items() if not k.startswith("_")}
