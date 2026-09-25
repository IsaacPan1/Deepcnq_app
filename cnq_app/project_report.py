"""Plot, HTML report and results zip for a population/cohort projection.

Consumes the dict from :mod:`project` plus the answered queries, writes
``report.html``, ``results.zip`` and ``projection.csv`` into the job dir, and
returns a JSON-safe summary for the UI. Headless Agg backend (runs in the job).
"""
from __future__ import annotations

import base64
import html
import io
import json
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import model_io  # noqa: E402
import project  # noqa: E402

_COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _time_label(unit) -> str:
    return f"Time ({unit})" if unit else "Time"


def curve_plot(proj: dict, queries: dict, unit) -> bytes:
    t = np.asarray(proj["time"], float)
    D = np.asarray(proj["events"], float)
    lo = np.asarray(proj["events_lo"], float)
    hi = np.asarray(proj["events_hi"], float)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.fill_between(t, lo, hi, color=_COLORS[0], alpha=0.2, label="95% band")
    ax.plot(t, D, color=_COLORS[0], lw=2, label="Expected cumulative events")
    for q in queries.get("events_at", []):
        if not q["out_of_range"]:
            ax.axvline(q["time"], color=_COLORS[3], ls=":", alpha=0.7)
            ax.scatter([q["time"]], [q["events"]], color=_COLORS[3], zorder=5)
    tf = queries.get("time_for")
    if tf and not tf.get("out_of_range") and tf.get("time") is not None:
        ax.axhline(tf["target"], color=_COLORS[2], ls=":", alpha=0.7)
        ax.scatter([tf["time"]], [tf["target"]], color=_COLORS[2], zorder=5,
                   label=f"{tf['target']:g} events by {tf['time']:.3g}")
    ax.set_xlabel(_time_label(unit))
    ax.set_ylabel("Cumulative events")
    ax.set_ylim(bottom=0)
    src = "population (scaled training KM)" if proj["source"] == "population" else "cohort (model)"
    ax.set_title(f"Projected cumulative events — {src}\n(supported to t={proj['horizon']:.3g})")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def _projection_csv(proj: dict) -> str:
    rows = ["time,events,events_lo,events_hi"]
    for t, d, lo, hi in zip(proj["time"], proj["events"], proj["events_lo"], proj["events_hi"]):
        rows.append(f"{t:.6g},{d:.6g},{lo:.6g},{hi:.6g}")
    return "\n".join(rows) + "\n"


def _queries_html(proj: dict, queries: dict, unit) -> str:
    parts = []
    ea = queries.get("events_at") or []
    if ea:
        rows = "".join(
            f"<tr><td class='lbl'>{q['time']:g}</td><td>"
            + ("beyond follow-up" if q["out_of_range"]
               else f"{q['events']:.1f} ({q['lo']:.1f}–{q['hi']:.1f})")
            + "</td></tr>" for q in ea)
        parts.append(f"<h2>Events by time</h2><table class='metrics'>"
                     f"<tr><th>Time ({html.escape(str(unit))})</th><th>Expected events (95%)</th></tr>"
                     f"{rows}</table>")
    tf = queries.get("time_for")
    if tf:
        if tf.get("out_of_range"):
            body = (f"Only ~{tf['max_events']:.0f} events occur within observed follow-up "
                    f"(to t={tf['horizon']:.3g}), so {tf['target']:g} isn't reachable without "
                    f"extrapolating the censored tail.")
        else:
            body = (f"<b>{tf['target']:g}</b> events are expected by "
                    f"<b>{tf['time']:.3g}</b> {html.escape(str(unit or ''))} "
                    f"(95% {tf['time_lo']:.3g}–{tf['time_hi']:.3g}).")
        parts.append(f"<h2>Time for a target</h2><p>{body}</p>")
    return "".join(parts)


def build_html(meta: dict, proj: dict, queries: dict, images: dict[str, bytes]) -> str:
    summary = model_io.summarize(meta)
    unit = meta.get("time_unit")
    src = "Population (N scaled to the training Kaplan–Meier curve)" \
        if proj["source"] == "population" else "Cohort (aggregated model predictions)"
    head_rows = [
        ("Basis", src),
        ("Subjects (N)", f"{proj['n']:.0f}"),
        ("Follow-up horizon", f"{proj['horizon']:.3g} {html.escape(str(unit or ''))}"),
        ("Max events within follow-up", f"{proj['max_events']:.1f}"),
        ("Model", html.escape(str(summary.get("model")))),
    ]
    head = "".join(f"<tr><td class='lbl'>{html.escape(l)}</td><td>{v}</td></tr>"
                   for l, v in head_rows)
    plot_html = "".join(
        f"<figure><img src='data:image/png;base64,{_b64(d)}' alt='projection'></figure>"
        for d in images.values())
    caveats = "".join(f"<li>{html.escape(c)}</li>" for c in project.CAVEATS)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>CNQ projection report</title>
<style>
  body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 2rem auto;
         max-width: 1000px; color: #1c2530; padding: 0 1rem; }}
  h1 {{ font-size: 1.5rem; }} h2 {{ margin-top: 2rem; border-bottom: 2px solid #e3e8ef;
        padding-bottom: .3rem; }}
  table.metrics {{ border-collapse: collapse; width: 100%; font-size: .9rem; }}
  table.metrics td, table.metrics th {{ border: 1px solid #dce1e8; padding: .35rem .6rem;
        text-align: right; }}
  table.metrics td.lbl {{ text-align: left; font-weight: 600; background: #fafbfd; }}
  figure {{ margin: 1rem 0; border: 1px solid #e3e8ef; border-radius: 8px; padding: .6rem; }}
  figure img {{ max-width: 100%; }}
  ul.caveats li {{ margin-bottom: .5rem; }}
</style></head><body>
<h1>CNQ — projection report</h1>
<table class="metrics">{head}</table>
<h2>Projected trend</h2>{plot_html}
{_queries_html(proj, queries, unit)}
<h2>How to read this</h2><ul class="caveats">{caveats}</ul>
</body></html>"""


def build(job_dir: Path, meta: dict, proj: dict, queries: dict) -> dict:
    job_dir = Path(job_dir)
    unit = meta.get("time_unit")
    images = {"projection": curve_plot(proj, queries, unit)}
    html_report = build_html(meta, proj, queries, images)
    csv_text = _projection_csv(proj)

    (job_dir / "report.html").write_text(html_report, encoding="utf-8")
    (job_dir / "projection.csv").write_text(csv_text, encoding="utf-8")
    img_dir = job_dir / "plots"
    img_dir.mkdir(exist_ok=True)
    for name, data in images.items():
        (img_dir / f"{name}.png").write_bytes(data)

    payload = {
        "kind": "project",
        "source": proj["source"],
        "n": proj["n"],
        "horizon": proj["horizon"],
        "max_events": proj["max_events"],
        "last_is_censored": proj["last_is_censored"],
        "time_unit": unit,
        "curve": {"time": proj["time"], "events": proj["events"],
                  "events_lo": proj["events_lo"], "events_hi": proj["events_hi"]},
        "events_at": queries.get("events_at", []),
        "time_for": queries.get("time_for"),
        "model_summary": model_io.summarize(meta),
        "caveats": project.CAVEATS,
    }
    (job_dir / "results.json").write_text(json.dumps(payload, indent=2, default=str),
                                          encoding="utf-8")
    with zipfile.ZipFile(job_dir / "results.zip", "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("report.html", html_report)
        zf.writestr("projection.csv", csv_text)
        zf.writestr("results.json", json.dumps(payload, indent=2, default=str))
        zf.writestr("model.json", json.dumps(meta, indent=2, default=str))
        for name, data in images.items():
            zf.writestr(f"plots/{name}.png", data)

    return {**payload,
            "_report_path": str(job_dir / "report.html"),
            "_zip_path": str(job_dir / "results.zip"),
            "_projection_path": str(job_dir / "projection.csv")}
