"""End-to-end smoke test for the CNQ web app.

Drives the real JobManager (background thread, per-epoch progress hook, plots,
report, zip) on the shipped sample data using the smoke config's tiny
architecture so it finishes in seconds on CPU.
"""
import sys
import time
import zipfile
from pathlib import Path

import pytest

APP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_DIR))

import paths  # noqa: E402
from deepquantreg.config import load_config  # noqa: E402


def _smoke_custom():
    cfg = load_config(paths.SMOKE_CONFIG)
    arch, train = cfg["architecture"], cfg["training"]
    return {
        "hidden_dim": arch["hidden_dim"], "layers": arch["layers"],
        "dropout": arch["dropout"], "grid_size": arch["grid_size"],
        "learning_rate": train["learning_rate"], "weight_decay": train["weight_decay"],
        "batch_size": train["batch_size"], "maximum_epochs": train["maximum_epochs"],
        "patience": train["patience"],
    }


def _run_job(manager, config, timeout=180):
    job = manager.start(config)
    deadline = time.time() + timeout
    while job.state in ("pending", "running"):
        if time.time() > deadline:
            manager.cancel(job.id)
            raise AssertionError(f"job timed out in state {job.state} / {job.step}")
        time.sleep(0.2)
    return job


def _base_config(models, n_splits=2):
    return {
        "csv_path": str(paths.SAMPLE_DATA),
        "duration_col": "survival_time",
        "event_col": "died",
        "feature_cols": [f"feat_{i}" for i in range(12)],
        "models": models,
        "quantiles": [0.1, 0.25, 0.5, 0.75, 0.9],
        "ratio": [65, 15, 20],
        "n_splits": n_splits,
        "seed": 42,
        "deterministic": False,
        "mode": "custom",
        "custom": _smoke_custom(),
        "preset": None,
    }


def test_end_to_end_smoke():
    from jobs import JobManager

    assert paths.SAMPLE_DATA.exists(), "sample data must be generated"
    manager = JobManager()
    models = ["KAN_gaps", "TransformerPS_gaps", "MLP_multiQ"]
    job = _run_job(manager, _base_config(models, n_splits=2))

    assert job.state == "done", f"job failed: {job.error}"
    results = job.results
    assert results is not None

    # per-model summaries with the required metrics
    for m in models:
        s = results["summaries"][m]
        assert s["pinball_mean"]["mean"] is not None
        assert len(s["pinball_per_tau"]) == 5
        assert s["icp_80"]["mean"] is not None
        assert s["adjacent_crossing"]["mean"] is not None
        assert len(s["pinball_per_split"]) == 2  # repeated splits
    # gap models must not cross (non-crossing parameterization)
    assert results["summaries"]["KAN_gaps"]["adjacent_crossing"]["mean"] == 0.0

    # Uno C available (median quantile present)
    assert results["summaries"]["KAN_gaps"]["uno_c"]["mean"] is not None

    # artifacts written
    assert (job.dir / "report.html").exists()
    assert (job.dir / "results.zip").exists()
    for name in ["km_curve", "training_curves", "calibration", "quantile_profiles",
                 "coverage_by_width", "importance_heatmap", "pinball_boxplots"]:
        assert (job.dir / "plots" / f"{name}.png").exists(), f"missing plot {name}"

    # zip contains report + plots
    with zipfile.ZipFile(job.dir / "results.zip") as zf:
        names = zf.namelist()
        assert "report.html" in names
        assert "results.json" in names
        assert any(n.startswith("plots/") for n in names)

    # report embeds base64 images and is self-contained
    html = (job.dir / "report.html").read_text(encoding="utf-8")
    assert "data:image/png;base64," in html


def test_quantile_validation_rejects_missing_required():
    from server import validate_run

    cfg = _base_config(["KAN_gaps"], n_splits=1)
    cfg["quantiles"] = [0.2, 0.5, 0.8]  # missing 0.1 and 0.9
    errors = validate_run(cfg)
    assert any("0.1" in e for e in errors)
    assert any("0.9" in e for e in errors)


def test_cancellation():
    from jobs import JobManager

    manager = JobManager()
    cfg = _base_config(["Transformer_KAN_gaps"], n_splits=3)
    cfg["custom"]["maximum_epochs"] = 500  # long enough to cancel mid-run
    cfg["custom"]["patience"] = 500
    job = manager.start(cfg)
    # wait until training actually starts
    for _ in range(200):
        if job.phase == "train" and job.epoch > 0:
            break
        time.sleep(0.1)
    assert manager.cancel(job.id)
    for _ in range(300):
        if job.state in ("cancelled", "done", "error"):
            break
        time.sleep(0.1)
    assert job.state == "cancelled", f"expected cancelled, got {job.state}"


# --------------------------------------------------------------------------- #
# API-level end-to-end (boots the real HTTP server in a thread)
# --------------------------------------------------------------------------- #
def _boot_server():
    import threading
    from http.server import ThreadingHTTPServer
    from server import Handler

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd, port


def _http_json(url, data=None, headers=None, method=None):
    import json as _json
    import urllib.request

    req = urllib.request.Request(url, data=data, headers=headers or {}, method=method)
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.status, resp.read(), resp.headers.get("Content-Type")


def test_api_end_to_end():
    import json as _json
    import urllib.request

    httpd, port = _boot_server()
    base = f"http://127.0.0.1:{port}"
    try:
        # index + static served from static/
        status, body, ctype = _http_json(f"{base}/")
        assert status == 200 and b"Censored Non-crossing" in body
        status, _, _ = _http_json(f"{base}/app.js")
        assert status == 200

        # health reports everything present in this environment
        status, body, _ = _http_json(f"{base}/api/health")
        health = _json.loads(body)
        assert health["ok"], f"missing packages: {health['missing']}"

        # upload the sample CSV as a raw body
        csv_bytes = paths.SAMPLE_DATA.read_bytes()
        status, body, _ = _http_json(
            f"{base}/api/upload", data=csv_bytes,
            headers={"Content-Type": "text/csv", "X-Filename": "sample.csv"})
        info = _json.loads(body)
        assert info["n_rows"] == 300

        # run a tiny job via the API
        cfg = _base_config(["KAN_gaps", "MLP_multiQ"], n_splits=1)
        cfg["csv_path"] = info["csv_path"]
        status, body, _ = _http_json(
            f"{base}/api/run", data=_json.dumps(cfg).encode(),
            headers={"Content-Type": "application/json"})
        job_id = _json.loads(body)["job_id"]

        # poll status until done
        deadline = time.time() + 180
        state = None
        while time.time() < deadline:
            _, body, _ = _http_json(f"{base}/api/status?id={job_id}")
            s = _json.loads(body)
            state = s["state"]
            if state in ("done", "error", "cancelled"):
                break
            time.sleep(0.3)
        assert state == "done", f"job ended in state {state}"

        # results + downloads
        _, body, _ = _http_json(f"{base}/api/results?id={job_id}")
        results = _json.loads(body)
        assert results["summaries"]["KAN_gaps"]["pinball_mean"]["mean"] is not None

        status, body, ctype = _http_json(f"{base}/api/report?id={job_id}")
        assert status == 200 and b"data:image/png;base64," in body
        status, body, ctype = _http_json(f"{base}/api/zip?id={job_id}")
        assert status == 200 and ctype == "application/zip" and len(body) > 1000
    finally:
        httpd.shutdown()
        httpd.server_close()
