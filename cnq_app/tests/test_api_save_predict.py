"""API payload-level tests for saving models and predicting.

These boot the real HTTP server and send **exactly the JSON the browser sends**
(see static/app.js) so a UI-vs-API field/name mismatch can't slip through again.
A single multi-model, 2-split run is trained once for the module, then every
save/predict request is exercised against it.
"""
import json
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

APP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_DIR))

import paths  # noqa: E402
from deepquantreg.config import load_config  # noqa: E402

FEATURES = [f"feat_{i}" for i in range(12)]


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


def _http(url, data=None, headers=None, method=None):
    """Return (status, body_bytes) without raising on 4xx/5xx."""
    req = urllib.request.Request(url, data=data, headers=headers or {}, method=method)
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def _post(base, path, obj):
    return _http(base + path, data=json.dumps(obj).encode(),
                 headers={"Content-Type": "application/json"}, method="POST")


def _poll(base, job_id, timeout=300):
    deadline = time.time() + timeout
    while time.time() < deadline:
        st, body = _http(f"{base}/api/status?id={job_id}")
        s = json.loads(body)
        if s["state"] in ("done", "cancelled", "error"):
            return s
        time.sleep(0.3)
    raise AssertionError("job timed out")


@pytest.fixture(scope="module")
def trained_server():
    from http.server import ThreadingHTTPServer
    from server import Handler

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    threading.Thread(target=httpd.serve_forever, daemon=True).start()

    # upload the sample and train two models across two splits
    st, body = _http(f"{base}/api/upload", data=paths.SAMPLE_DATA.read_bytes(),
                     headers={"Content-Type": "text/csv", "X-Filename": "sample.csv"})
    assert st == 200, body
    csv_path = json.loads(body)["csv_path"]

    cfg = {
        "csv_path": csv_path, "duration_col": "survival_time", "event_col": "died",
        "feature_cols": FEATURES, "models": ["KAN_gaps", "TransformerPS_gaps"],
        "quantiles": [0.1, 0.25, 0.5, 0.75, 0.9], "ratio": [65, 15, 20], "n_splits": 2,
        "seed": 42, "deterministic": False, "mode": "custom", "custom": _smoke_custom(),
        "preset": None,
    }
    st, body = _post(base, "/api/run", cfg)
    assert st == 200, body
    run_id = json.loads(body)["job_id"]
    assert _poll(base, run_id)["state"] == "done"
    yield base, run_id, csv_path
    httpd.shutdown()


def _save(base, run_id, model, bundle_type, name, split_index=None):
    """Exactly the payload static/app.js onSave() sends."""
    body = {"source_job_id": run_id, "model": model, "bundle_type": bundle_type, "name": name}
    if split_index is not None:
        body["split_index"] = split_index
    return _post(base, "/api/save", body)


def _saved_names(base):
    st, body = _http(f"{base}/api/models")
    return [m["name"] for m in json.loads(body)["models"]]


# --------------------------------------------------------------------------- #
# save: all three bundle types with a multi-model run
# --------------------------------------------------------------------------- #
def test_save_final_bundle(trained_server):
    base, run_id, _ = trained_server
    st, body = _save(base, run_id, "KAN_gaps", "final", "api-final")
    assert st == 200, body
    assert _poll(base, json.loads(body)["job_id"])["state"] == "done"
    assert "api-final" in _saved_names(base)
    st, blob = _http(f"{base}/api/models/download?name=api-final")
    assert st == 200 and blob[:2] == b"PK"  # a real zip


def test_save_ensemble_bundle(trained_server):
    base, run_id, _ = trained_server
    st, body = _save(base, run_id, "TransformerPS_gaps", "ensemble", "api-ensemble")
    assert st == 200, body
    s = _poll(base, json.loads(body)["job_id"])
    assert s["state"] == "done", s
    st, res = _http(f"{base}/api/results?id={json.loads(body)['job_id']}")
    assert json.loads(res)["summary"]["n_members"] == 2  # both splits bundled
    assert "api-ensemble" in _saved_names(base)


def test_save_single_split_bundle(trained_server):
    base, run_id, _ = trained_server
    st, body = _save(base, run_id, "KAN_gaps", "single_split", "api-split1", split_index=1)
    assert st == 200, body
    assert _poll(base, json.loads(body)["job_id"])["state"] == "done"
    assert "api-split1" in _saved_names(base)


def test_save_accepts_display_name(trained_server):
    # "KAN-CNQ" is the UI label; the server must map it back to KAN_gaps.
    base, run_id, _ = trained_server
    st, body = _save(base, run_id, "KAN-CNQ", "single_split", "api-display", split_index=0)
    assert st == 200, body
    assert _poll(base, json.loads(body)["job_id"])["state"] == "done"
    assert "api-display" in _saved_names(base)


# --------------------------------------------------------------------------- #
# save: early 400s before any job starts
# --------------------------------------------------------------------------- #
def test_save_missing_model_is_400(trained_server):
    base, run_id, _ = trained_server
    st, body = _post(base, "/api/save",
                     {"source_job_id": run_id, "bundle_type": "final", "name": "x"})
    assert st == 400 and b"model" in body


def test_save_unknown_model_is_400(trained_server):
    base, run_id, _ = trained_server
    st, body = _save(base, run_id, "NotARealModel", "final", "x")
    assert st == 400 and b"unknown model" in body


def test_save_model_not_in_run_is_400(trained_server):
    # MLP_singleQ is a real model but wasn't trained in this run.
    base, run_id, _ = trained_server
    st, body = _save(base, run_id, "MLP_singleQ", "final", "x")
    assert st == 400


def test_save_bad_split_index_is_400(trained_server):
    base, run_id, _ = trained_server
    st, body = _save(base, run_id, "KAN_gaps", "single_split", "x", split_index=9)
    assert st == 400 and b"range" in body


def test_save_bad_bundle_type_is_400(trained_server):
    base, run_id, _ = trained_server
    st, body = _save(base, run_id, "KAN_gaps", "nonsense", "x")
    assert st == 400


def test_save_missing_source_is_400(trained_server):
    base, _, _ = trained_server
    st, body = _post(base, "/api/save",
                     {"model": "KAN_gaps", "bundle_type": "final", "name": "x"})
    assert st == 400


# --------------------------------------------------------------------------- #
# predict: exact app.js payloads, saved model and uploaded file
# --------------------------------------------------------------------------- #
def _ensure_saved(base, run_id, name="api-pred"):
    if name not in _saved_names(base):
        st, body = _save(base, run_id, "KAN_gaps", "ensemble", name)
        assert st == 200, body
        assert _poll(base, json.loads(body)["job_id"])["state"] == "done"
    return name


def test_predict_with_saved_model(trained_server):
    base, run_id, _ = trained_server
    name = _ensure_saved(base, run_id)

    st, body = _http(f"{base}/api/predict/upload-data", data=paths.SAMPLE_DATA.read_bytes(),
                     headers={"Content-Type": "text/csv", "X-Filename": "new.csv"})
    assert st == 200, body
    pdata = json.loads(body)

    payload = {
        "model_ref": {"name": name}, "csv_path": pdata["csv_path"],
        "feature_map": {f: f for f in FEATURES},
        "id_col": None, "time_col": "survival_time", "event_col": "died",
    }
    st, body = _post(base, "/api/predict/validate", payload)
    assert st == 200, body
    v = json.loads(body)
    assert v["errors"] == [] and v["summary"]["has_external"] is True

    st, body = _post(base, "/api/predict/run", payload)
    assert st == 200, body
    pid = json.loads(body)["job_id"]
    assert _poll(base, pid)["state"] == "done"

    st, res = _http(f"{base}/api/results?id={pid}")
    r = json.loads(res)
    assert r["n"] > 0 and r["external"]["available"] is True

    st, csv = _http(f"{base}/api/predictions?id={pid}")
    assert st == 200 and b"q_0.5" in csv and b"out_of_range" in csv


def test_predict_with_uploaded_model(trained_server):
    base, run_id, _ = trained_server
    name = _ensure_saved(base, run_id)

    st, blob = _http(f"{base}/api/models/download?name={name}")
    assert st == 200
    st, body = _http(f"{base}/api/predict/upload-model", data=blob,
                     headers={"Content-Type": "application/octet-stream",
                              "X-Filename": f"{name}.cnqmodel"})
    assert st == 200, body
    model_path = json.loads(body)["model_path"]

    st, body = _http(f"{base}/api/predict/upload-data", data=paths.SAMPLE_DATA.read_bytes(),
                     headers={"Content-Type": "text/csv", "X-Filename": "new.csv"})
    csv_path = json.loads(body)["csv_path"]

    payload = {
        "model_ref": {"path": model_path}, "csv_path": csv_path,
        "feature_map": {f: f for f in FEATURES},
        "id_col": None, "time_col": None, "event_col": None,
    }
    st, body = _post(base, "/api/predict/validate", payload)
    assert st == 200 and json.loads(body)["errors"] == []
    st, body = _post(base, "/api/predict/run", payload)
    assert st == 200, body
    assert _poll(base, json.loads(body)["job_id"])["state"] == "done"


def test_predict_missing_feature_is_flagged(trained_server):
    # Drop a required feature from the mapping -> validation error, no job.
    base, run_id, _ = trained_server
    name = _ensure_saved(base, run_id)
    st, body = _http(f"{base}/api/predict/upload-data", data=paths.SAMPLE_DATA.read_bytes(),
                     headers={"Content-Type": "text/csv", "X-Filename": "new.csv"})
    csv_path = json.loads(body)["csv_path"]
    payload = {
        "model_ref": {"name": name}, "csv_path": csv_path,
        "feature_map": {f: f for f in FEATURES[1:]},  # missing feat_0
        "id_col": None, "time_col": None, "event_col": None,
    }
    st, body = _post(base, "/api/predict/validate", payload)
    assert st == 200
    assert any(e["code"] == "missing_features" for e in json.loads(body)["errors"])
