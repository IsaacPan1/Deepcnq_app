"""Demo assets: on-demand generation (no torch), the build-model job, and the
committed bundle. Data-generation tests redirect demo.COMMITTED_DIR / CACHE_DIR
to a temp folder so they never touch the committed assets.
"""
import json
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pandas as pd
import pytest

APP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_DIR))

import demo  # noqa: E402 (torch-free)
import model_io  # noqa: E402 (torch-light)
import paths  # noqa: E402


@pytest.fixture
def demo_dirs(monkeypatch, tmp_path):
    """Point demo generation at empty temp folders (committed + cache)."""
    committed, cache = tmp_path / "demo", tmp_path / "demo_cache"
    monkeypatch.setattr(demo, "COMMITTED_DIR", committed)
    monkeypatch.setattr(demo, "CACHE_DIR", cache)
    return committed, cache


def _http(url, data=None, headers=None, method=None):
    req = urllib.request.Request(url, data=data, headers=headers or {}, method=method)
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def _post(base, path, obj):
    return _http(base + path, data=json.dumps(obj).encode(),
                 headers={"Content-Type": "application/json"}, method="POST")


def _boot():
    from server import Handler
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, f"http://127.0.0.1:{httpd.server_address[1]}"


def _poll(base, job_id, timeout=300):
    end = time.time() + timeout
    while time.time() < end:
        st, body = _http(f"{base}/api/status?id={job_id}")
        s = json.loads(body)
        if s["state"] in ("done", "cancelled", "error"):
            return s
        time.sleep(0.3)
    raise AssertionError("job timed out")


# --------------------------------------------------------------------------- #
# data generation (no torch)
# --------------------------------------------------------------------------- #
def test_ensure_demo_data_is_byte_identical(demo_dirs):
    _, cache = demo_dirs
    r1 = demo.ensure_demo_data()
    assert set(r1["generated"]) == set(demo.CSV_FILES.values())
    first = {v: (cache / v).read_bytes() for v in demo.CSV_FILES.values()}

    for v in demo.CSV_FILES.values():
        (cache / v).unlink()
    demo.ensure_demo_data()
    for v in demo.CSV_FILES.values():
        assert (cache / v).read_bytes() == first[v], f"{v} not byte-identical across generations"

    # structure of the new-subjects file
    new = pd.read_csv(cache / demo.CSV_FILES["new"])
    assert len(new) == 20 and "note" in new.columns
    assert "time" not in new.columns and "event" not in new.columns
    assert new["note"].str.contains("out-of-range").sum() == 2


def test_concurrent_generation_produces_one_valid_set(demo_dirs):
    _, cache = demo_dirs
    errors = []

    def worker():
        try:
            demo.ensure_demo_data()
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors
    # every file exists exactly once and parses with the right row counts
    for key, name in demo.CSV_FILES.items():
        assert (cache / name).exists()
        pd.read_csv(cache / name)  # valid CSV
    assert len(pd.read_csv(cache / demo.CSV_FILES["train"])) == demo.N_TRAIN


def test_demo_endpoints_autogenerate_without_scripts(demo_dirs):
    httpd, base = _boot()
    try:
        st, body = _http(f"{base}/api/demo/train")
        assert st == 200 and b"make_demo" not in body and b"not found" not in body
        for key in ("new", "outcomes", "shifted"):
            st, body = _http(f"{base}/api/demo/data?name={key}")
            assert st == 200, body
            info = json.loads(body)
            assert info["n_rows"] > 0 and b"make_demo" not in body
        # files really were written to the (temp) cache
        for name in demo.CSV_FILES.values():
            assert (demo.CACHE_DIR / name).exists()
    finally:
        httpd.shutdown()


def test_sample_endpoint_regenerates(monkeypatch, tmp_path):
    # Point the committed sample at a missing path and the cache at temp.
    monkeypatch.setattr(paths, "SAMPLE_DATA", tmp_path / "nope.csv")
    monkeypatch.setattr(demo, "CACHE_DIR", tmp_path / "demo_cache")
    httpd, base = _boot()
    try:
        st, body = _http(f"{base}/api/sample")
        assert st == 200 and b"make_demo" not in body and len(body) > 100
    finally:
        httpd.shutdown()


# --------------------------------------------------------------------------- #
# build-model endpoint (needs torch)
# --------------------------------------------------------------------------- #
def test_build_endpoint_starts_job_attaches_and_registers(demo_dirs, monkeypatch):
    monkeypatch.setattr(demo, "DEFAULT_EPOCHS", 5)  # tiny; leaves a window to attach
    httpd, base = _boot()
    try:
        st, body = _http(f"{base}/api/demo/build", method="POST")
        assert st == 200, body
        job_id = json.loads(body)["job_id"]

        # a second build while the first runs must attach, not start a new job
        st, body2 = _http(f"{base}/api/demo/build", method="POST")
        assert st == 200
        again = json.loads(body2)
        assert again["job_id"] == job_id and again.get("attached") is True

        assert _poll(base, job_id)["state"] == "done"

        # bundle written to the temp cache, loads, and now appears without needs_build
        assert demo.resolve_model() is not None
        model_io.load_bundle(demo.resolve_model())
        st, body = _http(f"{base}/api/models")
        entry = next(m for m in json.loads(body)["models"] if m.get("is_demo"))
        assert not entry.get("needs_build") and entry.get("n_features") == 12
    finally:
        httpd.shutdown()


def test_demo_model_cannot_be_deleted():
    httpd, base = _boot()
    try:
        st, _ = _http(f"{base}/api/models?id={demo.DEMO_ID}", method="DELETE")
        assert st == 403
    finally:
        httpd.shutdown()


# ---- id resolution (torch-free: the model resolves before predict imports) ----
def _demo_csv(base):
    return json.loads(_http(f"{base}/api/demo/data?name=new")[1])["csv_path"]


def test_predict_unbuilt_demo_returns_400(demo_dirs):
    httpd, base = _boot()
    try:
        payload = {"model_id": demo.DEMO_ID, "csv_path": _demo_csv(base), "feature_map": {},
                   "id_col": "subject_id", "time_col": None, "event_col": None}
        st, body = _post(base, "/api/predict/run", payload)
        assert st == 400 and b"isn't built yet" in body
    finally:
        httpd.shutdown()


def test_predict_with_demo_label_as_id_is_rejected(demo_dirs):
    # Sending the display label instead of the id must be a clean 400.
    httpd, base = _boot()
    try:
        payload = {"model_id": demo.DEMO_LABEL, "csv_path": _demo_csv(base), "feature_map": {},
                   "id_col": "subject_id", "time_col": None, "event_col": None}
        st, body = _post(base, "/api/predict/run", payload)
        assert st == 400 and b"no longer exists" in body
    finally:
        httpd.shutdown()


def test_predict_demo_by_id_committed_and_cache(demo_dirs, monkeypatch):
    import shutil
    monkeypatch.setattr(demo, "DEFAULT_EPOCHS", 3)
    httpd, base = _boot()
    try:
        st, body = _http(f"{base}/api/demo/build", method="POST")
        assert st == 200 and _poll(base, json.loads(body)["job_id"])["state"] == "done"
        payload = {"model_id": demo.DEMO_ID, "csv_path": _demo_csv(base), "feature_map": {},
                   "id_col": "subject_id", "time_col": None, "event_col": None}

        st, body = _post(base, "/api/predict/run", payload)      # freshly built demo_cache bundle
        assert st == 200 and _poll(base, json.loads(body)["job_id"])["state"] == "done"

        committed = demo.COMMITTED_DIR / demo.MODEL_FILE          # now as a committed bundle
        committed.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(demo.cached(demo.MODEL_FILE)), str(committed))
        assert demo.demo_source() == "demo"
        st, body = _post(base, "/api/predict/run", payload)
        assert st == 200 and _poll(base, json.loads(body)["job_id"])["state"] == "done"
    finally:
        httpd.shutdown()


# --------------------------------------------------------------------------- #
# the committed bundle (present only after `python make_demo.py` was run)
# --------------------------------------------------------------------------- #
_COMMITTED = paths.DEMO_MODEL.exists() and (paths.DEMO_DIR / "demo_new_subjects.csv").exists()
needs_committed = pytest.mark.skipif(not _COMMITTED, reason="committed demo assets not present")


@needs_committed
def test_committed_bundle_predicts_20_flags_2():
    import predict
    bundle = model_io.load_bundle(paths.DEMO_MODEL)
    new = pd.read_csv(paths.DEMO_DIR / "demo_new_subjects.csv")
    r = predict.run_prediction(bundle, new, {}, id_col="subject_id")
    assert r["n"] == 20 and r["out_of_range_count"] == 2
    assert "note" not in r["columns"]


@needs_committed
def test_committed_bundle_external_validation_both_files():
    import predict
    bundle = model_io.load_bundle(paths.DEMO_MODEL)
    for name in ("demo_new_subjects_with_outcomes.csv", "demo_shifted_population.csv"):
        r = predict.run_prediction(bundle, pd.read_csv(paths.DEMO_DIR / name), {},
                                   id_col="subject_id", time_col="time", event_col="event")
        assert r["external"] and r["external"]["available"], name
