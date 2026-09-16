"""Demo assets: generation, the committed bundle, and delete protection.

`test_make_demo_quick` regenerates everything at tiny size/epoch count and
checks the generated bundle end to end. The committed-asset tests skip until
`make_demo.py` has been run and the files committed. Needs torch + deepcnq.
"""
import sys
import threading
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd
import pytest

APP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_DIR))

import make_demo  # noqa: E402
import model_io  # noqa: E402
import paths  # noqa: E402
import predict  # noqa: E402

DEMO = APP_DIR / "demo"
_COMMITTED = (paths.DEMO_MODEL.exists()
              and (DEMO / "demo_new_subjects.csv").exists()
              and (DEMO / "demo_new_subjects_with_outcomes.csv").exists()
              and (DEMO / "demo_shifted_population.csv").exists())
needs_committed = pytest.mark.skipif(
    not _COMMITTED, reason="demo assets not generated yet; run `python make_demo.py`")


# --------------------------------------------------------------------------- #
# generation (tiny)
# --------------------------------------------------------------------------- #
def test_make_demo_quick(tmp_path):
    info = make_demo.build_all(tmp_path, n_train=250, n_new=20, n_outcomes=150,
                               n_shifted=150, epochs=1, deterministic=True)
    for key in ("train", "new", "outcomes", "shifted", "model"):
        assert (tmp_path / make_demo.FILES[key]).exists(), f"missing {key}"
    assert (tmp_path / "README.md").exists()

    # new subjects: 20 rows, note column, NO outcomes, exactly 2 out-of-range notes
    new = pd.read_csv(tmp_path / make_demo.FILES["new"])
    assert len(new) == 20
    assert "note" in new.columns
    assert "time" not in new.columns and "event" not in new.columns
    assert new["note"].str.contains("out-of-range").sum() == 2

    # the generated bundle loads and predicts all 20, flagging exactly 2, ignoring `note`
    bundle = model_io.load_bundle(tmp_path / make_demo.FILES["model"])
    r = predict.run_prediction(bundle, new, {}, id_col="subject_id")
    assert r["n"] == 20
    assert r["out_of_range_count"] == 2
    assert "note" not in r["columns"]  # extra column ignored

    # external validation runs on BOTH outcome files
    for key in ("outcomes", "shifted"):
        frame = pd.read_csv(tmp_path / make_demo.FILES[key])
        res = predict.run_prediction(bundle, frame, {}, id_col="subject_id",
                                     time_col="time", event_col="event")
        assert res["external"] and res["external"]["available"], key

    # info dict carries a model size + timing (for DEMO.md)
    assert info["model"]["size_mb"] > 0 and info["model"]["seconds"] >= 0


# --------------------------------------------------------------------------- #
# the committed bundle
# --------------------------------------------------------------------------- #
@needs_committed
def test_committed_bundle_predicts_20_flags_2():
    bundle = model_io.load_bundle(paths.DEMO_MODEL)
    new = pd.read_csv(DEMO / "demo_new_subjects.csv")
    assert len(new) == 20 and "note" in new.columns
    r = predict.run_prediction(bundle, new, {}, id_col="subject_id")
    assert r["n"] == 20
    assert r["out_of_range_count"] == 2
    assert "note" not in r["columns"]


@needs_committed
def test_committed_bundle_external_validation_both_files():
    bundle = model_io.load_bundle(paths.DEMO_MODEL)
    for name in ("demo_new_subjects_with_outcomes.csv", "demo_shifted_population.csv"):
        frame = pd.read_csv(DEMO / name)
        r = predict.run_prediction(bundle, frame, {}, id_col="subject_id",
                                   time_col="time", event_col="event")
        assert r["external"] and r["external"]["available"], name
        assert r["external"]["coverage_80"] is not None


# --------------------------------------------------------------------------- #
# delete protection (API)
# --------------------------------------------------------------------------- #
def test_demo_model_cannot_be_deleted():
    from http.server import ThreadingHTTPServer
    from server import Handler, DEMO_MODEL_NAME

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        req = urllib.request.Request(
            f"{base}/api/models?name={urllib.parse.quote(DEMO_MODEL_NAME)}", method="DELETE")
        try:
            urllib.request.urlopen(req, timeout=30)
            raise AssertionError("delete should have been rejected")
        except urllib.error.HTTPError as exc:
            assert exc.code == 403
    finally:
        httpd.shutdown()
