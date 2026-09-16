"""Bundle format round-trip and corruption/incompatibility handling.

These need torch + deepquantreg, so they run in the app's virtualenv (or any
env with the scientific stack) on a machine where deepcnq is available.
"""
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

APP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_DIR))

import model_io  # noqa: E402
import save_model  # noqa: E402
from deepquantreg import build_model  # noqa: E402
from deepquantreg.training import predict_quantiles  # noqa: E402

FEATURES = ["a", "b", "c", "d"]
QUANTILES = [0.1, 0.5, 0.9]


def _build_args(input_dim=4):
    return {"name": "MLP_multiQ_gaps", "input_dim": input_dim, "quantiles": list(QUANTILES),
            "hidden_dim": 16, "layers": 2, "dropout": 0.0, "activation": "relu",
            "nhead": 4, "grid_size": 5}


def _make_bundle(path, build_args=None):
    build_args = build_args or _build_args()
    model = build_model(build_args)
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    scaler = {"mean": [0.0] * len(FEATURES), "scale": [1.0] * len(FEATURES)}
    save_model.build_bundle(
        path, bundle_type="single_split", model_name="MLP_multiQ_gaps",
        build_args=build_args, feature_names=FEATURES, quantiles=QUANTILES,
        state_dicts=[state], scalers=[scaler],
        feature_ranges={f: {"min": -3.0, "max": 3.0} for f in FEATURES},
        training={"n": 100, "events": 50, "censoring_pct": 50.0},
        event_mapping=None, metrics={}, time_unit="months", seed=0)
    return model


def _rewrite(src, dst, mutate, fix_manifest=True):
    with zipfile.ZipFile(src) as z:
        data = {n: z.read(n) for n in z.namelist()}
    meta = json.loads(data["model.json"])
    mutate(meta)
    mb = json.dumps(meta).encode()
    data["model.json"] = mb
    if fix_manifest:
        man = json.loads(data["manifest.json"])
        man["files"]["model.json"] = model_io._sha256(mb)
        data["manifest.json"] = json.dumps(man).encode()
    with zipfile.ZipFile(dst, "w") as z:
        for n, b in data.items():
            z.writestr(n, b)


# --------------------------------------------------------------------------- #
def test_round_trip_predictions_match(tmp_path):
    path = tmp_path / "m.cnqmodel"
    model = _make_bundle(path)
    loaded = model_io.load_bundle(path)
    assert len(loaded.models) == 1
    x = np.random.RandomState(0).randn(7, len(FEATURES)).astype("float32")
    a = predict_quantiles(model, x)
    b = predict_quantiles(loaded.models[0], x)
    assert np.allclose(a, b, atol=1e-6)


def test_meta_has_autofilled_fields(tmp_path):
    path = tmp_path / "m.cnqmodel"
    _make_bundle(path)
    meta = model_io.read_meta(path)
    assert meta["format_version"] == model_io.FORMAT_VERSION
    assert meta["log_time"]["inverse"] == "exp"
    assert "torch" in meta["versions"] and "numpy" in meta["versions"]
    assert meta["created_at"]
    assert meta["bundle"]["type"] == "single_split" and meta["bundle"]["n_members"] == 1
    assert meta["member_scalers"] and len(meta["member_scalers"]) == 1


def test_tampered_model_json_is_rejected(tmp_path):
    path = tmp_path / "m.cnqmodel"
    _make_bundle(path)
    bad = tmp_path / "bad.cnqmodel"
    # Change model.json but DON'T fix the manifest hash -> corruption detected.
    _rewrite(path, bad, lambda m: m.update(seed=999), fix_manifest=False)
    with pytest.raises(model_io.BundleError, match="corrupt"):
        model_io.load_bundle(bad)


def test_incompatible_format_version_is_rejected(tmp_path):
    path = tmp_path / "m.cnqmodel"
    _make_bundle(path)
    future = tmp_path / "future.cnqmodel"
    _rewrite(path, future, lambda m: m.update(format_version=999), fix_manifest=True)
    with pytest.raises(model_io.BundleError, match="format version"):
        model_io.load_bundle(future)


def test_bad_zip_is_rejected(tmp_path):
    junk = tmp_path / "junk.cnqmodel"
    junk.write_bytes(b"not a zip file")
    with pytest.raises(model_io.BundleError):
        model_io.read_meta(junk)


def test_weights_mismatch_is_rejected(tmp_path):
    # A bundle whose build_args declare a different input_dim than the weights.
    path = tmp_path / "m.cnqmodel"
    _make_bundle(path, build_args=_build_args(input_dim=4))
    mismatched = tmp_path / "mm.cnqmodel"
    _rewrite(path, mismatched,
             lambda m: m["build_args"].update(input_dim=6), fix_manifest=True)
    with pytest.raises(model_io.BundleError):
        model_io.load_bundle(mismatched)
