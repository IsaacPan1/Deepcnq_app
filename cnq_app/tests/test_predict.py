"""End-to-end save -> reload -> predict tests on the sample data.

Trains tiny models (few epochs) once per module, bundles each type, reloads and
checks predictions reproduce the in-session models. Needs torch + deepquantreg.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

APP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_DIR))

import grids  # noqa: E402
import model_io  # noqa: E402
import pipeline  # noqa: E402
import predict  # noqa: E402
import presets  # noqa: E402
import save_model  # noqa: E402
from deepquantreg import build_model  # noqa: E402
from deepquantreg.training import predict_quantiles  # noqa: E402

SAMPLE = APP_DIR / "sample_data" / "sample.csv"
MODEL = "MLP_multiQ_gaps"   # gaps => non-crossing; MLP => fast to train
FEATURES = [f"feat_{i}" for i in range(12)]
CUSTOM = {"hidden_dim": 16, "layers": 2, "dropout": 0.0, "grid_size": 5,
          "learning_rate": 1e-3, "weight_decay": 0.0, "batch_size": 64,
          "maximum_epochs": 3, "patience": 5}


@pytest.fixture(scope="module")
def trained():
    raw = pipeline.load_frame(SAMPLE)
    frame = pipeline.build_frame(raw, "survival_time", "died", FEATURES)
    quantiles = grids.build_grid("standard")
    resolved = presets.resolve(MODEL, preset=None, custom=CUSTOM, quantiles=quantiles)
    seed, ratio = 42, (65, 15, 20)
    outs, preps = [], []
    for s in range(2):
        prepared = pipeline.prepare(frame, FEATURES, ratio, seed + s)
        out = pipeline.train_model_on_split(MODEL, resolved, prepared, seed + s, s,
                                            deterministic=True, keep_for_plots=True)
        outs.append(out)
        preps.append(prepared)
    return SimpleNamespace(raw=raw, frame=frame, quantiles=quantiles, resolved=resolved,
                           seed=seed, outs=outs, preps=preps)


def _build_args(tr):
    return save_model.build_args_for(MODEL, tr.resolved, len(FEATURES), tr.quantiles)


def _common(tr):
    return dict(model_name=MODEL, build_args=_build_args(tr), feature_names=FEATURES,
                quantiles=tr.quantiles,
                feature_ranges=save_model.feature_ranges(tr.frame, FEATURES),
                training=save_model.training_stats(tr.frame), event_mapping=None,
                metrics={}, time_unit=None, seed=tr.seed)


def _single_bundle(tr, path, idx=0):
    out, prepared = tr.outs[idx], tr.preps[idx]
    save_model.build_bundle(path, bundle_type="single_split", state_dicts=[out.state_dict],
                            scalers=[save_model.scaler_dict(prepared.scaler)], **_common(tr))
    return path


def _ensemble_bundle(tr, path):
    save_model.build_bundle(
        path, bundle_type="ensemble",
        state_dicts=[o.state_dict for o in tr.outs],
        scalers=[save_model.scaler_dict(p.scaler) for p in tr.preps], **_common(tr))
    return path


def _test_rows(tr, prepared):
    ids = [int(i) for i in prepared.test["subject_id"]]
    return tr.frame.iloc[ids][FEATURES].reset_index(drop=True)


# --------------------------------------------------------------------------- #
def test_single_split_round_trip(trained, tmp_path):
    tr = trained
    path = _single_bundle(tr, tmp_path / "s.cnqmodel")
    loaded = model_io.load_bundle(path)
    a = predict_quantiles(tr.outs[0].keep_model, tr.preps[0].test["X"])
    b = predict_quantiles(loaded.models[0], tr.preps[0].test["X"])
    assert np.allclose(a, b, atol=1e-6)


def test_column_order_invariance(trained, tmp_path):
    tr = trained
    loaded = model_io.load_bundle(_single_bundle(tr, tmp_path / "s.cnqmodel"))
    raw = _test_rows(tr, tr.preps[0])
    r1 = predict.run_prediction(loaded, raw, {})
    r2 = predict.run_prediction(loaded, raw[list(reversed(FEATURES))], {})
    assert np.allclose(np.asarray(r1["pred_time"]), np.asarray(r2["pred_time"]), atol=1e-6)


def test_scaling_uses_training_scaler(trained, tmp_path):
    tr = trained
    loaded = model_io.load_bundle(_single_bundle(tr, tmp_path / "s.cnqmodel"))
    raw = _test_rows(tr, tr.preps[0])
    r = predict.run_prediction(loaded, raw, {})
    manual = predict_quantiles(
        loaded.models[0],
        tr.preps[0].scaler.transform(raw[FEATURES].to_numpy(float)).astype("float32"))
    assert np.allclose(np.log(np.asarray(r["pred_time"])), manual, atol=1e-6)


def test_single_row_matches_batch(trained, tmp_path):
    tr = trained
    loaded = model_io.load_bundle(_single_bundle(tr, tmp_path / "s.cnqmodel"))
    raw = _test_rows(tr, tr.preps[0])
    r_all = predict.run_prediction(loaded, raw, {})
    r_one = predict.run_prediction(loaded, raw.iloc[[0]].reset_index(drop=True), {})
    assert np.allclose(np.asarray(r_one["pred_time"])[0],
                       np.asarray(r_all["pred_time"])[0], atol=1e-6)


def test_ensemble_round_trip_and_non_crossing(trained, tmp_path):
    tr = trained
    loaded = model_io.load_bundle(_ensemble_bundle(tr, tmp_path / "e.cnqmodel"))
    assert len(loaded.models) == 2
    raw = _test_rows(tr, tr.preps[0])
    r = predict.run_prediction(loaded, raw, {})

    # in-session ensemble: each member standardised with its own scaler, averaged on log scale
    raw_x = raw[FEATURES].to_numpy(float)
    members = [predict_quantiles(o.keep_model, p.scaler.transform(raw_x).astype("float32"))
               for o, p in zip(tr.outs, tr.preps)]
    manual = np.mean(members, axis=0)
    assert np.allclose(np.log(np.asarray(r["pred_time"])), manual, atol=1e-6)

    # non-crossing: predicted times non-decreasing across tau for every subject
    pt = np.asarray(r["pred_time"])
    assert np.all(np.diff(pt, axis=1) >= -1e-6)


def test_final_refit_round_trip(trained, tmp_path):
    tr = trained
    refit = save_model.refit_final(MODEL, tr.frame, FEATURES, tr.resolved, tr.quantiles, tr.seed,
                                   deterministic=True)
    path = tmp_path / "f.cnqmodel"
    common = _common(tr)
    common["build_args"] = refit["build_args"]
    save_model.build_bundle(path, bundle_type="final", state_dicts=[refit["state_dict"]],
                            scalers=[refit["scaler"]], **common)
    loaded = model_io.load_bundle(path)
    ref = build_model(refit["build_args"])
    ref.load_state_dict(refit["state_dict"], strict=True)
    ref.eval()
    x = np.random.RandomState(0).randn(6, len(FEATURES)).astype("float32")
    assert np.allclose(predict_quantiles(ref, x), predict_quantiles(loaded.models[0], x), atol=1e-6)


def test_validate_missing_feature(trained, tmp_path):
    tr = trained
    meta = model_io.read_meta(_single_bundle(tr, tmp_path / "s.cnqmodel"))
    raw = _test_rows(tr, tr.preps[0]).drop(columns=[FEATURES[0]])
    res = predict.validate(meta, raw)
    assert any(e["code"] == "missing_features" for e in res["errors"])


def test_validate_text_value_in_feature(trained, tmp_path):
    tr = trained
    meta = model_io.read_meta(_single_bundle(tr, tmp_path / "s.cnqmodel"))
    raw = _test_rows(tr, tr.preps[0]).copy()
    raw[FEATURES[0]] = "not-a-number"
    res = predict.validate(meta, raw)
    assert any(e["code"] == "feature_not_numeric" for e in res["errors"])


def test_auto_map_exact_names(trained, tmp_path):
    tr = trained
    meta = model_io.read_meta(_single_bundle(tr, tmp_path / "s.cnqmodel"))
    raw = _test_rows(tr, tr.preps[0])                      # columns are the exact feature names
    res = predict.validate(meta, raw)
    assert res["summary"]["n_matched"] == len(FEATURES)
    assert not any(e["code"] == "missing_features" for e in res["errors"])


def test_renamed_columns_with_manual_map_identical(trained, tmp_path):
    tr = trained
    loaded = model_io.load_bundle(_single_bundle(tr, tmp_path / "s.cnqmodel"))
    raw = _test_rows(tr, tr.preps[0])
    rename = {f: f"col{i}" for i, f in enumerate(FEATURES)}
    renamed = raw.rename(columns=rename)
    manual = {f: rename[f] for f in FEATURES}
    a = predict.run_prediction(loaded, raw, {})
    b = predict.run_prediction(loaded, renamed, manual)
    assert np.allclose(np.asarray(a["pred_time"]), np.asarray(b["pred_time"]), atol=1e-6)


def test_position_mapping_identical(trained, tmp_path):
    tr = trained
    loaded = model_io.load_bundle(_single_bundle(tr, tmp_path / "s.cnqmodel"))
    raw = _test_rows(tr, tr.preps[0])
    ordered = raw[FEATURES].copy()                          # same order, non-matching names
    ordered.columns = [f"z{i}" for i in range(len(FEATURES))]
    posmap = {f: f"z{i}" for i, f in enumerate(FEATURES)}   # what "match by position" produces
    a = predict.run_prediction(loaded, raw, {})
    b = predict.run_prediction(loaded, ordered, posmap)
    assert np.allclose(np.asarray(a["pred_time"]), np.asarray(b["pred_time"]), atol=1e-6)


def test_duplicate_mapping_is_rejected(trained, tmp_path):
    tr = trained
    path = _single_bundle(tr, tmp_path / "s.cnqmodel")
    meta = model_io.read_meta(path)
    loaded = model_io.load_bundle(path)
    raw = _test_rows(tr, tr.preps[0])
    dup = {FEATURES[0]: "feat_0", FEATURES[1]: "feat_0"}   # two features -> one column
    res = predict.validate(meta, raw, dup)
    assert any(e["code"] == "duplicate_mapping" for e in res["errors"])
    with pytest.raises(ValueError):
        predict.run_prediction(loaded, raw, dup)


def test_external_validation_runs(trained, tmp_path):
    tr = trained
    loaded = model_io.load_bundle(_single_bundle(tr, tmp_path / "s.cnqmodel"))
    r = predict.run_prediction(loaded, tr.raw, {}, time_col="survival_time", event_col="died")
    ext = r["external"]
    assert ext and ext["available"]
    assert ext["pinball_mean"] is not None
    assert ext["coverage_80"] is not None
    assert ext["calibration"] and str(0.5) in ext["calibration"]
    assert len(ext["predicted_survival"]["time"]) == len(ext["predicted_survival"]["survival"])
