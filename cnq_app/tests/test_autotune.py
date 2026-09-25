"""Auto-tune: data-driven defaults, search space, and the search itself.

Needs torch + deepcnq (imports pipeline/presets/autotune). Runs on a machine with
the scientific stack; the search trains tiny models, so it takes a little while.
"""
import sys
from pathlib import Path

import pytest

APP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_DIR))

import autotune  # noqa: E402
import grids  # noqa: E402
import pipeline  # noqa: E402
import presets  # noqa: E402

SAMPLE = APP_DIR / "sample_data" / "sample.csv"
FEATURES = [f"feat_{i}" for i in range(12)]


def _frame():
    raw = pipeline.load_frame(SAMPLE)
    return pipeline.build_frame(raw, "survival_time", "died", FEATURES)


# --------------------------------------------------------------------------- #
# heuristic + space
# --------------------------------------------------------------------------- #
def test_defaults_scale_with_n():
    assert presets.defaults(300)["hidden_dim"] == 32
    assert presets.defaults(300)["dropout"] > presets.defaults(5000)["dropout"]
    assert presets.defaults(1500)["hidden_dim"] == 64
    assert presets.defaults(5000)["layers"] == 3


def test_auto_space_excludes_transformers_by_default():
    sp = presets.auto_space(1500, 12)
    assert set(sp) == {"KAN_gaps", "MLP_multiQ_gaps"}
    with_tx = presets.auto_space(1500, 12, ["KAN_gaps", "TransformerPS_gaps"])
    assert "TransformerPS_gaps" in with_tx
    assert set(presets.auto_space(1500, 12, ["nonsense"])) == {"KAN_gaps", "MLP_multiQ_gaps"}


def test_mlp_cnq_is_selectable_and_trainable():
    assert "MLP_multiQ_gaps" in presets.APP_MODELS
    # validate_run (server) accepts it now
    from server import validate_run
    cfg = {"models": ["MLP_multiQ_gaps"], "quantiles": [0.1, 0.5, 0.9],
           "duration_col": "t", "event_col": "e", "feature_cols": ["a"],
           "ratio": [65, 15, 20], "n_splits": 1, "mode": "custom", "custom": {},
           "csv_path": str(SAMPLE)}
    errors = validate_run(cfg)
    assert not any("unknown model" in e for e in errors)


# --------------------------------------------------------------------------- #
# the search
# --------------------------------------------------------------------------- #
def test_run_search_ranks_and_restricts():
    res = autotune.run_search(_frame(), FEATURES, grids.build_grid("standard"), (65, 15, 20), 42,
                              enabled_models=["KAN_gaps", "MLP_multiQ_gaps"], n_trials=4,
                              n_splits=1, deterministic=True)
    lb = res["leaderboard"]
    assert lb and res["best"] == lb[0] and lb[0]["rank"] == 1
    assert all(lb[i]["valid_pinball"] <= lb[i + 1]["valid_pinball"] for i in range(len(lb) - 1))
    assert set(t["model"] for t in lb) <= {"KAN_gaps", "MLP_multiQ_gaps"}
    # the winner's reported config uses the full epoch budget, not the search value
    assert res["best"]["custom"]["maximum_epochs"] >= 200


def test_run_search_cancel_returns_partial():
    calls = {"n": 0}

    def cancel():
        calls["n"] += 1
        return calls["n"] > 2   # cancel after a couple of checks

    res = autotune.run_search(_frame(), FEATURES, grids.build_grid("standard"), (65, 15, 20), 42,
                              enabled_models=["MLP_multiQ_gaps"], n_trials=10, n_splits=1,
                              deterministic=True, cancel=cancel)
    assert len(res["leaderboard"]) < 10   # stopped early, best-so-far still returned
