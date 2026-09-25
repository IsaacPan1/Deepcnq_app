"""Tests for quantile-grid generation and a smoke run on a dense (5%) grid."""
import sys
import time
import zipfile
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_DIR))

import paths  # noqa: E402
import grids  # noqa: E402
import presets  # noqa: E402
from deepquantreg.config import load_config  # noqa: E402


# --------------------------------------------------------------------------- #
# grid generation
# --------------------------------------------------------------------------- #
def test_standard_grid():
    assert grids.build_grid("standard") == [0.1, 0.25, 0.5, 0.75, 0.9]


def test_every10_count_and_required():
    g = grids.build_grid("every10")
    assert len(g) == 9
    assert g == [round(i / 100, 4) for i in range(10, 91, 10)]
    for r in (0.1, 0.5, 0.9):
        assert r in g


def test_every5_count_and_boundaries():
    g = grids.build_grid("every5")
    assert len(g) == 19
    assert g[0] == 0.05 and g[-1] == 0.95
    assert not grids.has_extreme(g)  # 0.05 / 0.95 are boundaries, not "extreme"


def test_every1_count_and_extreme():
    g = grids.build_grid("every1")
    assert len(g) == 99
    assert grids.has_extreme(g)


def test_rounding_has_no_float_artefacts():
    g = grids.build_grid("every5")
    assert 0.15 in g
    for v in g:
        assert v == round(v, 4)
        assert len(repr(v)) <= 6  # e.g. not 0.15000000000000002


def test_custom_merges_required_dedupes_and_sorts():
    g = grids.build_grid("custom", custom=[0.3, 0.3, 0.7, 0.2])
    assert g == sorted({0.2, 0.3, 0.7, 0.1, 0.5, 0.9})
    for r in (0.1, 0.5, 0.9):
        assert r in g


def test_custom_drops_out_of_range():
    g = grids.build_grid("custom", custom=[0, 1, 1.5, -0.2, 0.4])
    assert g == sorted({0.4, 0.1, 0.5, 0.9})


def test_custom_rounds_artefact_input():
    g = grids.build_grid("custom", custom=[0.1 + 0.05])  # 0.15000000000000002
    assert 0.15 in g and 0.15000000000000002 not in [round(x, 12) for x in g if x != 0.15]


def test_resolve_prefers_grid_then_list():
    assert grids.resolve({"quantile_grid": {"kind": "every10"}}) == grids.build_grid("every10")
    assert grids.resolve({"quantiles": [0.2, 0.8]}) == sorted({0.2, 0.8, 0.1, 0.5, 0.9})
    assert grids.resolve({}) == grids.STANDARD


def test_standard_present():
    assert grids.standard_present(grids.build_grid("every10")) == [0.1, 0.5, 0.9]
    assert grids.standard_present(grids.build_grid("every5")) == [0.1, 0.25, 0.5, 0.75, 0.9]


# --------------------------------------------------------------------------- #
# data-driven defaults
# --------------------------------------------------------------------------- #
def test_defaults_scale_with_sample_size():
    small = presets.defaults(200)
    large = presets.defaults(5000)
    assert small["hidden_dim"] < large["hidden_dim"]
    assert small["dropout"] >= large["dropout"]      # more regularisation on small data
    assert small["batch_size"] <= large["batch_size"]


def test_tune_models_are_tabular_only():
    assert presets.TABULAR_MODELS == ("KAN_gaps", "MLP_multiQ_gaps")
    assert presets.TUNE_ALLOWED_MODELS == presets.TABULAR_MODELS
    for m in presets.TRANSFORMER_MODELS:
        assert m not in presets.TUNE_ALLOWED_MODELS


# --------------------------------------------------------------------------- #
# smoke run on the 5% grid
# --------------------------------------------------------------------------- #
def _smoke_custom():
    cfg = load_config(paths.SMOKE_CONFIG)
    a, t = cfg["architecture"], cfg["training"]
    return {"hidden_dim": a["hidden_dim"], "layers": a["layers"], "dropout": a["dropout"],
            "grid_size": a["grid_size"], "learning_rate": t["learning_rate"],
            "weight_decay": t["weight_decay"], "batch_size": t["batch_size"],
            "maximum_epochs": t["maximum_epochs"], "patience": t["patience"]}


def test_smoke_run_with_5pct_grid():
    from jobs import JobManager

    cfg = {
        "csv_path": str(paths.SAMPLE_DATA),
        "duration_col": "survival_time", "event_col": "died",
        "feature_cols": [f"feat_{i}" for i in range(12)],
        "models": ["KAN_gaps"],
        "quantile_grid": {"kind": "every5"},
        "ratio": [65, 15, 20], "n_splits": 1, "seed": 42,
        "deterministic": False, "mode": "custom", "custom": _smoke_custom(), "preset": None,
        "time_unit": "months",
    }
    manager = JobManager()
    job = manager.start(cfg)
    deadline = time.time() + 180
    while job.state in ("pending", "running"):
        if time.time() > deadline:
            manager.cancel(job.id)
            raise AssertionError(f"timed out: {job.state}/{job.step}")
        time.sleep(0.2)
    assert job.state == "done", f"job failed: {job.error}"

    results = job.results
    assert results["config"]["quantiles"] == grids.build_grid("every5")
    assert results["config"]["quantile_grid"]["n_levels"] == 19
    # per-tau summary spans the full grid; report shows only the standard five.
    assert len(results["summaries"]["KAN_gaps"]["pinball_per_tau"]) == 19

    # dense grid -> survival-curve plot + predictions in the zip
    assert (job.dir / "plots" / "survival_curves.png").exists()
    with zipfile.ZipFile(job.dir / "results.zip") as zf:
        names = zf.namelist()
        assert any(n.startswith("predictions/") for n in names)
    assert (job.dir / "predictions" / "KAN_gaps.csv").exists()

    # report shows the standard levels and the full-grid mean label
    html = (job.dir / "report.html").read_text(encoding="utf-8")
    assert "mean over full grid" in html
    assert "Individual survival curves" in html
