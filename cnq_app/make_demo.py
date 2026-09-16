"""Generate the demonstration data and model for the CNQ app.

Everything here is **simulated** with the package's own generator
(`deepquantreg.data.generate_simulation`) — no real patient data. It writes four
CSVs and one trained model bundle into `cnq_app/demo/`:

* `demo_train.csv`               — ~1,500 subjects with outcomes, to train on;
* `demo_new_subjects.csv`        — 20 subjects, covariates only (the "predict for
                                   new patients" case), with an extra `note`
                                   column and 2 deliberately out-of-range rows;
* `demo_new_subjects_with_outcomes.csv` — ~500 subjects with outcomes, same
                                   population, for external validation;
* `demo_shifted_population.csv`  — ~500 subjects with outcomes from a *shifted*
                                   population (a different survival distribution),
                                   to show calibration degrade;
* `demo_model.cnqmodel`          — a final model refit on `demo_train.csv`.

The base population is `Weibull_Uniform10D_v1`: Weibull survival times whose
**shape** depends on the covariates (`shape = 2 + 2·Σx²/10`), ~50% censoring. The
shifted population uses `Gamma_Uniform10D_v1` — the same covariate generator but a
different survival-time distribution, so a model trained on the Weibull data is
miscalibrated on it. Covariates are the simulator's own `x0…x11`. Times are in
"months" (a label only; the numbers are the simulator's).

Run it from `cnq_app/`:

    python make_demo.py                 # full: real sizes, ~40 epochs
    python make_demo.py --quick         # tiny sizes/epochs (smoke)
    python make_demo.py --model KAN_gaps
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import paths  # noqa: E402 -- discovers the repo and puts deepquantreg on sys.path
paths.require_repo()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import grids  # noqa: E402
import pipeline  # noqa: E402
import presets  # noqa: E402
import save_model  # noqa: E402
from deepquantreg.data import generate_simulation  # noqa: E402

DEMO_DIR = paths.DEMO_DIR
TIME_UNIT = "months"

# Population scenarios (see module docstring).
BASE_SCENARIO = "Weibull_Uniform10D_v1"     # covariate-dependent Weibull shape, ~50% censoring
SHIFT_SCENARIO = "Gamma_Uniform10D_v1"      # same covariates, different survival distribution

# Fixed, distinct seeds so every file is reproducible and independent.
SEED_TRAIN = 101
SEED_NEW = 202
SEED_OUTCOMES = 303
SEED_SHIFTED = 404
SEED_MODEL = 1234

# Default demo model. TransformerPS_gaps is the paper's attention model; swap to
# KAN_gaps with --model if it trains faster on a given CPU.
DEFAULT_MODEL = "TransformerPS_gaps"
DEMO_CUSTOM = {
    "hidden_dim": 32, "layers": 2, "dropout": 0.0, "grid_size": 5,
    "learning_rate": 1e-3, "weight_decay": 0.0, "batch_size": 128,
    "maximum_epochs": 40, "patience": 8,
}
# A 5% grid (0.05..0.95, 19 levels) so the app draws survival curves in the demo.
DEMO_GRID_KIND = "every5"

FILES = {
    "train": "demo_train.csv",
    "new": "demo_new_subjects.csv",
    "outcomes": "demo_new_subjects_with_outcomes.csv",
    "shifted": "demo_shifted_population.csv",
    "model": "demo_model.cnqmodel",
}


# --------------------------------------------------------------------------- #
# simulation helpers
# --------------------------------------------------------------------------- #
def _simulate(scenario: str, n: int, seed: int) -> pd.DataFrame:
    """Return a frame with columns subject_id, time, event, x0…x11 (time = duration)."""
    frame = generate_simulation(scenario, n, seed)
    covars = [c for c in frame.columns if c not in {"duration", "event", "subject_id"}]
    out = pd.DataFrame()
    out["subject_id"] = frame["subject_id"].to_numpy()
    out["time"] = frame["duration"].round(4).to_numpy()
    out["event"] = frame["event"].astype(int).to_numpy()
    for c in covars:
        out[c] = frame[c].round(6).to_numpy()
    return out


def _covariates(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if c not in {"subject_id", "time", "event", "note"}]


def _censoring_pct(frame: pd.DataFrame) -> float:
    return round((1.0 - frame["event"].mean()) * 100, 1)


# --------------------------------------------------------------------------- #
# the four CSVs
# --------------------------------------------------------------------------- #
def write_train(out_dir: Path, n: int) -> pd.DataFrame:
    frame = _simulate(BASE_SCENARIO, n, SEED_TRAIN)
    frame["subject_id"] = [f"S{i:04d}" for i in range(len(frame))]
    frame.to_csv(out_dir / FILES["train"], index=False)
    return frame


def write_new_subjects(out_dir: Path, n: int, train: pd.DataFrame) -> pd.DataFrame:
    """20 subjects, covariates only, with a `note` column and 2 out-of-range rows."""
    frame = _simulate(BASE_SCENARIO, n, SEED_NEW).drop(columns=["time", "event"])
    frame["subject_id"] = [f"P{i + 1:02d}" for i in range(len(frame))]
    frame["note"] = "typical new patient"

    covars = _covariates(train)
    # Clamp every new subject into the training range first, so that *only* the
    # rows we plant below are out of range (otherwise a stray normal draw could
    # also fall outside and make the flagged count nondeterministic).
    for c in covars:
        frame[c] = frame[c].clip(float(train[c].min()), float(train[c].max()))

    # Push a covariate on two subjects well past its training range, to trigger
    # the out-of-range flag. Use x0 (a roughly N(0,1) covariate).
    feat = "x0" if "x0" in covars else covars[0]
    hi = float(train[feat].max())
    lo = float(train[feat].min())
    span = max(hi - lo, 1.0)
    if len(frame) >= 2:
        frame.loc[0, feat] = round(hi + 1.5 * span, 6)
        frame.loc[0, "note"] = f"{feat} deliberately above the training range (out-of-range demo)"
        frame.loc[1, feat] = round(lo - 1.5 * span, 6)
        frame.loc[1, "note"] = f"{feat} deliberately below the training range (out-of-range demo)"

    # column order: id, covariates…, note (no time/event)
    frame = frame[["subject_id", *covars, "note"]]
    frame.to_csv(out_dir / FILES["new"], index=False)
    return frame


def write_outcomes(out_dir: Path, n: int) -> pd.DataFrame:
    frame = _simulate(BASE_SCENARIO, n, SEED_OUTCOMES)
    frame["subject_id"] = [f"V{i:04d}" for i in range(len(frame))]
    frame.to_csv(out_dir / FILES["outcomes"], index=False)
    return frame


def write_shifted(out_dir: Path, n: int) -> pd.DataFrame:
    frame = _simulate(SHIFT_SCENARIO, n, SEED_SHIFTED)
    frame["subject_id"] = [f"W{i:04d}" for i in range(len(frame))]
    frame.to_csv(out_dir / FILES["shifted"], index=False)
    return frame


# --------------------------------------------------------------------------- #
# the model
# --------------------------------------------------------------------------- #
def train_model(out_dir: Path, train: pd.DataFrame, *, model_name: str, epochs: int,
                deterministic: bool = True) -> dict:
    """Train `model_name` as a final refit on the demo training data and save the
    bundle. Returns a small info dict (path, size, seconds, metrics)."""
    covars = _covariates(train)
    frame = pipeline.build_frame(train, "time", "event", covars, id_col="subject_id")
    quantiles = grids.build_grid(DEMO_GRID_KIND)
    custom = {**DEMO_CUSTOM, "maximum_epochs": int(epochs)}
    resolved = presets.resolve(model_name, preset=None, custom=custom, quantiles=quantiles)

    t0 = time.time()
    refit = save_model.refit_final(model_name, frame, covars, resolved, quantiles,
                                   SEED_MODEL, deterministic=deterministic)
    seconds = round(time.time() - t0, 1)

    out_path = out_dir / FILES["model"]
    save_model.build_bundle(
        out_path, bundle_type="final", model_name=model_name,
        build_args=refit["build_args"], feature_names=covars, quantiles=quantiles,
        state_dicts=[refit["state_dict"]], scalers=[refit["scaler"]],
        feature_ranges=save_model.feature_ranges(frame, covars),
        training=save_model.training_stats(frame), event_mapping=None,
        metrics=refit["metrics"], time_unit=TIME_UNIT, seed=SEED_MODEL)
    size_mb = round(out_path.stat().st_size / (1024 * 1024), 3)
    return {"path": out_path, "size_mb": size_mb, "seconds": seconds,
            "metrics": refit["metrics"], "model": model_name,
            "n_levels": len(quantiles), "n_features": len(covars)}


# --------------------------------------------------------------------------- #
# README for the demo folder
# --------------------------------------------------------------------------- #
def write_readme(out_dir: Path, sizes: dict) -> None:
    text = f"""# Demo data (all simulated)

Every file here is **simulated** by the `deepquantreg` package's own generator
(`generate_simulation`) — there is no real patient data. Times are labelled
"months" for the demo, but the numbers are the simulator's.

The base population (`{BASE_SCENARIO}`) has Weibull survival times whose *shape*
depends on the covariates, with about 50% censoring. Covariates are the
simulator's `x0…x11`.

| File | Rows | Contents |
|------|------|----------|
| `demo_train.csv` | {sizes['train']} | subjects with `time`, `event` and covariates — train a model on this |
| `demo_new_subjects.csv` | {sizes['new']} | new subjects, **covariates only** (no outcomes). Has an extra `note` column (ignored by the app) and **2 rows with a covariate outside the training range** to show the out-of-range flag |
| `demo_new_subjects_with_outcomes.csv` | {sizes['outcomes']} | new subjects **with** outcomes, same population — for external validation |
| `demo_shifted_population.csv` | {sizes['shifted']} | subjects with outcomes from a **shifted** population (`{SHIFT_SCENARIO}`: a different survival distribution) — calibration degrades here |

`demo_model.cnqmodel` is a final model (refit on all of `demo_train.csv`).

Regenerate everything with `python make_demo.py` from `cnq_app/` (see
[DEVELOPER.md](../DEVELOPER.md)).
"""
    (out_dir / "README.md").write_text(text, encoding="utf-8")


# --------------------------------------------------------------------------- #
# orchestration
# --------------------------------------------------------------------------- #
def build_all(out_dir: Path | None = None, *, model_name: str = DEFAULT_MODEL,
              n_train: int = 1500, n_new: int = 20, n_outcomes: int = 500,
              n_shifted: int = 500, epochs: int = DEMO_CUSTOM["maximum_epochs"],
              deterministic: bool = True, with_model: bool = True) -> dict:
    out_dir = Path(out_dir) if out_dir else DEMO_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    train = write_train(out_dir, n_train)
    new = write_new_subjects(out_dir, n_new, train)
    outcomes = write_outcomes(out_dir, n_outcomes)
    shifted = write_shifted(out_dir, n_shifted)
    sizes = {"train": len(train), "new": len(new), "outcomes": len(outcomes),
             "shifted": len(shifted)}
    write_readme(out_dir, sizes)

    info = {"sizes": sizes, "censoring": {
        "train": _censoring_pct(train), "outcomes": _censoring_pct(outcomes),
        "shifted": _censoring_pct(shifted)}}
    if with_model:
        info["model"] = train_model(out_dir, train, model_name=model_name,
                                    epochs=epochs, deterministic=deterministic)
    return info


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Generate the CNQ demo data and model.")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"model to train (default {DEFAULT_MODEL})")
    parser.add_argument("--epochs", type=int, default=DEMO_CUSTOM["maximum_epochs"])
    parser.add_argument("--quick", action="store_true",
                        help="tiny sizes/epochs for a fast smoke run")
    parser.add_argument("--no-model", action="store_true", help="write CSVs only")
    args = parser.parse_args(argv)

    kwargs = dict(model_name=args.model, epochs=args.epochs, with_model=not args.no_model)
    if args.quick:
        kwargs.update(n_train=200, n_new=20, n_outcomes=120, n_shifted=120, epochs=2)

    info = build_all(**kwargs)
    print("Demo data written to", DEMO_DIR)
    for key, n in info["sizes"].items():
        print(f"  {FILES[key]}: {n} rows")
    print("  censoring %:", info["censoring"])
    if info.get("model"):
        m = info["model"]
        print(f"Demo model: {m['model']} — {m['size_mb']} MB, trained in {m['seconds']} s, "
              f"{m['n_levels']} quantile levels, {m['n_features']} features")
        print("  training metrics:", m["metrics"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
