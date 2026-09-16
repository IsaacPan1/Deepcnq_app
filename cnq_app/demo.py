"""Demo data + model, generated on demand so the app never needs a terminal.

Two halves:

* **Data** (`ensure_demo_data`, `resolve_csv`, `ensure_sample_data`) — pure numpy,
  **no torch**. Regenerates any missing CSV with fixed seeds so the content is
  identical every time. Writes atomically under a lock into a writable cache
  (`demo_cache/`, gitignored); committed copies in `demo/` are preferred if
  present. The server calls these on every demo/sample request, so a missing file
  is regenerated on the spot rather than reported as an error.

* **Model** (`build_demo_model`) — needs torch; run from a background job. It
  ensures the data, refits the demo model on all of it and saves the bundle into
  the cache. `resolve_model` prefers a freshly built cache bundle, else the
  committed one.

The survival simulation is reimplemented here in numpy (matching the shapes of
the package's own generator: covariate-dependent Weibull shape, ~50% censoring)
precisely so data generation does not import `deepquantreg` (which pulls in
torch). deepcnq itself is never touched.
"""
from __future__ import annotations

import os
import tempfile
import threading
from pathlib import Path

import numpy as np
import pandas as pd

import paths

# Committed (preferred, read-only) and writable cache locations.
COMMITTED_DIR = paths.DEMO_DIR                 # cnq_app/demo/
CACHE_DIR = paths.HERE / "demo_cache"          # cnq_app/demo_cache/ (gitignored)

MODEL_FILE = "demo_model.cnqmodel"
CSV_FILES = {
    "train": "demo_train.csv",
    "new": "demo_new_subjects.csv",
    "outcomes": "demo_new_subjects_with_outcomes.csv",
    "shifted": "demo_shifted_population.csv",
}
SAMPLE_FILE = "sample.csv"

TIME_UNIT = "months"

# Fixed, distinct seeds — reproducible and independent files.
SEED_TRAIN, SEED_NEW, SEED_OUTCOMES, SEED_SHIFTED, SEED_SAMPLE, SEED_MODEL = 101, 202, 303, 404, 7, 1234
N_TRAIN, N_NEW, N_OUTCOMES, N_SHIFTED, N_SAMPLE = 1500, 20, 500, 500, 300

# Demo model config (built by a job; small enough for ~2 min on a laptop CPU).
DEFAULT_MODEL = "TransformerPS_gaps"
DEFAULT_EPOCHS = 40
DEMO_GRID_KIND = "every5"     # a 5% grid so survival curves render
DEMO_CUSTOM = {
    "hidden_dim": 32, "layers": 2, "dropout": 0.0, "grid_size": 5,
    "learning_rate": 1e-3, "weight_decay": 0.0, "batch_size": 128,
    "maximum_epochs": DEFAULT_EPOCHS, "patience": 8,
}

_data_lock = threading.Lock()


class DemoError(RuntimeError):
    """A demo asset could not be generated (e.g. the folder isn't writable)."""


class DemoCancelled(RuntimeError):
    """The demo model build was cancelled before training began."""


# --------------------------------------------------------------------------- #
# numpy survival simulation (no deepquantreg / torch)
# --------------------------------------------------------------------------- #
_CORR = np.array([[1, .8, 0, 0, 0, 0], [.8, 1, 0, 0, 0, 0], [0, 0, 1, .6, .6, 0],
                  [0, 0, .6, 1, .6, 0], [0, 0, .6, .6, 1, 0], [0, 0, 0, 0, 0, 1]], float)
_CHOL = np.linalg.cholesky(_CORR)
COVARS = [f"x{i}" for i in range(12)]


def _covariates(rng: np.random.RandomState, n: int) -> np.ndarray:
    normal = rng.normal(size=(n, 6)).dot(_CHOL.T)          # x0..x5 correlated normals
    binary = rng.binomial(1, 0.5, size=(n, 2)).astype(float)  # x6, x7
    uniform = rng.uniform(0, 1, size=(n, 4))               # x8..x11
    return np.hstack((normal, binary, uniform))


def _simulate(kind: str, n: int, seed: int) -> pd.DataFrame:
    """Return a frame with subject_id-less covariates x0..x11 + duration + event.

    ``kind='weibull'`` is the base population (covariate-dependent shape, ~50%
    censoring); ``kind='gamma'`` is the shifted population (different survival
    distribution). Deterministic for a given seed.
    """
    rng = np.random.RandomState(seed)
    x = _covariates(rng, n)
    sq = np.sum(x[:, :10] ** 2, axis=1) / 10.0
    if kind == "weibull":
        shape = 2.0 + 2.0 * sq
        scale = (4.0 * x[:, 1] + 4.0) ** 2
        event_time = rng.weibull(shape) * scale
        censor_max = 35.0
    elif kind == "gamma":
        shape = 2.0 + sq
        scale = (5.0 * x[:, 1] + 2.0) ** 2
        event_time = rng.gamma(shape, scale)
        censor_max = 60.0
    else:
        raise ValueError(f"unknown simulation kind {kind!r}")
    event_time = np.clip(event_time, 0.1, 100000.0)
    censor_time = rng.uniform(np.zeros(n), np.full(n, censor_max))
    observed = np.minimum(event_time, censor_time)
    frame = pd.DataFrame(np.round(x, 6), columns=COVARS)
    frame["duration"] = np.round(np.clip(observed, 0.1, 100000.0), 4)
    frame["event"] = (event_time <= censor_time).astype(int)
    return frame


# --------------------------------------------------------------------------- #
# the four demo frames + sample (deterministic)
# --------------------------------------------------------------------------- #
def _train_frame() -> pd.DataFrame:
    f = _simulate("weibull", N_TRAIN, SEED_TRAIN)
    return _with_ids(f, "S", outcomes=True)


def _outcomes_frame() -> pd.DataFrame:
    f = _simulate("weibull", N_OUTCOMES, SEED_OUTCOMES)
    return _with_ids(f, "V", outcomes=True)


def _shifted_frame() -> pd.DataFrame:
    f = _simulate("gamma", N_SHIFTED, SEED_SHIFTED)
    return _with_ids(f, "W", outcomes=True)


def _new_frame(train: pd.DataFrame) -> pd.DataFrame:
    """20 subjects, covariates only, a `note` column, and exactly 2 out-of-range."""
    f = _simulate("weibull", N_NEW, SEED_NEW)[COVARS].copy()
    # Clamp every subject into the training range so only the planted rows are OOR.
    for c in COVARS:
        f[c] = f[c].clip(float(train[c].min()), float(train[c].max())).round(6)
    f.insert(0, "subject_id", [f"P{i + 1:02d}" for i in range(len(f))])
    f["note"] = "typical new patient"
    feat = "x0"
    hi, lo = float(train[feat].max()), float(train[feat].min())
    span = max(hi - lo, 1.0)
    f.loc[0, feat] = round(hi + 1.5 * span, 6)
    f.loc[0, "note"] = f"{feat} deliberately above the training range (out-of-range demo)"
    f.loc[1, feat] = round(lo - 1.5 * span, 6)
    f.loc[1, "note"] = f"{feat} deliberately below the training range (out-of-range demo)"
    return f[["subject_id", *COVARS, "note"]]


def _sample_frame() -> pd.DataFrame:
    """A sample with the historic column names (feat_*, survival_time, died)."""
    f = _simulate("weibull", N_SAMPLE, SEED_SAMPLE)
    out = f[COVARS].copy()
    out.columns = [f"feat_{i}" for i in range(len(COVARS))]
    out["survival_time"] = f["duration"].to_numpy()
    out["died"] = f["event"].to_numpy()
    return out


def _with_ids(frame: pd.DataFrame, prefix: str, *, outcomes: bool) -> pd.DataFrame:
    out = pd.DataFrame()
    out["subject_id"] = [f"{prefix}{i:04d}" for i in range(len(frame))]
    out["time"] = frame["duration"].to_numpy()
    out["event"] = frame["event"].to_numpy()
    for c in COVARS:
        out[c] = frame[c].to_numpy()
    return out


# --------------------------------------------------------------------------- #
# path resolution
# --------------------------------------------------------------------------- #
def committed(name: str) -> Path:
    return COMMITTED_DIR / name


def cached(name: str) -> Path:
    return CACHE_DIR / name


def _present(name: str) -> bool:
    return committed(name).exists() or cached(name).exists()


def _resolve(name: str) -> "Path | None":
    if committed(name).exists():
        return committed(name)
    if cached(name).exists():
        return cached(name)
    return None


def resolve_csv(key: str) -> Path:
    """Path to demo CSV `key`, generating it into the cache if missing."""
    name = CSV_FILES[key]
    if not _present(name):
        ensure_demo_data()
    path = _resolve(name)
    if path is None:
        raise DemoError(f"could not produce {name}")
    return path


def resolve_model() -> "Path | None":
    """The demo model to use: a freshly built cache bundle wins over the committed
    one (it was built for this environment); otherwise the committed bundle."""
    if cached(MODEL_FILE).exists():
        return cached(MODEL_FILE)
    if committed(MODEL_FILE).exists():
        return committed(MODEL_FILE)
    return None


# --------------------------------------------------------------------------- #
# atomic write
# --------------------------------------------------------------------------- #
def _atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
        os.close(fd)
        try:
            frame.to_csv(tmp, index=False, lineterminator="\n")
            os.replace(tmp, path)     # atomic rename; a concurrent writer just wins the race
        except BaseException:
            try:
                os.remove(tmp)
            except OSError:
                pass
            raise
    except OSError as exc:
        raise DemoError(f"could not write demo data to {path.parent}: {exc}") from exc


# --------------------------------------------------------------------------- #
# ensure data (no torch)
# --------------------------------------------------------------------------- #
def ensure_demo_data() -> dict:
    """Generate any demo CSV missing from BOTH the committed folder and the cache.

    Fast, torch-free, idempotent and byte-identical across runs (fixed seeds).
    Safe under concurrency: a lock serialises generation and files are written via
    an atomic rename. Returns ``{"generated": [filenames]}``.
    """
    with _data_lock:
        missing = {k: v for k, v in CSV_FILES.items() if not _present(v)}
        if not missing:
            return {"generated": []}
        # `new` needs the training frame for its covariate ranges.
        train = _read_or_make_train() if "new" in missing else None
        frames = {}
        if "train" in missing:
            frames["train"] = train if train is not None else _read_or_make_train()
        if "new" in missing:
            frames["new"] = _new_frame(train)
        if "outcomes" in missing:
            frames["outcomes"] = _outcomes_frame()
        if "shifted" in missing:
            frames["shifted"] = _shifted_frame()
        written = []
        for key, frame in frames.items():
            _atomic_write_csv(cached(CSV_FILES[key]), frame)
            written.append(CSV_FILES[key])
        if written:
            _write_cache_readme()
        return {"generated": written}


def ensure_sample_data() -> Path:
    """Ensure the sample CSV exists (regenerate into the cache if missing)."""
    committed_sample = paths.SAMPLE_DATA
    if committed_sample.exists():
        return committed_sample
    with _data_lock:
        if not cached(SAMPLE_FILE).exists():
            _atomic_write_csv(cached(SAMPLE_FILE), _sample_frame())
        return cached(SAMPLE_FILE)


def _read_or_make_train() -> pd.DataFrame:
    for p in (committed(CSV_FILES["train"]), cached(CSV_FILES["train"])):
        if p.exists():
            return pd.read_csv(p)
    return _train_frame()


def _write_cache_readme() -> None:
    try:
        (CACHE_DIR / "README.md").write_text(
            "# demo_cache (generated)\n\nSimulated demo data generated on demand by "
            "`demo.ensure_demo_data()`. Safe to delete — it will be regenerated. The "
            "committed copies in `../demo/` are used instead when present.\n",
            encoding="utf-8")
    except OSError:
        pass


# --------------------------------------------------------------------------- #
# build the model (needs torch; run from a job)
# --------------------------------------------------------------------------- #
def build_demo_model(progress_cb=None, cancel_flag=None, *, model_name: str = DEFAULT_MODEL,
                     epochs: int = DEFAULT_EPOCHS, deterministic: bool = True,
                     out_path=None) -> dict:
    """Ensure the data, refit the demo model on all of it, and save the bundle.

    ``progress_cb(step, frac)`` and ``cancel_flag()`` are optional coarse hooks;
    inside a job, fine-grained epoch progress and mid-fit cancellation come from
    the job manager's ``predict_quantiles`` hook. Returns
    ``{path, size_mb, seconds, metrics, ...}``.
    """
    import time

    import grids
    import pipeline
    import presets
    import save_model

    def tick(step, frac):
        if progress_cb:
            progress_cb(step, frac)

    def check_cancel():
        if cancel_flag and cancel_flag():
            raise DemoCancelled()

    tick("Preparing demo data", 0.02)
    ensure_demo_data()
    check_cancel()

    train = pd.read_csv(resolve_csv("train"))
    frame = pipeline.build_frame(train, "time", "event", COVARS, id_col="subject_id")
    quantiles = grids.build_grid(DEMO_GRID_KIND)
    custom = {**DEMO_CUSTOM, "maximum_epochs": int(epochs)}
    resolved = presets.resolve(model_name, preset=None, custom=custom, quantiles=quantiles)

    tick("Training the demo model", 0.1)
    check_cancel()
    t0 = time.time()
    refit = save_model.refit_final(model_name, frame, COVARS, resolved, quantiles,
                                   SEED_MODEL, deterministic=deterministic)
    seconds = round(time.time() - t0, 1)

    out_path = Path(out_path) if out_path else cached(MODEL_FILE)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_model.build_bundle(
        out_path, bundle_type="final", model_name=model_name,
        build_args=refit["build_args"], feature_names=COVARS, quantiles=quantiles,
        state_dicts=[refit["state_dict"]], scalers=[refit["scaler"]],
        feature_ranges=save_model.feature_ranges(frame, COVARS),
        training=save_model.training_stats(frame), event_mapping=None,
        metrics=refit["metrics"], time_unit=TIME_UNIT, seed=SEED_MODEL)
    tick("Done", 1.0)
    size_mb = round(out_path.stat().st_size / (1024 * 1024), 3)
    return {"path": out_path, "size_mb": size_mb, "seconds": seconds,
            "metrics": refit["metrics"], "model": model_name,
            "n_levels": len(quantiles), "n_features": len(COVARS)}


def preview(path, n: int = 5) -> dict:
    """A CSV preview (columns, numeric/binary detection, head) using pandas only
    — no torch, so the demo-data endpoints stay torch-free. Same shape as
    ``server.preview_csv``."""
    frame = pd.read_csv(path)
    columns = list(frame.columns)
    numeric = [c for c in columns if pd.api.types.is_numeric_dtype(frame[c])]
    binaryish = []
    for c in columns:
        try:
            vals = pd.unique(frame[c].dropna())
            if len(vals) <= 2 and set(vals).issubset({0, 1, 0.0, 1.0}):
                binaryish.append(c)
        except (TypeError, ValueError):
            pass
    head = frame.head(n).astype(object).where(frame.head(n).notna(), None).values.tolist()
    return {"columns": columns, "numeric_columns": numeric, "binary_columns": binaryish,
            "n_rows": int(len(frame)), "preview": {"columns": columns, "rows": head}}


def censoring_pct(key: str) -> float:
    frame = pd.read_csv(resolve_csv(key))
    return round((1.0 - frame["event"].mean()) * 100, 1)


def write_all_csvs(out_dir) -> dict:
    """Write all four demo CSVs (+ README) into ``out_dir`` (used by make_demo.py
    to produce committable copies in demo/). Returns the row counts."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train = _train_frame()
    frames = {"train": train, "new": _new_frame(train),
              "outcomes": _outcomes_frame(), "shifted": _shifted_frame()}
    sizes = {}
    for key, frame in frames.items():
        _atomic_write_csv(out_dir / CSV_FILES[key], frame)
        sizes[key] = int(len(frame))
    (out_dir / "README.md").write_text(readme_text(sizes), encoding="utf-8")
    return sizes


def readme_text(sizes: dict) -> str:
    return f"""# Demo data (all simulated)

Every file here is **simulated** (numpy, no real patient data); the app can also
regenerate it on demand. The base population has Weibull survival times whose
*shape* depends on the covariates `x0…x11`, ~50% censoring. Times are labelled
"months".

| File | Rows | Contents |
|------|------|----------|
| `demo_train.csv` | {sizes.get('train', '?')} | subjects with `time`, `event` and covariates — train a model on this |
| `demo_new_subjects.csv` | {sizes.get('new', '?')} | new subjects, **covariates only**; extra `note` column (ignored) and **2 out-of-range rows** |
| `demo_new_subjects_with_outcomes.csv` | {sizes.get('outcomes', '?')} | new subjects **with** outcomes, same population — external validation |
| `demo_shifted_population.csv` | {sizes.get('shifted', '?')} | subjects with outcomes from a **shifted** population (a different survival distribution) — calibration degrades |

`demo_model.cnqmodel` is a final model (refit on all of `demo_train.csv`).
Rebuild everything from the app (Predict tab → Build demo model) or, for
developers, `python make_demo.py` (see [../DEVELOPER.md](../DEVELOPER.md)).
"""
