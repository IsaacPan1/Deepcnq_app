# CNQ App — developer notes

Technical documentation for the local web app that wraps the `deepquantreg`
package. End-user instructions live in [README.md](README.md).

## Layout

The app and the upstream repo are **sibling directories**:

```
deepquant/
  deepcnq/    # the BIG-S2/deepcnq clone — unmodified upstream, never edited
  cnq_app/    # this app
```

**The app never creates, modifies or deletes anything under `deepcnq/`.** It
imports the `deepquantreg` package directly from the repo's `src/` and reads the
configs from the repo's `configs/`; it does not install or vendor the package. If
an upstream API change breaks the app, fix it here in `cnq_app/`, never in the
repo.

## Repo discovery (`paths.py`)

`paths.py` resolves the repo root in this order:

1. the `CNQ_REPO` environment variable, if set;
2. the sibling `../deepcnq` next to the app;
3. otherwise it fails with a clear message explaining both options.

It validates that `src/deepquantreg/__init__.py` and
`configs/final/real/cnq.yaml` exist, then inserts `<repo>/src` at the front of
`sys.path`. Discovery is attempted at import but never raises there: on failure
`paths.REPO_ROOT is None` and `paths.REPO_ERROR` holds the message, so the server
still starts and reports the problem via `/api/health` and a page banner.

Exposed names: `REPO_ROOT`, `CONFIGS_DIR`, `SRC`, `CNQ_PRESET_CONFIG`,
`SMOKE_CONFIG` (all `None` when the repo is missing); app-local `STATIC`,
`SAMPLE_DATA`, `REQUIREMENTS`, `OUTPUT_DIR`; and helpers `require_repo()`,
`repo_version()` (git commit + dirty flag, else `VERSION.txt`, else `"unknown"`)
and `repo_info()`.

Set `CNQ_REPO` to point at a repo elsewhere:

```bash
CNQ_REPO=/path/to/deepcnq python -m server          # macOS/Linux
set CNQ_REPO=C:\path\to\deepcnq && python -m server  # Windows cmd
```

## Running the server directly

`run.py` is the first-run launcher (venv + deps); once dependencies are present
you can run the server module directly from `cnq_app/`:

```bash
python -m server            # http://127.0.0.1:8000/
python -m server 9000 -b 0.0.0.0
```

It also works from any working directory, because all paths resolve from
`Path(__file__)` / `paths`:

```bash
python /abs/path/to/cnq_app/server.py 8002
```

`python -m server` takes an optional positional **port** (default `8000`) and
`-b/--bind` (default `127.0.0.1`), prints the URL and the repo path/version,
warns about missing packages, and exits cleanly on Ctrl+C. A busy port prints a
one-line suggestion.

### Lazy imports

Only the standard library plus the stdlib-only `paths` module is imported when
`server.py` loads. The scientific stack (pandas, torch, matplotlib) and the
`pipeline`/`jobs` modules import lazily inside the request handlers, so the
server always starts. `GET /api/health` reports package availability **and**
whether the repo was found (with its path and version); the page shows a banner
if anything is missing.

### Progress hook

Per-epoch progress and cancellation are implemented by wrapping
`deepquantreg.training.trainer.predict_quantiles`. `trainer.fit` calls that
function once per epoch to score the validation split, so the wrapper is a clean
place to count epochs and to abort a fit when the user cancels.

## Outputs

Everything the app writes stays under `cnq_app/`:

- `.venv/` — the launcher's virtual environment
- `jobs/` — per-run output directories (train / save / predict), plus
  `jobs/uploads/` for uploaded CSVs and models. Each training run also writes an
  `artifacts/` subfolder (per-split weights + scalers) so it can be saved later.
  Pruned to the most recent `CNQ_KEEP_RUNS` runs (default 20).
- `models/` — saved `.cnqmodel` bundles (never pruned)
- `demo/` — simulated demo CSVs + `demo_model.cnqmodel` (see below)
- `sample_data/` — the bundled sample dataset
- `__pycache__/`, `.pytest_cache/` — caches

`results.json` and the HTML report both record the repo path and version
(commit hash or `VERSION.txt`).

## Dependencies

`cnq_app/requirements.txt` lists the app's own runtime deps (numpy, pandas,
scikit-learn, matplotlib, PyYAML). PyTorch is **not** listed there — `run.py`
installs it from the CPU/CUDA wheel index. The launcher only ever uses
`cnq_app/requirements.txt`, never the repo's, and the install stamp hashes that
file.

```bash
python run.py            # set up (if needed) and launch
python run.py 8001       # different port
python run.py --cuda     # CUDA torch wheels
python run.py --reinstall
```

## Data validation (`validation.py`)

`validation.validate(df, duration_col, event_col, feature_cols, id_col=None, *,
ratio, seed, n_splits, event_positive)` returns `{"errors", "warnings",
"summary"}`. Each message is a dict with a `code`, a plain-language `message`
and, where relevant, `column`/`count`/`values`/`examples`. Errors block a run;
warnings must be acknowledged in the UI.

It is called from three places: `POST /api/upload` (with a server-guessed
mapping), `POST /api/validate` (live, as the mapping changes) and `POST /api/run`
(which refuses to start while any error remains). The rules map directly onto the
`deepquantreg` requirements the app depends on:

**Errors** (block the run)

| Rule | Why (repo requirement) |
|------|------------------------|
| duration column not numeric | `prepare_real` scales/log-transforms a numeric time |
| event column not 0/1 | KM censoring weights need a 0/1 indicator. With exactly two other values (`1`/`2`, `True`/`False`, `dead`/`alive`) the app offers a mapping (`event_positive`) instead of guessing |
| no features selected | the model needs at least one input to `StandardScaler` |
| a selected feature is non-numeric | features are standardised with `StandardScaler` (numeric only); the message shows example values |
| no events at all | KM/IPCW and Uno C need observed events |
| fewer than 10 events in any split (checked after splitting with the chosen ratio/seed) | each split's train/valid/test must estimate weights and metrics |
| duplicate IDs (when an ID column is chosen) | one row per subject |

**Warnings** (must be confirmed)

| Rule | Effect |
|------|--------|
| duration ≤ 0 | excluded (log time needs duration > 0) |
| missing / non-numeric cells in selected columns | excluded, with a per-column count |
| a feature with a single constant value | no information; StandardScaler gives it zero variance |
| fewer than 100 events in total | results may be unstable |
| more than 80% censoring | estimates may be unreliable |
| the event or duration column also selected as a feature | outcome leakage |

**Summary**: rows before/after exclusions, events, censoring %, duration range and
number of features.

The duration `> 0` and 0/1-event exclusions are applied in `pipeline.build_frame`
(rows are dropped, not errored), and `event_positive`/`id_col` are threaded
through `build_frame`. `results.json` and the report include a **Data** section
(file name, rows used/excluded with reasons, events, censoring %, time unit,
event mapping, features).

## Settings: presets & quantile grids (`presets.py`, `grids.py`)

**Starting settings (presets).** `presets.PRESET_META` holds the approximate
size and censoring of each paper cohort (published figures for the public
benchmarks; the app never combines user data with them). `preset_label` renders
`"METABRIC (≈1,900 subjects, 42% censored)"` (subject counts rounded to two
significant figures), and `preset_meta()` ships these to the front end in
`/api/config`. `suggest_preset(n_subjects, censoring_pct)` picks the closest
cohort: the main distance is `|log(n_user) − log(n_preset)|`, with censoring as a
secondary term (`+ 0.3 · |Δcensoring|/100`) so it only breaks near-ties. The
suggestion is computed from the validation summary and returned as
`suggested_preset` on `/api/upload` and `/api/validate`; the UI marks that option
"suggested" and pre-selects it until the user changes it.

**Quantile grids.** `grids.py` is the single source of truth. `build_grid(kind,
custom)` supports `standard`, `every10`, `every5`, `every1` and `custom`,
building levels by integer stepping and `round(x, 4)` so there are no
floating-point artefacts. `merge_required` always adds `0.1/0.5/0.9`, drops
out-of-range values, sorts and de-duplicates. `resolve(cfg)` prefers a
`quantile_grid` spec, else a raw `quantiles` list. `has_extreme` flags levels
below 0.05 or above 0.95 (the UI warns; the bounds themselves are not extreme),
and `standard_present` returns the standard five that appear in a grid.

The client (`app.js`) mirrors `build_grid` only to show the live level count and
the extreme-levels warning; the server rebuilds the grid authoritatively in
`/api/run` (`grids.resolve`) and `jobs` records `{kind, n_levels}` in
`results.json`. With more than five levels:

- metric tables and calibration show only the standard levels present, while the
  "Pinball (mean over full grid)" row is the average over every level;
- an **Individual survival curves** plot (`plots.survival_curves_plot`) draws
  `S(t) = 1 − τ` against predicted time for a few subjects;
- full-grid per-subject predictions (original-scale times) are written to
  `predictions/<model>.csv` inside the results zip.

## Saving models & prediction (`model_io.py`, `save_model.py`, `predict.py`, `predict_report.py`)

A trained model can be bundled as a `.cnqmodel` file and reused to predict on new
subjects. Saving and prediction run through the same `JobManager` as training
(the job's `kind` is `save` or `predict`), so they get progress and cancellation
for free; a refit and external validation can be slow, prediction itself is fast.

### Bundle format (`.cnqmodel`)

A `.cnqmodel` is a plain **zip** with three members:

| Member | Contents |
|--------|----------|
| `weights.pt` | `torch.save` of a **list of state_dicts** — one per ensemble member (length 1 for a final / single-split model). Tensors only. |
| `model.json` | all non-weight metadata (see below). |
| `manifest.json` | `{"algorithm": "sha256", "files": {"model.json": …, "weights.pt": …}}` — the SHA-256 of the other two members. |

`model.json` records everything needed to rebuild the model and reproduce
predictions: `format_version`, the model name and the exact **`build_args`**
passed to `deepquantreg.build_model`, the ordered `feature_names`, the training
`scaler` (mean/scale) and `member_scalers`, the `quantiles`, the `log_time`
convention (`exp` inverse), the `bundle` type + member count, per-feature
`feature_ranges` (training min/max, for out-of-range flags), `missing_handling`,
training size/events/censoring, the `event_mapping`, training `metrics`, the
`deepcnq` version, library `versions`, `device`, `seed` and `created_at`.

`save_bundle(path, state_dicts, meta)` fills the auto fields (`format_version`,
`log_time`, `versions`, `created_at`), hashes the members and writes them into a
temp file it then renames into place (never a half-written bundle).
`load_bundle(path)` verifies both hashes and the format version, loads the
weights, rebuilds each member with `build_model(build_args)` and
`load_state_dict(strict=True)`, and returns rebuilt models + meta + warnings.
`read_meta(path)` returns just the verified `model.json` **without importing
torch**, so listing saved models (`GET /api/models`) stays light.

### Why `weights_only` + JSON, not pickle

Weights load with `torch.load(..., weights_only=True, map_location="cpu")`.
`weights_only=True` refuses to unpickle arbitrary Python objects, so opening a
bundle can only ever materialise tensors — a `.cnqmodel` someone emailed you
can't execute code on load. That is only safe because the bundle stores **no
Python objects**: all metadata is plain JSON and the weights are pure tensors.
Nothing is pickled, so there's no class/version coupling between the saving and
loading environments beyond what `build_args` + `build_model` reconstruct.

### Versioning and compatibility

`FORMAT_VERSION` (an int) gates the layout. A bundle whose `format_version` is
**newer** than the app is refused with a clear error; older is accepted. Hash
mismatches, a non-zip file, or missing members all raise `BundleError` with a
plain message. Two differences are **warnings, not errors** (returned on the
`LoadedBundle`): a different deepcnq commit, or a different torch version, than
the bundle was saved with — predictions should still match but may drift if model
code changed.

### Per-member scalers (ensembles)

Each repeated split is standardised with **its own** training `StandardScaler`
(fit on that split's train rows), so an ensemble member only produces correct
outputs when fed data scaled by *its* scaler. Bundles therefore store
`member_scalers` (one per member) alongside a top-level `scaler` (member 0, for
display). `predict.ensemble_predict_log` standardises the raw features with each
member's scaler in turn, then averages predictions **on the log scale**. Because
every `_gaps` model emits non-decreasing log-quantiles, the mean of those
sequences is also non-decreasing — the ensemble stays non-crossing without any
re-sorting. Final and single-split bundles simply carry one scaler / one member.

### How a run enables saving

To bundle the *exact* in-session weights (so saved predictions match to 1e-6),
each training run persists per-split artifacts under its job dir
(`artifacts/weights/<model>__<split>.pt`, `artifacts/scalers.json`,
`artifacts/feature_ranges.json`, `artifacts/context.json`,
`artifacts/training.json`). A `save` job reads those to build **single-split** or
**ensemble** bundles without retraining; a **final** bundle instead refits on all
rows via `pipeline.prepare_all` (a single train/valid split; 15% held out for
early stopping) in `save_model.refit_final`.

### Prediction validation & outputs

`predict.validate` mirrors the numeric rules in `validation.py` but is anchored
to the bundle's required feature set: missing features and fully non-numeric
feature columns are **errors**; rows with a missing feature value are **excluded,
never imputed** (a warning); subjects with any feature outside the training range
are flagged (`out_of_range`) with per-feature counts. `predict.run_prediction`
writes the predictions frame (`q_<τ>`, `median`, 80% interval + width,
`out_of_range`); when time/event columns are given, `external_validation`
computes IPCW pinball, 50/80% coverage, Uno C and per-τ calibration with a
Kaplan–Meier censoring estimate **from the new data**, plus a KM-vs-mean-predicted
survival curve. `predict_report.build` renders the plots, the self-contained HTML
report and a results zip (predictions CSV, figures, report, and a copy of
`model.json`).

### Model + predict API

| Endpoint | Purpose |
|----------|---------|
| `GET /api/models` | list saved bundles (`read_meta` each; torch-free) |
| `GET /api/models/download?name=` | download a saved `.cnqmodel` |
| `DELETE /api/models?name=` | delete a saved bundle |
| `POST /api/save` | start a `save` job `{source_job_id, model, bundle_type, split_index?, name}` |
| `POST /api/predict/upload-model` | store an uploaded `.cnqmodel`, return its summary |
| `POST /api/predict/upload-data` | store an uploaded CSV, return preview + columns |
| `POST /api/predict/validate` | validate a mapping against a bundle |
| `POST /api/predict/run` | start a `predict` job |
| `GET /api/predictions?id=` | download a predict job's `predictions.csv` |

Saved bundles live in `cnq_app/models/` (`paths.MODELS_DIR`); bundle names are
sanitised (`model_io.safe_name`) to a path-safe form. Predict jobs reuse the
existing `/api/report` and `/api/zip` job-download routes.

### Job folder cleanup (`CNQ_KEEP_RUNS`)

After every job the manager prunes `jobs/` to the most recent **`CNQ_KEEP_RUNS`**
run folders (default 20; `0` or negative disables it). Only 12-char hex job ids
are considered — `jobs/uploads/` and anything else is left alone. Saved
`.cnqmodel` bundles live outside `jobs/`, so they are never touched by pruning.

## Demo assets (`make_demo.py`, `demo/`)

`make_demo.py` generates the shipped demonstration data and model — all
**simulated** with the package's own `deepquantreg.data.generate_simulation`
(same generator as `sample_data/generate.py`). It writes into `cnq_app/demo/`:

- `demo_train.csv` (~1,500) — base population `Weibull_Uniform10D_v1` (Weibull
  times whose shape depends on the covariates, ~50% censoring);
- `demo_new_subjects.csv` (20) — covariates only, plus a `note` column and **2
  rows clamped outside the training range** to demonstrate the out-of-range flag;
- `demo_new_subjects_with_outcomes.csv` (~500) — same population, with outcomes;
- `demo_shifted_population.csv` (~500) — `Gamma_Uniform10D_v1`: same covariate
  generator but a different survival distribution, so a Weibull-trained model is
  miscalibrated on it (the "population shift" demo);
- `demo_model.cnqmodel` — a **final** bundle (refit on all of `demo_train.csv`),
  on the 5% quantile grid (so survival curves render), trained deterministically
  with a fixed seed and small settings (target <~2 min on a laptop CPU).

Covariates are the simulator's own `x0…x11`; the time unit is "months". Every
file uses a fixed, distinct seed. `make_demo.py --quick` uses tiny sizes/epochs
for a smoke run; `--model KAN_gaps` swaps the architecture; `--no-model` writes
only the CSVs. `build_all(...)` returns the sizes, censoring and (model) size /
seconds / metrics for filling in DEMO.md.

Regenerate and commit the assets after a deepcnq change that alters the model:

```bash
cd cnq_app
python make_demo.py           # writes demo/*.csv + demo/demo_model.cnqmodel
git add demo/                 # commit the regenerated assets
```

**Registration & protection.** The demo model lives in `demo/` (not `models/`)
and is *registered on startup, not moved*: `GET /api/models` lists it first under
the reserved name **`Demo model (simulated data)`** with `is_demo: true`. It
resolves to `paths.DEMO_MODEL` in predict/download, and `DELETE /api/models` on
that name returns **403** — it can't be deleted. The Predict tab shows a **demo**
badge and hides Delete for it; the demo new-subject files are served by
`GET /api/demo/data?name={new|outcomes|shifted}` and the training CSV by
`GET /api/demo/train`.

**Rebuild.** If the committed bundle can't load (e.g. a torch/deepcnq mismatch),
`read_meta` may still succeed but `load_bundle` fails at predict time. The UI then
offers **Rebuild demo model**, which `POST /api/demo/rebuild` runs as a normal job
(`kind: "demo"` → `JobManager._execute_demo` → `make_demo.train_model` on
`demo_train.csv`) with progress, overwriting `demo_model.cnqmodel`, then retries.

## Tests

```bash
cd cnq_app
python -m pytest tests -q
```

`tests/test_smoke.py` covers the full pipeline through the `JobManager`
(background thread, progress hook, plots, report, zip), an **API-level**
end-to-end test that boots the real server in a thread, quantile validation, and
cancellation. `tests/test_validation.py` covers every data-validation rule with
small synthetic frames plus an API flow that uploads a messy CSV (1/2 event
coding, a text column, blank cells), checks the messages, fixes the mapping and
completes a run. Both use the repo's smoke config (`configs/smoke/simulated.yaml`)
and the bundled sample data. `tests/test_model_io.py` covers the bundle
round-trip and every corruption/incompatibility path (tampered hash, bad zip,
future `format_version`, weights/architecture mismatch). `tests/test_predict.py`
trains tiny models on the sample data and checks that each saved bundle type
(single split, ensemble, refit) reloads and predicts identically to the
in-session model (within 1e-6), that predictions are invariant to CSV column
order, that a single row matches the same row inside a batch, that ensembles stay
non-crossing, that missing-feature / text-in-feature mappings error, and that
external validation runs.

## Building a release

```bash
cd cnq_app
python make_release.py          # writes ../../deepquant.zip
python make_release.py --allow-dirty
```

The archive contains a top-level `deepquant/` with `deepcnq/` (everything tracked
by `git ls-files`, `.git` excluded, plus a generated `deepcnq/VERSION.txt`) and
`cnq_app/` (excluding `.venv/`, `jobs/`, `__pycache__/`, `*.pyc`,
`.pytest_cache/` and other outputs). `VERSION.txt` is written into the archive
only — never onto disk in the repo. The build refuses a dirty repo unless
`--allow-dirty` is passed.

## Updating the repo

```bash
cd cnq_app
python update_repo.py       # git pull --ff-only, then the smoke test
```

Manual equivalent:

```bash
cd deepcnq && git pull
```

## File table

| File | Purpose |
|------|---------|
| `paths.py` | repo discovery (`CNQ_REPO`/sibling), `sys.path`, version helpers |
| `run.py` | first-run launcher: builds `.venv`, installs deps, stamps, launches |
| `run.bat` / `run.sh` | double-click wrappers that find Python and call `run.py` |
| `server.py` | `python -m server` entry point; CLI, `/api/health`, static serving, raw-body upload |
| `jobs.py` | background job manager, progress hook, cancellation, orchestration |
| `pipeline.py` | data prep (`prepare_real` replica), training, metrics, importance; `prepare_all` for refits |
| `model_io.py` | `.cnqmodel` bundle read/write, hash + version checks, `safe_name` |
| `save_model.py` | assemble bundles from a run; refit-on-all-data for final models |
| `predict.py` | prediction: validation, per-member scaling, ensemble averaging, external validation |
| `predict_report.py` | prediction plots, HTML report and results zip |
| `validation.py` | data-spec checks (errors/warnings/summary) used by upload, validate and run |
| `presets.py` | resolve hyper-parameters from `cnq.yaml`; preset labels + closest-cohort suggestion |
| `grids.py` | quantile-grid generation (kinds, rounding, required levels, standard subset) |
| `plots.py` | the seven matplotlib figures |
| `report.py` | self-contained HTML report + results zip |
| `update_repo.py` | fast-forward the repo, then run the smoke test |
| `make_release.py` | build the self-contained `deepquant.zip` |
| `make_demo.py` | generate the simulated demo CSVs + `demo_model.cnqmodel` |
| `static/` | front end (HTML/CSS/JS) |
| `sample_data/` | bundled simulated dataset + its generator |
| `demo/` | simulated demo data + model (generated by `make_demo.py`) |
| `requirements.txt` | the app's own runtime dependencies (torch installed by `run.py`) |
| `tests/test_smoke.py` | end-to-end + API + quantile-validation + cancellation tests |
| `tests/test_validation.py` | per-rule data-validation tests + a messy-CSV API flow |
| `tests/test_grids.py` | quantile-grid generation, preset suggestion, and a 5%-grid smoke run |
| `tests/test_model_io.py` | bundle round-trip + corruption/incompatibility handling |
| `tests/test_predict.py` | save→reload→predict parity, column-order/scaling invariance, ensembles, errors, external validation |
| `tests/test_api_save_predict.py` | HTTP payload-level save (all bundle types, display name, 400s) + predict |
| `tests/test_demo.py` | demo generation, committed bundle predict/flags, delete protection |
