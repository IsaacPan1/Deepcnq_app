# Demo data (all simulated)

Every file here is **simulated** (numpy, no real patient data); the app can also
regenerate it on demand. The base population has Weibull survival times whose
*shape* depends on the covariates `x0…x11`, ~50% censoring. Times are labelled
"months".

| File | Rows | Contents |
|------|------|----------|
| `demo_train.csv` | 1500 | subjects with `time`, `event` and covariates — train a model on this |
| `demo_new_subjects.csv` | 20 | new subjects, **covariates only**; extra `note` column (ignored) and **2 out-of-range rows** |
| `demo_new_subjects_with_outcomes.csv` | 500 | new subjects **with** outcomes, same population — external validation |
| `demo_shifted_population.csv` | 500 | subjects with outcomes from a **shifted** population (a different survival distribution) — calibration degrades |

`demo_model.cnqmodel` is a final model (refit on all of `demo_train.csv`).
Rebuild everything from the app (Predict tab → Build demo model) or, for
developers, `python make_demo.py` (see [../DEVELOPER.md](../DEVELOPER.md)).
