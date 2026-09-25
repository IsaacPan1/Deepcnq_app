## Start the app

1. Unzip `deepquant.zip` anywhere you like.
   (Avoid folders synced by OneDrive, iCloud or Dropbox.)
2. Open a terminal:
   - **Windows:** Start menu → type `cmd` → Enter
   - **Mac:** Cmd+Space → type `Terminal` → Enter
3. Go to the app folder, replacing `path/to/deepquant` with where you unzipped it:

**Windows**
```
cd /d path\to\deepquant\cnq_app
python run.py
```

**Mac / Linux**
```
cd path/to/deepquant/cnq_app
python3 run.py
```

> **Tip:** type `cd ` (with a space), then drag the `cnq_app` folder
> into the terminal window. The full path is filled in for you. Press Enter.

The first time takes a few minutes while it installs what it needs. Wait until you see:

```
Serving CNQ app on 127.0.0.1 port 8000 (http://127.0.0.1:8000/) ...
```

Then open **http://127.0.0.1:8000/** in your browser.

Keep the terminal open while using the app. Press **Ctrl+C** in the terminal to stop it.

## Try the demo

Want to see it work before using your own data? Everything below happens **in the
browser** — no terminal needed. It all uses **simulated** demo data.

1. On the **Train a model** tab, click **Use demo training data**, pick a model,
   set **Quantile levels → Every 5%**, and **Start training**.
2. Switch to the **Predict new subjects** tab and choose **Demo model (simulated
   data)**. If it says *"not built yet"*, click **Build demo model** once (about
   2 minutes) — the demo data and model are created for you automatically.
3. Under **Or use demo new subjects**, click **20 new patients** and **Run
   prediction** — note the two subjects flagged out-of-range.
4. Try **500 with outcomes** (coverage close to nominal) and then **500 shifted
   population** (calibration gets worse) to see why you validate on your own data.

A full presenter walkthrough is in [DEMO.md](DEMO.md), and a complete user manual
(train, auto-tune, save, predict, project — with a glossary and troubleshooting)
is in **[MANUAL.docx](MANUAL.docx)**.

## Open it again later

Run the same two lines again:

**Windows**
```
cd /d path\to\deepquant\cnq_app
python run.py
```

**Mac / Linux**
```
cd path/to/deepquant/cnq_app
python3 run.py
```

## Preparing your data

Use one CSV with a single header row and **one row per subject**. Once the app
is running you can download a template from the upload step (or the sample
dataset).

| Column | What it is | Example |
|--------|-----------|---------|
| Follow-up time | Time until the event or last contact — a positive number | `12.4` |
| Event indicator | `1` = event happened, `0` = censored (still event-free) | `1` |
| Covariates | One or more numeric predictors | `age = 58` |
| Subject ID *(optional)* | A label for each subject; not used as a predictor | `S001` |

Checklist:

- No blank cells in the columns you use.
- One row per subject (unique IDs if you include an ID column).
- A single header row at the top.
- Times are positive; the event column is `0` or `1`.

Example:

```
subject_id,time,event,age,biomarker,treatment
S001,12.4,1,58,2.7,1
S002,36.0,0,64,1.2,0
```

If your event column uses two other values (for example `1`/`2`,
`dead`/`alive`, or `True`/`False`), the app asks which value means the event
happened — it won't guess. You can also type a time unit (such as `months`) so
it appears on the plots.

**How much data?** Aim for at least about **100 events** (rows with a `1`), and
enough that every split keeps at least 10 events. More events give steadier
results — the app checks this and warns you before training.

## Settings

**Choose settings** (default): pick a model and its hyperparameters. **Starting
settings** reuse the values the paper tuned on a public dataset of similar size —
SUPPORT, FLCHAIN, GBSG, GBSG500, METABRIC or NKI70; after you upload, the closest
one is marked **suggested** (your data is never combined with those datasets).
Choose **Custom** to set values yourself — these are **pre-filled from your data's
size and covariate count**, so they're a sensible starting point you can edit.

**Auto-tune** (alternative): let the app search several models and settings on
*your* data and pick the best. Tick the **model families** to try — **KAN-CNQ and
MLP-CNQ** by default (transformers are opt-in, only for nonstandard /
cross-modality data) — choose a **search effort** (Quick / Standard / Thorough),
and click **Find best settings**. Each candidate is scored by validation IPCW
pinball (with a reduced training budget); you get a **leaderboard**, and **Use
best settings** fills in the winner so you can review and **Start training**
normally. The search is bounded and cancellable — cancelling still shows the best
found so far.

**Quantile levels** are the points on each subject's survival-time distribution
the model predicts. The default (0.1, 0.25, 0.5, 0.75, 0.9) is a good start;
"Every 10% / 5% / 1%" or a custom list give a smoother predicted curve, and
`0.1`, `0.5` and `0.9` are always included. Very low or high levels (below 0.05
or above 0.95) rely on few events and can be unreliable. With a dense grid the
tables still show the five standard levels (plus the average over the whole
grid), and the report adds individual survival curves.

## Save a model and predict new subjects

Once you have trained a model you can **save** it and reuse it later to predict
survival-time quantiles for new subjects — no retraining needed.

### Save a model

After a run finishes, use the **Save a model** panel on the results:

1. Choose which trained **model** to save.
2. Choose **what to save**:
   - **Final model: refit on all data** — retrains the chosen model on *every*
     row (holding out 15% for early stopping). This is usually what you want for
     a model you'll reuse. Training runs again, with a progress bar.
   - **Ensemble of the repeated-split models** — keeps every repeated split's
     model and averages their predictions. Only useful if you trained with more
     than one repeated split.
   - **Single split** — saves one split's model.
3. Give it a **name** and click **Save model**. You get a `.cnqmodel` file to
   download, and the model now appears in the **Predict** tab.

> Only the most recent **20** training runs are kept on disk (the
> `CNQ_KEEP_RUNS` setting). Save any model you want to keep — its run folder may
> be cleaned up automatically once older runs pile up. A saved `.cnqmodel` is
> self-contained and is **not** affected by that cleanup.

### Predict new subjects

Switch to the **Predict** tab:

1. **Choose a model** — pick one of your saved models (its summary is shown), or
   upload a `.cnqmodel` file.
2. **Upload new subjects** — a CSV containing the model's feature columns.
3. **Map columns** — a table lists every model feature (with its training range)
   next to a dropdown of your file's columns. Columns are auto-matched by name
   (exact first, then ignoring case, spaces, hyphens and underscores); it shows
   e.g. *"11 of 11 features matched"* and highlights any unmatched row. If the
   names don't line up you can:
   - **Download template for this model** — a CSV with exactly the columns the
     model expects (rename your data to match), then re-upload it;
   - **Match by position** — pair features to columns in order (it asks you to
     confirm the pairs first, since a wrong order gives wrong predictions);
   - pick columns yourself from each dropdown (a likely column may be marked
     *"suggested"*).

   Optionally choose a subject **ID** column, and — only if your new subjects have
   *known* outcomes — a **time** and **event** column for external validation.
4. **Run** the prediction. You get a results table and downloads for the
   **predictions CSV**, an **HTML report** (plots), and a **zip** of everything.

The predictions CSV has, per subject: each quantile `q_<τ>`, the `median`, the
80% interval (`interval80_low`/`high`/`width`) and an `out_of_range` flag.

### What the predictions mean (and don't)

- **Between-subject spread, not model uncertainty.** The quantile interval for a
  subject describes how survival time *varies between similar subjects* at those
  covariate values. It is **not** the model's own uncertainty about its
  parameters — that isn't quantified here. A narrow interval means similar
  subjects tend to have similar outcomes, not that the model is "confident".
- **Population shift.** Predictions assume the new subjects come from a
  population *similar to the training data*. If the new cohort differs (different
  hospital, era, inclusion criteria), all predictions can be off — even ones that
  look in-range.
- **Out-of-range flags.** A subject whose feature values fall outside the
  training range is marked `out_of_range` in the output. Those predictions are
  **extrapolations** and are less reliable.
- **External validation needs unseen data.** The optional time/event metrics
  (IPCW pinball, interval coverage, Uno C-index, calibration) are only meaningful
  on subjects the model was **not** trained on. They are computed with a
  censoring estimate from the *new* data and are labelled as external validation
  in the report — never mix in rows the model already saw.

## Project population trends

The **Project trends** tab answers aggregate questions instead of per-subject
ones:

- **How many events by a time?** enter one or more time points → expected
  cumulative events (with a confidence band).
- **How long until a target?** enter a number of events → the time by which they
  are expected.

Pick a model, then choose:

- **Population size N** *(main result)* — scales the model's training-population
  survival curve (Kaplan–Meier) to N subjects. No cohort upload needed.
- **Upload a cohort** — uses the model to project *these* subjects' covariates
  (same column-mapping table as Predict). Useful when your cohort's mix differs
  from the training population.

You get a cumulative-events-over-time chart with a band, answers to your
questions, and downloads (projection CSV, report, zip).

**What to keep in mind:**

- The band is **aggregate outcome/sampling uncertainty** (Greenwood for the
  population curve, a Poisson-binomial count for a cohort) — *not* the model's own
  uncertainty about its parameters.
- Projection is only shown **up to the observed follow-up** (the last event time
  in the training data). A time — or a target number of events — beyond that is
  flagged and not projected; extrapolating the censored tail would need a
  parametric dropout/enrollment model (not in this version). Using a **dense
  quantile grid** (Every 5% / 1%) when you train gives a smoother, further-reaching
  curve.
- Population projection assumes the N subjects resemble the training population.