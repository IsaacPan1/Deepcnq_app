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

**Starting settings** reuse the hyperparameters the paper tuned on a public
dataset of similar size — SUPPORT, FLCHAIN, GBSG, GBSG500, METABRIC or NKI70,
each shown with its rough size and censoring. After you upload, the closest one
is marked **suggested**. Only the settings are reused; your data is never
combined with those datasets. Choose **Custom** to set the values yourself.

**Quantile levels** are the points on each subject's survival-time distribution
the model predicts. The default (0.1, 0.25, 0.5, 0.75, 0.9) is a good start;
"Every 10% / 5% / 1%" or a custom list give a smoother predicted curve, and
`0.1`, `0.5` and `0.9` are always included. Very low or high levels (below 0.05
or above 0.95) rely on few events and can be unreliable. With a dense grid the
tables still show the five standard levels (plus the average over the whole
grid), and the report adds individual survival curves.