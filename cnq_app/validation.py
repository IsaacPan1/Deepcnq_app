"""Data specification checks for uploaded survival CSVs.

``validate`` inspects a DataFrame against a chosen column mapping and returns
plain-language ``errors`` (which block a run), ``warnings`` (which the user must
acknowledge) and a ``summary``. Each message names the column and the number of
affected rows so the UI can render an actionable panel.

The rules follow the deepquantreg requirements the app relies on:

* duration must be numeric and > 0 (the model works on log survival time);
* the event indicator must be 0/1 (Kaplan--Meier IPCW weights need it);
* features must be numeric (they are standardised with StandardScaler).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

MIN_EVENTS_TOTAL = 100
MIN_EVENTS_PER_SPLIT = 10
HIGH_CENSORING = 0.80


def _msg(code: str, message: str, **extra: Any) -> dict:
    out = {"code": code, "message": message}
    out.update(extra)
    return out


def _examples(series: pd.Series, k: int = 3) -> list[str]:
    vals = pd.unique(series.dropna())
    return [str(v) for v in vals[:k]]


def _jsonify(value: Any) -> Any:
    """Make a distinct column value safe to send as JSON and echo back."""
    if isinstance(value, (bool, str)):
        return value
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value)
    return str(value)


def _event_series(raw: pd.Series, event_positive: Any):
    """Return (event 0/1 float series or None, info dict) for the event column.

    ``info`` carries an error code / distinct values when the column cannot be
    read as 0/1 without guessing.
    """
    nonnull = raw.dropna()
    distinct = list(pd.unique(nonnull))
    is_bool = raw.dtype == bool or any(isinstance(v, (bool, np.bool_)) for v in distinct)

    if event_positive is not None:
        mapped = raw.map(lambda v: np.nan if pd.isna(v) else (1.0 if v == event_positive else 0.0))
        return mapped.astype(float), {"mapping_applied": True, "positive": _jsonify(event_positive)}

    numeric = pd.to_numeric(raw, errors="coerce")
    num_distinct = set(numeric.dropna().unique())
    if not is_bool and num_distinct and num_distinct <= {0.0, 1.0}:
        return numeric.astype(float), {}

    if len(distinct) == 2:
        return None, {"code": "event_two_values", "values": [_jsonify(v) for v in distinct]}

    bad = int((~numeric.isin([0.0, 1.0]) & raw.notna()).sum())
    return None, {"code": "event_bad_values", "values": [_jsonify(v) for v in distinct[:6]],
                  "count": bad}


def _normalize_ratio(ratio) -> tuple[float, float, float]:
    total = float(sum(ratio))
    if total <= 0:
        raise ValueError("split ratio must be positive")
    return tuple(float(r) / total for r in ratio)  # type: ignore[return-value]


def _split_event_counts(used: pd.DataFrame, ratio, seed: int) -> dict[str, int]:
    """Events in each of train/valid/test, replicating pipeline.prepare's split."""
    from sklearn.model_selection import train_test_split

    train_f, valid_f, test_f = _normalize_ratio(ratio)
    train_valid, test = train_test_split(used, test_size=test_f, random_state=seed)
    valid_rel = valid_f / (train_f + valid_f)
    train, valid = train_test_split(train_valid, test_size=valid_rel, random_state=seed)
    return {
        "training": int(train["event"].sum()),
        "validation": int(valid["event"].sum()),
        "test": int(test["event"].sum()),
    }


def validate(df: pd.DataFrame, duration_col: str, event_col: str, feature_cols: list[str],
             id_col: str | None = None, *, ratio=(65, 15, 20), seed: int = 42,
             n_splits: int = 1, event_positive: Any = None) -> dict:
    errors: list[dict] = []
    warnings: list[dict] = []
    feature_cols = list(feature_cols or [])
    n_before = int(len(df))

    # ---- columns exist -------------------------------------------------
    wanted = [c for c in [duration_col, event_col, *feature_cols, id_col] if c]
    missing_cols = [c for c in wanted if c not in df.columns]
    for c in missing_cols:
        errors.append(_msg("missing_column", f"Column '{c}' is not in the file.", column=c))
    if duration_col not in df.columns or event_col not in df.columns:
        # Cannot do anything meaningful without duration/event.
        return {"errors": errors, "warnings": warnings,
                "summary": {"rows_before": n_before, "rows_used": 0, "rows_excluded": n_before,
                            "n_events": 0, "censoring_pct": None, "duration_min": None,
                            "duration_max": None, "n_features": len(feature_cols)}}

    # ---- no features ---------------------------------------------------
    if not feature_cols:
        errors.append(_msg("no_features",
                           "No feature columns are selected. Choose at least one covariate."))

    # ---- duration numeric ---------------------------------------------
    duration = pd.to_numeric(df[duration_col], errors="coerce")
    if duration.notna().sum() == 0:
        errors.append(_msg("duration_not_numeric",
                           f"The duration column '{duration_col}' is not numeric "
                           f"(examples: {', '.join(_examples(df[duration_col])) or 'none'}).",
                           column=duration_col, count=int(df[duration_col].notna().sum())))

    # ---- event 0/1 (or two-value mapping) -----------------------------
    event, event_info = _event_series(df[event_col], event_positive)
    if event_info.get("code") == "event_two_values":
        a, b = event_info["values"]
        errors.append(_msg("event_two_values",
                           f"The event column '{event_col}' has two values ({a} and {b}). "
                           f"Choose which one means the event happened.",
                           column=event_col, values=event_info["values"]))
    elif event_info.get("code") == "event_bad_values":
        shown = ", ".join(str(v) for v in event_info["values"])
        errors.append(_msg("event_bad_values",
                           f"The event column '{event_col}' must be 0 (censored) or 1 (event). "
                           f"Found values: {shown}. {event_info['count']} row(s) are affected.",
                           column=event_col, count=event_info["count"]))

    # ---- features numeric ---------------------------------------------
    feature_numeric: dict[str, pd.Series] = {}
    for f in feature_cols:
        if f not in df.columns:
            continue
        coerced = pd.to_numeric(df[f], errors="coerce")
        feature_numeric[f] = coerced
        if coerced.notna().sum() == 0:
            errors.append(_msg("feature_not_numeric",
                               f"Feature '{f}' is not numeric "
                               f"(examples: {', '.join(_examples(df[f])) or 'none'}).",
                               column=f, count=int(df[f].notna().sum()),
                               examples=_examples(df[f])))

    # ---- duplicate IDs -------------------------------------------------
    if id_col and id_col in df.columns:
        dup = int(df[id_col].dropna().duplicated(keep=False).sum())
        if dup:
            errors.append(_msg("duplicate_ids",
                               f"The ID column '{id_col}' repeats: {dup} row(s) share an ID. "
                               "Each subject must appear once.", column=id_col, count=dup))

    # ---- build the 'used' frame + exclusion accounting ----------------
    work = pd.DataFrame(index=df.index)
    work["duration"] = duration
    work["event"] = event if event is not None else np.nan
    for f in feature_cols:
        if f in feature_numeric:
            work[f] = feature_numeric[f]
    used_cols = ["duration"] + (["event"] if event is not None else []) + \
                [f for f in feature_cols if f in feature_numeric]

    # missing / non-numeric per column -> excluded rows (warning)
    def _missing_count(col_key, source):
        return int(source.isna().sum())

    missing = {}
    missing[duration_col] = _missing_count("duration", work["duration"])
    if event is not None:
        missing[event_col] = _missing_count("event", work["event"])
    for f in feature_cols:
        if f in feature_numeric:
            missing[f] = _missing_count(f, work[f])
    if id_col and id_col in df.columns:
        missing[id_col] = int(df[id_col].isna().sum())
    for col, count in missing.items():
        if count:
            warnings.append(_msg("missing_values",
                                 f"Column '{col}' has {count} blank or non-numeric cell(s); "
                                 "those rows will be excluded.", column=col, count=count))

    # duration <= 0 (warning; excluded)
    nonpos = int((work["duration"] <= 0).sum())
    if nonpos:
        warnings.append(_msg("nonpositive_duration",
                             f"{nonpos} row(s) have a duration of 0 or less in '{duration_col}'; "
                             "they will be excluded (the model uses log time).",
                             column=duration_col, count=nonpos))

    complete = work[used_cols].notna().all(axis=1)
    if id_col and id_col in df.columns:
        complete &= df[id_col].notna()
    positive = work["duration"] > 0
    used = work[complete & positive]
    rows_used = int(len(used))

    # ---- event counts / censoring -------------------------------------
    n_events = int(used["event"].sum()) if (event is not None and rows_used) else 0
    censoring = (1.0 - n_events / rows_used) if rows_used else None

    if event is not None and rows_used and n_events == 0:
        errors.append(_msg("no_events",
                           f"There are no events (all rows are censored) in '{event_col}'. "
                           "The model needs at least some observed events."))
    if event is not None and rows_used and 0 < n_events < MIN_EVENTS_TOTAL:
        warnings.append(_msg("few_events_total",
                             f"Only {n_events} events in total (fewer than {MIN_EVENTS_TOTAL}). "
                             "Results may be unstable.", count=n_events))
    if censoring is not None and censoring > HIGH_CENSORING and n_events > 0:
        warnings.append(_msg("high_censoring",
                             f"{censoring * 100:.0f}% of rows are censored (over "
                             f"{HIGH_CENSORING * 100:.0f}%). Estimates may be unreliable.",
                             count=rows_used - n_events))

    # ---- constant features (warning) ----------------------------------
    for f in feature_cols:
        if f in feature_numeric and rows_used:
            if used[f].nunique(dropna=True) <= 1:
                warnings.append(_msg("constant_feature",
                                     f"Feature '{f}' has the same value in every used row; "
                                     "it adds no information.", column=f))

    # ---- event/duration also chosen as a feature (warning) ------------
    if duration_col in feature_cols:
        warnings.append(_msg("duration_is_feature",
                             f"The duration column '{duration_col}' is also selected as a "
                             "feature. This leaks the outcome; remove it from features.",
                             column=duration_col))
    if event_col in feature_cols:
        warnings.append(_msg("event_is_feature",
                             f"The event column '{event_col}' is also selected as a feature. "
                             "This leaks the outcome; remove it from features.", column=event_col))

    # ---- fewer than 10 events in any split ----------------------------
    if event is not None and n_events > 0 and rows_used >= 3:
        worst = {}
        try:
            for i in range(max(1, int(n_splits))):
                counts = _split_event_counts(used, ratio, int(seed) + i)
                for part, c in counts.items():
                    worst[part] = min(worst.get(part, c), c)
        except Exception:  # noqa: BLE001 - splitting can fail on tiny frames
            worst = {}
        for part, c in worst.items():
            if c < MIN_EVENTS_PER_SPLIT:
                t, v, te = (int(round(x)) for x in ratio)
                errors.append(_msg("few_events_split",
                                   f"With a {t}/{v}/{te} split (seed {seed}), the {part} set has "
                                   f"only {c} event(s); each split needs at least "
                                   f"{MIN_EVENTS_PER_SPLIT}. Use fewer/larger splits or more data.",
                                   count=c, part=part))

    summary = {
        "rows_before": n_before,
        "rows_used": rows_used,
        "rows_excluded": n_before - rows_used,
        "n_events": n_events,
        "censoring_pct": round(censoring * 100, 1) if censoring is not None else None,
        "duration_min": float(used["duration"].min()) if rows_used else None,
        "duration_max": float(used["duration"].max()) if rows_used else None,
        "n_features": len(feature_cols),
    }
    if event_info.get("mapping_applied"):
        summary["event_positive"] = event_info["positive"]
    return {"errors": errors, "warnings": warnings, "summary": summary}
