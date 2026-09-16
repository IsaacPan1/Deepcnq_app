"""Predict survival-time quantiles for new subjects from a saved ``.cnqmodel``.

The flow mirrors training exactly so predictions are reproducible:

* columns are matched to the bundle's ordered feature names (every training
  feature is required; extras are ignored);
* rows with a missing / non-numeric feature value are excluded, never imputed;
* features are standardised with the bundle's *training* scaler (mean/scale),
  fed in the bundle's order, and each ensemble member is evaluated with the
  package's ``predict_quantiles``;
* member predictions are averaged on the log scale (which preserves the
  non-crossing of the ``_gaps`` models) and mapped back to time with ``exp``.

When the new CSV also carries time and event columns, ``external_validation``
scores the predictions with IPCW metrics whose censoring distribution is
estimated from the *new* data -- clearly external validation, never the
training set.
"""
from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

import pipeline
from deepquantreg.metrics import CensoringKM
from deepquantreg.metrics.survival import uno_c_index, weighted_pinball_mean
from deepquantreg.training import predict_quantiles  # package binding: NOT the job progress hook


# --------------------------------------------------------------------------- #
# column matching
# --------------------------------------------------------------------------- #
def auto_feature_map(feature_names: list[str], columns: list[str]) -> dict[str, str]:
    """Best-effort map bundle feature -> CSV column, by exact then case-insensitive name."""
    lower = {c.lower(): c for c in columns}
    out: dict[str, str] = {}
    for f in feature_names:
        if f in columns:
            out[f] = f
        elif f.lower() in lower:
            out[f] = lower[f.lower()]
    return out


def _resolve_map(meta: dict, columns: list[str], feature_map: dict | None) -> dict[str, str]:
    features = list(meta.get("feature_names") or [])
    fmap = dict(feature_map or {})
    auto = auto_feature_map(features, columns)
    for f in features:
        fmap.setdefault(f, auto.get(f))
    # Only keep entries pointing at a real column.
    return {f: c for f, c in fmap.items() if c in columns}


# --------------------------------------------------------------------------- #
# validation (reuses the numeric rules validation.py applies at training time)
# --------------------------------------------------------------------------- #
def _msg(code: str, message: str, **extra: Any) -> dict:
    return {"code": code, "message": message, **extra}


def validate(meta: dict, raw: pd.DataFrame, feature_map: dict | None = None,
             *, id_col: str | None = None, time_col: str | None = None,
             event_col: str | None = None, event_positive: Any = None) -> dict:
    """Errors block prediction; warnings are advisory. Mirrors validation.py's
    numeric rules but is anchored to the bundle's required feature set."""
    errors: list[dict] = []
    warnings: list[dict] = []
    features = list(meta.get("feature_names") or [])
    columns = list(raw.columns)
    fmap = _resolve_map(meta, columns, feature_map)

    missing = [f for f in features if f not in fmap]
    if missing:
        errors.append(_msg("missing_features",
                           "The saved model needs these feature column(s) that aren't "
                           f"mapped: {', '.join(missing)}.", features=missing))

    # numeric checks on mapped features
    numeric: dict[str, pd.Series] = {}
    for f in features:
        col = fmap.get(f)
        if not col:
            continue
        coerced = pd.to_numeric(raw[col], errors="coerce")
        numeric[f] = coerced
        if coerced.notna().sum() == 0:
            examples = [str(v) for v in pd.unique(raw[col].dropna())[:3]]
            errors.append(_msg("feature_not_numeric",
                               f"Feature '{f}' (column '{col}') has no numeric values "
                               f"(examples: {', '.join(examples) or 'none'}).", column=col))

    if id_col and id_col in columns:
        dup = int(raw[id_col].dropna().duplicated(keep=False).sum())
        if dup:
            warnings.append(_msg("duplicate_ids",
                                 f"The ID column '{id_col}' repeats in {dup} row(s).",
                                 column=id_col, count=dup))

    # rows excluded for a missing feature value (only when all features numeric)
    rows_used = 0
    n_excluded = 0
    if not missing and not any(e["code"] == "feature_not_numeric" for e in errors):
        mat = pd.DataFrame({f: numeric[f] for f in features})
        complete = mat.notna().all(axis=1)
        rows_used = int(complete.sum())
        n_excluded = int(len(raw) - rows_used)
        per_col = {f: int(numeric[f].isna().sum()) for f in features if int(numeric[f].isna().sum())}
        for f, c in per_col.items():
            warnings.append(_msg("missing_values",
                                 f"Feature '{f}' has {c} blank/non-numeric cell(s); those "
                                 "rows are excluded from prediction (not imputed).",
                                 column=fmap[f], count=c))
        # out-of-range vs training min/max
        oor = _out_of_range_counts(meta, mat[complete], features)
        total_oor = int(oor.get("_any_rows", 0))
        if total_oor:
            per = ", ".join(f"{f}: {n}" for f, n in oor["per_feature"].items() if n)
            warnings.append(_msg("out_of_range",
                                 f"{total_oor} subject(s) have at least one feature outside the "
                                 f"training range ({per}). They are flagged in the output; "
                                 "predictions there are extrapolations.",
                                 count=total_oor, per_feature=oor["per_feature"]))

    summary = {
        "rows_before": int(len(raw)),
        "rows_used": rows_used,
        "rows_excluded": n_excluded,
        "n_features": len(features),
        "has_external": bool(time_col and event_col),
    }
    return {"errors": errors, "warnings": warnings, "summary": summary, "feature_map": fmap}


def _out_of_range_counts(meta: dict, feat_df: pd.DataFrame, features: list[str]) -> dict:
    ranges = meta.get("feature_ranges") or {}
    per_feature: dict[str, int] = {}
    any_mask = np.zeros(len(feat_df), dtype=bool)
    for f in features:
        r = ranges.get(f)
        if not r:
            per_feature[f] = 0
            continue
        lo, hi = float(r["min"]), float(r["max"])
        col = feat_df[f].to_numpy(float)
        out = (col < lo) | (col > hi)
        per_feature[f] = int(out.sum())
        any_mask |= out
    return {"per_feature": per_feature, "_any_rows": int(any_mask.sum()), "_any_mask": any_mask}


# --------------------------------------------------------------------------- #
# prediction
# --------------------------------------------------------------------------- #
def _apply_scaler(raw_x: np.ndarray, scaler: dict, n_features: int) -> np.ndarray:
    mean = np.asarray((scaler or {}).get("mean"), dtype=float)
    scale = np.asarray((scaler or {}).get("scale"), dtype=float)
    if mean.shape != (n_features,) or scale.shape != (n_features,):
        raise ValueError("bundle scaler mean/scale do not match the feature count")
    scale = np.where(scale == 0, 1.0, scale)             # match StandardScaler's zero-variance rule
    return ((raw_x - mean) / scale).astype("float32")


def _member_scalers(meta: dict) -> list[dict]:
    """One scaler per member. Ensembles carry ``member_scalers`` (each split was
    standardised with its own training scaler); other bundles fall back to the
    single top-level scaler."""
    members = meta.get("member_scalers")
    if members:
        return list(members)
    return [meta.get("scaler") or {}]


def ensemble_predict_log(models: list, raw_x: np.ndarray, scalers: list[dict],
                         n_features: int, device: str = "cpu") -> np.ndarray:
    """Average member predictions on the log scale.

    Each member is standardised with *its own* training scaler, then evaluated;
    because every ``_gaps`` member returns non-decreasing log-quantiles, their
    elementwise mean is also non-decreasing -- the ensemble stays non-crossing
    without any re-sorting.
    """
    acc = None
    for i, model in enumerate(models):
        scaler = scalers[i] if i < len(scalers) else scalers[-1]
        x = _apply_scaler(raw_x, scaler, n_features)
        pred = np.asarray(predict_quantiles(model, x, device), dtype=float)
        acc = pred if acc is None else acc + pred
    return acc / len(models)


def run_prediction(bundle, raw: pd.DataFrame, feature_map: dict, *,
                   id_col: str | None = None, time_col: str | None = None,
                   event_col: str | None = None, event_positive: Any = None,
                   device: str = "cpu",
                   progress: Optional[Callable[[str, float], None]] = None) -> dict:
    """Predict quantiles for every usable row and (optionally) externally validate.

    ``bundle`` is a ``model_io.LoadedBundle``. ``feature_map`` maps each bundle
    feature to a CSV column (see ``validate``). ``progress(step, frac)`` is an
    optional UI hook.
    """
    def tick(step: str, frac: float):
        if progress:
            progress(step, frac)

    meta = bundle.meta
    features = list(meta.get("feature_names") or [])
    quantiles = [float(q) for q in meta.get("quantiles") or []]
    fmap = _resolve_map(meta, list(raw.columns), feature_map)
    missing = [f for f in features if f not in fmap]
    if missing:
        raise ValueError(f"missing required feature column(s): {', '.join(missing)}")

    tick("Preparing data", 0.1)
    # numeric feature matrix, original index preserved
    numeric = {f: pd.to_numeric(raw[fmap[f]], errors="coerce") for f in features}
    mat = pd.DataFrame(numeric, index=raw.index)
    complete = mat.notna().all(axis=1)
    feat_df = mat[complete].reset_index(drop=True)
    kept_index = raw.index[complete]
    n = len(feat_df)
    if n == 0:
        raise ValueError("no rows left to predict after excluding missing feature values")

    # subject ids
    if id_col and id_col in raw.columns:
        ids = raw.loc[kept_index, id_col].to_numpy()
    else:
        ids = np.arange(n)

    tick("Running the model", 0.4)
    raw_x = feat_df[features].to_numpy(dtype=float)                   # bundle feature order
    scalers = _member_scalers(meta)
    pred_log = ensemble_predict_log(bundle.models, raw_x, scalers, len(features), device)
    pred_time = np.exp(pred_log)                                      # (n, Q), log-time -> time

    # out-of-range flags aligned to feat_df
    oor = _out_of_range_counts(meta, feat_df, features)
    oor_mask = oor["_any_mask"]

    tick("Assembling predictions", 0.7)
    frame = _predictions_frame(ids, pred_time, quantiles, oor_mask, id_named=bool(id_col))

    external = None
    if time_col and event_col and time_col in raw.columns and event_col in raw.columns:
        tick("External validation", 0.85)
        external = external_validation(
            raw.loc[kept_index], pred_log, quantiles, time_col, event_col,
            event_positive=event_positive)

    tick("Done", 1.0)
    preview_cols = list(frame.columns)
    preview_rows = frame.head(20).astype(object).where(frame.head(20).notna(), None).values.tolist()
    return {
        "n": n,
        "rows_excluded": int(len(raw) - n),
        "quantiles": quantiles,
        "feature_map": fmap,
        "subject_ids": [_py(v) for v in ids],
        "pred_log": pred_log,
        "pred_time": pred_time,
        "out_of_range_mask": oor_mask.tolist(),
        "out_of_range_count": int(oor_mask.sum()),
        "per_feature_oor": oor["per_feature"],
        "predictions_csv": frame.to_csv(index=False),
        "columns": preview_cols,
        "preview_rows": preview_rows,
        "median_time": np.exp(pred_log[:, _median_idx(quantiles)]).tolist(),
        "width80": _interval_width(pred_time, quantiles).tolist(),
        "external": external,
    }


def _median_idx(quantiles: list[float]) -> int:
    idx = pipeline._index_of(quantiles, 0.5)
    if idx is None:
        raise ValueError("the model's quantile grid has no 0.5 (median) level")
    return idx


def _tau_idx(quantiles: list[float], tau: float) -> int:
    idx = pipeline._index_of(quantiles, tau)
    if idx is None:
        raise ValueError(f"the model's quantile grid has no {tau} level")
    return idx


def _interval_width(pred_time: np.ndarray, quantiles: list[float]) -> np.ndarray:
    lo, hi = _tau_idx(quantiles, 0.1), _tau_idx(quantiles, 0.9)
    return pred_time[:, hi] - pred_time[:, lo]


def _predictions_frame(ids, pred_time, quantiles, oor_mask, *, id_named: bool) -> pd.DataFrame:
    cols: dict[str, Any] = {"subject_id": ids}
    for j, q in enumerate(quantiles):
        cols[f"q_{q:g}"] = pred_time[:, j]
    med = _median_idx(quantiles)
    lo, hi = _tau_idx(quantiles, 0.1), _tau_idx(quantiles, 0.9)
    cols["median"] = pred_time[:, med]
    cols["interval80_low"] = pred_time[:, lo]
    cols["interval80_high"] = pred_time[:, hi]
    cols["interval80_width"] = pred_time[:, hi] - pred_time[:, lo]
    cols["out_of_range"] = oor_mask.astype(int)
    return pd.DataFrame(cols)


# --------------------------------------------------------------------------- #
# external validation (IPCW; censoring estimated from the NEW data)
# --------------------------------------------------------------------------- #
def external_validation(raw_used: pd.DataFrame, pred_log: np.ndarray, quantiles: list[float],
                        time_col: str, event_col: str, *, event_positive: Any = None) -> dict:
    """Score predictions against observed outcomes in the new data.

    IPCW weights use a Kaplan--Meier censoring model fitted on *these* rows, so
    every metric is genuine external validation.
    """
    time = pd.to_numeric(raw_used[time_col], errors="coerce").to_numpy(float)
    if event_positive is not None:
        event = raw_used[event_col].map(
            lambda v: np.nan if pd.isna(v) else (1.0 if v == event_positive else 0.0)).to_numpy(float)
    else:
        event = pd.to_numeric(raw_used[event_col], errors="coerce").to_numpy(float)

    valid = np.isfinite(time) & np.isfinite(event) & (time > 0) & np.isin(event, [0.0, 1.0])
    n_valid = int(valid.sum())
    if n_valid < 5 or event[valid].sum() == 0:
        return {"available": False,
                "note": "Not enough rows with a valid positive time and 0/1 event "
                        "(and at least one event) for external validation."}

    t = time[valid]
    e = event[valid].astype(int)
    preds = pred_log[valid]

    horizon = float(np.max(t[e == 1]))
    horizon_log = float(np.log(max(horizon, 1e-12)))
    km = CensoringKM(0.005).fit(t, e)
    weights = km.weights(t, e)

    pin_per_tau = pipeline.per_tau_pinball(t, preds, weights, quantiles, horizon_log)
    pin_mean = weighted_pinball_mean(np.log(np.maximum(t, 1e-12)), preds, weights,
                                     quantiles, horizon_log)
    cov_80 = pipeline.interval_coverage(t, preds, weights, quantiles, 0.1, 0.9, horizon_log)
    cov_50 = pipeline.interval_coverage(t, preds, weights, quantiles, 0.25, 0.75, horizon_log)

    from deepquantreg.diagnostics.quantiles import calibration_by_tau
    calib = calibration_by_tau(t, preds, quantiles, weights)

    try:
        median_log = preds[:, _median_idx(quantiles)]
        uno = uno_c_index(t, e, t, e, median_log, horizon)
    except Exception:  # noqa: BLE001 - concordance can fail on degenerate data
        uno = None

    km_curve = pipeline.km_survival_curve(t, e)
    grid, mean_surv = mean_predicted_survival(preds, quantiles, horizon)

    return {
        "available": True,
        "n": n_valid,
        "n_events": int(e.sum()),
        "horizon": horizon,
        "pinball_mean": _py(pin_mean),
        "pinball_per_tau": [{"tau": float(q), "value": _py(v)}
                            for q, v in zip(quantiles, pin_per_tau)],
        "coverage_80": _py(cov_80),
        "coverage_50": _py(cov_50),
        "uno_c": _py(uno) if uno is not None else None,
        "calibration": {str(k): _py(v) for k, v in calib.items()},
        "km": km_curve,
        "predicted_survival": {"time": grid.tolist(), "survival": mean_surv.tolist()},
        "note": "External validation on the uploaded data. The censoring distribution is "
                "estimated from these rows, not from the training data.",
    }


def mean_predicted_survival(pred_log: np.ndarray, quantiles: list[float],
                            horizon: float, n_grid: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """Mean predicted survival curve S(t) implied by the quantile predictions.

    For subject i, the tau-quantile of time q_tau satisfies P(T<=q_tau)=tau, i.e.
    S_i(q_tau)=1-tau. Interpolating tau against predicted times gives S_i on a
    common grid; averaging over subjects yields the population curve to overlay
    on the Kaplan--Meier estimate of the new outcomes.
    """
    q = np.asarray(quantiles, float)
    times = np.exp(pred_log)                    # (n, Q), increasing across columns
    grid = np.linspace(0.0, max(horizon, 1e-6), n_grid)
    surv = np.empty((times.shape[0], n_grid))
    for i in range(times.shape[0]):
        tau_at_t = np.interp(grid, times[i], q, left=q[0], right=q[-1])
        surv[i] = 1.0 - tau_at_t
    return grid, np.clip(surv.mean(axis=0), 0.0, 1.0)


def _py(v):
    if v is None:
        return None
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (bytes, np.bytes_)):
        return v.decode("utf-8", "replace")
    if isinstance(v, np.generic):
        return v.item()
    return v
