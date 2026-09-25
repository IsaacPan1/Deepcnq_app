"""Data preparation, training and evaluation built on the deepquantreg package.

Everything here imports deepquantreg directly (never scripts/train.py). The
``prepare`` function replicates ``prepare_real`` from scripts/train.py: a
training-fitted StandardScaler for the features and a training-fitted
Kaplan--Meier censoring model for the IPCW weights.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import paths  # noqa: F401 -- puts deepquantreg on sys.path
from deepquantreg import build_model
from deepquantreg.data.preprocessing import PreparedData
from deepquantreg.metrics import CensoringKM
from deepquantreg.metrics.survival import (
    pinball, uno_c_index, weighted_mean, weighted_pinball_mean,
)
from deepquantreg.diagnostics.quantiles import (
    adjacent_crossing_rate, calibration_by_tau, interval_width, outer_crossing_rate,
)
from deepquantreg.training import fit
from deepquantreg.training import predict_quantiles as _predict_original

REQUIRED_QUANTILES = (0.1, 0.5, 0.9)


# --------------------------------------------------------------------------- #
# Data preparation
# --------------------------------------------------------------------------- #
def load_frame(path) -> pd.DataFrame:
    return pd.read_csv(path)


def build_frame(raw: pd.DataFrame, duration_col: str, event_col: str,
                feature_cols: list[str], id_col: str | None = None,
                event_positive=None) -> pd.DataFrame:
    """Rename the mapped columns to the package's canonical duration/event/x.

    Rows with missing/non-numeric cells in the selected columns, and rows whose
    duration is <= 0, are excluded (they are reported as validation warnings).
    ``event_positive`` maps a two-value event column onto 1/0; ``id_col`` labels
    each row's ``subject_id`` (otherwise a 0..n index is used).
    """
    wanted = [duration_col, event_col, *feature_cols, *( [id_col] if id_col else [])]
    missing = [c for c in wanted if c not in raw.columns]
    if missing:
        raise ValueError(f"columns not found in CSV: {missing}")
    if not feature_cols:
        raise ValueError("select at least one feature column")
    frame = pd.DataFrame()
    frame["duration"] = pd.to_numeric(raw[duration_col], errors="coerce").astype(float)
    if event_positive is not None:
        frame["event"] = raw[event_col].map(
            lambda v: np.nan if pd.isna(v) else (1.0 if v == event_positive else 0.0))
    else:
        frame["event"] = pd.to_numeric(raw[event_col], errors="coerce").astype(float)
    for col in feature_cols:
        frame[col] = pd.to_numeric(raw[col], errors="coerce").astype(float)
    if id_col:
        frame["subject_id"] = raw[id_col].to_numpy()
        frame = frame.dropna(subset=[c for c in frame.columns if c != "subject_id"])
    else:
        frame = frame.dropna()
    frame = frame[frame["duration"] > 0].reset_index(drop=True)  # exclude non-positive durations
    if frame.empty:
        raise ValueError("no usable rows after excluding missing values and non-positive durations")
    uniq = set(np.unique(frame["event"].to_numpy()))
    if not uniq.issubset({0.0, 1.0}):
        raise ValueError("event column must be binary (1=observed event, 0=censored)")
    if frame["event"].sum() == 0:
        raise ValueError("event column has no observed events (all censored)")
    frame["event"] = frame["event"].astype(int)
    if not id_col:
        frame["subject_id"] = np.arange(len(frame))
    return frame


def split_ratio_to_fracs(ratio: tuple[float, float, float]) -> tuple[float, float, float]:
    total = float(sum(ratio))
    if total <= 0:
        raise ValueError("split ratio must sum to a positive number")
    return tuple(r / total for r in ratio)  # type: ignore[return-value]


def prepare(frame: pd.DataFrame, feature_cols: list[str], ratio: tuple[float, float, float],
            split_seed: int, epsilon: float = 0.005) -> PreparedData:
    """Split, scale and weight one dataset. Mirrors prepare_real in train.py.

    The split ratio is (train, valid, test). Test is carved off first, then
    valid from the remainder, matching the nested train_test_split in
    data/preprocessing.prepare_splits but with a user-chosen ratio.
    """
    train_f, valid_f, test_f = split_ratio_to_fracs(ratio)
    train_valid, test = train_test_split(frame, test_size=test_f, random_state=split_seed)
    valid_rel = valid_f / (train_f + valid_f)
    train, valid = train_test_split(train_valid, test_size=valid_rel, random_state=split_seed)

    scaler = StandardScaler().fit(train[feature_cols])
    km = CensoringKM(epsilon).fit(train.duration, train.event)

    def pack(part: pd.DataFrame) -> dict:
        time, event = part.duration.to_numpy(float), part.event.to_numpy(int)
        return {
            "X": scaler.transform(part[feature_cols]).astype("float32"),
            "time": time,
            "event": event,
            "weights": km.weights(time, event).astype("float32"),
            "subject_id": part.subject_id.to_numpy(),
        }

    return PreparedData(pack(train), pack(valid), pack(test), scaler, km,
                        list(feature_cols), split_seed)


def prepare_all(frame: pd.DataFrame, feature_cols: list[str], valid_frac: float,
                seed: int, epsilon: float = 0.005):
    """Single train/valid split over ALL rows, for refitting a final model.

    Used when saving a "final" bundle: the chosen architecture is retrained on
    everything, holding out ``valid_frac`` (default 15%) purely for early
    stopping. The scaler and censoring KM are fitted on the train part only, as
    in :func:`prepare`. Returns ``(train, valid, scaler, km)`` -- no test set.
    """
    train, valid = train_test_split(frame, test_size=valid_frac, random_state=seed)
    scaler = StandardScaler().fit(train[feature_cols])
    km = CensoringKM(epsilon).fit(train.duration, train.event)

    def pack(part: pd.DataFrame) -> dict:
        time, event = part.duration.to_numpy(float), part.event.to_numpy(int)
        return {
            "X": scaler.transform(part[feature_cols]).astype("float32"),
            "time": time,
            "event": event,
            "weights": km.weights(time, event).astype("float32"),
            "subject_id": part.subject_id.to_numpy(),
        }

    return pack(train), pack(valid), scaler, km


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def _horizon(prepared: PreparedData) -> float:
    tr = prepared.train
    return float(np.max(tr["time"][tr["event"] == 1]))


def per_tau_pinball(time, predictions_log, weights, quantiles, horizon_log) -> list[float]:
    y = np.log(np.maximum(np.asarray(time, float), 1e-12))
    preds = np.asarray(predictions_log, float)
    y_t = np.minimum(y, horizon_log)
    preds_t = np.minimum(preds, horizon_log)
    return [weighted_mean(pinball(y_t, preds_t[:, j], tau), weights)
            for j, tau in enumerate(quantiles)]


def interval_coverage(time, predictions_log, weights, quantiles, lo_tau, hi_tau, horizon_log):
    """Weighted coverage of the [lo_tau, hi_tau] prediction interval (ICP)."""
    quantiles = list(quantiles)
    lo = _index_of(quantiles, lo_tau)
    hi = _index_of(quantiles, hi_tau)
    if lo is None or hi is None:
        return None
    y = np.minimum(np.log(np.maximum(np.asarray(time, float), 1e-12)), horizon_log)
    preds = np.minimum(np.asarray(predictions_log, float), horizon_log)
    inside = ((y >= preds[:, lo]) & (y <= preds[:, hi])).astype(float)
    return weighted_mean(inside, weights)


def _index_of(quantiles, tau):
    for j, q in enumerate(quantiles):
        if np.isclose(q, tau):
            return j
    return None


def evaluate_split(prepared: PreparedData, test_pred, quantiles, include_unoc=True) -> dict:
    """Full metric bundle for one trained model on one split's test set."""
    km = prepared.censoring
    test = prepared.test
    horizon = _horizon(prepared)
    horizon_log = float(np.log(max(horizon, 1e-12)))
    weights = km.weights(test["time"], test["event"])

    result: dict[str, Any] = {}
    result["pinball_mean"] = weighted_pinball_mean(
        np.log(np.maximum(test["time"], 1e-12)), test_pred, weights, quantiles, horizon_log)
    result["pinball_per_tau"] = per_tau_pinball(
        test["time"], test_pred, weights, quantiles, horizon_log)
    result["icp_80"] = interval_coverage(
        test["time"], test_pred, weights, quantiles, 0.1, 0.9, horizon_log)
    result["icp_50"] = interval_coverage(
        test["time"], test_pred, weights, quantiles, 0.25, 0.75, horizon_log)
    result["adjacent_crossing"] = adjacent_crossing_rate(test_pred)
    result["outer_crossing"] = outer_crossing_rate(test_pred)
    result["calibration"] = calibration_by_tau(test["time"], test_pred, quantiles, weights)

    median = _index_of(quantiles, 0.5)
    if include_unoc and median is not None:
        result["uno_c"] = uno_c_index(
            prepared.train["time"], prepared.train["event"], test["time"], test["event"],
            np.asarray(test_pred)[:, median], horizon)
    else:
        result["uno_c"] = None
    return result


def permutation_importance(model, prepared: PreparedData, quantiles, *, n_repeats=3,
                           seed=0) -> np.ndarray:
    """Increase in per-tau test pinball when each feature is permuted.

    Returns an array of shape (n_features, n_quantiles). Larger = more important.
    """
    test = prepared.test
    km = prepared.censoring
    horizon_log = float(np.log(max(_horizon(prepared), 1e-12)))
    weights = km.weights(test["time"], test["event"])
    base_pred = _predict_original(model, test["X"])
    base = np.array(per_tau_pinball(test["time"], base_pred, weights, quantiles, horizon_log))

    rng = np.random.RandomState(seed)
    n_features = test["X"].shape[1]
    importance = np.zeros((n_features, len(quantiles)))
    for f in range(n_features):
        deltas = np.zeros((n_repeats, len(quantiles)))
        for r in range(n_repeats):
            perturbed = test["X"].copy()
            perturbed[:, f] = perturbed[rng.permutation(len(perturbed)), f]
            pred = _predict_original(model, perturbed)
            deltas[r] = np.array(
                per_tau_pinball(test["time"], pred, weights, quantiles, horizon_log)) - base
        importance[f] = deltas.mean(axis=0)
    return importance


def km_survival_curve(time, event) -> dict:
    """Standard Kaplan--Meier estimate of the outcome survival function S(t).

    Also returns the Greenwood variance of S(t) (``var``), the last observed
    event time (``horizon`` -- the furthest the estimate is supported) and whether
    follow-up extends past it via censoring (``last_is_censored``). Population
    projection uses these for the trend, its confidence band and the range guard.
    """
    time = np.asarray(time, float)
    event = np.asarray(event, int)
    order = np.argsort(time)
    time, event = time[order], event[order]
    uniq = np.unique(time)
    surv, gvar = 1.0, 0.0                      # gvar = cumulative Greenwood sum
    values, var_out = [], []
    last_event_time = 0.0
    n = len(time)
    for t in uniq:
        at_risk = int(np.sum(time >= t))
        d = int(np.sum((time == t) & (event == 1)))
        if at_risk > 0 and d > 0:
            surv *= 1.0 - d / at_risk
            if at_risk - d > 0:
                gvar += d / (at_risk * (at_risk - d))
            last_event_time = float(t)
        values.append(surv)
        var_out.append(surv * surv * gvar)     # Var(S(t)) = S(t)^2 * sum d/(n(n-d))
    last_time = float(uniq[-1]) if len(uniq) else 0.0
    return {
        "time": [0.0, *[float(t) for t in uniq]],
        "survival": [1.0, *[float(v) for v in values]],
        "var": [0.0, *[float(v) for v in var_out]],
        "n": int(n),
        "n_events": int(event.sum()),
        "horizon": last_event_time,
        "last_is_censored": bool(last_time > last_event_time),
    }


# --------------------------------------------------------------------------- #
# One training run (model x split)
# --------------------------------------------------------------------------- #
@dataclass
class TrainOutput:
    model_name: str
    split_index: int
    best_epoch: int
    epochs_ran: int
    early_stopped: bool
    history: list[dict]
    metrics: dict
    test_pred: np.ndarray
    quantiles: list[float]
    # kept only for the representative (first) split to power detailed plots
    keep_model: Any = None
    prepared: Any = None
    # CPU copy of the trained weights, always captured so a later "save model"
    # step can bundle the exact in-session model (single split / ensemble).
    state_dict: Any = None
    # best validation IPCW-pinball (early-stopping metric) -- auto-tune ranks on this.
    valid_pinball: Any = None


def train_model_on_split(model_name: str, resolved: dict, prepared: PreparedData,
                         seed: int, split_index: int, *, device="cpu",
                         deterministic=False, keep_for_plots=False,
                         include_unoc=True) -> TrainOutput:
    quantiles = [float(q) for q in resolved["quantiles"]]
    architecture = dict(resolved["architecture"])
    architecture.update(name=model_name, input_dim=len(prepared.feature_columns),
                        quantiles=quantiles)
    model = build_model(architecture)
    training = resolved["training"]
    model, state = fit(
        model, prepared.train, prepared.valid, quantiles, seed=seed,
        max_epochs=int(training["maximum_epochs"]), batch_size=int(training["batch_size"]),
        lr=float(training["learning_rate"]), weight_decay=float(training["weight_decay"]),
        optimizer=training["optimizer"], scheduler=training["scheduler"],
        patience=int(training["patience"]), device=device, deterministic=deterministic,
    )
    test_pred = _predict_original(model, prepared.test["X"], device)
    metrics = evaluate_split(prepared, test_pred, quantiles, include_unoc=include_unoc)
    cpu_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    valid = state.get("best_valid_trainG_pinball_mean")
    return TrainOutput(
        model_name=model_name, split_index=split_index, best_epoch=state["best_epoch"],
        epochs_ran=state["epochs_ran"], early_stopped=state["early_stopped"],
        history=state["history"], metrics=metrics, test_pred=test_pred, quantiles=quantiles,
        keep_model=model if keep_for_plots else None,
        prepared=prepared if keep_for_plots else None,
        state_dict=cpu_state,
        valid_pinball=float(valid) if valid is not None else None,
    )


# --------------------------------------------------------------------------- #
# Aggregation across splits
# --------------------------------------------------------------------------- #
def _mean_std(values):
    arr = np.array([v for v in values if v is not None], float)
    if arr.size == 0:
        return {"mean": None, "std": None}
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=0))}


def summarize_model(outputs: list[TrainOutput], quantiles) -> dict:
    per_tau = np.array([o.metrics["pinball_per_tau"] for o in outputs], float)
    summary = {
        "pinball_mean": _mean_std([o.metrics["pinball_mean"] for o in outputs]),
        "pinball_per_tau": [
            {"tau": float(q), **_mean_std(per_tau[:, j])} for j, q in enumerate(quantiles)
        ],
        "icp_80": _mean_std([o.metrics["icp_80"] for o in outputs]),
        "icp_50": _mean_std([o.metrics["icp_50"] for o in outputs]),
        "uno_c": _mean_std([o.metrics["uno_c"] for o in outputs]),
        "adjacent_crossing": _mean_std([o.metrics["adjacent_crossing"] for o in outputs]),
        "outer_crossing": _mean_std([o.metrics["outer_crossing"] for o in outputs]),
        "best_epoch": _mean_std([o.best_epoch for o in outputs]),
        "best_epochs": [o.best_epoch for o in outputs],
        "pinball_per_split": [o.metrics["pinball_mean"] for o in outputs],
    }
    return summary
