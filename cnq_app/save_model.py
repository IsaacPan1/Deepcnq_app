"""Assemble ``.cnqmodel`` bundles from a finished training run.

Three bundle types (see the spec / README):

* ``final``    -- retrain the chosen architecture on ALL rows, holding out 15%
                  for early stopping (same seed), and bundle that one model;
* ``ensemble`` -- bundle every repeated split's weights; predictions average on
                  the log scale (non-crossing preserved);
* ``single_split`` -- bundle one chosen split's weights.

The functions here are pure (no job / HTTP dependencies) so the round-trip test
can bundle in-session models directly and compare. ``jobs.py`` calls them from a
background job, feeding weights/scalers it persisted during the run.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

import paths
import pipeline
from deepquantreg import build_model
from deepquantreg.training import fit, predict_quantiles

MISSING_HANDLING = "exclude_row"  # rows with a missing feature value are dropped, never imputed


# --------------------------------------------------------------------------- #
# small extractors
# --------------------------------------------------------------------------- #
def scaler_dict(scaler) -> dict:
    """{'mean': [...], 'scale': [...]} from a fitted sklearn StandardScaler."""
    return {"mean": [float(x) for x in scaler.mean_],
            "scale": [float(x) for x in scaler.scale_]}


def feature_ranges(frame: pd.DataFrame, feature_cols: list[str]) -> dict:
    return {f: {"min": float(frame[f].min()), "max": float(frame[f].max())}
            for f in feature_cols}


def training_stats(frame: pd.DataFrame) -> dict:
    n = int(len(frame))
    events = int(frame["event"].sum())
    censoring = round((1.0 - events / n) * 100, 1) if n else None
    return {"n": n, "events": events, "censoring_pct": censoring}


def build_args_for(model_name: str, resolved: dict, n_features: int,
                   quantiles: list[float]) -> dict:
    """The exact dict passed to ``build_model`` (ModelConfig fields only)."""
    arch = dict(resolved["architecture"])
    arch.update(name=model_name, input_dim=int(n_features), quantiles=[float(q) for q in quantiles])
    return arch


# --------------------------------------------------------------------------- #
# meta assembly + write
# --------------------------------------------------------------------------- #
def build_bundle(path, *, bundle_type: str, model_name: str, build_args: dict,
                 feature_names: list[str], quantiles: list[float], state_dicts: list,
                 scalers: list[dict], feature_ranges: dict, training: dict,
                 event_mapping: dict | None, metrics: dict, time_unit: str | None,
                 seed: int, device: str = "cpu"):
    """Assemble ``meta`` and write the bundle via :func:`model_io.save_bundle`.

    ``scalers`` is one scaler dict per member (``state_dicts`` aligned): the
    first is the primary/display scaler, the full list is stored as
    ``member_scalers`` so predict standardises each member with its own scaler.
    """
    import model_io

    if len(scalers) != len(state_dicts):
        raise ValueError("need one scaler per member (state_dict)")
    meta = {
        "model": model_name,
        "build_args": build_args,
        "feature_names": list(feature_names),
        "quantiles": [float(q) for q in quantiles],
        "scaler": scalers[0],
        "member_scalers": list(scalers),
        "feature_ranges": feature_ranges,
        "missing_handling": MISSING_HANDLING,
        "training": training,
        "event_mapping": event_mapping,
        "metrics": metrics,
        "time_unit": (time_unit or None),
        "bundle": {"type": bundle_type},
        "device": device,
        "seed": int(seed),
        "deepcnq": paths.repo_info(),
    }
    return model_io.save_bundle(path, state_dicts=state_dicts, meta=meta)


# --------------------------------------------------------------------------- #
# final: refit on all data
# --------------------------------------------------------------------------- #
def refit_final(model_name: str, frame: pd.DataFrame, feature_cols: list[str],
                resolved: dict, quantiles: list[float], seed: int, *,
                valid_frac: float = 0.15, deterministic: bool = False,
                device: str = "cpu") -> dict:
    """Retrain ``model_name`` on all rows (15% held out for early stopping).

    ``model_name`` is passed explicitly -- the ``resolved`` architecture dict from
    ``presets.resolve`` has no ``name`` key (the name is added at build time), so
    deriving it from ``resolved`` would give ``None``. Returns
    ``{state_dict, scaler, metrics, build_args}``; ``state_dict`` is on CPU. The
    job manager's per-epoch progress hook wraps ``fit`` automatically when this
    runs inside a job; called bare (tests) it just trains.
    """
    if not model_name:
        raise ValueError("refit_final needs a model name")
    quantiles = [float(q) for q in quantiles]
    train, valid, scaler, km = pipeline.prepare_all(frame, feature_cols, valid_frac, seed)

    build_args = build_args_for(model_name, resolved, len(feature_cols), quantiles)
    model = build_model(build_args)
    training = resolved["training"]
    model, state = fit(
        model, train, valid, quantiles, seed=seed,
        max_epochs=int(training["maximum_epochs"]), batch_size=int(training["batch_size"]),
        lr=float(training["learning_rate"]), weight_decay=float(training["weight_decay"]),
        optimizer=training["optimizer"], scheduler=training["scheduler"],
        patience=int(training["patience"]), device=device, deterministic=deterministic,
    )

    # Validation metrics on the held-out 15% (IPCW pinball over the grid).
    valid_pred = np.asarray(predict_quantiles(model, valid["X"], device), float)
    horizon = float(np.max(train["time"][train["event"] == 1]))
    horizon_log = float(np.log(max(horizon, 1e-12)))
    from deepquantreg.metrics.survival import weighted_pinball_mean
    pinball = weighted_pinball_mean(
        np.log(np.maximum(valid["time"], 1e-12)), valid_pred, valid["weights"],
        quantiles, horizon_log)

    cpu_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    metrics = {
        "source": "refit_holdout",
        "valid_pinball_mean": float(pinball),
        "n_valid": int(len(valid["time"])),
        "best_epoch": int(state.get("best_epoch", -1)),
        "epochs_ran": int(state.get("epochs_ran", -1)),
        "early_stopped": bool(state.get("early_stopped", False)),
    }
    return {"state_dict": cpu_state, "scaler": scaler_dict(scaler), "metrics": metrics,
            "build_args": build_args}
