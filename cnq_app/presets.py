"""Resolve model hyper-parameters and pick data-driven defaults.

Hyper-parameters come from the data (``defaults`` / ``auto_space`` + auto-tune),
not from other datasets. Only the shipped cnq.yaml ``common`` block (generic
optimizer / scheduler / batch settings) is read; the old per-dataset "paper
presets" have been removed.
"""
from __future__ import annotations

from typing import Any

import paths
from deepquantreg.config import load_config

# Every model name the backend can build/validate (kept broad for tests and for
# possible future multimodal use). Transformers remain valid here but are not
# offered in the tabular UI or the tuner.
APP_MODELS = ("KAN_gaps", "MLP_multiQ_gaps", "TransformerPS_gaps",
              "Transformer_KAN_gaps", "MLP_multiQ")
TRANSFORMER_MODELS = ("TransformerPS_gaps", "Transformer_KAN_gaps")

# This app takes pure tabular CSVs, so it only offers (and auto-tunes over) KAN and
# the non-crossing MLP. Transformers help only for multimodal data (text /
# annotations), which this app doesn't ingest, so they're excluded here.
TABULAR_MODELS = ("KAN_gaps", "MLP_multiQ_gaps")
TUNE_DEFAULT_MODELS = TABULAR_MODELS
TUNE_ALLOWED_MODELS = TABULAR_MODELS

# Display labels the front end shows (mirror of MODEL_INFO in static/app.js). The
# API stores/uses the internal names on the left; these let the server map a
# display label back to its internal name for robustness.
MODEL_DISPLAY = {
    "KAN_gaps": "KAN-CNQ",
    "TransformerPS_gaps": "Trans-CNQ",
    "Transformer_KAN_gaps": "TransKAN-CNQ",
    "MLP_multiQ_gaps": "MLP-CNQ",
    "MLP_multiQ": "MLP multi-quantile",
    "MLP_singleQ": "MLP single-quantile",
}


def resolve_model_name(name, allowed) -> "str | None":
    """Map an internal or display model name to an internal name within ``allowed``.

    Accepts the internal name (``KAN_gaps``), a case-insensitive variant, or the
    UI display label (``KAN-CNQ``). Returns ``None`` when it cannot be resolved to
    one of the ``allowed`` models, so callers can reject it cleanly.
    """
    if not name:
        return None
    allowed = list(allowed)
    if name in allowed:
        return name
    by_lower = {a.lower(): a for a in allowed}
    if name.lower() in by_lower:
        return by_lower[name.lower()]
    display_to_internal = {label.lower(): key for key, label in MODEL_DISPLAY.items()}
    internal = display_to_internal.get(name.lower())
    return internal if internal in allowed else None

# Generic training defaults (not dataset-specific). The shipped cnq.yaml only
# supplies the shared ``common`` block (optimizer / scheduler / batch); the old
# per-dataset "paper presets" have been removed -- hyper-parameters now come from
# the data (``defaults`` / auto-tune), not from other cohorts.
_DEFAULT_ARCH = {"hidden_dim": 100, "layers": 2, "dropout": 0.0, "grid_size": 5}
_DEFAULT_TRAIN = {"learning_rate": 1e-4, "weight_decay": 0.0, "patience": 10}
NHEAD = 4


def _load() -> dict[str, Any]:
    return load_config(paths.CNQ_PRESET_CONFIG)


def defaults(n: int, p: int | None = None, censoring: float | None = None) -> dict[str, Any]:
    """Data-driven default hyper-parameters from sample size ``n`` and covariate
    count ``p``. Small data -> narrower net, more regularisation, smaller batch;
    large data -> wider/deeper, less regularisation. Values stay divisible by
    ``nhead`` so they're valid for the transformer models too. Used as the Manual
    default (pre-filled after upload) and as the centre of the auto-tune space.
    """
    n = int(n or 0)
    if n < 500:
        hidden, layers, dropout, wd, batch, patience = 32, 2, 0.2, 1e-3, 32, 20
    elif n < 2000:
        hidden, layers, dropout, wd, batch, patience = 64, 2, 0.1, 1e-4, 64, 12
    else:
        hidden, layers, dropout, wd, batch, patience = 128, 3, 0.0, 0.0, 128, 10
    if p and p >= 20:
        hidden = max(hidden, 64)
    return {"hidden_dim": hidden, "layers": layers, "dropout": dropout, "grid_size": 5,
            "learning_rate": 1e-3, "weight_decay": wd, "batch_size": batch,
            "maximum_epochs": 300, "patience": patience}


def auto_space(n: int, p: int | None = None,
               enabled_models: "list[str] | None" = None) -> dict[str, dict]:
    """Per-model candidate values for the auto-tune search, centred on
    :func:`defaults`. Only the meaningful architecture axes vary; epochs / patience
    / batch are fixed by the caller (short during search, full when reported)."""
    base = defaults(n, p)
    h = base["hidden_dim"]
    small = int(n or 0) < 500
    common = {
        "hidden_dim": sorted({max(16, h // 2), h, h * 2}),
        "layers": [1, 2] if small else [2, 3],
        "dropout": [0.1, 0.2, 0.3] if small else [0.0, 0.1, 0.2],
        "learning_rate": [3e-4, 1e-3, 3e-3],
        "weight_decay": [0.0, 1e-4, 1e-3],
        "grid_size": [5],
    }
    models = [m for m in (enabled_models or TUNE_DEFAULT_MODELS) if m in TUNE_ALLOWED_MODELS]
    if not models:
        models = list(TUNE_DEFAULT_MODELS)
    return {m: {k: list(v) for k, v in common.items()} for m in models}


def resolve(model: str, *, preset: str | None, custom: dict | None,
            quantiles: list[float]) -> dict[str, Any]:
    """Build the (architecture, training) dicts for one model.

    Follows scripts/train.py: ``common`` supplies shared training knobs, the
    per-dataset/per-model entry supplies overrides, and anything missing falls
    back to the source defaults.
    """
    cfg = _load()
    common = cfg.get("common", {})
    if custom is not None:
        selected = dict(custom)
        base_train = {
            "optimizer": custom.get("optimizer", common.get("optimizer", "AdamW")),
            "scheduler": custom.get("scheduler", common.get("scheduler", "CosineAnnealingLR")),
            "batch_size": int(custom.get("batch_size", common.get("batch_size", 64))),
            "maximum_epochs": int(custom.get("maximum_epochs", common.get("maximum_epochs", 500))),
        }
    else:
        if preset is None:
            raise ValueError("either a preset or custom hyper-parameters are required")
        selected = cfg.get("configs", {}).get(preset, {}).get(model, {})
        base_train = {
            "optimizer": common.get("optimizer", "AdamW"),
            "scheduler": common.get("scheduler", "CosineAnnealingLR"),
            "batch_size": int(common.get("batch_size", 64)),
            "maximum_epochs": int(common.get("maximum_epochs", 500)),
        }

    architecture = {
        "hidden_dim": int(selected.get("hidden_dim", _DEFAULT_ARCH["hidden_dim"])),
        "layers": int(selected.get("layers", _DEFAULT_ARCH["layers"])),
        "dropout": float(selected.get("dropout", _DEFAULT_ARCH["dropout"])),
        "grid_size": int(selected.get("grid_size", _DEFAULT_ARCH["grid_size"])),
        "nhead": NHEAD,
        "activation": "relu",
    }
    training = {
        **base_train,
        "learning_rate": float(selected.get("learning_rate", _DEFAULT_TRAIN["learning_rate"])),
        "weight_decay": float(selected.get("weight_decay", _DEFAULT_TRAIN["weight_decay"])),
        "patience": int(selected.get("patience", common.get("patience", _DEFAULT_TRAIN["patience"]))),
    }

    if model in TRANSFORMER_MODELS and architecture["hidden_dim"] % NHEAD:
        raise ValueError(
            f"{model}: hidden_dim ({architecture['hidden_dim']}) must be divisible by nhead ({NHEAD})")
    return {"architecture": architecture, "training": training, "quantiles": list(quantiles)}
