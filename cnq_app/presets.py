"""Resolve model hyper-parameters from the shipped cnq.yaml presets.

The presets file (configs/final/real/cnq.yaml) is JSON with a ``common`` block
of shared training settings and a ``configs`` block keyed by dataset name, each
holding per-model overrides. This mirrors the ``common``/``configs`` handling in
scripts/train.py so the app produces the same architecture/training dicts.
"""
from __future__ import annotations

import math
from typing import Any

import paths
from deepquantreg.config import load_config

# Models the UI exposes. The three CNQ models plus the optional MLP baseline.
APP_MODELS = ("KAN_gaps", "TransformerPS_gaps", "Transformer_KAN_gaps", "MLP_multiQ")
TRANSFORMER_MODELS = ("TransformerPS_gaps", "Transformer_KAN_gaps")

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

# Approximate size / censoring for each paper cohort, used only to label the
# "Starting settings" options and to suggest the closest one. These are the
# published figures for the public benchmark datasets; the app never combines
# the user's data with them.
PRESET_META = {
    "support":  {"label": "SUPPORT",  "n": 8873, "censoring": 32},
    "flchain":  {"label": "FLCHAIN",  "n": 7874, "censoring": 72},
    "gbsg":     {"label": "GBSG",     "n": 2232, "censoring": 43},
    "gbsg500":  {"label": "GBSG500",  "n": 500,  "censoring": 56},
    "metabric": {"label": "METABRIC", "n": 1904, "censoring": 42},
    "nki70":    {"label": "NKI70",    "n": 144,  "censoring": 67},
}

# Fallback architecture/training used when a preset has no entry for a model
# (the shipped presets only cover the three CNQ models, not MLP_multiQ).
_DEFAULT_ARCH = {"hidden_dim": 100, "layers": 2, "dropout": 0.0, "grid_size": 5}
_DEFAULT_TRAIN = {"learning_rate": 1e-4, "weight_decay": 0.0, "patience": 10}
NHEAD = 4


def _load() -> dict[str, Any]:
    return load_config(paths.CNQ_PRESET_CONFIG)


def preset_names() -> list[str]:
    """Dataset preset keys available in cnq.yaml (support, flchain, ...)."""
    return list(_load().get("configs", {}).keys())


def default_quantiles() -> list[float]:
    common = _load().get("common", {})
    return [float(q) for q in common.get("quantiles", [0.1, 0.25, 0.5, 0.75, 0.9])]


def _approx(n: int) -> int:
    """Round a subject count to two significant figures for display."""
    if n <= 0:
        return 0
    factor = 10 ** (int(math.floor(math.log10(n))) - 1)
    return int(round(n / factor) * factor)


def preset_label(name: str) -> str:
    """e.g. 'METABRIC (≈1,900 subjects, 42% censored)'."""
    meta = PRESET_META.get(name)
    if not meta:
        return name
    return f"{meta['label']} (≈{_approx(meta['n']):,} subjects, {meta['censoring']}% censored)"


def preset_meta() -> dict[str, Any]:
    """Per-preset metadata (label, n, censoring) for the front end."""
    return {name: {**PRESET_META[name], "display": preset_label(name)}
            for name in preset_names() if name in PRESET_META}


def suggest_preset(n_subjects: int | None, censoring_pct: float | None) -> str | None:
    """Closest paper cohort: log subject count is the main distance, censoring second."""
    names = [n for n in preset_names() if n in PRESET_META]
    if not names or not n_subjects or n_subjects <= 0:
        return None
    log_user = math.log(n_subjects)

    def score(name: str) -> float:
        meta = PRESET_META[name]
        log_diff = abs(log_user - math.log(meta["n"]))
        cens_diff = 0.0
        if censoring_pct is not None:
            cens_diff = abs(censoring_pct - meta["censoring"]) / 100.0
        return log_diff + 0.3 * cens_diff  # log dominates; censoring breaks near-ties

    return min(names, key=score)


def preset_summary() -> dict[str, Any]:
    """Everything the front end needs to render the preset picker."""
    cfg = _load()
    common = cfg.get("common", {})
    configs = cfg.get("configs", {})
    return {
        "presets": {name: configs[name] for name in configs},
        "preset_meta": preset_meta(),
        "common": {
            "quantiles": [float(q) for q in common.get("quantiles", default_quantiles())],
            "optimizer": common.get("optimizer", "AdamW"),
            "scheduler": common.get("scheduler", "CosineAnnealingLR"),
            "batch_size": int(common.get("batch_size", 64)),
            "maximum_epochs": int(common.get("maximum_epochs", 500)),
            "patience": int(common.get("patience", 10)),
        },
        "models": list(APP_MODELS),
        "transformer_models": list(TRANSFORMER_MODELS),
    }


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
