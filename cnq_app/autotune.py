"""Bounded hyper-parameter / model search ("auto-tune").

Random search over KAN + non-crossing MLP (transformers opt-in), sized to the
data by :func:`presets.auto_space`. Each trial trains one (or a few) split(s) with
a **reduced** epoch budget and is ranked by **validation** IPCW pinball (the
early-stopping metric, held out from the untouched test split). Returns a
leaderboard and the winner; the winner's reported config uses the *full* epoch
budget so the user can train it properly afterwards.

Everything is bounded (trial count + optional wall-clock) and cancellable at
trial boundaries, so a cancel still yields the best-so-far leaderboard.
"""
from __future__ import annotations

import random
import time
from typing import Callable, Optional

import numpy as np

import pipeline
import presets

# Reduced budget used *during* the search (the winner is retrained fully later).
SEARCH_EPOCHS = 60
SEARCH_PATIENCE = 8


def _eval_config(model: str, custom: dict, frame, feature_cols, quantiles, ratio,
                 seed: int, n_splits: int, deterministic: bool, device: str,
                 cancel: Optional[Callable[[], bool]]) -> "dict | None":
    """Train ``model`` at ``custom`` over ``n_splits`` and average the validation
    and test pinball. Returns None if it can't be evaluated (or was cancelled)."""
    try:
        resolved = presets.resolve(model, preset=None, custom=custom, quantiles=quantiles)
    except Exception:  # noqa: BLE001 - invalid combo (e.g. hidden % nhead); skip
        return None
    valid_scores, test_scores = [], []
    for s in range(n_splits):
        if cancel and cancel():
            break
        prepared = pipeline.prepare(frame, feature_cols, ratio, seed + s)
        out = pipeline.train_model_on_split(
            model, resolved, prepared, seed + s, s, device=device,
            deterministic=deterministic, keep_for_plots=False, include_unoc=False)
        if out.valid_pinball is not None:
            valid_scores.append(out.valid_pinball)
        test_scores.append(out.metrics["pinball_mean"])
    if not valid_scores:
        return None
    return {"valid_pinball": float(np.mean(valid_scores)),
            "test_pinball": float(np.mean(test_scores)),
            "n_splits": len(valid_scores)}


def run_search(frame, feature_cols, quantiles, ratio, seed, *, enabled_models=None,
               n_trials: int = 12, n_splits: int = 1, deterministic: bool = False,
               device: str = "cpu", time_budget: "float | None" = None,
               progress: Optional[Callable] = None,
               cancel: Optional[Callable[[], bool]] = None) -> dict:
    n, p = len(frame), len(feature_cols)
    space = presets.auto_space(n, p, enabled_models)
    models = list(space.keys())
    full = presets.defaults(n, p)             # epochs/patience/batch for the reported config
    rng = random.Random(int(seed))
    started = time.time()

    leaderboard: list[dict] = []
    seen: set = set()
    for i in range(int(n_trials)):
        if cancel and cancel():
            break
        if time_budget and leaderboard and (time.time() - started) > time_budget:
            break
        model = rng.choice(models)
        tuned = {k: rng.choice(v) for k, v in space[model].items()}
        key = (model, tuple(sorted(tuned.items())))
        if key in seen:                       # don't waste budget on duplicates
            continue
        seen.add(key)

        eval_cfg = {**tuned, "batch_size": full["batch_size"],
                    "maximum_epochs": SEARCH_EPOCHS, "patience": SEARCH_PATIENCE}
        scored = _eval_config(model, eval_cfg, frame, feature_cols, quantiles, ratio,
                              seed + 1000 * (i + 1), n_splits, deterministic, device, cancel)
        if scored is None:
            continue
        # the config we report/apply uses the FULL epoch budget
        report_cfg = {**tuned, "batch_size": full["batch_size"],
                      "maximum_epochs": full["maximum_epochs"], "patience": full["patience"]}
        leaderboard.append({"model": model, "custom": report_cfg, **scored})
        if progress:
            best = min(leaderboard, key=lambda t: t["valid_pinball"])
            progress(len(leaderboard), int(n_trials), best)

    leaderboard.sort(key=lambda t: t["valid_pinball"])
    for rank, t in enumerate(leaderboard, 1):
        t["rank"] = rank
    return {"leaderboard": leaderboard, "best": leaderboard[0] if leaderboard else None,
            "models": models, "n_splits": int(n_splits), "n_trials": int(n_trials)}
