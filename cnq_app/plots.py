"""Matplotlib figures for the results report.

Every function returns PNG bytes so figures can be embedded (base64) into a
self-contained HTML report and also written straight into the results zip.
Rendering uses the non-interactive Agg backend so it works headless in the
background training thread.
"""
from __future__ import annotations

import io
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import paths  # noqa: F401,E402
import grids  # noqa: E402
from deepquantreg.diagnostics.quantiles import interval_width  # noqa: E402
from deepquantreg.metrics.survival import weighted_mean  # noqa: E402
from deepquantreg.metrics import CensoringKM  # noqa: E402

_COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def _fig_to_png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def _color(i: int) -> str:
    return _COLORS[i % len(_COLORS)]


def _time_label(time_unit: str | None) -> str:
    return f"Time ({time_unit})" if time_unit else "Time"


# --------------------------------------------------------------------------- #
def km_curve_plot(km: dict, time_unit: str | None = None) -> bytes:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.step(km["time"], km["survival"], where="post", color=_color(0), lw=2)
    ax.set_xlabel(_time_label(time_unit))
    ax.set_ylabel("Survival probability S(t)")
    ax.set_ylim(0, 1.02)
    ax.set_title(f"Kaplan–Meier outcome curve (n={km['n']}, events={km['n_events']})")
    ax.grid(alpha=0.3)
    return _fig_to_png(fig)


def training_curves_plot(outputs: dict) -> bytes:
    """Per-epoch training curves for each model's representative (first) split."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for i, (model, runs) in enumerate(outputs.items()):
        history = runs[0].history
        epochs = [h["epoch"] for h in history]
        axes[0].plot(epochs, [h["train_objective"] for h in history],
                     color=_color(i), label=model)
        axes[0].plot(epochs, [h["valid_objective"] for h in history],
                     color=_color(i), ls="--", alpha=0.7)
        axes[1].plot(epochs, [h["valid_trainG_pinball_mean"] for h in history],
                     color=_color(i), label=model)
        best = runs[0].best_epoch
        if 0 <= best < len(history):
            axes[1].scatter([best], [history[best]["valid_trainG_pinball_mean"]],
                            color=_color(i), zorder=5, marker="o")
    axes[0].set_title("Objective (solid=train, dashed=valid)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("IPCW pinball objective")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)
    axes[1].set_title("Validation selection metric (● = best epoch)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("valid_trainG_pinball_mean")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    return _fig_to_png(fig)


def calibration_plot(outputs: dict, quantiles: list) -> bytes:
    """Nominal tau vs mean empirical coverage across splits, per model.

    Shows only the standard levels present, even for a denser grid.
    """
    levels = grids.standard_present(quantiles) or list(quantiles)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], color="grey", ls=":", label="ideal")
    for i, (model, runs) in enumerate(outputs.items()):
        # average calibration across splits at each standard tau
        cals = np.array([[r.metrics["calibration"][str(float(q))] for q in levels]
                         for r in runs])
        ax.plot(levels, cals.mean(axis=0), marker="o", color=_color(i), label=model)
    ax.set_xlabel("Nominal quantile level τ")
    ax.set_ylabel("Empirical coverage  P(Y ≤ q_τ)")
    ax.set_title("Per-τ calibration")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return _fig_to_png(fig)


def quantile_profiles_plot(rep_prepared: dict, outputs: dict, quantiles: list,
                           n_subjects: int = 6, time_unit: str | None = None) -> bytes:
    """Predicted quantile curves (in time) for a handful of test subjects.

    Uses the first selected model's representative split.
    """
    model = next(iter(outputs))
    prepared = rep_prepared[model]
    pred = outputs[model][0].test_pred  # (subjects, quantiles) in log-time
    test = prepared.test
    n = min(n_subjects, pred.shape[0])
    rng = np.random.RandomState(0)
    idx = rng.choice(pred.shape[0], size=n, replace=False)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for k, subj in enumerate(idx):
        times = np.exp(pred[subj])
        ax.plot(quantiles, times, marker="o", color=_color(k), alpha=0.85,
                label=f"subj {test['subject_id'][subj]}")
        obs = test["time"][subj]
        marker = "x" if test["event"][subj] == 1 else "^"
        ax.scatter([0.5], [obs], color=_color(k), marker=marker, s=60, zorder=5)
    ax.set_xlabel("Quantile level τ")
    ax.set_ylabel(f"Predicted survival time ({time_unit})" if time_unit
                  else "Predicted survival time")
    ax.set_title(f"Individual quantile profiles — {model}\n(× observed event, ▲ censored, at τ=0.5)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    return _fig_to_png(fig)


def survival_curves_plot(rep_prepared: dict, outputs: dict, quantiles: list,
                         n_subjects: int = 6, time_unit: str | None = None) -> bytes:
    """Predicted survival S(t) = 1 - τ against predicted time, for a few subjects.

    Uses the dense grid of the first selected model's representative split; observed
    times are marked (× event, ▲ censored).
    """
    model = next(iter(outputs))
    prepared = rep_prepared[model]
    pred = outputs[model][0].test_pred  # (subjects, quantiles) in log-time
    test = prepared.test
    taus = np.asarray(quantiles, float)
    surv = 1.0 - taus  # survival probability at each predicted quantile time
    n = min(n_subjects, pred.shape[0])
    rng = np.random.RandomState(0)
    idx = rng.choice(pred.shape[0], size=n, replace=False)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for k, subj in enumerate(idx):
        times = np.exp(pred[subj])  # original-scale predicted times, increasing in τ
        ax.step(times, surv, where="post", color=_color(k), alpha=0.85,
                label=f"subj {test['subject_id'][subj]}")
        obs = test["time"][subj]
        marker = "x" if test["event"][subj] == 1 else "^"
        ax.scatter([obs], [0.5], color=_color(k), marker=marker, s=55, zorder=5)
    ax.set_xlabel(f"Time ({time_unit})" if time_unit else "Time")
    ax.set_ylabel("Predicted survival  S(t) = 1 − τ")
    ax.set_ylim(0, 1.02)
    ax.set_title(f"Individual survival curves — {model}\n(× observed event, ▲ censored, at S=0.5)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    return _fig_to_png(fig)


def coverage_by_width_plot(rep_prepared: dict, outputs: dict, quantiles: list) -> bytes:
    """80% interval coverage within terciles of interval width, per model."""
    lo = _index_of(quantiles, 0.1)
    hi = _index_of(quantiles, 0.9)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    width_labels = ["narrow", "medium", "wide"]
    x = np.arange(3)
    n_models = len(outputs)
    bar_w = 0.8 / max(n_models, 1)
    for i, (model, runs) in enumerate(outputs.items()):
        prepared = rep_prepared[model]
        pred = runs[0].test_pred
        test = prepared.test
        km = prepared.censoring
        weights = km.weights(test["time"], test["event"])
        widths = interval_width(pred)
        y = np.log(np.maximum(test["time"], 1e-12))
        inside = ((y >= pred[:, lo]) & (y <= pred[:, hi])).astype(float)
        terts = np.quantile(widths, [1 / 3, 2 / 3])
        bins = np.digitize(widths, terts)
        cov = []
        for b in range(3):
            mask = bins == b
            cov.append(weighted_mean(inside[mask], weights[mask]) if mask.sum() else np.nan)
        ax.bar(x + i * bar_w, cov, width=bar_w, color=_color(i), label=model)
    ax.axhline(0.8, color="grey", ls=":", label="nominal 80%")
    ax.set_xticks(x + bar_w * (n_models - 1) / 2)
    ax.set_xticklabels(width_labels)
    ax.set_xlabel("Interval-width tercile")
    ax.set_ylabel("80% interval coverage")
    ax.set_ylim(0, 1.05)
    ax.set_title("Coverage by interval-width tercile")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    return _fig_to_png(fig)


def importance_heatmap(importance: dict, feature_cols: list, quantiles: list) -> bytes:
    """Permutation importance heatmap: features × τ, one panel per model."""
    models = list(importance.keys())
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(max(4.5 * n, 5), max(0.35 * len(feature_cols) + 1.5, 3)),
                             squeeze=False)
    vmax = max((np.abs(np.array(importance[m])).max() for m in models), default=1.0) or 1.0
    for i, model in enumerate(models):
        ax = axes[0][i]
        data = np.array(importance[model])
        im = ax.imshow(data, aspect="auto", cmap="viridis", vmin=0, vmax=vmax)
        ax.set_xticks(range(len(quantiles)))
        ax.set_xticklabels([f"{q:g}" for q in quantiles], fontsize=8)
        ax.set_yticks(range(len(feature_cols)))
        ax.set_yticklabels(feature_cols, fontsize=7)
        ax.set_xlabel("τ")
        ax.set_title(model, fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Permutation importance (Δ pinball when feature shuffled)", y=1.02)
    return _fig_to_png(fig)


def pinball_boxplots(outputs: dict) -> bytes:
    """Distribution of test pinball_mean across repeated splits, per model."""
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    models = list(outputs.keys())
    data = [[r.metrics["pinball_mean"] for r in outputs[m]] for m in models]
    bp = ax.boxplot(data, patch_artist=True, showmeans=True)
    ax.set_xticks(range(1, len(models) + 1))
    ax.set_xticklabels(models)
    for i, box in enumerate(bp["boxes"]):
        box.set_facecolor(_color(i))
        box.set_alpha(0.5)
    ax.set_ylabel("Test IPCW pinball (mean over τ)")
    ax.set_title(f"Pinball across {len(data[0])} repeated split(s)")
    ax.grid(alpha=0.3, axis="y")
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right", fontsize=8)
    return _fig_to_png(fig)


def _index_of(quantiles, tau):
    for j, q in enumerate(quantiles):
        if np.isclose(q, tau):
            return j
    return None


# --------------------------------------------------------------------------- #
def generate_all(*, km_curve, outputs, quantiles, rep_prepared, importance,
                 feature_cols, summaries, time_unit: str | None = None) -> dict[str, bytes]:
    """Render every figure. Returns {name: png_bytes}."""
    figures = {
        "km_curve": km_curve_plot(km_curve, time_unit=time_unit),
        "training_curves": training_curves_plot(outputs),
        "calibration": calibration_plot(outputs, quantiles),
        "quantile_profiles": quantile_profiles_plot(rep_prepared, outputs, quantiles,
                                                    time_unit=time_unit),
        "coverage_by_width": coverage_by_width_plot(rep_prepared, outputs, quantiles),
        "importance_heatmap": importance_heatmap(importance, feature_cols, quantiles),
        "pinball_boxplots": pinball_boxplots(outputs),
    }
    # A dense grid earns a survival-curve view of individual predictions.
    if len(quantiles) > 5:
        figures["survival_curves"] = survival_curves_plot(
            rep_prepared, outputs, quantiles, time_unit=time_unit)
    return figures
