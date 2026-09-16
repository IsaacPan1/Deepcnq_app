"""Quantile-grid generation shared by the server, jobs and report.

Grids are built with integer stepping and rounding so there are no
floating-point artefacts (e.g. 0.15000000000000002). The required levels
0.1/0.5/0.9 are always merged in, then the grid is sorted and de-duplicated.
"""
from __future__ import annotations

# The five levels shown in metric tables / headline numbers even for dense grids.
STANDARD = [0.1, 0.25, 0.5, 0.75, 0.9]
# Levels the models/metrics require (ICP needs 0.1/0.9, Uno C needs 0.5).
REQUIRED = (0.1, 0.5, 0.9)
PRECISION = 4  # decimal places; keeps generated levels clean

GRID_KINDS = ("standard", "every10", "every5", "every1", "custom")
GRID_LABELS = {
    "standard": "Standard: 0.1, 0.25, 0.5, 0.75, 0.9",
    "every10": "Every 10%: 0.1 to 0.9",
    "every5": "Every 5%: 0.05 to 0.95",
    "every1": "Every 1%: 0.01 to 0.99",
    "custom": "Custom list",
}


def _round(x: float) -> float:
    return round(float(x), PRECISION)


def merge_required(values) -> list[float]:
    """Round, add the required levels, drop out-of-range, sort and de-duplicate."""
    vals = [_round(v) for v in values] + [_round(r) for r in REQUIRED]
    return sorted({v for v in vals if 0.0 < v < 1.0})


def build_grid(kind: str, custom=None) -> list[float]:
    if kind == "standard":
        base = list(STANDARD)
    elif kind == "every10":
        base = [i / 100 for i in range(10, 91, 10)]
    elif kind == "every5":
        base = [i / 100 for i in range(5, 96, 5)]
    elif kind == "every1":
        base = [i / 100 for i in range(1, 100)]
    elif kind == "custom":
        base = []
        for v in (custom or []):
            try:
                base.append(float(v))
            except (TypeError, ValueError):
                continue
    else:
        raise ValueError(f"unknown quantile grid {kind!r}; choose from {', '.join(GRID_KINDS)}")
    return merge_required(base)


def standard_present(quantiles) -> list[float]:
    """Standard levels that appear in the grid, in order."""
    present = []
    for s in STANDARD:
        if any(abs(s - q) < 1e-9 for q in quantiles):
            present.append(s)
    return present


def has_extreme(quantiles) -> bool:
    """True if any level is below 0.05 or above 0.95."""
    return any(q < 0.05 - 1e-9 or q > 0.95 + 1e-9 for q in quantiles)


def resolve(cfg: dict) -> list[float]:
    """Return the canonical quantile list for a run config.

    Prefers an explicit ``quantile_grid`` spec ({"kind", "custom"}); otherwise
    falls back to a raw ``quantiles`` list (also merged/rounded).
    """
    spec = cfg.get("quantile_grid")
    if spec:
        return build_grid(spec.get("kind", "standard"), spec.get("custom"))
    if cfg.get("quantiles"):
        return merge_required(cfg["quantiles"])
    return list(STANDARD)
