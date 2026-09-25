"""Population-level projections of cumulative events over time.

Two directions, from a trained CNQ model:

* **A. Events by a time** -- fix time points, get the expected number of events.
* **B. Time for a target** -- fix a number of events (with a sample size), get the
  time by which they are expected.

Two sources of the trend:

* **Population (from N)** -- the training data's Kaplan--Meier survival stored in
  the bundle, scaled to the requested cohort size: ``D(t) = N * (1 - S_KM(t))``.
  The confidence band uses the KM Greenwood variance. This is covariate-free and
  is the headline "how many events in N subjects" result.
* **Cohort (from covariates)** -- aggregate the model's per-subject predicted
  survival ``S_i(t)``: ``D(t) = sum_i (1 - S_i(t))``. The band is the
  Poisson--binomial variance ``sum_i F_i(t)(1-F_i(t))``. This adjusts the trend for
  a specific cohort whose covariate mix differs from training.

Both are **range-guarded**: projection is only reported up to the observed
follow-up horizon (last KM event time). Beyond it -- or for a target that needs
more events than are reachable within follow-up -- the query is flagged
out-of-range (v1 does not parametrically extrapolate the censored tail).

This module is pure numpy: population projection needs no torch, so it stays
testable and light. Only the cohort path (fed pre-computed predictions) touches
model output.
"""
from __future__ import annotations

import numpy as np

Z = 1.959963984540054  # ~95% normal quantile


def _pack(time, D, D_lo, D_hi, horizon, n, *, source, last_is_censored=False) -> dict:
    time = np.asarray(time, float)
    D = np.asarray(D, float)
    horizon = float(horizon)
    return {
        "source": source,
        "n": float(n),
        "horizon": horizon,
        "last_is_censored": bool(last_is_censored),
        "max_events": float(np.interp(horizon, time, D)),
        "time": time.tolist(),
        "events": D.tolist(),
        "events_lo": np.asarray(D_lo, float).tolist(),
        "events_hi": np.asarray(D_hi, float).tolist(),
    }


# --------------------------------------------------------------------------- #
# population: scaled training KM
# --------------------------------------------------------------------------- #
def population_projection(meta: dict, n_subjects: float) -> dict:
    ts = meta.get("training_survival") or {}
    if not ts.get("time"):
        raise ValueError("this model has no stored training survival curve; re-save "
                         "the model to enable population projection")
    n = float(n_subjects)
    if n <= 0:
        raise ValueError("sample size N must be positive")
    t = np.asarray(ts["time"], float)
    surv = np.asarray(ts["survival"], float)
    var = np.asarray(ts.get("var") or np.zeros_like(surv), float)
    horizon = float(ts.get("horizon") or t[-1])

    se = np.sqrt(np.maximum(var, 0.0))
    surv_lo = np.clip(surv - Z * se, 0.0, 1.0)
    surv_hi = np.clip(surv + Z * se, 0.0, 1.0)
    events = n * (1.0 - surv)
    events_lo = n * (1.0 - surv_hi)          # more survival -> fewer events
    events_hi = n * (1.0 - surv_lo)
    return _pack(t, events, events_lo, events_hi, horizon, n,
                 source="population", last_is_censored=ts.get("last_is_censored", False))


# --------------------------------------------------------------------------- #
# cohort: aggregate model per-subject survival
# --------------------------------------------------------------------------- #
def cohort_projection(pred_log, quantiles, horizon: float, *, n_grid: int = 200) -> dict:
    """``pred_log`` is (n_subjects, n_quantiles) log-time predictions; ``quantiles``
    the tau levels. ``horizon`` caps the trend at the training follow-up."""
    q = np.asarray(quantiles, float)
    times = np.exp(np.asarray(pred_log, float))     # (n, Q) predicted event times
    n = times.shape[0]
    horizon = float(horizon) if horizon and horizon > 0 else float(np.max(times))
    grid = np.linspace(0.0, horizon, n_grid)

    # F_i(t) = P(T_i <= t): interpolate tau against the subject's predicted times;
    # clamp outside the covered quantile range (no silent extrapolation of tau).
    F = np.empty((n, grid.size))
    for i in range(n):
        F[i] = np.interp(grid, times[i], q, left=q[0], right=q[-1])
    events = F.sum(axis=0)
    se = np.sqrt(np.maximum((F * (1.0 - F)).sum(axis=0), 0.0))   # Poisson-binomial
    events_lo = np.clip(events - Z * se, 0.0, n)
    events_hi = np.clip(events + Z * se, 0.0, n)
    return _pack(grid, events, events_lo, events_hi, horizon, n, source="cohort")


# --------------------------------------------------------------------------- #
# queries
# --------------------------------------------------------------------------- #
def events_at_times(proj: dict, times) -> list[dict]:
    """Direction A: expected events (with band) at each requested time."""
    t = np.asarray(proj["time"], float)
    D = np.asarray(proj["events"], float)
    lo = np.asarray(proj["events_lo"], float)
    hi = np.asarray(proj["events_hi"], float)
    horizon = proj["horizon"]
    out = []
    for tt in times:
        tt = float(tt)
        oor = tt > horizon + 1e-9
        out.append({
            "time": tt,
            "events": None if oor else float(np.interp(tt, t, D)),
            "lo": None if oor else float(np.interp(tt, t, lo)),
            "hi": None if oor else float(np.interp(tt, t, hi)),
            "out_of_range": oor,
        })
    return out


def time_for_events(proj: dict, k: float) -> dict:
    """Direction B: the time by which ``k`` events are expected (with band).

    ``events`` is monotone non-decreasing in time, so this inverts by
    interpolation. If ``k`` exceeds the events reachable within follow-up, it is
    out of range (needs parametric tail extrapolation, not in v1)."""
    t = np.asarray(proj["time"], float)
    D = np.asarray(proj["events"], float)
    lo = np.asarray(proj["events_lo"], float)
    hi = np.asarray(proj["events_hi"], float)
    k = float(k)
    if k < 0:
        raise ValueError("target number of events must be non-negative")
    if k > proj["max_events"] + 1e-9:
        return {"target": k, "out_of_range": True, "max_events": proj["max_events"],
                "horizon": proj["horizon"], "time": None, "time_lo": None, "time_hi": None}
    # np.interp needs an increasing x; events curves are non-decreasing.
    return {
        "target": k,
        "out_of_range": False,
        "time": float(np.interp(k, D, t)),
        "time_lo": float(np.interp(k, hi, t)),   # the upper events curve reaches k sooner
        "time_hi": float(np.interp(k, lo, t)),
    }


# Caveats shown with every projection (aggregate/outcome uncertainty, not the
# model's own epistemic uncertainty; and the follow-up limit).
CAVEATS = [
    "The trend is an expected cumulative-events curve; the band is aggregate "
    "outcome/sampling uncertainty (Poisson-binomial for a cohort, Greenwood for the "
    "population), not the model's own uncertainty about its parameters.",
    "Projection is only supported up to the observed follow-up horizon (the last "
    "event time in the training data). Beyond it, and for targets needing more "
    "events than occur within follow-up, results are not shown -- extrapolating the "
    "censored tail would need a parametric dropout/enrollment model.",
    "Population projection assumes the N subjects resemble the training population; "
    "a cohort projection assumes the uploaded subjects were drawn similarly.",
]
