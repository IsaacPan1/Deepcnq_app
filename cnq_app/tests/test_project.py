"""Projection math (torch-free): population + cohort curves and the A/B queries."""
import sys
from pathlib import Path

import numpy as np
import pytest

APP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_DIR))

import project  # noqa: E402 (pure numpy)


def _meta():
    return {"training_survival": {
        "time": [0, 1, 2, 3, 4], "survival": [1.0, 0.8, 0.6, 0.4, 0.2],
        "var": [0.0, 0.001, 0.002, 0.003, 0.004], "horizon": 4.0, "last_is_censored": False}}


def test_population_scaled_and_monotone():
    p = project.population_projection(_meta(), 100)
    D = np.asarray(p["events"])
    assert np.all(np.diff(D) >= -1e-9)                 # monotone non-decreasing
    assert abs(p["max_events"] - 80.0) < 1e-6          # 100 * (1 - S(horizon)=0.2)
    lo, hi = np.asarray(p["events_lo"]), np.asarray(p["events_hi"])
    assert np.all(lo <= D + 1e-9) and np.all(D <= hi + 1e-9)


def test_population_requires_curve():
    with pytest.raises(ValueError):
        project.population_projection({"training_survival": {}}, 100)
    with pytest.raises(ValueError):
        project.population_projection(_meta(), 0)      # N must be positive


def test_events_at_times_and_range_guard():
    p = project.population_projection(_meta(), 100)
    a = project.events_at_times(p, [2, 5])
    assert abs(a[0]["events"] - 40.0) < 1e-6 and not a[0]["out_of_range"]
    assert a[0]["lo"] <= a[0]["events"] <= a[0]["hi"]
    assert a[1]["out_of_range"] and a[1]["events"] is None   # beyond horizon=4


def test_time_for_events_inverse_and_guard():
    p = project.population_projection(_meta(), 100)
    b = project.time_for_events(p, 50)
    assert not b["out_of_range"] and abs(b["time"] - 2.5) < 1e-6
    # inverse consistency: events at the returned time == the target
    assert abs(project.events_at_times(p, [b["time"]])[0]["events"] - 50.0) < 1e-6
    assert b["time_lo"] <= b["time"] <= b["time_hi"]
    assert project.time_for_events(p, 90)["out_of_range"]    # exceeds max_events=80


def test_cohort_projection_monotone_and_bounded():
    pred_log = np.log(np.array([[1.0, 2.0, 4.0], [1.5, 3.0, 6.0]]))  # 2 subjects, q=.1,.5,.9
    c = project.cohort_projection(pred_log, [0.1, 0.5, 0.9], horizon=5.0)
    D = np.asarray(c["events"])
    assert c["n"] == 2.0 and c["max_events"] <= 2.0 + 1e-9
    assert np.all(np.diff(D) >= -1e-9)
    lo, hi = np.asarray(c["events_lo"]), np.asarray(c["events_hi"])
    assert np.all(lo <= D + 1e-9) and np.all(D <= hi + 1e-9)


def test_time_for_zero_events_is_time_zero():
    p = project.population_projection(_meta(), 100)
    assert abs(project.time_for_events(p, 0.0)["time"]) < 1e-9
