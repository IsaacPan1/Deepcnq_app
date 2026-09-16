"""Unit tests for validation.validate plus an API test with a messy CSV."""
import io
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

APP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_DIR))

import paths  # noqa: E402
import validation  # noqa: E402
from deepquantreg.config import load_config  # noqa: E402


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def make_df(n=160, events=120, features=3, id_col=False, seed=0):
    rng = np.random.RandomState(seed)
    df = pd.DataFrame({"time": rng.uniform(1, 100, n).round(3)})
    ev = np.zeros(n, dtype=int)
    ev[:events] = 1
    rng.shuffle(ev)
    df["status"] = ev
    for i in range(features):
        df[f"x{i}"] = rng.normal(size=n).round(3)
    if id_col:
        df["pid"] = [f"S{i:04d}" for i in range(n)]
    return df


def feats(k=3):
    return [f"x{i}" for i in range(k)]


def codes(result, kind):
    return {m["code"] for m in result[kind]}


# --------------------------------------------------------------------------- #
# baseline: a clean dataset produces no errors and no warnings
# --------------------------------------------------------------------------- #
def test_clean_dataset_passes():
    r = validation.validate(make_df(), "time", "status", feats())
    assert r["errors"] == []
    assert r["warnings"] == []
    s = r["summary"]
    assert s["rows_before"] == 160 and s["rows_used"] == 160 and s["rows_excluded"] == 0
    assert s["n_events"] == 120 and s["n_features"] == 3
    assert s["duration_min"] is not None and s["duration_max"] is not None


# --------------------------------------------------------------------------- #
# error rules
# --------------------------------------------------------------------------- #
def test_duration_not_numeric():
    df = make_df()
    df["time"] = ["not-a-number"] * len(df)
    r = validation.validate(df, "time", "status", feats())
    assert "duration_not_numeric" in codes(r, "errors")


def test_event_two_values_offers_mapping_then_resolves():
    df = make_df(events=120)
    df["status"] = np.where(df["status"] == 1, 2, 1)  # code events as 1/2
    r = validation.validate(df, "time", "status", feats())
    two = [e for e in r["errors"] if e["code"] == "event_two_values"]
    assert two and set(map(str, two[0]["values"])) == {"1", "2"}
    # applying the mapping (2 == event) clears the error
    r2 = validation.validate(df, "time", "status", feats(), event_positive=2)
    assert "event_two_values" not in codes(r2, "errors")
    assert r2["summary"]["n_events"] == 120


def test_event_two_values_strings():
    df = make_df()
    df["status"] = np.where(df["status"] == 1, "dead", "alive")
    r = validation.validate(df, "time", "status", feats())
    two = [e for e in r["errors"] if e["code"] == "event_two_values"]
    assert two and set(map(str, two[0]["values"])) == {"dead", "alive"}


def test_event_bad_values_three_distinct():
    df = make_df()
    df.loc[df.index[:10], "status"] = 2  # now {0,1,2}
    r = validation.validate(df, "time", "status", feats())
    assert "event_bad_values" in codes(r, "errors")


def test_no_features():
    r = validation.validate(make_df(), "time", "status", [])
    assert "no_features" in codes(r, "errors")


def test_feature_not_numeric_shows_examples():
    df = make_df()
    df["x0"] = ["low", "med", "high"] * (len(df) // 3) + ["low"] * (len(df) % 3)
    r = validation.validate(df, "time", "status", feats())
    bad = [e for e in r["errors"] if e["code"] == "feature_not_numeric"]
    assert bad and bad[0]["column"] == "x0" and bad[0]["examples"]


def test_no_events():
    df = make_df(events=0)
    r = validation.validate(df, "time", "status", feats())
    assert "no_events" in codes(r, "errors")


def test_fewer_than_ten_events_per_split():
    df = make_df(n=40, events=30)
    r = validation.validate(df, "time", "status", feats(), ratio=(65, 15, 20), seed=42)
    assert "few_events_split" in codes(r, "errors")


def test_duplicate_ids():
    df = make_df(id_col=True)
    df.loc[df.index[1], "pid"] = df.loc[df.index[0], "pid"]  # duplicate one ID
    r = validation.validate(df, "time", "status", feats(), id_col="pid")
    dup = [e for e in r["errors"] if e["code"] == "duplicate_ids"]
    assert dup and dup[0]["count"] >= 2


# --------------------------------------------------------------------------- #
# warning rules
# --------------------------------------------------------------------------- #
def test_nonpositive_duration_warning():
    df = make_df()
    df.loc[df.index[:5], "time"] = 0
    df.loc[df.index[5:8], "time"] = -3
    r = validation.validate(df, "time", "status", feats())
    w = [x for x in r["warnings"] if x["code"] == "nonpositive_duration"]
    assert w and w[0]["count"] == 8
    assert r["summary"]["rows_used"] == 152


def test_missing_values_warning_per_column():
    df = make_df()
    df.loc[df.index[:4], "x1"] = np.nan
    df.loc[df.index[:2], "time"] = np.nan
    r = validation.validate(df, "time", "status", feats())
    miss = {w["column"]: w["count"] for w in r["warnings"] if w["code"] == "missing_values"}
    assert miss.get("x1") == 4 and miss.get("time") == 2


def test_constant_feature_warning():
    df = make_df()
    df["x0"] = 7.0
    r = validation.validate(df, "time", "status", feats())
    w = [x for x in r["warnings"] if x["code"] == "constant_feature"]
    assert w and w[0]["column"] == "x0"


def test_few_events_total_warning():
    df = make_df(n=200, events=90)
    r = validation.validate(df, "time", "status", feats())
    assert "few_events_total" in codes(r, "warnings")


def test_high_censoring_warning():
    df = make_df(n=200, events=20)
    r = validation.validate(df, "time", "status", feats())
    assert "high_censoring" in codes(r, "warnings")


def test_outcome_column_selected_as_feature():
    df = make_df()
    r = validation.validate(df, "time", "status", ["x0", "status", "time"])
    assert "event_is_feature" in codes(r, "warnings")
    assert "duration_is_feature" in codes(r, "warnings")


def test_summary_counts_after_exclusions():
    df = make_df(n=100, events=60)
    df.loc[df.index[:10], "x0"] = np.nan  # excluded
    r = validation.validate(df, "time", "status", feats())
    s = r["summary"]
    assert s["rows_before"] == 100 and s["rows_used"] == 90
    assert s["rows_excluded"] == 10 and s["n_features"] == 3


# --------------------------------------------------------------------------- #
# API test: messy CSV -> messages -> fix mapping -> completed run
# --------------------------------------------------------------------------- #
def _boot_server():
    import threading
    from http.server import ThreadingHTTPServer
    from server import Handler

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, port


def _http(url, data=None, headers=None):
    import json
    import urllib.request

    req = urllib.request.Request(url, data=data, headers=headers or {})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read())


def _messy_csv_bytes():
    rng = np.random.RandomState(1)
    n = 170
    df = pd.DataFrame({
        "pid": [f"S{i:04d}" for i in range(n)],
        "time": rng.uniform(1, 120, n).round(2),
    })
    ev = np.zeros(n, dtype=int)
    ev[:120] = 1
    rng.shuffle(ev)
    df["event"] = np.where(ev == 1, 2, 1)            # 1/2 coding, 2 = event
    df["age"] = rng.randint(40, 80, n)
    df["bio"] = rng.normal(size=n).round(3)
    df.loc[df.index[:6], "bio"] = np.nan             # blank cells
    df["note"] = rng.choice(["low", "med", "high"], n)  # text column
    return df.to_csv(index=False).encode("utf-8")


def test_bad_csv_api_flow():
    import json

    httpd, port = _boot_server()
    base = f"http://127.0.0.1:{port}"
    try:
        up = _http(f"{base}/api/upload", data=_messy_csv_bytes(),
                   headers={"Content-Type": "text/csv", "X-Filename": "messy.csv"})
        csv_path = up["csv_path"]

        # Validate a deliberately wrong mapping: text column as a feature, blanks included.
        bad = _http(f"{base}/api/validate",
                    data=json.dumps({
                        "csv_path": csv_path, "duration_col": "time", "event_col": "event",
                        "feature_cols": ["age", "bio", "note"], "id_col": "pid",
                        "ratio": [65, 15, 20], "seed": 42, "n_splits": 1,
                    }).encode(), headers={"Content-Type": "application/json"})
        assert "event_two_values" in codes(bad, "errors")
        assert "feature_not_numeric" in codes(bad, "errors")
        assert "missing_values" in codes(bad, "warnings")

        # Fix the mapping: numeric features, map event value 2 -> event.
        good = _http(f"{base}/api/validate",
                     data=json.dumps({
                         "csv_path": csv_path, "duration_col": "time", "event_col": "event",
                         "feature_cols": ["age", "bio"], "id_col": "pid",
                         "event_positive": 2, "ratio": [65, 15, 20], "seed": 42, "n_splits": 1,
                     }).encode(), headers={"Content-Type": "application/json"})
        assert good["errors"] == []
        # 6 rows have a blank bio and are excluded; the rest are used.
        assert good["summary"]["rows_used"] == 164
        expected_events = good["summary"]["n_events"]
        assert expected_events > 100

        # Run to completion with the fixed mapping (tiny smoke architecture).
        smoke = load_config(paths.SMOKE_CONFIG)
        custom = {
            "hidden_dim": smoke["architecture"]["hidden_dim"],
            "layers": smoke["architecture"]["layers"],
            "dropout": smoke["architecture"]["dropout"],
            "grid_size": smoke["architecture"]["grid_size"],
            "learning_rate": smoke["training"]["learning_rate"],
            "weight_decay": smoke["training"]["weight_decay"],
            "batch_size": smoke["training"]["batch_size"],
            "maximum_epochs": smoke["training"]["maximum_epochs"],
            "patience": smoke["training"]["patience"],
        }
        run = _http(f"{base}/api/run", data=json.dumps({
            "csv_path": csv_path, "file_name": "messy.csv",
            "duration_col": "time", "event_col": "event", "event_positive": 2,
            "feature_cols": ["age", "bio"], "id_col": "pid", "time_unit": "months",
            "models": ["KAN_gaps"], "quantiles": [0.1, 0.25, 0.5, 0.75, 0.9],
            "ratio": [65, 15, 20], "n_splits": 1, "seed": 42, "deterministic": False,
            "mode": "custom", "custom": custom,
        }).encode(), headers={"Content-Type": "application/json"})
        job_id = run["job_id"]

        deadline = time.time() + 180
        state = None
        while time.time() < deadline:
            s = _http(f"{base}/api/status?id={job_id}")
            state = s["state"]
            if state in ("done", "error", "cancelled"):
                break
            time.sleep(0.3)
        assert state == "done", f"run ended in {state}"

        results = _http(f"{base}/api/results?id={job_id}")
        data = results["data"]
        assert data["rows_used"] == 164          # 170 - 6 blank bio rows
        assert data["rows_excluded"] == 6
        assert data["n_events"] == expected_events
        assert data["time_unit"] == "months"
        assert data["event_mapping"]["positive_value"] == 2
        assert data["id_col"] == "pid"
    finally:
        httpd.shutdown()
        httpd.server_close()
