"""Local single-user web server for the CNQ training app.

Run with ``python -m server [port] [-b BIND]`` from inside ``cnq_app``. It uses
only the standard library ThreadingHTTPServer with a SimpleHTTPRequestHandler
subclass to serve the ``static/`` front end, and accepts CSV uploads as the raw
request body (no multipart parsing).

Only the standard library (plus the stdlib-only ``paths`` module) is imported at
module level. The scientific stack (pandas, torch, matplotlib) and the app's
``pipeline``/``jobs`` modules are imported lazily inside the handlers, so the
server always starts and can report — via ``/api/health`` and a banner on the
page — when a package is missing or the deepcnq repository cannot be found.
Filesystem locations come from ``paths`` (repo discovery + app-local dirs),
never from the current working directory.
"""
from __future__ import annotations

import argparse
import errno
import importlib.util
import json
import os
import re
import uuid
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import paths  # stdlib-only; discovers the repo and puts deepquantreg on sys.path

STATIC_DIR = paths.STATIC
SAMPLE_DATA = paths.SAMPLE_DATA
UPLOAD_DIR = paths.OUTPUT_DIR / "uploads"
MAX_UPLOAD = 64 * 1024 * 1024  # 64 MB raw-body cap

# Packages the app needs; import name -> pip name (for the health banner).
# torch is installed by run.py from the wheel index; deepquantreg comes from the repo.
REQUIRED_PACKAGES = {
    "numpy": "numpy", "pandas": "pandas", "sklearn": "scikit-learn",
    "matplotlib": "matplotlib", "yaml": "PyYAML", "torch": "torch",
    "deepquantreg": "deepquantreg (from the deepcnq repo)",
}

# Lazily created; importing jobs pulls in the whole scientific stack.
_MANAGER = None


def _get_manager():
    global _MANAGER
    if _MANAGER is None:
        from jobs import JobManager  # lazy: heavy import
        _MANAGER = JobManager()
    return _MANAGER


def check_packages() -> dict:
    """Report which required packages are importable, without importing them."""
    status = {}
    for name in REQUIRED_PACKAGES:
        try:
            status[name] = importlib.util.find_spec(name) is not None
        except (ImportError, ValueError):
            status[name] = False
    missing = [name for name, ok in status.items() if not ok]
    return {
        "ok": not missing,
        "packages": status,
        "missing": missing,
        "missing_pip": [REQUIRED_PACKAGES[name] for name in missing],
    }


def health_report() -> dict:
    """Combined health: package availability plus repo discovery status."""
    pkg = check_packages()
    repo = paths.repo_info()
    pkg["repo"] = repo
    pkg["ok"] = pkg["ok"] and repo["found"]
    return pkg


# --------------------------------------------------------------------------- #
# Template CSV
# --------------------------------------------------------------------------- #
TEMPLATE_CSV = (
    "subject_id,time,event,age,biomarker,treatment\n"
    "S001,12.4,1,58,2.7,1\n"
    "S002,36.0,0,64,1.2,0\n"
)


# --------------------------------------------------------------------------- #
# CSV preview + guessing + validation (import the scientific stack lazily)
# --------------------------------------------------------------------------- #
def preview_csv(path: Path, n: int = 5) -> dict:
    import pandas as pd
    import pipeline

    frame = pipeline.load_frame(path)
    columns = list(frame.columns)
    # is_numeric_dtype handles pandas extension dtypes (e.g. StringDtype in pandas 3).
    numeric = [c for c in columns if pd.api.types.is_numeric_dtype(frame[c])]
    binaryish = []
    for c in columns:
        try:
            vals = pd.unique(frame[c].dropna())
            if len(vals) <= 2 and set(vals).issubset({0, 1, 0.0, 1.0}):
                binaryish.append(c)
        except (TypeError, ValueError):
            pass
    head = frame.head(n).astype(object).where(frame.head(n).notna(), None).values.tolist()
    return {
        "columns": columns,
        "numeric_columns": numeric,
        "binary_columns": binaryish,
        "n_rows": int(len(frame)),
        "preview": {"columns": columns, "rows": head},
    }


def guess_mapping(columns: list[str], numeric: list[str], binary: list[str]) -> dict:
    """Best-effort default column mapping (the UI can override every choice)."""
    numeric = numeric or list(columns)
    duration = next((c for c in numeric
                     if re.search(r"time|dur|surv|month|day|year|follow|fu", c, re.I)),
                    numeric[0] if numeric else (columns[0] if columns else None))
    event_re = r"event|status|death|died|dead|censor|relaps|recur"
    # Prefer a binary column whose name looks like an event, then any name match,
    # then the first binary column, so a 0/1 covariate isn't mistaken for the event.
    event = (next((c for c in binary if re.search(event_re, c, re.I)), None)
             or next((c for c in columns if re.search(event_re, c, re.I)), None)
             or (binary[0] if binary else None))
    id_col = next((c for c in columns
                   if re.fullmatch(r"(subject_?id|patient_?id|case_?id|id)", c, re.I)
                   or re.search(r"_id$", c, re.I)), None)
    features = [c for c in numeric if c not in {duration, event, id_col}]
    return {"duration_col": duration, "event_col": event, "id_col": id_col,
            "feature_cols": features}


def run_validation(cfg: dict) -> dict:
    """Load the CSV named in cfg and run validation.validate against the mapping.

    Also attaches ``suggested_preset`` (closest paper cohort by size/censoring).
    """
    import pipeline
    import presets
    import validation

    frame = pipeline.load_frame(cfg["csv_path"])
    result = validation.validate(
        frame, cfg.get("duration_col"), cfg.get("event_col"),
        cfg.get("feature_cols") or [], cfg.get("id_col") or None,
        ratio=tuple(cfg.get("ratio") or (65, 15, 20)),
        seed=int(cfg.get("seed", 42)), n_splits=int(cfg.get("n_splits", 1)),
        event_positive=cfg.get("event_positive"))
    summary = result.get("summary", {})
    result["suggested_preset"] = presets.suggest_preset(
        summary.get("rows_used"), summary.get("censoring_pct"))
    return result


def validate_run(cfg: dict) -> list[str]:
    import pipeline
    import presets

    errors = []
    quantiles = cfg.get("quantiles") or []
    try:
        quantiles = sorted(float(q) for q in quantiles)
    except (TypeError, ValueError):
        return ["quantiles must be numbers"]
    if any(not 0 < q < 1 for q in quantiles):
        errors.append("all quantiles must lie strictly in (0, 1)")
    if any(a >= b for a, b in zip(quantiles, quantiles[1:])):
        errors.append("quantiles must be strictly increasing / unique")
    for required in pipeline.REQUIRED_QUANTILES:
        if not any(abs(q - required) < 1e-9 for q in quantiles):
            errors.append(f"quantiles must include {required} (needed for ICP / Uno C)")
    cfg["quantiles"] = quantiles

    models = cfg.get("models") or []
    if not models:
        errors.append("select at least one model")
    for m in models:
        if m not in presets.APP_MODELS:
            errors.append(f"unknown model {m!r}")
    if not cfg.get("duration_col"):
        errors.append("map the duration column")
    if not cfg.get("event_col"):
        errors.append("map the event column")
    if not cfg.get("feature_cols"):
        errors.append("select at least one feature column")
    ratio = cfg.get("ratio") or [65, 15, 20]
    if len(ratio) != 3 or sum(float(r) for r in ratio) <= 0 or any(float(r) < 0 for r in ratio):
        errors.append("split ratio must be three non-negative numbers")
    if int(cfg.get("n_splits", 1)) < 1:
        errors.append("number of splits must be >= 1")
    mode = cfg.get("mode", "preset")
    if mode == "preset" and not cfg.get("preset"):
        errors.append("choose a preset or switch to custom hyper-parameters")
    if not cfg.get("csv_path") or not Path(cfg["csv_path"]).exists():
        errors.append("upload a CSV first")
    return errors


# --------------------------------------------------------------------------- #
class Handler(SimpleHTTPRequestHandler):
    """Serve static/ for everything except the /api/* routes."""

    server_version = "CNQApp/1.0"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(STATIC_DIR), **kwargs)

    def log_message(self, fmt, *args):  # keep the console quiet
        pass

    # ---- helpers ----
    def _send_json(self, obj, status=200):
        body = json.dumps(obj, default=_json_default).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_bytes(self, data: bytes, content_type: str, status=200, filename=None):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        if filename:
            self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.end_headers()
        self.wfile.write(data)

    def _read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length", 0))
        if length > MAX_UPLOAD:
            raise ValueError("upload exceeds size limit")
        return self.rfile.read(length) if length else b""

    def _error(self, message, status=400):
        self._send_json({"error": message}, status=status)

    def _repo_ready(self) -> bool:
        """Guard endpoints that need the repo; send a clean error if it's missing."""
        if paths.REPO_ROOT is None:
            self._error("deepcnq repository not found. " + (paths.REPO_ERROR or ""), 503)
            return False
        return True

    # ---- routing ----
    def do_GET(self):
        parsed = urlparse(self.path)
        route = parsed.path
        if not route.startswith("/api/"):
            return super().do_GET()  # static file serving from static/
        query = parse_qs(parsed.query)
        try:
            if route == "/api/health":
                return self._send_json(health_report())
            if route == "/api/config":
                return self._handle_config()
            if route == "/api/status":
                return self._handle_status(query)
            if route == "/api/results":
                return self._handle_results(query)
            if route == "/api/report":
                return self._handle_download(query, "report.html", "text/html",
                                             "cnq_report.html", inline=True)
            if route == "/api/zip":
                return self._handle_download(query, "results.zip", "application/zip",
                                             "cnq_results.zip")
            if route == "/api/sample":
                if not SAMPLE_DATA.exists():
                    return self._error("sample data not found", 404)
                return self._send_bytes(SAMPLE_DATA.read_bytes(), "text/csv",
                                        filename="sample_survival.csv")
            if route == "/api/template":
                return self._send_bytes(TEMPLATE_CSV.encode("utf-8"), "text/csv",
                                        filename="cnq_template.csv")
            return self._error("not found", 404)
        except Exception as exc:  # noqa: BLE001
            return self._error(f"{type(exc).__name__}: {exc}", 500)

    def do_POST(self):
        parsed = urlparse(self.path)
        route = parsed.path
        query = parse_qs(parsed.query)
        try:
            if route == "/api/upload":
                return self._handle_upload()
            if route == "/api/validate":
                return self._handle_validate()
            if route == "/api/run":
                return self._handle_run()
            if route == "/api/cancel":
                return self._handle_cancel(query)
            return self._error("not found", 404)
        except Exception as exc:  # noqa: BLE001
            return self._error(f"{type(exc).__name__}: {exc}", 500)

    # ---- handlers ----
    def _handle_config(self):
        if not self._repo_ready():
            return
        import pipeline
        import presets

        self._send_json({
            **presets.preset_summary(),
            "preset_names": presets.preset_names(),
            "required_quantiles": list(pipeline.REQUIRED_QUANTILES),
            "has_sample": SAMPLE_DATA.exists(),
        })

    def _handle_upload(self):
        if not self._repo_ready():
            return
        data = self._read_body()
        if not data:
            return self._error("empty upload")
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        upload_id = uuid.uuid4().hex[:12]
        path = UPLOAD_DIR / f"{upload_id}.csv"
        path.write_bytes(data)
        try:
            info = preview_csv(path)
        except Exception as exc:  # noqa: BLE001
            path.unlink(missing_ok=True)
            return self._error(f"could not parse CSV: {exc}")
        info["csv_path"] = str(path)
        info["upload_id"] = upload_id
        info["file_name"] = self.headers.get("X-Filename") or f"{upload_id}.csv"
        # Guess a mapping and validate it immediately so the UI opens pre-filled.
        guess = guess_mapping(info["columns"], info["numeric_columns"], info["binary_columns"])
        info["guess"] = guess
        try:
            info["validation"] = run_validation({"csv_path": str(path), **guess})
        except Exception as exc:  # noqa: BLE001
            info["validation"] = {"errors": [{"code": "validation_failed",
                                              "message": f"could not validate: {exc}"}],
                                  "warnings": [], "summary": {}}
        self._send_json(info)

    def _handle_validate(self):
        if not self._repo_ready():
            return
        cfg = json.loads(self._read_body() or b"{}")
        if not cfg.get("csv_path") or not Path(cfg["csv_path"]).exists():
            return self._error("upload a CSV first")
        try:
            self._send_json(run_validation(cfg))
        except Exception as exc:  # noqa: BLE001
            self._error(f"validation failed: {exc}", 500)

    def _handle_run(self):
        if not self._repo_ready():
            return
        import presets

        import grids

        cfg = json.loads(self._read_body() or b"{}")
        if cfg.get("mode") != "custom":
            cfg["custom"] = None
        else:
            cfg["preset"] = None
        # Build the canonical quantile grid server-side (rounding + required levels).
        cfg["quantiles"] = grids.resolve(cfg)
        errors = validate_run(cfg)
        if errors:
            return self._error("; ".join(errors))
        # Re-validate the data itself; refuse to start while there are errors.
        try:
            data_report = run_validation(cfg)
        except Exception as exc:  # noqa: BLE001
            return self._error(f"could not validate data: {exc}")
        if data_report["errors"]:
            return self._error("; ".join(m["message"] for m in data_report["errors"]))
        manager = _get_manager()
        if manager.active() is not None:
            return self._error("a job is already running", 409)
        try:
            # fail fast on bad hyper-parameters (e.g. hidden_dim not divisible by nhead)
            for m in cfg["models"]:
                presets.resolve(m, preset=cfg.get("preset"), custom=cfg.get("custom"),
                                quantiles=cfg["quantiles"])
        except Exception as exc:  # noqa: BLE001
            return self._error(str(exc))
        job = manager.start(cfg)
        self._send_json({"job_id": job.id, **job.status()})

    def _handle_status(self, query):
        job = _get_manager().get(_one(query, "id"))
        if job is None:
            return self._error("unknown job", 404)
        self._send_json(job.status())

    def _handle_results(self, query):
        job = _get_manager().get(_one(query, "id"))
        if job is None or job.results is None:
            return self._error("results not ready", 404)
        payload = {k: v for k, v in job.results.items() if not k.startswith("_")}
        self._send_json(payload)

    def _handle_cancel(self, query):
        ok = _get_manager().cancel(_one(query, "id"))
        self._send_json({"cancelled": ok})

    def _handle_download(self, query, filename, content_type, download_name, inline=False):
        job = _get_manager().get(_one(query, "id"))
        if job is None:
            return self._error("unknown job", 404)
        target = job.dir / filename
        if not target.exists():
            return self._error("file not ready", 404)
        self._send_bytes(target.read_bytes(), content_type,
                         filename=None if inline else download_name)


class Server(ThreadingHTTPServer):
    # http.server sets allow_reuse_address = True, but on Windows SO_REUSEADDR
    # lets a second process bind an already-listening port, which would hide a
    # busy port instead of raising EADDRINUSE. Disable reuse there so the
    # port-busy message actually fires; keep it on POSIX to avoid TIME_WAIT.
    allow_reuse_address = os.name != "nt"


def _one(query, key):
    values = query.get(key)
    return values[0] if values else None


def _json_default(obj):
    # numpy may be absent in a minimal environment; fall back to str.
    try:
        import numpy as np
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except ImportError:
        pass
    return str(obj)


# --------------------------------------------------------------------------- #
def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m server", description="Local CNQ training web app.")
    parser.add_argument("port", nargs="?", type=int, default=8000,
                        help="port to listen on (default: 8000)")
    parser.add_argument("-b", "--bind", default="127.0.0.1",
                        help="address to bind (default: 127.0.0.1)")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    try:
        httpd = Server((args.bind, args.port), Handler)
    except OSError as exc:
        if exc.errno == errno.EADDRINUSE:
            print(f"Port {args.port} is already in use - try another, "
                  f"e.g. python -m server {args.port + 1}")
        else:
            print(f"Could not bind {args.bind}:{args.port} - {exc}")
        return 1

    display_host = "127.0.0.1" if args.bind in ("0.0.0.0", "::", "") else args.bind
    print(f"Serving CNQ app on {args.bind}:{args.port} "
          f"(http://{display_host}:{args.port}/)")
    repo = paths.repo_info()
    if repo["found"]:
        print(f"Repository: {repo['path']} ({repo['version_str']})")
    else:
        print("WARNING: deepcnq repository not found - the app cannot train until "
              "it is available.")
        print("  Set CNQ_REPO or place the repo at ../deepcnq. "
              "See /api/health for details.")
    packages = check_packages()
    if not packages["ok"]:
        print(f"WARNING: missing packages: {', '.join(packages['missing_pip'])}")
    print("Press Ctrl+C to stop.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
