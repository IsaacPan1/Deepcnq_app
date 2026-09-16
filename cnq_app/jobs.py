"""Background training jobs with per-epoch progress and cancellation.

Progress is reported by wrapping ``deepquantreg.training.trainer.predict_quantiles``.
``trainer.fit`` calls that function exactly once per epoch (to score the
validation split), so the wrapper is a clean hook to count epochs and to abort a
run mid-fit when the user cancels -- raising inside it unwinds ``fit`` cleanly.
"""
from __future__ import annotations

import json
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

import paths
import pipeline
import plots
import report
import presets
from deepquantreg.training import trainer as _trainer

RUNS_DIR = paths.OUTPUT_DIR


class Cancelled(Exception):
    """Raised inside the patched predict hook to abort training on cancel."""


# --------------------------------------------------------------------------- #
# predict_quantiles wrapping (per-epoch progress + cancel)
# --------------------------------------------------------------------------- #
_ACTIVE: Optional["Job"] = None
_orig_predict = _trainer.predict_quantiles


def _patched_predict(model, features, device="cpu", batch_size=4096):
    job = _ACTIVE
    if job is not None and job._counting:
        if job.cancel_event.is_set():
            raise Cancelled()
        job.epoch += 1
        job._touch()
    return _orig_predict(model, features, device, batch_size)


# Install once; fit() looks up predict_quantiles as a module global at call time.
_trainer.predict_quantiles = _patched_predict


# --------------------------------------------------------------------------- #
# Job
# --------------------------------------------------------------------------- #
@dataclass
class Job:
    id: str
    config: dict
    state: str = "pending"        # pending|running|done|cancelled|error
    phase: str = "queued"          # queued|prepare|train|plots|report|done
    step: str = "Queued"
    error: str = ""
    n_models: int = 0
    n_splits: int = 0
    unit_index: int = 0            # completed model*split units
    total_units: int = 0
    current_model: str = ""
    current_split: int = 0
    epoch: int = 0
    max_epochs: int = 0
    started_at: float = 0.0
    finished_at: float = 0.0
    results: Optional[dict] = None
    cancel_event: threading.Event = field(default_factory=threading.Event)
    _counting: bool = False
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def _touch(self):
        pass  # progress fields are plain reads; lock only guards status snapshots

    @property
    def dir(self) -> Path:
        return RUNS_DIR / self.id

    def progress(self) -> float:
        # Training occupies 0..0.92; plots/report/done are set in status().
        if self.total_units == 0:
            return 0.0
        frac_within = 0.0
        if self._counting and self.max_epochs:
            frac_within = min(self.epoch / self.max_epochs, 1.0)
        base = (self.unit_index + frac_within) / self.total_units
        return 0.92 * min(base, 1.0)

    def status(self) -> dict:
        with self._lock:
            prog = self.progress()
            if self.phase in ("plots",):
                prog = 0.93
            elif self.phase == "report":
                prog = 0.97
            elif self.phase == "done":
                prog = 1.0
            return {
                "id": self.id,
                "state": self.state,
                "phase": self.phase,
                "step": self.step,
                "error": self.error,
                "progress": round(prog, 4),
                "current_model": self.current_model,
                "current_split": self.current_split,
                "n_splits": self.n_splits,
                "n_models": self.n_models,
                "unit_index": self.unit_index,
                "total_units": self.total_units,
                "epoch": self.epoch,
                "max_epochs": self.max_epochs,
                "elapsed": round((self.finished_at or time.time()) - self.started_at, 1)
                if self.started_at else 0.0,
                "has_results": self.results is not None,
            }


# --------------------------------------------------------------------------- #
# Manager
# --------------------------------------------------------------------------- #
class JobManager:
    """Single-user manager: at most one active job at a time."""

    def __init__(self):
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        RUNS_DIR.mkdir(parents=True, exist_ok=True)

    def get(self, job_id: str) -> Optional[Job]:
        return self._jobs.get(job_id)

    def active(self) -> Optional[Job]:
        for job in self._jobs.values():
            if job.state in ("pending", "running"):
                return job
        return None

    def start(self, config: dict) -> Job:
        if self.active() is not None:
            raise RuntimeError("a job is already running; cancel it first")
        job = Job(id=uuid.uuid4().hex[:12], config=config)
        job.dir.mkdir(parents=True, exist_ok=True)
        self._jobs[job.id] = job
        threading.Thread(target=self._run, args=(job,), daemon=True).start()
        return job

    def cancel(self, job_id: str) -> bool:
        job = self.get(job_id)
        if job is None or job.state not in ("pending", "running"):
            return False
        job.cancel_event.set()
        return True

    # ----------------------------------------------------------------- run
    def _run(self, job: Job):
        global _ACTIVE
        _ACTIVE = job
        job.state = "running"
        job.started_at = time.time()
        try:
            self._execute(job)
            if job.cancel_event.is_set():
                job.state = "cancelled"
                job.step = "Cancelled"
            else:
                job.state = "done"
                job.phase = "done"
                job.step = "Finished"
        except Cancelled:
            job.state = "cancelled"
            job.step = "Cancelled"
        except Exception as exc:  # noqa: BLE001 - surface any failure to the UI
            job.state = "error"
            job.error = f"{type(exc).__name__}: {exc}"
            job.step = "Failed"
            (job.dir / "error.txt").write_text(traceback.format_exc())
        finally:
            job.finished_at = time.time()
            job._counting = False
            _ACTIVE = None

    def _execute(self, job: Job):
        cfg = job.config
        job.phase = "prepare"
        job.step = "Loading data and building splits"

        id_col = cfg.get("id_col") or None
        event_positive = cfg.get("event_positive")
        raw = pipeline.load_frame(cfg["csv_path"])
        # Validation summary/warnings (reasons for excluded rows) for the report.
        import validation
        data_report = validation.validate(
            raw, cfg["duration_col"], cfg["event_col"], cfg["feature_cols"], id_col,
            ratio=tuple(cfg["ratio"]), seed=int(cfg["seed"]),
            n_splits=int(cfg["n_splits"]), event_positive=event_positive)
        frame = pipeline.build_frame(raw, cfg["duration_col"], cfg["event_col"],
                                     cfg["feature_cols"], id_col=id_col,
                                     event_positive=event_positive)
        feature_cols = cfg["feature_cols"]
        import grids
        quantiles = grids.resolve(cfg)
        ratio = tuple(float(r) for r in cfg["ratio"])
        n_splits = int(cfg["n_splits"])
        seed = int(cfg["seed"])
        models = list(cfg["models"])
        deterministic = bool(cfg.get("deterministic", False))
        epsilon = float(cfg.get("epsilon", 0.005))
        time_unit = (cfg.get("time_unit") or "").strip()
        self._data_section = self._build_data_section(cfg, raw, frame, data_report, id_col,
                                                      event_positive, time_unit)

        job.n_models = len(models)
        job.n_splits = n_splits
        job.total_units = len(models) * n_splits

        # Resolve hyper-parameters per model up-front (fail fast on bad config).
        resolved = {m: presets.resolve(m, preset=cfg.get("preset"), custom=cfg.get("custom"),
                                        quantiles=quantiles) for m in models}

        km_curve = pipeline.km_survival_curve(frame["duration"], frame["event"])

        # outputs[model] -> list of TrainOutput across splits
        outputs: dict[str, list[pipeline.TrainOutput]] = {m: [] for m in models}
        rep_models: dict[str, Any] = {}
        rep_prepared: dict[str, Any] = {}

        job.phase = "train"
        for split_index in range(n_splits):
            if job.cancel_event.is_set():
                raise Cancelled()
            split_seed = seed + split_index
            prepared = pipeline.prepare(frame, feature_cols, ratio, split_seed, epsilon)
            for model_name in models:
                if job.cancel_event.is_set():
                    raise Cancelled()
                job.current_model = model_name
                job.current_split = split_index + 1
                job.max_epochs = int(resolved[model_name]["training"]["maximum_epochs"])
                job.epoch = 0
                job.step = (f"Training {model_name} — split {split_index + 1}/{n_splits}")
                keep = split_index == 0
                job._counting = True
                try:
                    out = pipeline.train_model_on_split(
                        model_name, resolved[model_name], prepared, split_seed, split_index,
                        deterministic=deterministic, keep_for_plots=keep,
                        include_unoc=bool(cfg.get("include_unoc", True)))
                finally:
                    job._counting = False
                outputs[model_name].append(out)
                if keep:
                    rep_models[model_name] = out.keep_model
                    rep_prepared[model_name] = out.prepared
                    out.keep_model = None  # drop the large reference from the stored output
                    out.prepared = None
                job.unit_index += 1

        # ---- summarize -------------------------------------------------
        summaries = {m: pipeline.summarize_model(outputs[m], quantiles) for m in models}

        # ---- plots -----------------------------------------------------
        job.phase = "plots"
        job.step = "Generating plots"
        importance = {}
        for model_name in models:
            imp = pipeline.permutation_importance(
                rep_models[model_name], rep_prepared[model_name], quantiles, seed=seed)
            importance[model_name] = imp

        images = plots.generate_all(
            km_curve=km_curve, outputs=outputs, quantiles=quantiles,
            rep_prepared=rep_prepared, importance=importance, feature_cols=feature_cols,
            summaries=summaries, time_unit=time_unit)

        # persist images
        img_dir = job.dir / "plots"
        img_dir.mkdir(exist_ok=True)
        for name, data in images.items():
            (img_dir / f"{name}.png").write_bytes(data)

        # ---- full-grid predictions (representative split, per model) ---
        predictions = self._build_predictions(rep_prepared, outputs, quantiles)
        pred_dir = job.dir / "predictions"
        pred_dir.mkdir(exist_ok=True)
        for name, csv_text in predictions.items():
            (pred_dir / name).write_text(csv_text, encoding="utf-8")

        # ---- assemble results -----------------------------------------
        results = self._assemble_results(job, frame, feature_cols, quantiles, ratio,
                                         n_splits, seed, models, summaries, outputs,
                                         km_curve, importance)
        (job.dir / "results.json").write_text(json.dumps(results, indent=2, default=_json_default))

        # ---- report + zip ---------------------------------------------
        job.phase = "report"
        job.step = "Building report and results archive"
        html = report.build_html_report(results, images)
        (job.dir / "report.html").write_text(html, encoding="utf-8")
        report.build_zip(job.dir / "results.zip", results, images, html,
                         csv_path=cfg["csv_path"], predictions=predictions)

        results["_report_path"] = str(job.dir / "report.html")
        results["_zip_path"] = str(job.dir / "results.zip")
        job.results = results

    def _build_predictions(self, rep_prepared, outputs, quantiles) -> dict[str, str]:
        """CSV of full-grid predictions (original-scale times) per model, first split."""
        import pandas as pd

        out = {}
        for model, runs in outputs.items():
            prepared = rep_prepared.get(model)
            if prepared is None:
                continue
            pred_log = runs[0].test_pred  # (subjects, quantiles) in log-time
            test = prepared.test
            frame = pd.DataFrame({"subject_id": test["subject_id"],
                                  "observed_time": test["time"],
                                  "event": test["event"]})
            for j, q in enumerate(quantiles):
                frame[f"q{q:g}_time"] = np.exp(pred_log[:, j])
            out[f"{model}.csv"] = frame.to_csv(index=False)
        return out

    def _build_data_section(self, cfg, raw, frame, data_report, id_col, event_positive,
                            time_unit) -> dict:
        summary = data_report.get("summary", {})
        reasons = [w["message"] for w in data_report.get("warnings", [])
                   if w.get("code") in ("missing_values", "nonpositive_duration")]
        event_mapping = None
        if event_positive is not None:
            event_mapping = {"positive_value": event_positive,
                             "meaning": f"{cfg['event_col']} == {event_positive!r} -> event (1)"}
        return {
            "file_name": cfg.get("file_name") or "uploaded.csv",
            "rows_total": int(len(raw)),
            "rows_used": int(len(frame)),
            "rows_excluded": int(len(raw)) - int(len(frame)),
            "exclusion_reasons": reasons,
            "n_events": int(frame["event"].sum()),
            "censoring_pct": round(float(1.0 - frame["event"].mean()) * 100, 1),
            "duration_range": [float(frame["duration"].min()), float(frame["duration"].max())],
            "time_unit": time_unit or None,
            "event_mapping": event_mapping,
            "id_col": id_col,
            "duration_col": cfg["duration_col"],
            "event_col": cfg["event_col"],
            "features": list(cfg["feature_cols"]),
        }

    def _assemble_results(self, job, frame, feature_cols, quantiles, ratio, n_splits,
                          seed, models, summaries, outputs, km_curve, importance) -> dict:
        return {
            "job_id": job.id,
            "repo": paths.repo_info(),
            "data": getattr(self, "_data_section", {}),
            "config": {
                "models": models,
                "quantiles": quantiles,
                "quantile_grid": {
                    "kind": (job.config.get("quantile_grid") or {}).get("kind", "standard"),
                    "n_levels": len(quantiles),
                },
                "ratio": list(ratio),
                "n_splits": n_splits,
                "seed": seed,
                "preset": job.config.get("preset"),
                "custom": job.config.get("custom"),
                "duration_col": job.config["duration_col"],
                "event_col": job.config["event_col"],
                "feature_cols": feature_cols,
                "id_col": job.config.get("id_col") or None,
                "time_unit": (job.config.get("time_unit") or "").strip() or None,
                "event_positive": job.config.get("event_positive"),
            },
            "resolved": {m: presets.resolve(m, preset=job.config.get("preset"),
                                            custom=job.config.get("custom"),
                                            quantiles=quantiles) for m in models},
            "dataset": {
                "n_rows": int(len(frame)),
                "n_features": len(feature_cols),
                "n_events": int(frame["event"].sum()),
                "censoring_rate": float(1.0 - frame["event"].mean()),
            },
            "km": km_curve,
            "summaries": summaries,
            "per_split": {
                m: [
                    {
                        "split_index": o.split_index,
                        "best_epoch": o.best_epoch,
                        "epochs_ran": o.epochs_ran,
                        "early_stopped": o.early_stopped,
                        "metrics": {k: v for k, v in o.metrics.items()},
                    }
                    for o in outputs[m]
                ]
                for m in models
            },
            "importance": {m: importance[m].tolist() for m in models},
        }


def _json_default(obj):
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"not serializable: {type(obj)}")
