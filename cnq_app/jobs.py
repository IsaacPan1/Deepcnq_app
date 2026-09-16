"""Background training jobs with per-epoch progress and cancellation.

Progress is reported by wrapping ``deepquantreg.training.trainer.predict_quantiles``.
``trainer.fit`` calls that function exactly once per epoch (to score the
validation split), so the wrapper is a clean hook to count epochs and to abort a
run mid-fit when the user cancels -- raising inside it unwinds ``fit`` cleanly.
"""
from __future__ import annotations

import json
import os
import shutil
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
import save_model
from deepquantreg.training import trainer as _trainer

RUNS_DIR = paths.OUTPUT_DIR
# Cleanup: keep only the most recent N job folders on disk. Configurable via the
# CNQ_KEEP_RUNS env var; 0 or negative disables pruning.
KEEP_RUNS = int(os.environ.get("CNQ_KEEP_RUNS", "20"))


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
    kind: str = "train"            # train|save|predict
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
    _override: Optional[float] = None   # explicit 0..1 progress for non-train jobs
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
            prog = self._override if self._override is not None else self.progress()
            if self.phase in ("plots",):
                prog = 0.93
            elif self.phase == "report":
                prog = 0.97
            elif self.phase == "done":
                prog = 1.0
            return {
                "id": self.id,
                "kind": self.kind,
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
        job = Job(id=uuid.uuid4().hex[:12], config=config, kind=config.get("kind", "train"))
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

    def _prune_runs(self, keep: int = KEEP_RUNS) -> None:
        """Keep only the most recent ``keep`` job folders on disk (by mtime).

        Job folders are 12-char hex ids; other entries under the runs dir (e.g.
        ``uploads``) are left alone. Best-effort -- never raises into a job.
        """
        if keep <= 0:
            return
        try:
            dirs = [d for d in RUNS_DIR.iterdir()
                    if d.is_dir() and len(d.name) == 12
                    and all(c in "0123456789abcdef" for c in d.name)]
            dirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
        except OSError:
            return
        for d in dirs[keep:]:
            shutil.rmtree(d, ignore_errors=True)

    # ----------------------------------------------------------------- run
    def _run(self, job: Job):
        global _ACTIVE
        _ACTIVE = job
        job.state = "running"
        job.started_at = time.time()
        try:
            if job.kind == "save":
                self._execute_save(job)
            elif job.kind == "predict":
                self._execute_predict(job)
            else:
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
            self._prune_runs()

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

        # Persist per-split weights + scalers + feature ranges so a later "Save
        # model" job can bundle the exact trained models, or refit on all data.
        art_dir = job.dir / "artifacts"
        (art_dir / "weights").mkdir(parents=True, exist_ok=True)
        (art_dir / "feature_ranges.json").write_text(
            json.dumps(save_model.feature_ranges(frame, feature_cols)))
        (art_dir / "training.json").write_text(json.dumps({
            **save_model.training_stats(frame),
            "event_mapping": self._data_section.get("event_mapping"),
        }))
        (art_dir / "context.json").write_text(json.dumps({
            "csv_path": cfg["csv_path"],
            "duration_col": cfg["duration_col"],
            "event_col": cfg["event_col"],
            "id_col": id_col,
            "event_positive": event_positive,
            "feature_cols": feature_cols,
        }))
        split_scalers: dict[str, dict] = {}

        job.phase = "train"
        for split_index in range(n_splits):
            if job.cancel_event.is_set():
                raise Cancelled()
            split_seed = seed + split_index
            prepared = pipeline.prepare(frame, feature_cols, ratio, split_seed, epsilon)
            split_scalers[str(split_index)] = save_model.scaler_dict(prepared.scaler)
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
                import torch
                torch.save(out.state_dict,
                           str(art_dir / "weights" / f"{model_name}__{split_index}.pt"))
                out.state_dict = None  # weights now on disk; free memory
                job.unit_index += 1

        (art_dir / "scalers.json").write_text(json.dumps(split_scalers))

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

    # ----------------------------------------------------------------- save
    def _reload_frame(self, context: dict):
        raw = pipeline.load_frame(context["csv_path"])
        return pipeline.build_frame(
            raw, context["duration_col"], context["event_col"], context["feature_cols"],
            id_col=context.get("id_col") or None,
            event_positive=context.get("event_positive"))

    @staticmethod
    def _split_metrics(results: dict, model: str, split_index: int) -> dict:
        for entry in results.get("per_split", {}).get(model, []):
            if entry.get("split_index") == split_index:
                return entry.get("metrics", {})
        return {}

    def _execute_save(self, job: Job):
        import torch
        import model_io

        cfg = job.config
        source_id = cfg["source_job_id"]
        model_name = cfg["model"]
        bundle_type = cfg["bundle_type"]
        name = _safe_name(cfg["name"])

        src_dir = RUNS_DIR / source_id
        results_path = src_dir / "results.json"
        if not results_path.exists():
            raise RuntimeError("the source training run was not found (nothing to save)")
        results = json.loads(results_path.read_text())
        rconfig = results["config"]
        if model_name not in rconfig["models"]:
            raise RuntimeError(f"model {model_name!r} was not part of that run")
        resolved = results["resolved"][model_name]
        quantiles = [float(q) for q in rconfig["quantiles"]]
        feature_cols = list(rconfig["feature_cols"])
        seed = int(rconfig["seed"])
        time_unit = rconfig.get("time_unit")

        art = src_dir / "artifacts"
        if not (art / "scalers.json").exists():
            raise RuntimeError("this run has no saved training artifacts (retrain to enable saving)")
        context = json.loads((art / "context.json").read_text())
        training = json.loads((art / "training.json").read_text())
        event_mapping = training.pop("event_mapping", None)
        ranges = json.loads((art / "feature_ranges.json").read_text())
        scalers_by_split = json.loads((art / "scalers.json").read_text())

        paths.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = paths.MODELS_DIR / f"{name}.cnqmodel"
        if out_path.exists():
            raise RuntimeError(f"a saved model named {name!r} already exists; choose another name")

        job.phase = "prepare"
        job.step = "Preparing to save the model"
        job._override = 0.05

        if bundle_type == "final":
            frame = self._reload_frame(context)
            job.n_models, job.n_splits, job.total_units = 1, 1, 1
            job.current_model, job.current_split = model_name, 1
            job.max_epochs = int(resolved["training"]["maximum_epochs"])
            job.epoch = 0
            job._override = None  # switch to epoch-based progress like a normal run
            job.phase = "train"
            job.step = f"Refitting {model_name} on all data"
            job._counting = True
            try:
                refit = save_model.refit_final(
                    model_name, frame, feature_cols, resolved, quantiles, seed,
                    deterministic=bool(rconfig.get("deterministic", False)))
            finally:
                job._counting = False
            state_dicts = [refit["state_dict"]]
            member_scalers = [refit["scaler"]]
            metrics = refit["metrics"]
            build_args = refit["build_args"]
            ranges = save_model.feature_ranges(frame, feature_cols)
            training = save_model.training_stats(frame)
        elif bundle_type == "single_split":
            split_index = int(cfg.get("split_index", 0))
            wpath = art / "weights" / f"{model_name}__{split_index}.pt"
            if not wpath.exists():
                raise RuntimeError(f"no saved weights for split {split_index}")
            state_dicts = [torch.load(str(wpath), weights_only=True, map_location="cpu")]
            member_scalers = [scalers_by_split[str(split_index)]]
            build_args = save_model.build_args_for(model_name, resolved, len(feature_cols), quantiles)
            metrics = self._split_metrics(results, model_name, split_index)
        elif bundle_type == "ensemble":
            n_splits = int(rconfig["n_splits"])
            state_dicts, member_scalers = [], []
            for s in range(n_splits):
                wpath = art / "weights" / f"{model_name}__{s}.pt"
                if not wpath.exists():
                    raise RuntimeError(f"missing weights for split {s}; cannot build the ensemble")
                state_dicts.append(torch.load(str(wpath), weights_only=True, map_location="cpu"))
                member_scalers.append(scalers_by_split[str(s)])
            build_args = save_model.build_args_for(model_name, resolved, len(feature_cols), quantiles)
            metrics = results["summaries"].get(model_name, {})
        else:
            raise RuntimeError(f"unknown bundle type {bundle_type!r}")

        job.phase = "report"
        job.step = "Writing the model bundle"
        job._override = 0.95
        save_model.build_bundle(
            out_path, bundle_type=bundle_type, model_name=model_name, build_args=build_args,
            feature_names=feature_cols, quantiles=quantiles, state_dicts=state_dicts,
            scalers=member_scalers, feature_ranges=ranges, training=training,
            event_mapping=event_mapping, metrics=metrics, time_unit=time_unit, seed=seed)
        meta = model_io.read_meta(out_path)
        job.results = {
            "kind": "save",
            "bundle_name": name,
            "bundle_file": out_path.name,
            "summary": model_io.summarize(meta),
            "_bundle_path": str(out_path),
        }

    # ----------------------------------------------------------------- predict
    def _resolve_bundle_path(self, cfg: dict) -> Path:
        ref = cfg.get("model_ref") or {}
        if ref.get("path"):
            p = Path(ref["path"])
            if not p.exists():
                raise RuntimeError("the uploaded model file was not found")
            return p
        if ref.get("name"):
            p = paths.MODELS_DIR / f"{_safe_name(ref['name'])}.cnqmodel"
            if not p.exists():
                raise RuntimeError(f"saved model {ref['name']!r} was not found")
            return p
        raise RuntimeError("no model was selected to predict with")

    def _execute_predict(self, job: Job):
        import model_io
        import predict as predict_mod
        import predict_report

        cfg = job.config
        bundle_path = self._resolve_bundle_path(cfg)
        job.phase = "prepare"
        job.step = "Loading the saved model"
        job._override = 0.05
        bundle = model_io.load_bundle(bundle_path)

        raw = pipeline.load_frame(cfg["csv_path"])

        def cb(step, frac):
            job.step = step
            job._override = 0.05 + 0.75 * float(frac)

        result = predict_mod.run_prediction(
            bundle, raw, cfg.get("feature_map") or {},
            id_col=cfg.get("id_col") or None, time_col=cfg.get("time_col") or None,
            event_col=cfg.get("event_col") or None, event_positive=cfg.get("event_positive"),
            progress=cb)

        job.phase = "report"
        job.step = "Building the prediction report"
        job._override = 0.9
        artifacts = predict_report.build(job.dir, bundle, result, cfg)
        job.results = {"kind": "predict", **artifacts}


def _safe_name(name: str) -> str:
    import model_io
    return model_io.safe_name(name)


def _json_default(obj):
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"not serializable: {type(obj)}")
