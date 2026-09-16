"""Read and write ``.cnqmodel`` bundles: a trained CNQ model plus everything
needed to reproduce its predictions on new subjects.

A bundle is a plain zip with three members:

* ``weights.pt`` -- ``torch.save`` of a *list of state_dicts* (one per ensemble
  member; length 1 for a final/single-split model). State dicts hold only
  tensors, so they load back with ``weights_only=True`` (no pickle of arbitrary
  Python objects -- see DEVELOPER.md for why).
* ``model.json`` -- all non-weight metadata: the exact ``build_model`` arguments,
  the ordered feature names, the training scaler (mean/scale), the quantile grid,
  the log-time output convention, per-feature training ranges, training
  size/events/censoring, the event mapping, training validation metrics, the
  deepcnq version, library versions, device, seed and creation time.
* ``manifest.json`` -- the SHA-256 of ``model.json`` and ``weights.pt`` so a
  corrupted or tampered bundle is caught on load.

``save_bundle`` writes one; ``read_meta`` cheaply returns just ``model.json``
(no torch import, used to list saved models); ``load_bundle`` verifies the
hashes and format version, rebuilds each model with the package's
``build_model`` and ``load_state_dict(strict=True)``, and warns -- but does not
fail -- when the current deepcnq commit or torch version differs from the
bundle's.
"""
from __future__ import annotations

import hashlib
import io
import json
import re
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import paths  # puts deepquantreg on sys.path; stdlib-only, safe to import early

# Bump when the on-disk layout changes in a backwards-incompatible way. Bundles
# with a HIGHER major than this app are refused; a lower one is accepted.
FORMAT_VERSION = 1

# How to read the model's raw outputs. Predictions are natural-log survival
# time; exp() maps them back to the original time scale.
LOG_TIME_CONVENTION = {
    "output": "log_time",
    "inverse": "exp",
    "note": "Model outputs are natural-log survival time; apply exp() for the time scale.",
}

BUNDLE_TYPES = ("final", "ensemble", "single_split")

WEIGHTS_NAME = "weights.pt"
MODEL_NAME = "model.json"
MANIFEST_NAME = "manifest.json"

# Keys the caller must supply in ``meta``; save_bundle fills the rest.
_REQUIRED_META_KEYS = (
    "model", "build_args", "feature_names", "scaler", "quantiles", "bundle",
    "feature_ranges", "missing_handling", "training", "event_mapping",
    "metrics", "time_unit", "device", "seed", "deepcnq",
)


class BundleError(Exception):
    """A bundle is missing, corrupted, or incompatible with this app."""


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def safe_name(name: str) -> str:
    """A filesystem-safe bundle name (no path traversal): letters, digits,
    space, dot, dash and underscore only, capped at 80 chars."""
    cleaned = re.sub(r"[^A-Za-z0-9 _.-]+", "_", (name or "").strip()).strip(" .")
    if not cleaned:
        raise BundleError("please provide a valid model name")
    return cleaned[:80]


def _library_versions() -> dict[str, str]:
    import numpy
    import pandas
    import torch

    try:
        import deepquantreg
        dq = getattr(deepquantreg, "__version__", "unknown")
    except Exception:  # noqa: BLE001
        dq = "unknown"
    return {
        "torch": torch.__version__,
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "deepquantreg": dq,
    }


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


# --------------------------------------------------------------------------- #
# save
# --------------------------------------------------------------------------- #
def save_bundle(path, *, state_dicts: list, meta: dict) -> Path:
    """Write a ``.cnqmodel`` bundle to ``path``.

    ``state_dicts`` is a list of torch state dicts (one per ensemble member;
    length 1 otherwise). ``meta`` supplies the training-specific fields (see
    ``_REQUIRED_META_KEYS``); this function adds ``format_version``, the
    log-time convention, library versions and the creation time, then writes
    ``weights.pt``, ``model.json`` and ``manifest.json`` into the zip.
    """
    import torch

    if not state_dicts:
        raise BundleError("save_bundle needs at least one state_dict")
    missing = [k for k in _REQUIRED_META_KEYS if k not in meta]
    if missing:
        raise BundleError(f"meta is missing required keys: {', '.join(missing)}")

    n_members = len(state_dicts)
    bundle_type = meta["bundle"].get("type")
    if bundle_type not in BUNDLE_TYPES:
        raise BundleError(f"bundle type must be one of {BUNDLE_TYPES}, got {bundle_type!r}")

    full = dict(meta)
    full["format_version"] = FORMAT_VERSION
    full["log_time"] = LOG_TIME_CONVENTION
    full["versions"] = _library_versions()
    full["created_at"] = _now_iso()
    # Trust the actual member count over whatever the caller guessed.
    full["bundle"] = {**meta["bundle"], "type": bundle_type, "n_members": n_members}

    # Serialize weights (tensors only) to bytes so we can hash before writing.
    cpu_states = [{k: v.detach().cpu() for k, v in sd.items()} for sd in state_dicts]
    buf = io.BytesIO()
    torch.save(cpu_states, buf)
    weights_bytes = buf.getvalue()

    model_bytes = json.dumps(full, indent=2, default=_json_default).encode("utf-8")
    manifest = {
        "algorithm": "sha256",
        "files": {MODEL_NAME: _sha256(model_bytes), WEIGHTS_NAME: _sha256(weights_bytes)},
    }
    manifest_bytes = json.dumps(manifest, indent=2).encode("utf-8")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(MODEL_NAME, model_bytes)
        zf.writestr(WEIGHTS_NAME, weights_bytes)
        zf.writestr(MANIFEST_NAME, manifest_bytes)
    tmp.replace(path)  # atomic-ish: never leave a half-written bundle behind
    return path


# --------------------------------------------------------------------------- #
# read metadata only (no torch)
# --------------------------------------------------------------------------- #
def read_meta(path) -> dict:
    """Return the verified ``model.json`` without loading any weights.

    Verifies the ``model.json`` hash and the format version, so listing saved
    models still catches corruption, but never imports torch.
    """
    try:
        with zipfile.ZipFile(path) as zf:
            names = set(zf.namelist())
            for required in (MODEL_NAME, WEIGHTS_NAME, MANIFEST_NAME):
                if required not in names:
                    raise BundleError(f"bundle is missing {required}")
            manifest = json.loads(zf.read(MANIFEST_NAME))
            model_bytes = zf.read(MODEL_NAME)
    except zipfile.BadZipFile as exc:
        raise BundleError(f"not a valid .cnqmodel file (bad zip): {exc}") from exc
    except OSError as exc:
        raise BundleError(f"could not open bundle: {exc}") from exc

    expected = (manifest.get("files") or {}).get(MODEL_NAME)
    if expected and _sha256(model_bytes) != expected:
        raise BundleError("model.json is corrupted (hash mismatch)")
    try:
        meta = json.loads(model_bytes)
    except ValueError as exc:
        raise BundleError(f"model.json is not valid JSON: {exc}") from exc
    _check_format_version(meta)
    return meta


def _check_format_version(meta: dict) -> None:
    fv = meta.get("format_version")
    if not isinstance(fv, int):
        raise BundleError("bundle has no valid format_version")
    if fv > FORMAT_VERSION:
        raise BundleError(
            f"this bundle is format version {fv}, but this app supports up to "
            f"{FORMAT_VERSION}. Update the app to load it.")


# --------------------------------------------------------------------------- #
# load (rebuild models)
# --------------------------------------------------------------------------- #
@dataclass
class LoadedBundle:
    models: list          # rebuilt nn.Modules, weights loaded, on CPU
    meta: dict
    warnings: list = field(default_factory=list)

    @property
    def quantiles(self) -> list:
        return list(self.meta.get("quantiles", []))

    @property
    def feature_names(self) -> list:
        return list(self.meta.get("feature_names", []))


def load_bundle(path) -> LoadedBundle:
    """Verify and load a bundle, rebuilding every member model.

    Hashes and the format version are checked; weights load with
    ``weights_only=True`` and ``map_location="cpu"``; each model is rebuilt from
    the stored ``build_args`` with the package's ``build_model`` and loaded with
    ``load_state_dict(strict=True)``. Commit / torch-version drift is a warning,
    not an error.
    """
    import torch
    from deepquantreg import build_model

    try:
        with zipfile.ZipFile(path) as zf:
            names = set(zf.namelist())
            for required in (MODEL_NAME, WEIGHTS_NAME, MANIFEST_NAME):
                if required not in names:
                    raise BundleError(f"bundle is missing {required}")
            manifest = json.loads(zf.read(MANIFEST_NAME))
            model_bytes = zf.read(MODEL_NAME)
            weights_bytes = zf.read(WEIGHTS_NAME)
    except zipfile.BadZipFile as exc:
        raise BundleError(f"not a valid .cnqmodel file (bad zip): {exc}") from exc
    except OSError as exc:
        raise BundleError(f"could not open bundle: {exc}") from exc

    files = manifest.get("files") or {}
    if files.get(MODEL_NAME) and _sha256(model_bytes) != files[MODEL_NAME]:
        raise BundleError("model.json is corrupted (hash mismatch)")
    if files.get(WEIGHTS_NAME) and _sha256(weights_bytes) != files[WEIGHTS_NAME]:
        raise BundleError("weights.pt is corrupted (hash mismatch)")

    meta = json.loads(model_bytes)
    _check_format_version(meta)

    build_args = meta.get("build_args")
    if not isinstance(build_args, dict):
        raise BundleError("bundle model.json has no build_args")

    try:
        states = torch.load(io.BytesIO(weights_bytes), weights_only=True, map_location="cpu")
    except Exception as exc:  # noqa: BLE001 - torch raises many types on bad checkpoints
        raise BundleError(f"could not read weights.pt: {exc}") from exc
    if not isinstance(states, (list, tuple)) or not states:
        raise BundleError("weights.pt does not contain a non-empty list of state_dicts")

    models = []
    for i, sd in enumerate(states):
        try:
            model = build_model(build_args)
        except Exception as exc:  # noqa: BLE001
            raise BundleError(
                f"could not rebuild the model from build_args ({type(exc).__name__}: {exc})") from exc
        try:
            model.load_state_dict(sd, strict=True)
        except Exception as exc:  # noqa: BLE001
            raise BundleError(
                f"weights for member {i} do not match the model architecture "
                f"({type(exc).__name__}: {exc})") from exc
        model.eval()
        models.append(model)

    return LoadedBundle(models=models, meta=meta, warnings=_compat_warnings(meta))


def _compat_warnings(meta: dict) -> list:
    """Non-fatal warnings when the environment differs from the bundle's."""
    warnings: list[str] = []

    saved_versions = meta.get("versions") or {}
    try:
        import torch
        current_torch = torch.__version__
    except Exception:  # noqa: BLE001
        current_torch = None
    saved_torch = saved_versions.get("torch")
    if current_torch and saved_torch and current_torch != saved_torch:
        warnings.append(
            f"This bundle was saved with torch {saved_torch}; you are running torch "
            f"{current_torch}. Predictions should match but may differ slightly.")

    saved_commit = (meta.get("deepcnq") or {}).get("version")
    current_commit = paths.repo_version().get("version")
    if saved_commit and current_commit and saved_commit not in ("unknown", None) \
            and current_commit not in ("unknown", None) and saved_commit != current_commit:
        warnings.append(
            f"This bundle was saved against deepcnq {str(saved_commit)[:12]}; the current "
            f"checkout is {str(current_commit)[:12]}. If model code changed, predictions "
            f"may differ.")
    return warnings


# --------------------------------------------------------------------------- #
# listing summary
# --------------------------------------------------------------------------- #
def summarize(meta: dict) -> dict:
    """Compact record for the saved-model list / API."""
    training = meta.get("training") or {}
    bundle = meta.get("bundle") or {}
    return {
        "model": meta.get("model"),
        "bundle_type": bundle.get("type"),
        "n_members": bundle.get("n_members"),
        "created_at": meta.get("created_at"),
        "n": training.get("n"),
        "n_events": training.get("events"),
        "censoring_pct": training.get("censoring_pct"),
        "n_features": len(meta.get("feature_names") or []),
        "features": list(meta.get("feature_names") or []),
        "quantiles": list(meta.get("quantiles") or []),
        "time_unit": meta.get("time_unit"),
        "deepcnq": (meta.get("deepcnq") or {}).get("version_str"),
    }


def _json_default(obj: Any):
    # Keep numpy scalars/arrays JSON-friendly without importing numpy eagerly.
    import numpy as np

    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"not serializable: {type(obj)}")
