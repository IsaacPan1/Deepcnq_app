"""Locate the deepcnq repo (a sibling clone) and expose the deepquantreg package.

The app and the repo are sibling directories:

    deepquant/
      deepcnq/    # unmodified upstream clone (the app never edits it)
      cnq_app/    # this app

The repo root is resolved from the ``CNQ_REPO`` environment variable, else the
sibling ``../deepcnq`` next to this app. Its ``src/`` is inserted at the front of
``sys.path`` so the app can ``import deepquantreg`` directly -- the package is
neither installed nor vendored into cnq_app. Discovery is attempted at import but
degrades gracefully (``REPO_ROOT is None`` + ``REPO_ERROR``) so the server can
still start and report a clear "repository not found" message.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent            # cnq_app/
STATIC = HERE / "static"                           # app-local
SAMPLE_DATA = HERE / "sample_data" / "sample.csv"  # app-local
REQUIREMENTS = HERE / "requirements.txt"           # the app's OWN requirements
OUTPUT_DIR = HERE / "jobs"                          # all run outputs live here

# Markers a directory must contain to count as a valid deepcnq checkout.
_MARKER_PKG = Path("src") / "deepquantreg" / "__init__.py"
_MARKER_CFG = Path("configs") / "final" / "real" / "cnq.yaml"


class RepoNotFoundError(RuntimeError):
    """Raised when the deepcnq repository cannot be located or is incomplete."""


def _is_valid_repo(root: Path) -> bool:
    try:
        return (root / _MARKER_PKG).is_file() and (root / _MARKER_CFG).is_file()
    except OSError:
        return False


def _candidates() -> list[tuple[str, Path]]:
    out: list[tuple[str, Path]] = []
    env = os.environ.get("CNQ_REPO")
    if env:
        out.append(("CNQ_REPO", Path(env).expanduser()))
    out.append(("sibling ../deepcnq", HERE.parent / "deepcnq"))
    return out


def discover_repo() -> Path:
    tried = []
    for label, cand in _candidates():
        tried.append(f"  - {label}: {cand}")
        if _is_valid_repo(cand):
            return cand.resolve()
    raise RepoNotFoundError(
        "Could not find the deepcnq repository.\n"
        "Looked in:\n" + "\n".join(tried) + "\n\n"
        "Fix it by either:\n"
        "  1. placing this app so that ../deepcnq is the repo clone, or\n"
        "  2. setting the CNQ_REPO environment variable to the repo's path.\n"
        f"A valid repo contains {_MARKER_PKG.as_posix()} and {_MARKER_CFG.as_posix()}."
    )


# Attempt discovery at import; never raise here so importers stay robust.
try:
    REPO_ROOT: Path | None = discover_repo()
    REPO_ERROR: str | None = None
    SRC: Path | None = REPO_ROOT / "src"
    CONFIGS_DIR: Path | None = REPO_ROOT / "configs"
    CNQ_PRESET_CONFIG: Path | None = CONFIGS_DIR / "final" / "real" / "cnq.yaml"
    SMOKE_CONFIG: Path | None = CONFIGS_DIR / "smoke" / "simulated.yaml"
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
except RepoNotFoundError as exc:
    REPO_ROOT = None
    REPO_ERROR = str(exc)
    SRC = CONFIGS_DIR = CNQ_PRESET_CONFIG = SMOKE_CONFIG = None


def require_repo() -> Path:
    """Return the repo root or raise a clear RepoNotFoundError."""
    if REPO_ROOT is None:
        raise RepoNotFoundError(REPO_ERROR or "deepcnq repository not found")
    return REPO_ROOT


def _looks_like_hash(value: str) -> bool:
    return len(value) >= 7 and all(c in "0123456789abcdef" for c in value.lower())


def repo_version() -> dict:
    """Version of the repo: git commit + dirty flag, else VERSION.txt, else unknown."""
    if REPO_ROOT is None:
        return {"version": "unknown", "dirty": None, "source": "not-found", "date": None}
    if (REPO_ROOT / ".git").exists():
        try:
            commit = subprocess.run(
                ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
                capture_output=True, text=True, timeout=5)
            if commit.returncode == 0:
                status = subprocess.run(
                    ["git", "-C", str(REPO_ROOT), "status", "--porcelain"],
                    capture_output=True, text=True, timeout=5)
                date = subprocess.run(
                    ["git", "-C", str(REPO_ROOT), "log", "-1", "--format=%cI"],
                    capture_output=True, text=True, timeout=5)
                return {"version": commit.stdout.strip(),
                        "dirty": bool(status.stdout.strip()),
                        "source": "git",
                        "date": date.stdout.strip() or None}
        except (OSError, subprocess.SubprocessError):
            pass
    version_file = REPO_ROOT / "VERSION.txt"
    if version_file.is_file():
        # utf-8-sig strips a BOM if one slipped in when the file was written.
        text = version_file.read_text(encoding="utf-8-sig")
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        version = lines[0] if lines else "unknown"
        date = lines[1] if len(lines) > 1 else None
        return {"version": version, "dirty": None, "source": "VERSION.txt", "date": date}
    return {"version": "unknown", "dirty": None, "source": "unknown", "date": None}


def repo_version_str() -> str:
    v = repo_version()
    version = v["version"]
    if version == "unknown":
        return "unknown"
    short = version[:12] if _looks_like_hash(version) else version
    return f"{short}{'-dirty' if v.get('dirty') else ''}"


def repo_info() -> dict:
    """Everything the UI / report needs about the repo, without raising."""
    v = repo_version()
    return {
        "found": REPO_ROOT is not None,
        "path": str(REPO_ROOT) if REPO_ROOT is not None else None,
        "version": v["version"],
        "version_str": repo_version_str(),
        "dirty": v["dirty"],
        "source": v["source"],
        "date": v["date"],
        "error": REPO_ERROR,
    }
