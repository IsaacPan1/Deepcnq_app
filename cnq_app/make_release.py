#!/usr/bin/env python3
"""Build a self-contained ``deepquant.zip`` release (standard library only).

The archive holds a top-level ``deepquant/`` folder with two siblings:

    deepquant/
      deepcnq/    # the upstream repo, exactly as tracked by git (+ VERSION.txt)
      cnq_app/    # this app

Repo contents come from ``git ls-files`` (everything tracked, ``.git`` excluded).
A ``deepcnq/VERSION.txt`` recording the commit hash and date is written **into the
archive only** -- never onto disk in the repo, which is left untouched. App
outputs (``.venv/``, ``jobs/``, caches, test artefacts) are excluded.

The build refuses to run against a dirty repo unless ``--allow-dirty`` is given.
The repo is located via ``paths`` (``CNQ_REPO`` or the sibling ``../deepcnq``).
"""
from __future__ import annotations

import argparse
import subprocess
import zipfile
from pathlib import Path

import paths

APP_DIR = paths.HERE                       # deepquant/cnq_app
DEPLOY_ROOT = APP_DIR.parent               # deepquant/
OUTPUT = DEPLOY_ROOT.parent / "deepquant.zip"
ARCHIVE_ROOT = "deepquant"                  # top-level folder inside the zip

APP_EXCLUDE_DIRS = {".venv", "jobs", "__pycache__", ".pytest_cache", ".git",
                    ".mypy_cache", ".ipynb_checkpoints", "_runs"}
APP_EXCLUDE_FILES = {"deepquant.zip", "server.out", "server.log", ".DS_Store"}
APP_EXCLUDE_SUFFIXES = {".pyc", ".pyo"}


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(repo), *args],
                          capture_output=True, text=True, timeout=30)


def repo_is_dirty(repo: Path) -> bool:
    result = _git(repo, "status", "--porcelain")
    return result.returncode == 0 and bool(result.stdout.strip())


def version_text(repo: Path) -> str:
    commit = _git(repo, "rev-parse", "HEAD")
    date = _git(repo, "log", "-1", "--format=%cI")
    if commit.returncode == 0:
        return f"{commit.stdout.strip()}\n{date.stdout.strip()}\n"
    return "unknown\n"


def repo_files(repo: Path) -> list[str]:
    """Tracked files, relative to the repo root (POSIX paths)."""
    listed = _git(repo, "ls-files", "-z")
    if listed.returncode != 0:
        raise SystemExit(f"ERROR: 'git ls-files' failed in {repo}: {listed.stderr.strip()}")
    return [name for name in listed.stdout.split("\0") if name]


def app_excluded(rel: Path) -> bool:
    parts = rel.parts
    if any(part in APP_EXCLUDE_DIRS for part in parts):
        return True
    if rel.name in APP_EXCLUDE_FILES:
        return True
    return rel.suffix in APP_EXCLUDE_SUFFIXES


def app_files() -> list[Path]:
    return [p for p in sorted(APP_DIR.rglob("*"))
            if p.is_file() and not app_excluded(p.relative_to(APP_DIR))]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Build deepquant.zip.")
    parser.add_argument("--allow-dirty", action="store_true",
                        help="build even if the repo has uncommitted changes")
    args = parser.parse_args(argv)

    try:
        repo = paths.require_repo()
    except paths.RepoNotFoundError as exc:
        raise SystemExit(f"ERROR: {exc}")

    has_git = (repo / ".git").exists()
    if has_git and repo_is_dirty(repo) and not args.allow_dirty:
        raise SystemExit(
            f"ERROR: the repository at {repo} has uncommitted changes.\n"
            "Commit/stash them, or pass --allow-dirty to build anyway.")
    if not has_git:
        print(f"NOTE: {repo} is not a git checkout; bundling its files as-is.")

    if OUTPUT.exists():
        OUTPUT.unlink()

    repo_count = app_count = 0
    with zipfile.ZipFile(OUTPUT, "w", zipfile.ZIP_DEFLATED) as zf:
        # ---- repo (tracked files only) ----
        if has_git:
            for rel in repo_files(repo):
                source = repo / rel
                if not source.is_file():
                    continue  # tracked but missing on disk; skip quietly
                zf.write(source, f"{ARCHIVE_ROOT}/deepcnq/{rel}")
                repo_count += 1
            zf.writestr(f"{ARCHIVE_ROOT}/deepcnq/VERSION.txt", version_text(repo))
        else:
            for source in sorted(repo.rglob("*")):
                rel = source.relative_to(repo)
                if source.is_file() and ".git" not in rel.parts:
                    zf.write(source, f"{ARCHIVE_ROOT}/deepcnq/{rel.as_posix()}")
                    repo_count += 1
            zf.writestr(f"{ARCHIVE_ROOT}/deepcnq/VERSION.txt", "unknown\n")

        # ---- app ----
        for source in app_files():
            rel = source.relative_to(APP_DIR)
            zf.write(source, f"{ARCHIVE_ROOT}/cnq_app/{rel.as_posix()}")
            app_count += 1

    size_mb = OUTPUT.stat().st_size / (1024 * 1024)
    print(f"Wrote {OUTPUT} ({size_mb:.2f} MB)")
    print(f"  deepcnq/: {repo_count} tracked files + VERSION.txt")
    print(f"  cnq_app/: {app_count} files")
    print("Top-level tree:")
    print(f"  {ARCHIVE_ROOT}/")
    print(f"    deepcnq/   (upstream repo, unmodified)")
    print(f"    cnq_app/   (the app)")
    print("Unzip anywhere, then:  cd deepquant/cnq_app && python run.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
