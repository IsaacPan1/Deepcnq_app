#!/usr/bin/env python3
"""Update the deepcnq repo to the latest upstream commit, then smoke-test the app.

Runs ``git -C <repo> pull --ff-only`` (a fast-forward only, so local history is
never rewritten), prints the old and new commit, and finally runs the app's smoke
test to confirm the update didn't break anything.

Manual equivalent:

    cd deepcnq && git pull
"""
from __future__ import annotations

import subprocess
import sys

import paths


def _git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args],
                          capture_output=True, text=True)


def main() -> int:
    try:
        repo = paths.require_repo()
    except paths.RepoNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    if not (repo / ".git").exists():
        print(f"ERROR: {repo} is not a git checkout; cannot pull.", file=sys.stderr)
        return 1

    before = _git(repo, "rev-parse", "HEAD").stdout.strip()
    print(f"Updating {repo}")
    print(f"  current commit: {before[:12]}")

    pull = _git(repo, "pull", "--ff-only")
    sys.stdout.write(pull.stdout)
    sys.stderr.write(pull.stderr)
    if pull.returncode != 0:
        print("ERROR: 'git pull --ff-only' failed. Resolve the repo state manually "
              "(cd deepcnq && git status).", file=sys.stderr)
        return pull.returncode

    after = _git(repo, "rev-parse", "HEAD").stdout.strip()
    if before == after:
        print(f"  already up to date at {after[:12]}")
    else:
        print(f"  updated: {before[:12]} -> {after[:12]}")

    print("\nRunning the smoke test ...")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_smoke.py", "-q"], cwd=str(paths.HERE))
    if result.returncode == 0:
        print("Smoke test passed.")
    else:
        print("Smoke test FAILED - see output above.", file=sys.stderr)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
