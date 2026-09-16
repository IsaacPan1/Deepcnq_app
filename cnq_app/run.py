#!/usr/bin/env python3
"""First-run launcher for the CNQ app (standard library only, cross-platform).

On first run it creates a local ``.venv``, installs PyTorch (CPU wheels on
Windows/Linux, plain PyPI on macOS, CUDA wheels with ``--cuda``) and the rest of
``requirements.txt``, verifies the scientific stack imports, then launches the
server. A stamp file records the requirements hash, torch variant and Python
version, so later runs skip straight to launching. Any extra arguments (a port,
``-b``) are passed through to ``python -m server``.

Later runs first sanity-check the existing ``.venv``: if its interpreter no
longer works (a moved/deleted base Python, exit 103 "No Python at ...") or was
built with a different Python major.minor than this run uses, or its stamp
doesn't match for any reason, the whole ``.venv`` is deleted and rebuilt from
scratch rather than installed into.

    python run.py                # set up if needed, then serve on 8000
    python run.py 8001           # serve on a different port
    python run.py --cuda         # install CUDA torch wheels
    python run.py --reinstall    # force a clean dependency install
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
VENV = HERE / ".venv"
REQUIREMENTS = HERE / "requirements.txt"  # the app's OWN requirements, never the repo's
STAMP = VENV / ".cnq_setup.json"

MIN_PYTHON = (3, 9)
MAX_TESTED_PYTHON = (3, 13)
# Interpreters we look for (newest first) when run.py itself is on an untested
# Python and we'd rather build the venv with something supported.
SUPPORTED_FALLBACKS = ((3, 13), (3, 12), (3, 11))

# torch install channels keyed by variant name (recorded in the stamp)
TORCH_INDEX = {
    "cpu": "https://download.pytorch.org/whl/cpu",
    "cu121": "https://download.pytorch.org/whl/cu121",
    # "default" -> plain PyPI (macOS), no index url
}


def fail(message: str, code: int = 1) -> "NoReturn":  # type: ignore[valid-type]
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(code)


def check_python() -> None:
    if sys.version_info[:2] < MIN_PYTHON:
        fail(f"Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+ is required, "
             f"but this is {platform.python_version()}.")


def torch_variant(cuda: bool) -> str:
    if platform.system() == "Darwin":
        return "default"  # macOS: PyPI wheels (CPU/MPS)
    return "cu121" if cuda else "cpu"


def venv_python() -> Path:
    if os.name == "nt":
        return VENV / "Scripts" / "python.exe"
    return VENV / "bin" / "python"


def query_version(cmd: list[str]) -> "tuple[int, int] | None":
    """Return the ``(major, minor)`` of the interpreter invoked by ``cmd``, or
    ``None`` if it can't be run at all.

    A venv whose base Python was moved or deleted prints "No Python at ..." and
    exits 103 (Windows) or fails to launch; a missing command raises OSError.
    All of these map to ``None`` so the caller can rebuild.
    """
    try:
        proc = subprocess.run([*cmd, "-c", "import sys; print(sys.version)"],
                              capture_output=True, text=True)
    except OSError:
        return None
    if proc.returncode != 0:
        return None
    token = proc.stdout.strip().split()[0] if proc.stdout.strip() else ""
    parts = token.split(".")
    try:
        return int(parts[0]), int(parts[1])
    except (IndexError, ValueError):
        return None


def find_supported_python() -> "tuple[list[str], tuple[int, int]] | None":
    """Look for a supported (<= MAX_TESTED) interpreter to build the venv with,
    since run.py itself is on an untested Python. Uses the ``py`` launcher on
    Windows and versioned ``pythonX.Y`` names on PATH elsewhere. Returns the
    command prefix and its ``(major, minor)``, or ``None`` if none is found.
    """
    for major, minor in SUPPORTED_FALLBACKS:
        cmd = (["py", f"-{major}.{minor}"] if os.name == "nt"
               else [f"python{major}.{minor}"])
        if query_version(cmd) == (major, minor):
            return cmd, (major, minor)
    return None


def base_python() -> "tuple[list[str], tuple[int, int]]":
    """The interpreter used to CREATE the venv, plus its ``(major, minor)``.

    Normally run.py's own interpreter. If that's newer than the tested range,
    prefer a supported interpreter when one is available, otherwise warn and
    carry on with the current one.
    """
    own = sys.version_info[:2]
    if own <= MAX_TESTED_PYTHON:
        return [sys.executable], own
    found = find_supported_python()
    if found:
        cmd, ver = found
        print(f"Python {platform.python_version()} is newer than the tested "
              f"{MAX_TESTED_PYTHON[0]}.{MAX_TESTED_PYTHON[1]}; building .venv with "
              f"'{' '.join(cmd)}' (Python {ver[0]}.{ver[1]}) instead.")
        return cmd, ver
    print(f"WARNING: Python {platform.python_version()} is newer than the latest "
          f"tested version ({MAX_TESTED_PYTHON[0]}.{MAX_TESTED_PYTHON[1]}) and no "
          f"{SUPPORTED_FALLBACKS[0][0]}.{SUPPORTED_FALLBACKS[0][1]}/"
          f"{SUPPORTED_FALLBACKS[-1][0]}.{SUPPORTED_FALLBACKS[-1][1]} interpreter "
          f"was found. Continuing; if dependency installation fails, install a "
          f"supported Python and try again.")
    return [sys.executable], own


def warn_if_synced_folder() -> None:
    """Warn once if the app lives inside a cloud-synced folder, which can lock
    or corrupt files in ``.venv`` while it syncs."""
    lowered = str(HERE).lower()
    for token, label in (("onedrive", "OneDrive"), ("icloud", "iCloud"),
                         ("dropbox", "Dropbox")):
        if token in lowered:
            print(f"WARNING: this app is inside a {label} folder. Cloud sync can "
                  f"lock or corrupt files in .venv. If setup misbehaves, move the "
                  f"app to a local, non-synced folder.")
            return


def requirements_hash() -> str:
    data = REQUIREMENTS.read_bytes() if REQUIREMENTS.exists() else b""
    return hashlib.sha256(data).hexdigest()


def current_stamp(base_ver: "tuple[int, int]", variant: str) -> dict:
    return {
        "requirements_sha256": requirements_hash(),
        "torch_variant": variant,
        "python": f"{base_ver[0]}.{base_ver[1]}",
    }


def stamp_matches(base_ver: "tuple[int, int]", variant: str) -> bool:
    if not STAMP.exists():
        return False
    try:
        saved = json.loads(STAMP.read_text())
    except (OSError, ValueError):
        return False
    return saved == current_stamp(base_ver, variant)


def venv_rebuild_reason(base_ver: "tuple[int, int]", variant: str,
                        reinstall: bool) -> "str | None":
    """Return ``None`` if the existing ``.venv`` can be reused as-is, otherwise a
    short reason describing why it must be rebuilt from scratch."""
    if reinstall:
        return "reinstall requested"
    if not VENV.exists():
        return "no environment yet"
    ver = query_version([str(venv_python())])
    if ver is None:
        return "its Python interpreter is missing or broken"
    if ver != base_ver:
        return (f"it was built with Python {ver[0]}.{ver[1]}, but this run needs "
                f"Python {base_ver[0]}.{base_ver[1]}")
    if not stamp_matches(base_ver, variant):
        return "its dependency stamp doesn't match"
    return None


def run(cmd, **kwargs) -> None:
    print("  $ " + " ".join(str(c) for c in cmd))
    result = subprocess.run(cmd, **kwargs)
    if result.returncode != 0:
        fail(f"command failed (exit {result.returncode}): {' '.join(str(c) for c in cmd)}")


def delete_venv() -> None:
    """Remove ``.venv`` so it can be rebuilt. If files are locked (a running
    server, or OneDrive/Dropbox syncing), ask the user to remove it manually."""
    try:
        shutil.rmtree(VENV)
    except OSError:
        print("Close any running CNQ app and delete the .venv folder manually, "
              "then run again.")
        raise SystemExit(1)


def create_venv(base_cmd: list[str]) -> None:
    print(f"Creating virtual environment in {VENV} ...")
    try:
        run([*base_cmd, "-m", "venv", str(VENV)])
    except SystemExit:
        fail("could not create a virtual environment. On Debian/Ubuntu you may need "
             "'sudo apt install python3-venv'.")


def install_dependencies(variant: str) -> None:
    vpy = str(venv_python())
    print("Upgrading pip ...")
    run([vpy, "-m", "pip", "install", "--upgrade", "pip"])

    print(f"Installing PyTorch (variant: {variant}) ...")
    torch_cmd = [vpy, "-m", "pip", "install", "torch"]
    index = TORCH_INDEX.get(variant)
    if index:
        torch_cmd += ["--index-url", index]
    run(torch_cmd)

    if REQUIREMENTS.exists():
        print("Installing requirements.txt ...")
        run([vpy, "-m", "pip", "install", "-r", str(REQUIREMENTS)])
    else:
        print(f"WARNING: {REQUIREMENTS} not found; skipping requirements install.")


def verify_imports() -> None:
    print("Verifying the scientific stack imports ...")
    check = ("import numpy, pandas, sklearn, torch, matplotlib; "
             "print('  torch', torch.__version__, '| cuda', torch.cuda.is_available())")
    run([str(venv_python()), "-c", check])


def setup(base_cmd: list[str], base_ver: "tuple[int, int]", variant: str,
          reinstall: bool) -> None:
    reason = venv_rebuild_reason(base_ver, variant, reinstall)
    if reason is None:
        print("Dependencies already installed and up to date - skipping setup.")
        return
    if VENV.exists():
        # Never install into a stale/broken environment - start clean.
        print(f"Rebuilding .venv from scratch ({reason}).")
        delete_venv()
    create_venv(base_cmd)
    install_dependencies(variant)
    verify_imports()
    STAMP.write_text(json.dumps(current_stamp(base_ver, variant), indent=2))
    print("Setup complete.")


def launch(server_args: list[str]) -> int:
    vpy = str(venv_python())
    cmd = [vpy, "-m", "server", *server_args]
    print("Launching the CNQ app ...")
    try:
        return subprocess.run(cmd, cwd=str(HERE)).returncode
    except KeyboardInterrupt:
        return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="run.py", description="Set up the local environment and launch the CNQ app.",
        epilog="Any other arguments (e.g. a port, -b) are passed through to the server.")
    parser.add_argument("--cuda", action="store_true",
                        help="install CUDA torch wheels (ignored on macOS)")
    parser.add_argument("--reinstall", action="store_true",
                        help="force a clean dependency install")
    args, server_args = parser.parse_known_args(argv)

    check_python()
    warn_if_synced_folder()
    base_cmd, base_ver = base_python()
    variant = torch_variant(args.cuda)
    setup(base_cmd, base_ver, variant, args.reinstall)
    return launch(server_args)


if __name__ == "__main__":
    raise SystemExit(main())
