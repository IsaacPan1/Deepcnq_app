#!/usr/bin/env python3
"""First-run launcher for the CNQ app (standard library only, cross-platform).

On first run it creates a local ``.venv``, installs PyTorch (CPU wheels on
Windows/Linux, plain PyPI on macOS, CUDA wheels with ``--cuda``) and the rest of
``requirements.txt``, verifies the scientific stack imports, then launches the
server. A stamp file records the requirements hash, torch variant and Python
version, so later runs skip straight to launching. Any extra arguments (a port,
``-b``) are passed through to ``python -m server``.

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
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
VENV = HERE / ".venv"
REQUIREMENTS = HERE / "requirements.txt"  # the app's OWN requirements, never the repo's
STAMP = VENV / ".cnq_setup.json"

MIN_PYTHON = (3, 9)
MAX_TESTED_PYTHON = (3, 12)

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
    if sys.version_info[:2] > MAX_TESTED_PYTHON:
        print(f"WARNING: Python {platform.python_version()} is newer than the latest "
              f"tested version ({MAX_TESTED_PYTHON[0]}.{MAX_TESTED_PYTHON[1]}). "
              f"If dependency installation fails, try a {MAX_TESTED_PYTHON[0]}."
              f"{MAX_TESTED_PYTHON[1]} interpreter.")


def torch_variant(cuda: bool) -> str:
    if platform.system() == "Darwin":
        return "default"  # macOS: PyPI wheels (CPU/MPS)
    return "cu121" if cuda else "cpu"


def venv_python() -> Path:
    if os.name == "nt":
        return VENV / "Scripts" / "python.exe"
    return VENV / "bin" / "python"


def requirements_hash() -> str:
    data = REQUIREMENTS.read_bytes() if REQUIREMENTS.exists() else b""
    return hashlib.sha256(data).hexdigest()


def current_stamp(variant: str) -> dict:
    return {
        "requirements_sha256": requirements_hash(),
        "torch_variant": variant,
        "python": platform.python_version(),
    }


def stamp_matches(variant: str) -> bool:
    if not venv_python().exists() or not STAMP.exists():
        return False
    try:
        saved = json.loads(STAMP.read_text())
    except (OSError, ValueError):
        return False
    return saved == current_stamp(variant)


def run(cmd, **kwargs) -> None:
    print("  $ " + " ".join(str(c) for c in cmd))
    result = subprocess.run(cmd, **kwargs)
    if result.returncode != 0:
        fail(f"command failed (exit {result.returncode}): {' '.join(str(c) for c in cmd)}")


def create_venv() -> None:
    if venv_python().exists():
        return
    print(f"Creating virtual environment in {VENV} ...")
    try:
        run([sys.executable, "-m", "venv", str(VENV)])
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


def setup(variant: str, reinstall: bool) -> None:
    if not reinstall and stamp_matches(variant):
        print("Dependencies already installed and up to date - skipping setup.")
        return
    if reinstall:
        print("Reinstall requested - refreshing dependencies.")
    create_venv()
    install_dependencies(variant)
    verify_imports()
    STAMP.write_text(json.dumps(current_stamp(variant), indent=2))
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
    variant = torch_variant(args.cuda)
    setup(variant, args.reinstall)
    return launch(server_args)


if __name__ == "__main__":
    raise SystemExit(main())
