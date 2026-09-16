#!/usr/bin/env bash
# Launch the CNQ app via run.py.
set -e
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if command -v python3 >/dev/null 2>&1; then
    PY=python3
elif command -v python >/dev/null 2>&1; then
    PY=python
else
    echo "" >&2
    echo "ERROR: Python was not found on your PATH." >&2
    echo "Install Python 3.9 or newer (e.g. 'brew install python' on macOS," >&2
    echo "or your Linux package manager), then re-run this script." >&2
    echo "" >&2
    exit 1
fi

exec "$PY" "$DIR/run.py" "$@"
