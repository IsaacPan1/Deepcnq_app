"""Tests for the reverse-proxy base-path logic.

The browser-side `computeAppBase` (static/app.js) resolves the directory the app
was served from so relative requests work both at the root and behind
jupyter-server-proxy (/proxy/8000/). These tests pin that logic:

* `test_compute_app_base_via_node` runs the *actual* app.js function in Node for
  the five canonical inputs (skips cleanly if Node is unavailable).
* `test_selfcheck_copy_in_sync` guards the standalone copy in selfcheck.html
  against drift, since that page must not depend on app.js.
* `test_ping_endpoint` boots the real server and checks /api/ping echoes the
  received path.
"""
import json
import re
import shutil
import subprocess
import sys
import threading
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

APP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_DIR))

APP_JS = APP_DIR / "static" / "app.js"
SELFCHECK = APP_DIR / "static" / "selfcheck.html"

# (pathname, expected APP_BASE) — the contract the UI depends on.
CASES = [
    ("/", "/"),
    ("/index.html", "/"),
    ("/proxy/8000/", "/proxy/8000/"),
    ("/proxy/8000", "/proxy/8000/"),            # missing trailing slash
    ("/user/isaac/proxy/8000/", "/user/isaac/proxy/8000/"),
]


def _extract_compute_app_base(source: str) -> str:
    """Return the `function computeAppBase(...) { ... }` text, brace-matched."""
    start = source.index("function computeAppBase")
    i = source.index("{", start)
    depth = 0
    for j in range(i, len(source)):
        if source[j] == "{":
            depth += 1
        elif source[j] == "}":
            depth -= 1
            if depth == 0:
                return source[start:j + 1]
    raise AssertionError("unbalanced braces in computeAppBase")


def test_compute_app_base_via_node():
    node = shutil.which("node")
    if not node:
        pytest.skip("node not available to execute the browser function")

    fn = _extract_compute_app_base(APP_JS.read_text(encoding="utf-8"))
    harness = (
        fn
        + "\nconst cases = " + json.dumps(CASES) + ";\n"
        + "for (const [inp, exp] of cases) {\n"
        + "  const got = computeAppBase(inp);\n"
        + "  if (got !== exp) { console.error(`FAIL ${inp}: ${got} != ${exp}`); process.exit(1); }\n"
        + "}\nconsole.log('OK');\n"
    )
    proc = subprocess.run([node, "-e", harness], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "OK" in proc.stdout


def test_selfcheck_copy_in_sync():
    """selfcheck.html inlines computeAppBase (it must not import app.js)."""
    app_fn = _extract_compute_app_base(APP_JS.read_text(encoding="utf-8"))
    sc_fn = _extract_compute_app_base(SELFCHECK.read_text(encoding="utf-8"))

    def norm(s):  # compare code only — comments/whitespace may differ
        s = re.sub(r"//[^\n]*", "", s)
        return re.sub(r"\s+", " ", s).strip()

    assert norm(app_fn) == norm(sc_fn), (
        "computeAppBase in selfcheck.html drifted from app.js; keep them identical"
    )


def _boot_server():
    from server import Handler
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd, port


def test_ping_endpoint():
    httpd, port = _boot_server()
    try:
        with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/api/ping", timeout=30) as resp:
            assert resp.status == 200
            body = json.loads(resp.read())
        assert body["ok"] is True
        assert body["path_seen"] == "/api/ping"
    finally:
        httpd.shutdown()
        httpd.server_close()
