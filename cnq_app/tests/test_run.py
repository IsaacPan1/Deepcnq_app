"""Tests for run.py's .venv robustness (broken/outdated environments).

These exercise the decision logic that decides whether an existing ``.venv``
can be reused or must be rebuilt from scratch, plus the deletion path. They do
NOT create real virtual environments or install anything.
"""
import json
import sys
from pathlib import Path

import pytest

APP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_DIR))

import run  # noqa: E402


@pytest.fixture
def fake_venv(tmp_path, monkeypatch):
    """Point run.py's global VENV/STAMP at a throwaway directory."""
    venv = tmp_path / ".venv"
    monkeypatch.setattr(run, "VENV", venv)
    monkeypatch.setattr(run, "STAMP", venv / ".cnq_setup.json")
    return venv


def _write_stamp(base_ver, variant):
    run.VENV.mkdir(parents=True, exist_ok=True)
    run.STAMP.write_text(json.dumps(run.current_stamp(base_ver, variant)))


# --------------------------------------------------------------------------- #
# query_version: the real parse + failure behavior
# --------------------------------------------------------------------------- #
def test_query_version_reads_running_interpreter():
    assert run.query_version([sys.executable]) == sys.version_info[:2]


def test_query_version_missing_interpreter_returns_none(tmp_path):
    missing = tmp_path / ("python.exe" if run.os.name == "nt" else "python")
    assert run.query_version([str(missing)]) is None


# --------------------------------------------------------------------------- #
# venv_rebuild_reason: reuse vs rebuild decision
# --------------------------------------------------------------------------- #
def test_reuses_matching_venv(fake_venv, monkeypatch):
    base_ver = (3, 12)
    # Pretend the venv interpreter runs and reports the matching version.
    monkeypatch.setattr(run, "query_version", lambda cmd: base_ver)
    _write_stamp(base_ver, "cpu")
    assert run.venv_rebuild_reason(base_ver, "cpu", reinstall=False) is None


def test_rebuilds_when_interpreter_missing(fake_venv, monkeypatch):
    """Simulate a .venv whose interpreter is gone (moved base Python / exit 103):
    the venv dir exists but its python fails to run."""
    base_ver = (3, 12)
    fake_venv.mkdir(parents=True)  # exists, but there is no python inside it
    _write_stamp(base_ver, "cpu")  # stamp is otherwise fine
    # query_version on the (nonexistent) venv python returns None for real.
    reason = run.venv_rebuild_reason(base_ver, "cpu", reinstall=False)
    assert reason is not None and "interpreter" in reason


def test_rebuilds_on_different_python_version(fake_venv, monkeypatch):
    """Simulate a .venv created with a different Python (e.g. 3.10) than the one
    this run would build with (3.12)."""
    base_ver = (3, 12)
    monkeypatch.setattr(run, "query_version", lambda cmd: (3, 10))
    _write_stamp(base_ver, "cpu")
    reason = run.venv_rebuild_reason(base_ver, "cpu", reinstall=False)
    assert reason is not None and "3.10" in reason and "3.12" in reason


def test_rebuilds_on_stamp_mismatch(fake_venv, monkeypatch):
    base_ver = (3, 12)
    monkeypatch.setattr(run, "query_version", lambda cmd: base_ver)
    _write_stamp(base_ver, "cpu")
    # A different torch variant means the stamp no longer matches.
    reason = run.venv_rebuild_reason(base_ver, "cu121", reinstall=False)
    assert reason is not None and "stamp" in reason


def test_reinstall_forces_rebuild(fake_venv, monkeypatch):
    base_ver = (3, 12)
    monkeypatch.setattr(run, "query_version", lambda cmd: base_ver)
    _write_stamp(base_ver, "cpu")
    assert run.venv_rebuild_reason(base_ver, "cpu", reinstall=True) is not None


def test_first_run_has_no_venv(fake_venv):
    assert run.venv_rebuild_reason((3, 12), "cpu", reinstall=False) == "no environment yet"


# --------------------------------------------------------------------------- #
# delete_venv: happy path and the locked-files message
# --------------------------------------------------------------------------- #
def test_delete_venv_removes_directory(fake_venv):
    fake_venv.mkdir(parents=True)
    (fake_venv / "marker.txt").write_text("x")
    run.delete_venv()
    assert not fake_venv.exists()


def test_delete_venv_locked_prints_manual_message(fake_venv, monkeypatch, capsys):
    fake_venv.mkdir(parents=True)

    def _locked(_path):
        raise PermissionError("in use")

    monkeypatch.setattr(run.shutil, "rmtree", _locked)
    with pytest.raises(SystemExit):
        run.delete_venv()
    out = capsys.readouterr().out
    assert "delete the .venv folder manually" in out


# --------------------------------------------------------------------------- #
# warn_if_synced_folder: warns once inside cloud-synced paths
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("segment,label", [
    ("OneDrive", "OneDrive"), ("iCloudDrive", "iCloud"), ("Dropbox", "Dropbox")])
def test_warns_inside_synced_folder(tmp_path, monkeypatch, capsys, segment, label):
    monkeypatch.setattr(run, "HERE", tmp_path / segment / "cnq_app")
    run.warn_if_synced_folder()
    assert label in capsys.readouterr().out


def test_no_warning_for_local_folder(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(run, "HERE", tmp_path / "projects" / "cnq_app")
    run.warn_if_synced_folder()
    assert capsys.readouterr().out == ""
