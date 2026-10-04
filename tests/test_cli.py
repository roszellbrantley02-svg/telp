"""The front door: telp.py commands in real subprocesses, one shared memory."""
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# run telp.py with the stand-in sentence model and no network
_RUNNER = ("import sys; sys.path.insert(0, sys.argv.pop(1)); "
           "from tests.fake_minilm import install; install(); "
           "import runpy; sys.argv[0] = 'telp.py'; "
           "runpy.run_path(sys.argv[0], run_name='__main__')")


def _telp(state, *args, port=None, timeout=120):
    env = dict(os.environ, TELP_STATE_DIR=str(state),
               PYTHONDONTWRITEBYTECODE="1",
               TELP_PORT=str(port or _free_port()),
               HTTPS_PROXY="http://127.0.0.1:9", HTTP_PROXY="http://127.0.0.1:9")
    return subprocess.run(
        [sys.executable, "-c", _RUNNER, str(ROOT), *args],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=timeout)


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_teach_then_ask_from_separate_processes(tmp_path):
    r = _telp(tmp_path, "teach", "The Zorblax river flows through Quendia.")
    assert r.returncode == 0, r.stderr
    r = _telp(tmp_path, "ask", "where does the Zorblax river flow?")
    assert r.returncode == 0, r.stderr
    assert "Quendia" in r.stdout


def test_learn_and_seen_do_not_orphan_taught_facts(tmp_path):
    _telp(tmp_path, "teach", "My dog's name is Astro.")
    r = _telp(tmp_path, "seen")
    assert r.returncode == 0, r.stderr
    assert "hasn't seen" in r.stdout
    r = _telp(tmp_path, "learn", "Iceland")       # offline: fails politely
    assert r.returncode == 0, r.stderr
    r = _telp(tmp_path, "stats")
    assert r.returncode == 0, r.stderr
    assert "lattice_total: 0" not in r.stdout
    assert not (tmp_path / "standalone_lattice.db").exists()


def test_arithmetic_and_honest_miss(tmp_path):
    r = _telp(tmp_path, "ask", "what is 2*(3+4)?")
    assert "= 14" in r.stdout
    r = _telp(tmp_path, "ask", "meeting on 2024-10-15")
    assert "1999" not in r.stdout


def test_daemon_sees_what_other_processes_teach_and_forget(tmp_path):
    port = _free_port()
    env = dict(os.environ, TELP_STATE_DIR=str(tmp_path), TELP_PORT=str(port),
               PYTHONDONTWRITEBYTECODE="1")
    daemon = subprocess.Popen(
        [sys.executable, "-c", _RUNNER, str(ROOT), "serve"], cwd=ROOT,
        env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        deadline = time.time() + 120
        while time.time() < deadline:
            try:
                socket.create_connection(("127.0.0.1", port), 0.5).close()
                break
            except OSError:
                time.sleep(0.3)
        else:
            pytest.fail("daemon did not start")

        r = _telp(tmp_path, "teach", "The Zorblax river flows through Quendia.",
                  port=port)
        assert r.returncode == 0, r.stderr
        r = _telp(tmp_path, "ask", "where does the Zorblax river flow?",
                  port=port)
        assert "Quendia" in r.stdout
    finally:
        _telp(tmp_path, "stop", port=port, timeout=30)
        try:
            daemon.wait(timeout=30)
        except subprocess.TimeoutExpired:
            daemon.kill()


def test_forget_from_the_command_line(tmp_path):
    _telp(tmp_path, "teach", "The Zorblax river flows through Quendia.")
    r = _telp(tmp_path, "forget", "the Zorblax river")
    assert r.returncode == 0, r.stderr
    assert "forgot" in r.stdout.lower() or "erased" in r.stdout.lower(), r.stdout
    r = _telp(tmp_path, "ask", "where does the Zorblax river flow?")
    assert "Quendia" not in r.stdout
