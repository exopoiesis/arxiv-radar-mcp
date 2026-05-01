"""Tests for the SSH-tunnel + stdio↔HTTP proxy.

We don't actually open a real tunnel — that requires sshd and a real
backend. Instead we mock subprocess.Popen and shutil.which to verify
the plumbing: command construction, port-readiness polling, cleanup.
"""
from __future__ import annotations

import socket

import pytest

from arxiv_radar_mcp import proxy


# ----- _pick_free_port -------------------------------------------------------


def test_pick_free_port_returns_usable_port():
    port = proxy._pick_free_port()
    assert 1024 <= port <= 65535
    # Should be re-bindable right after — kernel doesn't hold it.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", port))


# ----- _start_tunnel ---------------------------------------------------------


def test_start_tunnel_constructs_correct_ssh_command(monkeypatch):
    captured = {}

    class _FakePopen:
        def __init__(self, cmd, **kwargs):
            captured["cmd"] = cmd
            captured["kwargs"] = kwargs

        def poll(self):
            return None

    monkeypatch.setattr(proxy.subprocess, "Popen", _FakePopen)
    proxy._start_tunnel("ssh", "user@host", local=12345, remote=8765)

    cmd = captured["cmd"]
    assert cmd[0] == "ssh"
    assert "-N" in cmd
    assert "127.0.0.1:12345:127.0.0.1:8765" in cmd
    assert "user@host" in cmd
    # Sanity: ExitOnForwardFailure is set to surface remote port-already-in-use.
    assert any("ExitOnForwardFailure=yes" in arg for arg in cmd)


# ----- _wait_for_port --------------------------------------------------------


def test_wait_for_port_returns_true_when_listener_appears(monkeypatch):
    """Spin a tiny TCP server in a thread, verify _wait_for_port detects it."""
    import threading

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", 0))
    port = server.getsockname()[1]
    server.listen(1)

    def _accept_one():
        try:
            conn, _ = server.accept()
            conn.close()
        except OSError:
            pass

    t = threading.Thread(target=_accept_one, daemon=True)
    t.start()

    try:
        assert proxy._wait_for_port(port, timeout=2.0) is True
    finally:
        server.close()


def test_wait_for_port_returns_false_when_no_listener():
    # Pick a port nothing's listening on.
    free = proxy._pick_free_port()
    assert proxy._wait_for_port(free, timeout=0.5) is False


# ----- run_proxy: dispatcher -------------------------------------------------


def test_run_proxy_errors_when_ssh_binary_missing(monkeypatch):
    monkeypatch.setattr(proxy.shutil, "which", lambda _: None)
    rc = proxy.run_proxy(target="user@gomer", remote_port=8765, ssh_binary="ssh")
    assert rc == 2  # documented "ssh missing" exit code


def test_run_proxy_errors_when_tunnel_fails_to_open(monkeypatch):
    """If the tunnel never starts listening, we should bail out cleanly."""
    monkeypatch.setattr(proxy.shutil, "which", lambda _: "/usr/bin/ssh")

    class _SilentPopen:
        def __init__(self, *_, **__):
            self._terminated = False

        def poll(self):
            return None if not self._terminated else 0

        def terminate(self):
            self._terminated = True

        def wait(self, timeout=None):
            return 0

        def kill(self):
            self._terminated = True

    monkeypatch.setattr(proxy.subprocess, "Popen", _SilentPopen)
    monkeypatch.setattr(proxy, "_wait_for_port", lambda *_a, **_kw: False)

    rc = proxy.run_proxy(target="user@gomer", remote_port=8765, ssh_binary="ssh")
    assert rc == 3
