"""CLI-level argparse tests for arxiv_radar_mcp.__main__.

Covers transport selection, mutual exclusion, default mode dispatch.
We don't actually start any server here — we monkeypatch the entry
points and check what got called with what args.
"""
from __future__ import annotations

import sys

import pytest

from arxiv_radar_mcp import __main__ as cli


def _run(monkeypatch, argv, **patches):
    """Helper: set sys.argv, optionally monkeypatch entry-point fns,
    return cli.main()'s return code + a record of what got called."""
    monkeypatch.setattr(sys, "argv", ["arxiv-radar-mcp", *argv])
    record: dict = {}

    def _make_recorder(name):
        def _rec(*args, **kwargs):
            record[name] = {"args": args, "kwargs": kwargs}
            return 0 if name == "run_proxy" else None
        return _rec

    # We patch the *importable* function names in their original modules so
    # the deferred imports inside main() pick up the stub.
    monkeypatch.setattr("arxiv_radar_mcp.build_cache.build_cache",
                        _make_recorder("build_cache"))
    monkeypatch.setattr("arxiv_radar_mcp.server.serve",
                        _make_recorder("serve"))
    monkeypatch.setattr("arxiv_radar_mcp.server.serve_http",
                        _make_recorder("serve_http"))
    monkeypatch.setattr("corpus_core.proxy.run_proxy",
                        _make_recorder("run_proxy"))

    rc = cli.main()
    return rc, record


def test_default_mode_runs_stdio_server(monkeypatch):
    rc, rec = _run(monkeypatch, [])
    assert rc == 0
    assert "serve" in rec
    assert "serve_http" not in rec
    assert "run_proxy" not in rec


def test_transport_http_runs_http_server(monkeypatch):
    rc, rec = _run(monkeypatch, ["--transport", "http"])
    assert rc == 0
    assert "serve_http" in rec
    assert rec["serve_http"]["kwargs"]["host"] == "127.0.0.1"
    assert rec["serve_http"]["kwargs"]["port"] == 8765


def test_transport_http_custom_bind_and_port(monkeypatch):
    rc, rec = _run(monkeypatch, [
        "--transport", "http", "--bind", "0.0.0.0", "--port", "9000",
    ])
    assert rc == 0
    assert rec["serve_http"]["kwargs"]["host"] == "0.0.0.0"
    assert rec["serve_http"]["kwargs"]["port"] == 9000


def test_remote_mode_runs_proxy(monkeypatch):
    rc, rec = _run(monkeypatch, ["--remote", "user@gomer"])
    assert rc == 0
    assert "run_proxy" in rec
    assert rec["run_proxy"]["kwargs"]["target"] == "user@gomer"
    assert rec["run_proxy"]["kwargs"]["remote_port"] == 8765


def test_remote_with_custom_port(monkeypatch):
    rc, rec = _run(monkeypatch, [
        "--remote", "user@gomer", "--remote-port", "9999",
    ])
    assert rc == 0
    assert rec["run_proxy"]["kwargs"]["remote_port"] == 9999


def test_remote_and_transport_http_are_mutually_exclusive(monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "arxiv-radar-mcp", "--remote", "user@gomer", "--transport", "http",
    ])
    with pytest.raises(SystemExit):
        cli.main()


def test_build_cache_short_circuits_other_modes(monkeypatch):
    rc, rec = _run(monkeypatch, ["--build-cache", "--transport", "http"])
    assert rc == 0
    assert "build_cache" in rec
    # http server must NOT have been started
    assert "serve_http" not in rec
    assert "serve" not in rec
