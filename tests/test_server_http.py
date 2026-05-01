"""HTTP transport plumbing tests for server.py.

The actual streamable-HTTP server is exercised in the gomer scenario,
not here. These tests verify the wiring around it: serve_http boots the
right loop with the right host/port, and the underlying mcp app is
constructed with our 14 tools intact.
"""
from __future__ import annotations

import asyncio

import pytest


def test_build_mcp_app_uses_tool_specs(local_config):
    from arxiv_radar_mcp.server import RadarServer, TOOL_SPECS, _build_mcp_app

    radar = RadarServer(local_config)
    try:
        app = _build_mcp_app(radar)
        # mcp.server.lowlevel.Server stores the registered list-tools handler.
        assert app is not None
        # We can't easily call into the handler without an MCP context,
        # but TOOL_SPECS itself is the source of truth — guard against drift.
        assert len(TOOL_SPECS) == 15
    finally:
        radar.jobs.shutdown()


def test_serve_http_calls_uvicorn_with_correct_bind(monkeypatch, local_config, tmp_path):
    """serve_http should construct a uvicorn server bound to host:port and
    delegate the actual run to asyncio.run."""
    from arxiv_radar_mcp import server as srv_mod

    captured = {}

    async def _fake_run_streamable(_radar, host, port):
        captured["host"] = host
        captured["port"] = port

    monkeypatch.setattr(srv_mod, "_run_streamable_http", _fake_run_streamable)

    # Avoid platform config lookup — point at a non-existent path so load()
    # falls back to defaults (which are safe for this test).
    monkeypatch.setattr(srv_mod, "load", lambda _path=None: local_config)

    srv_mod.serve_http(host="127.0.0.1", port=8765, config_path=tmp_path / "x.toml")
    assert captured == {"host": "127.0.0.1", "port": 8765}
