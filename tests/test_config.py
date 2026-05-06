"""Config resolution tests.

Regression coverage for the precedence bug where an explicit `--config`
arg was silently dropped if the platformdirs default didn't exist (the
ternary `or ... if ... else None` parsed across the whole or-chain).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from arxiv_radar_mcp.config import Config, load


@pytest.fixture
def isolate_config_env(monkeypatch, tmp_path):
    """Make sure no real env var or platformdirs default leaks into the test."""
    monkeypatch.delenv("ARXIV_RADAR_CONFIG", raising=False)
    # Point the default-config probe at a tmp path that doesn't exist —
    # mimics a fresh machine.
    fake_default = tmp_path / "no_such_config.toml"
    monkeypatch.setattr("arxiv_radar_mcp.config._default_config_path",
                        lambda: fake_default)
    # Move cwd somewhere with no radar.toml.
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _write_minimal_toml(p: Path) -> None:
    # Use forward slashes — Windows backslashes are TOML escape sequences.
    safe_path = p.parent.as_posix()
    p.write_text(
        '[sources.local-test]\n'
        'type = "local"\n'
        f'path = "{safe_path}"\n'
        '\n'
        '[embeddings]\n'
        'model = "test-model-name"\n'
        '\n'
        '[reranker]\n'
        'enabled = false\n',
        encoding="utf-8",
    )


def test_explicit_config_path_is_honoured_when_default_missing(isolate_config_env):
    """Regression: explicit --config used to be dropped if platformdirs default
    didn't exist, due to ternary-over-or precedence."""
    cfg_file = isolate_config_env / "my-radar.toml"
    _write_minimal_toml(cfg_file)

    cfg = load(cfg_file)

    assert cfg.embeddings.model == "test-model-name"
    assert cfg.reranker.enabled is False
    assert cfg.sources[0].name == "local-test"
    assert cfg.sources[0].type == "local"


def test_env_var_used_when_no_explicit_arg(isolate_config_env, monkeypatch):
    cfg_file = isolate_config_env / "env-radar.toml"
    _write_minimal_toml(cfg_file)
    monkeypatch.setenv("ARXIV_RADAR_CONFIG", str(cfg_file))

    cfg = load(None)

    assert cfg.embeddings.model == "test-model-name"


def test_falls_back_to_defaults_when_nothing_resolves(isolate_config_env):
    """No explicit, no env, no default file, no cwd radar.toml → built-ins."""
    cfg = load(None)

    assert [(s.name, s.type, s.repo) for s in cfg.sources] == [
        ("chemistry", "github", "exopoiesis/arxiv-radar-chemistry"),
        ("chemical_engineering", "github", "exopoiesis/arxiv-radar-chem-eng"),
        ("physics", "github", "exopoiesis/arxiv-radar-physics"),
        ("polymer", "github", "exopoiesis/arxiv-radar-polymer"),
    ]
    # Default model is mxbai-embed-large-v1 (РЕШЕНИЕ-003).
    assert cfg.embeddings.model == "mixedbread-ai/mxbai-embed-large-v1"
    assert cfg.reranker.enabled is True
    assert cfg.reranker.model == "BAAI/bge-reranker-base"


def test_explicit_arg_wins_over_env_and_default(isolate_config_env, monkeypatch):
    explicit = isolate_config_env / "explicit.toml"
    explicit.write_text(
        '[embeddings]\nmodel = "explicit-wins"\n[reranker]\nenabled = false\n',
        encoding="utf-8",
    )
    env_file = isolate_config_env / "env.toml"
    env_file.write_text(
        '[embeddings]\nmodel = "env-loses"\n[reranker]\nenabled = false\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("ARXIV_RADAR_CONFIG", str(env_file))

    cfg = load(explicit)

    assert cfg.embeddings.model == "explicit-wins"
