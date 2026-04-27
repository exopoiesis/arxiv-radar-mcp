"""Configuration loader for arxiv-radar.

Resolution order for the config file:
  1. explicit --config <path> CLI arg
  2. $ARXIV_RADAR_CONFIG env var
  3. ~/.config/arxiv-radar/radar.toml  (or platformdirs equivalent on Windows / macOS)
  4. ./radar.toml in CWD (handy during development)

If nothing is found, returns built-in defaults that point at exopoiesis/
daily-arxiv-ai4chem via raw GitHub URLs.
"""
from __future__ import annotations

import os
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from platformdirs import user_cache_dir, user_config_dir


def _default_config_path() -> Path:
    return Path(user_config_dir("arxiv-radar", appauthor="exopoiesis")) / "radar.toml"


def _default_cache_dir() -> Path:
    return Path(user_cache_dir("arxiv-radar", appauthor="exopoiesis"))


@dataclass
class SourceConfig:
    """One domain feed: where to read papers-*.json from."""
    name: str
    type: str  # "github" or "local"
    repo: str | None = None       # e.g. "exopoiesis/daily-arxiv-ai4chem"
    branch: str = "main"
    path: Path | None = None      # for type="local"


@dataclass
class EmbeddingsConfig:
    model: str = "sentence-transformers/all-MiniLM-L6-v2"
    cache_dir: Path = field(default_factory=lambda: _default_cache_dir() / "embeddings")
    batch_size: int = 64


@dataclass
class ServerConfig:
    default_k: int = 10
    hybrid_rrf_k: int = 60


@dataclass
class Config:
    sources: list[SourceConfig]
    embeddings: EmbeddingsConfig
    server: ServerConfig

    @classmethod
    def defaults(cls) -> "Config":
        return cls(
            sources=[
                SourceConfig(
                    name="ai4chem",
                    type="github",
                    repo="exopoiesis/daily-arxiv-ai4chem",
                ),
            ],
            embeddings=EmbeddingsConfig(),
            server=ServerConfig(),
        )


def load(config_path: Path | None = None) -> Config:
    """Load config from the first existing path in the resolution order, else defaults."""
    candidate = (
        config_path
        or _env_path()
        or _default_config_path() if _default_config_path().exists() else None
    )
    if candidate is None:
        cwd = Path.cwd() / "radar.toml"
        if cwd.exists():
            candidate = cwd

    if candidate is None or not candidate.exists():
        return Config.defaults()

    with open(candidate, "rb") as f:
        data = tomllib.load(f)
    return _from_dict(data)


def _env_path() -> Path | None:
    p = os.environ.get("ARXIV_RADAR_CONFIG")
    return Path(p) if p else None


def _from_dict(data: dict) -> Config:
    sources_raw = data.get("sources", {})
    sources: list[SourceConfig] = []
    for name, body in sources_raw.items():
        sources.append(SourceConfig(
            name=name,
            type=body.get("type", "github"),
            repo=body.get("repo"),
            branch=body.get("branch", "main"),
            path=Path(body["path"]).expanduser() if body.get("path") else None,
        ))

    emb_raw = data.get("embeddings", {})
    emb = EmbeddingsConfig(
        model=emb_raw.get("model", EmbeddingsConfig.model),
        cache_dir=Path(emb_raw["cache_dir"]).expanduser() if emb_raw.get("cache_dir")
                  else EmbeddingsConfig().cache_dir,
        batch_size=emb_raw.get("batch_size", 64),
    )

    srv_raw = data.get("server", {})
    srv = ServerConfig(
        default_k=srv_raw.get("default_k", 10),
        hybrid_rrf_k=srv_raw.get("hybrid_rrf_k", 60),
    )

    return Config(sources=sources or Config.defaults().sources,
                  embeddings=emb, server=srv)
