"""Configuration loader for arxiv-radar.

Resolution order for the config file:
  1. explicit --config <path> CLI arg
  2. $ARXIV_RADAR_CONFIG env var
  3. ~/.config/arxiv-radar/radar.toml  (or platformdirs equivalent on Windows / macOS)
  4. ./radar.toml in CWD (handy during development)

If nothing is found, returns built-in defaults that point at the
exopoiesis/arxiv-radar-* source repos via raw GitHub URLs.
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
    repo: str | None = None       # e.g. "exopoiesis/arxiv-radar-chemistry"
    branch: str = "main"
    path: Path | None = None      # for type="local"


@dataclass
class EmbeddingsConfig:
    """Bi-encoder used for the dense retrieval index.

    Default: mxbai-embed-large-v1 — best CPU-only English embedding in the
    sub-1B-param tier (MTEB ~64.7), 1024 dims, ~57 MB cache for 14k papers.
    See embeddings.py:_QUERY_PREFIX for the registry of models that need
    instruction-style prefixes.

    target_dim:
      For models trained with Matryoshka Representation Learning (Qwen3,
      Nomic, mxbai-v2 …) you can truncate the native embedding to a smaller
      dim at encode time and re-normalize. Quality drops 1–2 MTEB points per
      halving but cache size and cosine speed scale linearly. Set to None to
      keep the model's native dim.
    """
    model: str = "mixedbread-ai/mxbai-embed-large-v1"
    cache_dir: Path = field(default_factory=lambda: _default_cache_dir() / "embeddings")
    batch_size: int = 64
    target_dim: int | None = None


# RerankerConfig now lives in corpus_core.reranker — re-exported here
# so existing `from arxiv_radar_mcp.config import RerankerConfig` keeps
# working for downstream code.
from corpus_core.reranker import RerankerConfig  # noqa: E402, F401


@dataclass
class ServerConfig:
    default_k: int = 10
    hybrid_rrf_k: int = 60


@dataclass
class RefreshConfig:
    """Daily auto-refresh of the abstract corpus from the arxiv-radar-* feeds.

    `enabled=True` (default): backend kicks off a background asyncio task
    that runs `refresh_sources()` every `interval_hours`. For sources whose
    `path` is a git working tree, the refresher does `git -C path pull`
    first, picking up new shards (and dropping pruned ones) automatically.

    `full_rebuild=True` (recommended for GPU server): re-encode the entire
    abstract index after every refresh. Robust to deletions in upstream.
    Costs scale with corpus size and embedding model; fine to do nightly on GPU.

    `full_rebuild=False` (recommended for CPU laptop): incremental — encode
    only new arxiv_ids, append to embeddings.npy. Cheap (~10 sec/50 papers
    on CPU), but can drift from upstream when papers get archived/deleted
    upstream. User can flip to full periodically via `--build-cache`.
    """
    enabled: bool = True
    interval_hours: int = 24
    full_rebuild: bool = True


@dataclass
class Config:
    sources: list[SourceConfig]
    embeddings: EmbeddingsConfig
    reranker: RerankerConfig
    server: ServerConfig
    refresh: RefreshConfig

    @classmethod
    def defaults(cls) -> "Config":
        return cls(
            sources=[
                SourceConfig(
                    name="chemistry",
                    type="github",
                    repo="exopoiesis/arxiv-radar-chemistry",
                ),
                SourceConfig(
                    name="chemical_engineering",
                    type="github",
                    repo="exopoiesis/arxiv-radar-chem-eng",
                ),
                SourceConfig(
                    name="electrochemistry",
                    type="github",
                    repo="exopoiesis/arxiv-radar-electrochemistry",
                ),
                SourceConfig(
                    name="physics",
                    type="github",
                    repo="exopoiesis/arxiv-radar-physics",
                ),
                SourceConfig(
                    name="polymer",
                    type="github",
                    repo="exopoiesis/arxiv-radar-polymer",
                ),
                SourceConfig(
                    name="sulfide_materials",
                    type="github",
                    repo="exopoiesis/arxiv-radar-sulfide-materials",
                ),
            ],
            embeddings=EmbeddingsConfig(),
            reranker=RerankerConfig(),
            server=ServerConfig(),
            refresh=RefreshConfig(),
        )


def load(config_path: Path | None = None) -> Config:
    """Load config from the first existing path in the resolution order, else defaults.

    Resolution order:
      1. explicit `config_path` arg (e.g. CLI `--config`)
      2. $ARXIV_RADAR_CONFIG env var
      3. platformdirs default (e.g. ~/.config/arxiv-radar/radar.toml)
      4. ./radar.toml in cwd
      5. built-in defaults
    """
    candidate: Path | None = config_path or _env_path()
    if candidate is None:
        default = _default_config_path()
        if default.exists():
            candidate = default
    if candidate is None:
        cwd_cfg = Path.cwd() / "radar.toml"
        if cwd_cfg.exists():
            candidate = cwd_cfg

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
    emb_defaults = EmbeddingsConfig()
    emb = EmbeddingsConfig(
        model=emb_raw.get("model", emb_defaults.model),
        cache_dir=Path(emb_raw["cache_dir"]).expanduser() if emb_raw.get("cache_dir")
                  else emb_defaults.cache_dir,
        batch_size=emb_raw.get("batch_size", emb_defaults.batch_size),
        target_dim=emb_raw.get("target_dim", emb_defaults.target_dim),
    )

    rer_raw = data.get("reranker", {})
    rer_defaults = RerankerConfig()
    rer = RerankerConfig(
        enabled=rer_raw.get("enabled", rer_defaults.enabled),
        model=rer_raw.get("model", rer_defaults.model),
        top_k_candidates=rer_raw.get("top_k_candidates", rer_defaults.top_k_candidates),
    )

    srv_raw = data.get("server", {})
    srv = ServerConfig(
        default_k=srv_raw.get("default_k", 10),
        hybrid_rrf_k=srv_raw.get("hybrid_rrf_k", 60),
    )

    ref_raw = data.get("refresh", {})
    ref_defaults = RefreshConfig()
    ref = RefreshConfig(
        enabled=ref_raw.get("enabled", ref_defaults.enabled),
        interval_hours=ref_raw.get("interval_hours", ref_defaults.interval_hours),
        full_rebuild=ref_raw.get("full_rebuild", ref_defaults.full_rebuild),
    )

    return Config(sources=sources or Config.defaults().sources,
                  embeddings=emb, reranker=rer, server=srv, refresh=ref)
