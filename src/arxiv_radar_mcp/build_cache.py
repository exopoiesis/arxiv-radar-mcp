"""arxiv-radar-mcp's `--build-cache` orchestrator.

Glue layer between the arxiv-radar-* fork loader and corpus_core's
Encoder/EmbeddingIndex. Lives here (not in corpus_core) because it
depends on this project's specific corpus schema and config.

Used by the CLI `arxiv-radar-mcp --build-cache` and by ad-hoc scripts.
The long-running server uses Encoder directly via RadarServer; the
shim `encode_query(text, config)` here is kept for one-shot scripts
only.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from arxiv_radar_mcp.config import Config, load
from arxiv_radar_mcp.corpus import Paper, load_all
from corpus_core.embeddings import Encoder

LOG = logging.getLogger(__name__)


def build_cache(config_path: Path | None = None) -> None:
    """One-shot: load corpus → encode → persist. Call from CLI or first use."""
    config = load(config_path)

    LOG.info("loading corpus...")
    papers = load_all(config)

    LOG.info(f"encoding {len(papers)} papers with {config.embeddings.model}")
    encoder = Encoder(config)
    paper_list = list(papers.values())
    matrix = encoder.encode_passages([p.search_text for p in paper_list])
    ids = [p.arxiv_id for p in paper_list]

    config.embeddings.cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(config.embeddings.cache_dir / "embeddings.npy", matrix)
    (config.embeddings.cache_dir / "index.json").write_text(
        json.dumps({
            "model": config.embeddings.model,
            "dims": int(matrix.shape[1]),
            "n": int(matrix.shape[0]),
            "row_for": {pid: i for i, pid in enumerate(ids)},
        }, indent=1),
        encoding="utf-8",
    )
    LOG.info(f"wrote {matrix.shape[0]} embeddings ({matrix.nbytes/1024/1024:.1f} MB) "
             f"to {config.embeddings.cache_dir}")


def encode_query(text: str, config: Config) -> np.ndarray:
    """Encode a single query. SLOW — instantiates the model fresh each call.

    Long-running callers (the MCP server) should use Encoder.encode_query
    instead, which keeps the model loaded.
    """
    return Encoder(config).encode_query(text)
