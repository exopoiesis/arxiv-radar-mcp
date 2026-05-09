"""CLI helper: rebuild the fulltext embedding index from cached sources.

Synchronous, prints summary. The MCP `reindex` tool wraps this in a
job for the conversational case; for scenario testing on gomer this
direct CLI is faster (no jobs round-trip).

Usage:
    python -m arxiv_radar_mcp.reindex_cli
    python -m arxiv_radar_mcp.reindex_cli --config /data/radar.toml
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from arxiv_radar_mcp.config import load
from corpus_core.embeddings import Encoder
from corpus_core.corpus_index import reindex


def main() -> int:
    parser = argparse.ArgumentParser(prog="arxiv-radar-reindex")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level), stream=sys.stderr,
                        format="[%(asctime)s %(levelname)s %(name)s] %(message)s",
                        datefmt="%H:%M:%S")

    config = load(args.config)
    fulltext_dir = config.embeddings.cache_dir.parent / "fulltext"
    encoder = Encoder(config)

    t0 = time.time()
    index = reindex(fulltext_dir, encoder)
    elapsed = time.time() - t0

    meta = index.metadata or {}
    summary = {
        "n_papers": meta.get("n_papers", 0),
        "n_chunks": index.matrix.shape[0],
        "dims": index.dims,
        "model": index.model_name,
        "max_seq_length": meta.get("max_seq_length"),
        "elapsed_seconds": round(elapsed, 1),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
