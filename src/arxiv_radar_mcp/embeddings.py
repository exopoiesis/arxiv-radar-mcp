"""Sentence-transformers embedding cache.

On `--build-cache`:
  1. Load corpus
  2. For each paper: encode `paper.search_text` (title + abstract)
  3. Persist:
       <cache_dir>/embeddings.npy        — float32, shape (N, D)
       <cache_dir>/index.json            — {arxiv_id: row_idx, ...}, plus model + dims
  4. Subsequent runs: mmap-load embeddings + index for fast cosine similarity.

Skeleton — wire up sentence-transformers + the actual encode loop in next session.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from arxiv_radar_mcp.config import Config
from arxiv_radar_mcp.corpus import Paper, load_all

LOG = logging.getLogger(__name__)


@dataclass
class EmbeddingIndex:
    """In-memory view over the cached embeddings.

    Use .vector(arxiv_id) to fetch a single embedding,
    .matrix is the full (N, D) array for batch cosine similarity,
    .row_for[arxiv_id] is the row index.
    """
    matrix: np.ndarray            # (N, D) float32, L2-normalized
    row_for: dict[str, int]
    model_name: str
    dims: int

    @classmethod
    def load(cls, cache_dir: Path) -> "EmbeddingIndex":
        idx = json.loads((cache_dir / "index.json").read_text(encoding="utf-8"))
        matrix = np.load(cache_dir / "embeddings.npy", mmap_mode="r")
        return cls(
            matrix=matrix,
            row_for=idx["row_for"],
            model_name=idx["model"],
            dims=idx["dims"],
        )

    def vector(self, arxiv_id: str) -> np.ndarray | None:
        i = self.row_for.get(arxiv_id)
        return None if i is None else self.matrix[i]


def build_cache(config_path: Path | None = None) -> None:
    """One-shot: load corpus → encode → persist. Call from CLI or first use."""
    from arxiv_radar_mcp.config import load
    config = load(config_path)

    LOG.info("loading corpus...")
    papers = load_all(config)

    LOG.info(f"encoding {len(papers)} papers with {config.embeddings.model}")
    matrix, ids = _encode(list(papers.values()), config)

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


def _encode(papers: list[Paper], config: Config) -> tuple[np.ndarray, list[str]]:
    """Encode title+abstract for each paper. L2-normalized output."""
    # Lazy import — sentence-transformers pulls in torch which is heavy.
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(config.embeddings.model)
    texts = [p.search_text for p in papers]
    ids = [p.arxiv_id for p in papers]

    matrix = model.encode(
        texts,
        batch_size=config.embeddings.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32, copy=False)
    return matrix, ids


def encode_query(text: str, config: Config) -> np.ndarray:
    """Encode a single query (search-time). Same model + L2-normalize."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(config.embeddings.model)
    return model.encode([text], normalize_embeddings=True).astype(np.float32)[0]
