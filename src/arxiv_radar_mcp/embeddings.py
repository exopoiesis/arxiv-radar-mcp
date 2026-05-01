"""Sentence-transformers embedding cache.

On `--build-cache`:
  1. Load corpus
  2. For each paper: encode `paper.search_text` (title + abstract), prefixed
     with the model's passage-prefix when the model expects one (E5 family).
  3. Persist:
       <cache_dir>/embeddings.npy        — float32, shape (N, D), L2-normalized
       <cache_dir>/index.json            — {arxiv_id: row_idx, ...}, plus model + dims
  4. Subsequent runs: mmap-load embeddings + index for fast cosine similarity.

At query-time, the SentenceTransformer instance lives on the long-running
server (Encoder) — loaded once, reused for every query. The legacy
`encode_query(text, config)` shim instantiates fresh per-call and is kept
only for ad-hoc / one-shot scripts.
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


# ---------------------------------------------------------------------------
# Instruction-prefix registry
# ---------------------------------------------------------------------------
# Several embedding families were trained with explicit query / passage
# prefixes. Forgetting them silently costs 5–15% recall. The registry maps
# canonical model names to their prescribed prefixes; unknown models get
# empty strings (no-op).

_QWEN3_QUERY_PREFIX = (
    "Instruct: Given a web search query, retrieve relevant passages "
    "that answer the query\nQuery: "
)

_QUERY_PREFIX = {
    # mxbai (Mixedbread) — query-only prefix, passages plain.
    "mixedbread-ai/mxbai-embed-large-v1":
        "Represent this sentence for searching relevant passages: ",
    "mixedbread-ai/mxbai-embed-large-v2":
        "Represent this sentence for searching relevant passages: ",

    # BGE family (BAAI) — query-only prefix.
    "BAAI/bge-large-en-v1.5":
        "Represent this sentence for searching relevant passages: ",
    "BAAI/bge-base-en-v1.5":
        "Represent this sentence for searching relevant passages: ",
    "BAAI/bge-small-en-v1.5":
        "Represent this sentence for searching relevant passages: ",

    # E5 family (Microsoft) — symmetric prefixes on both sides.
    "intfloat/e5-large-v2": "query: ",
    "intfloat/e5-base-v2":  "query: ",
    "intfloat/e5-small-v2": "query: ",
    "intfloat/multilingual-e5-large": "query: ",
    "intfloat/multilingual-e5-base":  "query: ",

    # Qwen3-Embedding — instruction-style query template, no doc prefix.
    "Qwen/Qwen3-Embedding-0.6B": _QWEN3_QUERY_PREFIX,
    "Qwen/Qwen3-Embedding-4B":   _QWEN3_QUERY_PREFIX,
    "Qwen/Qwen3-Embedding-8B":   _QWEN3_QUERY_PREFIX,

    # Microsoft Harrier-OSS — same Instruct/Query template (verified on
    # the 27B model card). Last-token pooling + L2 norm handled by
    # sentence-transformers. No prefix on the document side.
    "microsoft/harrier-oss-v1-0.6b": _QWEN3_QUERY_PREFIX,
    "microsoft/harrier-oss-v1-27b":  _QWEN3_QUERY_PREFIX,
}

_PASSAGE_PREFIX = {
    # E5 expects "passage: " on the document side too.
    "intfloat/e5-large-v2": "passage: ",
    "intfloat/e5-base-v2":  "passage: ",
    "intfloat/e5-small-v2": "passage: ",
    "intfloat/multilingual-e5-large": "passage: ",
    "intfloat/multilingual-e5-base":  "passage: ",
}


def query_prefix(model: str) -> str:
    """Return the query-side prefix for a model, or '' if none is registered."""
    return _QUERY_PREFIX.get(model, "")


def passage_prefix(model: str) -> str:
    """Return the passage-side prefix for a model, or '' if none is registered."""
    return _PASSAGE_PREFIX.get(model, "")


# ---------------------------------------------------------------------------
# Index (read-only, mmap)
# ---------------------------------------------------------------------------

@dataclass
class EmbeddingIndex:
    """In-memory view over a cached embedding matrix.

    Two flavours coexist:
      * abstract index: one row per paper, `row_for[arxiv_id] = row`,
        `metadata` empty.
      * fulltext index: rows are chunks (potentially many per paper).
        `row_for` maps arxiv_id → first row of that paper; `metadata`
        carries the per-row {arxiv_id, section, chunk_idx, n_chars}.

    Use .vector(arxiv_id) to fetch a single embedding (works for abstract
    index; for fulltext returns the first chunk's vector — query through
    `chunks_for(arxiv_id)` if you need them all).
    """
    matrix: np.ndarray                       # (N, D) float32, L2-normalized
    row_for: dict[str, int]
    model_name: str
    dims: int
    metadata: dict | None = None             # {chunks: [...], max_seq_length: int, ...}

    @classmethod
    def load(cls, cache_dir: Path) -> "EmbeddingIndex":
        idx = json.loads((cache_dir / "index.json").read_text(encoding="utf-8"))
        matrix = np.load(cache_dir / "embeddings.npy", mmap_mode="r")
        # Strip core fields; everything else is metadata (forward-compat).
        core = {"row_for", "model", "dims", "n"}
        metadata = {k: v for k, v in idx.items() if k not in core}
        return cls(
            matrix=matrix,
            row_for=idx["row_for"],
            model_name=idx["model"],
            dims=idx["dims"],
            metadata=metadata or None,
        )

    def vector(self, arxiv_id: str) -> np.ndarray | None:
        i = self.row_for.get(arxiv_id)
        return None if i is None else self.matrix[i]

    def chunks_for(self, arxiv_id: str) -> list[tuple[int, dict]]:
        """For a fulltext index: return [(row_idx, chunk_meta), ...] for one paper.

        Empty list for non-fulltext indexes or unknown arxiv_id.
        """
        if not self.metadata or "chunks" not in self.metadata:
            return []
        return [(i, c) for i, c in enumerate(self.metadata["chunks"])
                if c.get("arxiv_id") == arxiv_id]


# ---------------------------------------------------------------------------
# Encoder (lazy-load wrapper)
# ---------------------------------------------------------------------------

class Encoder:
    """Wraps SentenceTransformer with model-aware prefixes + lazy load.

    Designed to live on the long-running RadarServer: the heavy
    SentenceTransformer instance is loaded on first `encode_*` call and
    kept for the lifetime of the server, so subsequent queries pay no
    model-load cost.
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self._model = None  # type: ignore[var-annotated]

    @property
    def model_name(self) -> str:
        return self.config.embeddings.model

    def _ensure_loaded(self) -> None:
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            LOG.info(f"loading bi-encoder {self.model_name}...")
            self._model = SentenceTransformer(self.model_name)

    def encode_query(self, text: str) -> np.ndarray:
        """Encode a single query. L2-normalized, with model-specific prefix."""
        self._ensure_loaded()
        prefixed = query_prefix(self.model_name) + text
        vec = self._model.encode(  # type: ignore[union-attr]
            [prefixed],
            normalize_embeddings=True,
        ).astype(np.float32)[0]
        return _maybe_truncate(vec, self.config.embeddings.target_dim)

    def encode_passages(
        self, texts: list[str], show_progress: bool = True,
    ) -> np.ndarray:
        """Encode a batch of passages. L2-normalized, with model prefix if any."""
        self._ensure_loaded()
        prefix = passage_prefix(self.model_name)
        prepared = [prefix + t for t in texts] if prefix else texts
        matrix = self._model.encode(  # type: ignore[union-attr]
            prepared,
            batch_size=self.config.embeddings.batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32, copy=False)
        return _maybe_truncate(matrix, self.config.embeddings.target_dim)


def _maybe_truncate(arr: np.ndarray, target_dim: int | None) -> np.ndarray:
    """Matryoshka-style dim reduction: slice to target_dim, then re-L2-normalize.

    No-op when target_dim is None or already ≥ the model's native dim. Works on
    a single vector (1-D) or a batch (2-D, last axis = features).
    """
    if target_dim is None:
        return arr
    native = arr.shape[-1]
    if target_dim >= native:
        return arr
    sliced = arr[..., :target_dim]
    norms = np.linalg.norm(sliced, axis=-1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return (sliced / norms).astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# Cache build
# ---------------------------------------------------------------------------

def build_cache(config_path: Path | None = None) -> None:
    """One-shot: load corpus → encode → persist. Call from CLI or first use."""
    from arxiv_radar_mcp.config import load
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


# ---------------------------------------------------------------------------
# Legacy ad-hoc helpers (kept for one-shot scripts; prefer Encoder for serving)
# ---------------------------------------------------------------------------

def encode_query(text: str, config: Config) -> np.ndarray:
    """Encode a single query. SLOW — instantiates the model fresh each call.

    Long-running callers (the MCP server) should use Encoder.encode_query
    instead, which keeps the model loaded.
    """
    return Encoder(config).encode_query(text)
