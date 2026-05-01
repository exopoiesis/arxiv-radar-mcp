"""Cross-encoder reranker.

Cross-encoders score (query, passage) pairs jointly — they're slower than
bi-encoders (no caching, must re-run per pair) but the precision boost
over dense retrieval is large (≈ +5–15 nDCG@10 in published benchmarks).

We use them in `search_hybrid` as a final pass: pull top-K from RRF,
rerank, return top-k. Lazy-loaded so unit tests and the --build-cache
path don't pay the model-download cost.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from arxiv_radar_mcp.config import RerankerConfig

if TYPE_CHECKING:
    from arxiv_radar_mcp.corpus import Paper

LOG = logging.getLogger(__name__)


class Reranker:
    """Lazy CrossEncoder wrapper. One instance per server lifetime."""

    def __init__(self, config: RerankerConfig) -> None:
        self.config = config
        self._model = None  # type: ignore[var-annotated]

    @property
    def model_name(self) -> str:
        return self.config.model

    def _ensure_loaded(self) -> None:
        if self._model is None:
            from sentence_transformers import CrossEncoder
            LOG.info(f"loading cross-encoder {self.model_name}...")
            self._model = CrossEncoder(self.model_name)

    def rerank(
        self, query: str, candidates: list["Paper"], k: int,
    ) -> list[tuple["Paper", float]]:
        """Re-score (query, paper.search_text) pairs and return top-k by score."""
        if not candidates:
            return []
        self._ensure_loaded()
        pairs = [(query, p.search_text) for p in candidates]
        scores = self._model.predict(pairs, show_progress_bar=False)  # type: ignore[union-attr]
        ranked = sorted(
            zip(candidates, scores),
            key=lambda x: float(x[1]),
            reverse=True,
        )
        return [(p, float(s)) for p, s in ranked[:k]]
