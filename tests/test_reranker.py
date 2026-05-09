"""Reranker unit tests.

The Reranker class is kept on disk after the 2026-05-01 pivot
([РЕШЕНИЕ-010]) but no longer wired into RadarServer or any MCP tool.
We keep its tests around so the class stays viable if we ever need to
re-introduce hybrid search for a weaker bi-encoder.
"""
from __future__ import annotations

import pytest

from arxiv_radar_mcp.config import RerankerConfig
from arxiv_radar_mcp.corpus import Paper
from corpus_core.reranker import Reranker


def _make_paper(arxiv_id: str, title: str) -> Paper:
    return Paper(
        arxiv_id=arxiv_id, title=title, first_author="x", authors=["x"],
        abstract=f"abstract for {title}", primary_category="cs.LG",
        categories=[], published="2025-01-01", updated="2025-01-01",
        pdf_url="", topics=[], tags=[], domain="chemistry",
    )


class _FakeCrossEncoder:
    """Returns a fixed score per (query, passage) pair."""

    def __init__(self, scores_by_passage_substr: dict[str, float]) -> None:
        self.lookup = scores_by_passage_substr
        self.last_pairs: list[tuple[str, str]] | None = None

    def predict(self, pairs, show_progress_bar=False):
        self.last_pairs = [tuple(p) for p in pairs]
        out = []
        for _query, passage in pairs:
            for needle, score in self.lookup.items():
                if needle in passage:
                    out.append(score)
                    break
            else:
                out.append(0.0)
        return out


def test_reranker_reorders_by_score():
    rer = Reranker(RerankerConfig())
    fake = _FakeCrossEncoder({"alpha": 0.1, "beta": 0.9, "gamma": 0.5})
    rer._model = fake  # bypass lazy load

    candidates = [_make_paper("1", "alpha"), _make_paper("2", "beta"),
                  _make_paper("3", "gamma")]
    out = rer.rerank("query", candidates, k=3)

    assert [p.arxiv_id for p, _ in out] == ["2", "3", "1"]
    assert out[0][1] == pytest.approx(0.9)


def test_reranker_truncates_to_k():
    rer = Reranker(RerankerConfig())
    rer._model = _FakeCrossEncoder({f"t{i}": float(i) for i in range(5)})
    candidates = [_make_paper(str(i), f"t{i}") for i in range(5)]

    out = rer.rerank("q", candidates, k=2)
    assert len(out) == 2
    assert [p.arxiv_id for p, _ in out] == ["4", "3"]


def test_reranker_empty_candidates_returns_empty():
    rer = Reranker(RerankerConfig())
    out = rer.rerank("q", [], k=10)
    assert out == []
    assert rer._model is None


def test_reranker_passes_query_paired_with_search_text():
    rer = Reranker(RerankerConfig())
    fake = _FakeCrossEncoder({"alpha": 1.0})
    rer._model = fake
    candidates = [_make_paper("1", "alpha")]

    rer.rerank("my query", candidates, k=1)
    assert fake.last_pairs == [("my query", candidates[0].search_text)]
