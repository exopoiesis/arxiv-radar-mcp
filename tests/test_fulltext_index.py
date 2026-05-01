"""Tests for fulltext_index.py — search primitives over the chunk index.

reindex() requires the encoder to actually run; we cover its plumbing
with a fake encoder, but the heavy "encode 12K-token chunks" path is
exercised in the gomer scenario scripts (tmp/scenario_*.sh).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from arxiv_radar_mcp.embeddings import EmbeddingIndex
from arxiv_radar_mcp.fulltext_index import (load_chunk_texts, reindex,
                                            search_paper_semantic,
                                            search_paper_text,
                                            similar_to_paper)


# ----- helpers ---------------------------------------------------------------


def _make_chunk_index(rows: list[tuple[str, str, str]]) -> EmbeddingIndex:
    """rows = [(arxiv_id, section, text)]. Embeddings are deterministic
    one-hot per row — every row gets a unique direction so cosine returns
    sensible per-row scores when querying with one of those directions."""
    n = len(rows)
    dim = max(n, 4)
    matrix = np.zeros((n, dim), dtype=np.float32)
    chunks = []
    row_for: dict[str, int] = {}

    for i, (pid, section, _text) in enumerate(rows):
        matrix[i, i] = 1.0  # one-hot
        chunks.append({
            "arxiv_id": pid, "section": section,
            "chunk_idx": i, "n_chars": 100, "n_tokens_est": 25,
        })
        if pid not in row_for:
            row_for[pid] = i

    return EmbeddingIndex(
        matrix=matrix,
        row_for=row_for,
        model_name="fake/test",
        dims=dim,
        metadata={"chunks": chunks, "max_seq_length": 12_000, "n_papers": len(set(r[0] for r in rows))},
    )


# ----- search_paper_text -----------------------------------------------------


def test_search_paper_text_finds_matching_chunks():
    rows = [
        ("p1", "Methods", "We compute lattice parameters using DFT methods."),
        ("p1", "Results", "Results show high accuracy."),
        ("p2", "Methods", "We use neural networks here."),
    ]
    chunk_texts = [r[2] for r in rows]
    chunk_meta = _make_chunk_index(rows).metadata["chunks"]

    out = search_paper_text(chunk_texts, chunk_meta, "lattice parameters")
    assert len(out) == 1
    assert out[0]["arxiv_id"] == "p1"
    assert out[0]["section"] == "Methods"
    assert "lattice" in out[0]["snippet"].lower()


def test_search_paper_text_empty_query():
    out = search_paper_text(["text"], [{"arxiv_id": "p1", "section": "x"}],
                            "")
    assert out == []


def test_search_paper_text_token_AND_semantics():
    chunk_texts = ["alpha only", "alpha and beta together", "beta only"]
    meta = [{"arxiv_id": f"p{i}", "section": "x", "chunk_idx": 0, "n_chars": 10}
            for i in range(3)]
    out = search_paper_text(chunk_texts, meta, "alpha beta")
    assert len(out) == 1  # only the chunk with both tokens
    assert out[0]["arxiv_id"] == "p1"


# ----- search_paper_semantic -------------------------------------------------


def test_search_paper_semantic_returns_top_k_by_cosine():
    rows = [
        ("p1", "Methods", "text 1"),
        ("p1", "Results", "text 2"),
        ("p2", "Discussion", "text 3"),
    ]
    index = _make_chunk_index(rows)
    chunk_texts = [r[2] for r in rows]

    # Query vector matching row 1 ("Results" of p1)
    qvec = np.zeros(index.dims, dtype=np.float32)
    qvec[1] = 1.0

    out = search_paper_semantic(index, chunk_texts, qvec, k=2)
    assert len(out) == 2
    assert out[0]["arxiv_id"] == "p1"
    assert out[0]["section"] == "Results"
    assert out[0]["score"] == pytest.approx(1.0)


def test_search_paper_semantic_includes_snippet_when_text_provided():
    rows = [("p1", "Methods", "the methods chunk text")]
    index = _make_chunk_index(rows)
    chunk_texts = [rows[0][2]]
    qvec = np.zeros(index.dims, dtype=np.float32)
    qvec[0] = 1.0

    out = search_paper_semantic(index, chunk_texts, qvec)
    assert "methods chunk" in out[0]["snippet"]


def test_search_paper_semantic_empty_index():
    """No chunks → empty result, not an error."""
    empty = EmbeddingIndex(
        matrix=np.zeros((0, 4), dtype=np.float32),
        row_for={}, model_name="x", dims=4,
        metadata={"chunks": [], "max_seq_length": 12_000, "n_papers": 0},
    )
    qvec = np.zeros(4, dtype=np.float32)
    qvec[0] = 1.0
    assert search_paper_semantic(empty, None, qvec) == []


# ----- similar_to_paper ------------------------------------------------------


def test_similar_to_paper_excludes_source():
    rows = [
        ("p1", "Methods", "t1"),
        ("p1", "Results", "t2"),
        ("p2", "Methods", "t3"),
        ("p3", "Methods", "t4"),
    ]
    index = _make_chunk_index(rows)
    out = similar_to_paper(index, "p1", k=10)
    ids = {r["arxiv_id"] for r in out}
    assert "p1" not in ids


def test_similar_to_paper_returns_one_row_per_paper():
    """Even if multiple chunks of the same paper rank high, one row."""
    rows = [
        ("p1", "Methods", "t"),
        ("p2", "Methods", "t"),
        ("p2", "Results", "t"),
        ("p2", "Discussion", "t"),
    ]
    index = _make_chunk_index(rows)
    out = similar_to_paper(index, "p1", k=10)
    pids = [r["arxiv_id"] for r in out]
    assert pids.count("p2") == 1


def test_similar_to_paper_unknown_id_returns_empty():
    rows = [("p1", "Methods", "t")]
    index = _make_chunk_index(rows)
    assert similar_to_paper(index, "nonexistent") == []


# ----- load_chunk_texts (re-derive on demand) --------------------------------


def test_load_chunk_texts_rebuilds_from_sources(tmp_path: Path):
    sources = tmp_path / "sources"
    sources.mkdir()
    (sources / "p1.md").write_text(
        "## Methods\nMethods body\n\n## Results\nResults body\n",
        encoding="utf-8",
    )

    chunks_meta = [
        {"arxiv_id": "p1", "section": "Methods", "chunk_idx": 0, "n_chars": 12},
        {"arxiv_id": "p1", "section": "Results", "chunk_idx": 0, "n_chars": 12},
    ]
    index = EmbeddingIndex(
        matrix=np.zeros((2, 4), dtype=np.float32),
        row_for={"p1": 0}, model_name="x", dims=4,
        metadata={"chunks": chunks_meta, "max_seq_length": 12_000, "n_papers": 1},
    )

    texts = load_chunk_texts(tmp_path, index)
    assert len(texts) == 2
    assert "Methods body" in texts[0]
    assert "Results body" in texts[1]


def test_load_chunk_texts_handles_missing_source(tmp_path: Path):
    """If a meta entry references a vanished source file, return empty
    string for those rows rather than crashing."""
    (tmp_path / "sources").mkdir()
    chunks_meta = [{"arxiv_id": "ghost", "section": "x", "chunk_idx": 0,
                    "n_chars": 0}]
    index = EmbeddingIndex(
        matrix=np.zeros((1, 4), dtype=np.float32),
        row_for={"ghost": 0}, model_name="x", dims=4,
        metadata={"chunks": chunks_meta, "max_seq_length": 12_000, "n_papers": 1},
    )
    texts = load_chunk_texts(tmp_path, index)
    assert texts == [""]


# ----- reindex (with fake encoder) -------------------------------------------


class _FakeEncoder:
    """Stand-in for arxiv_radar_mcp.embeddings.Encoder in unit tests."""

    def __init__(self, dim: int = 4):
        self.dim = dim
        self.config = type("C", (), {"embeddings": type("E", (), {
            "batch_size": 32, "model": "fake/encoder"})()})()
        self._model = type("M", (), {"max_seq_length": 512})()

    @property
    def model_name(self):
        return "fake/encoder"

    def _ensure_loaded(self):
        pass

    def encode_passages(self, texts, show_progress=True):
        # Each chunk → unique one-hot direction in self.dim space.
        n = len(texts)
        dim = max(n, self.dim)
        matrix = np.zeros((n, dim), dtype=np.float32)
        for i in range(n):
            matrix[i, i % dim] = 1.0
        return matrix


def test_reindex_writes_npy_and_index_json(tmp_path: Path):
    sources = tmp_path / "sources"
    sources.mkdir()
    (sources / "p1.md").write_text(
        "## Methods\nm body\n\n## Results\nr body\n", encoding="utf-8")
    (sources / "p1.meta.json").write_text(json.dumps({
        "arxiv_id": "p1", "source": "html", "fetch_time": "2026-05-01T00:00:00",
        "n_chars": 50, "n_chunks_after_split": 0,
    }), encoding="utf-8")

    index = reindex(tmp_path, _FakeEncoder())

    assert (tmp_path / "embeddings.npy").exists()
    assert (tmp_path / "index.json").exists()
    payload = json.loads((tmp_path / "index.json").read_text())
    assert payload["model"] == "fake/encoder"
    assert payload["n_papers"] == 1
    assert len(payload["chunks"]) == 2  # Methods + Results
    assert index.matrix.shape[0] == 2   # one row per chunk; dim is _FakeEncoder's choice


def test_reindex_backfills_meta_n_chunks(tmp_path: Path):
    sources = tmp_path / "sources"
    sources.mkdir()
    (sources / "p1.md").write_text(
        "## A\nx\n\n## B\ny\n\n## C\nz\n", encoding="utf-8")
    (sources / "p1.meta.json").write_text(json.dumps({
        "arxiv_id": "p1", "source": "html", "fetch_time": "2026-05-01",
        "n_chars": 30, "n_chunks_after_split": 0,
    }), encoding="utf-8")

    reindex(tmp_path, _FakeEncoder())

    meta = json.loads((sources / "p1.meta.json").read_text())
    assert meta["n_chunks_after_split"] == 3
    assert "indexed_at" in meta


def test_reindex_no_sources_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        reindex(tmp_path, _FakeEncoder())
