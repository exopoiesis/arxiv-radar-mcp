"""Tests for fulltext_index.py — search primitives over the chunk index.

reindex() requires the encoder to actually run; we cover its plumbing
with a fake encoder, but the heavy "encode 12K-token chunks" path is
exercised in the gomer scenario scripts (tmp/scenario_*.sh).
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pytest

from corpus_core.embeddings import EmbeddingIndex
from corpus_core.corpus_index import (load_chunk_texts, reindex,
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


def test_search_paper_semantic_snippet_chars_extends_default():
    """U11: snippet_chars=600 returns ~600 chars instead of the default ~240,
    so researchers can extract recipes/numbers without 3-5 narrow queries."""
    long_text = "X " * 500   # 1000 chars
    rows = [("p1", "Methods", long_text)]
    index = _make_chunk_index(rows)
    qvec = np.zeros(index.dims, dtype=np.float32)
    qvec[0] = 1.0

    short = search_paper_semantic(index, [long_text], qvec, snippet_chars=240)
    long  = search_paper_semantic(index, [long_text], qvec, snippet_chars=600)
    # Default is shorter; explicit 600 returns more (within 10 chars margin
    # for ellipsis + whitespace stripping).
    assert len(long[0]["snippet"]) > len(short[0]["snippet"])
    assert len(long[0]["snippet"]) >= 590


def test_search_paper_text_snippet_chars_extends_default():
    chunk_texts = ["alpha beta " + "X " * 500]
    chunk_meta = [{"arxiv_id": "p1", "section": "Methods",
                   "chunk_idx": 0, "n_chars": 1010}]
    short = search_paper_text(chunk_texts, chunk_meta, "alpha", snippet_chars=240)
    long  = search_paper_text(chunk_texts, chunk_meta, "alpha", snippet_chars=600)
    assert len(long[0]["snippet"]) > len(short[0]["snippet"])


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
    """Stand-in for corpus_core.embeddings.Encoder in unit tests."""

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

    def encode_passages(self, texts, show_progress=True,
                        max_seq_length=512, batch_size=None):
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


# ----- adaptive bucketing ----------------------------------------------------


def test_reindex_buckets_chunks_by_length(tmp_path: Path):
    """Reindex log/index should reflect that short chunks land in the
    short bucket and long chunks in the long one — the size threshold
    being _REINDEX_BUCKETS[0][0] (default 512 tokens)."""
    sources = tmp_path / "sources"
    sources.mkdir()
    short_body = "## A\nshort\n"
    medium_body = "## B\n" + "para. " * 300 + "\n"          # ~300 tokens
    long_body  = "## C\n" + "para. " * 3000 + "\n"          # ~3000 tokens
    (sources / "p1.md").write_text(short_body + medium_body + long_body,
                                   encoding="utf-8")
    (sources / "p1.meta.json").write_text(json.dumps({
        "arxiv_id": "p1", "source": "html", "fetch_time": "2026-05-01",
        "n_chars": 1000, "n_chunks_after_split": 0,
    }), encoding="utf-8")

    # _FakeEncoder ignores max_seq_length but works for shape correctness.
    enc = _FakeEncoder()
    index = reindex(tmp_path, enc)

    chunks_meta = (index.metadata or {}).get("chunks", [])
    assert len(chunks_meta) == 3
    sections = [m["section"] for m in chunks_meta]
    assert sections == ["A", "B", "C"]
    # Token estimates increase strictly: short < medium < long.
    estimates = [m["n_tokens_est"] for m in chunks_meta]
    assert estimates == sorted(estimates)


def test_encode_bucketed_preserves_original_order(tmp_path: Path):
    """When chunks are split into buckets and encoded separately, the
    final matrix must put each row back at its original index — otherwise
    chunk_meta and embeddings.npy would be misaligned."""
    from corpus_core.chunker import Chunk
    from corpus_core.corpus_index import _encode_bucketed

    # Mix of short (≤512), medium (≤2048), and long (>2048) chunks.
    chunks = [
        Chunk(section="long",   chunk_idx=0, text="L" * 1, n_chars=1, n_tokens_est=3000),
        Chunk(section="short",  chunk_idx=0, text="s" * 1, n_chars=1, n_tokens_est=100),
        Chunk(section="medium", chunk_idx=0, text="m" * 1, n_chars=1, n_tokens_est=1500),
        Chunk(section="short2", chunk_idx=0, text="s" * 1, n_chars=1, n_tokens_est=200),
    ]

    enc = _FakeEncoder()
    matrix, stats = _encode_bucketed(enc, chunks)

    assert matrix.shape[0] == len(chunks)
    # Stats contain one row per bucket; counts must add up to total.
    total_in_buckets = sum(n for _label, n, _t in stats)
    assert total_in_buckets == len(chunks)
    # Short bucket should contain 2, medium 1, long 1.
    counts = {label: n for label, n, _t in stats}
    assert counts.get("≤512t") == 2
    assert counts.get("≤2048t") == 1
    assert counts.get("≤4096t") == 1


def test_encode_bucketed_empty_chunks():
    from corpus_core.corpus_index import _encode_bucketed
    enc = _FakeEncoder()
    matrix, stats = _encode_bucketed(enc, [])
    assert matrix.shape == (0, enc.dim) or matrix.shape[0] == 0
    assert stats == []


# ----- junk section filter --------------------------------------------------


@pytest.mark.parametrize("section,expected", [
    # Junk
    ("References", True),
    ("REFERENCES", True),
    ("Reference", True),
    ("Acknowledgments", True),
    ("Acknowledgements", True),
    ("V Acknowledgements", True),
    ("Bibliography", True),
    ("Data Availability", True),
    ("Data availability", True),
    ("Author Contributions", True),
    ("Funding", True),
    ("Competing Interests", True),
    ("Conflict of Interest", True),
    ("Supplementary Material", True),
    ("Supplementary Information", True),
    ("Appendix A", True),
    ("Appendix B", True),
    ("S10 References", True),
    # NOT junk
    ("Header", False),
    ("Abstract", False),
    ("Introduction", False),
    ("1 Introduction", False),
    ("Methods", False),
    ("Methodology", False),
    ("Results", False),
    ("Discussion", False),
    ("Conclusions", False),
    ("Body", False),
    ("3 Dataset", False),
    ("4.2 Force field dependency", False),
    ("S10 Incorporating Non-Adiabatic Effects into the Photochemistry of Liquid Water", False),
    ("", False),
])
def test_is_junk_section_classifies_correctly(section, expected):
    from corpus_core.corpus_index import is_junk_section
    assert is_junk_section(section) == expected, f"{section!r} → expected {expected}"


def test_search_paper_semantic_filters_junk():
    """Junk sections should be skipped in favour of clean ones from the
    oversample pool, even when junk has higher cosine score."""
    from corpus_core.corpus_index import search_paper_semantic

    # 6 rows: row 0 (References, top-score), rows 1-5 (clean sections)
    n = 6
    matrix = np.zeros((n, 8), dtype=np.float32)
    for i in range(n):
        matrix[i, 0] = 1.0 - i * 0.01      # row 0 highest, then descending
    chunks = [
        {"arxiv_id": "p1", "section": "References", "chunk_idx": 0, "n_chars": 100},
        {"arxiv_id": "p2", "section": "Methods", "chunk_idx": 0, "n_chars": 100},
        {"arxiv_id": "p3", "section": "Results", "chunk_idx": 0, "n_chars": 100},
        {"arxiv_id": "p4", "section": "Introduction", "chunk_idx": 0, "n_chars": 100},
        {"arxiv_id": "p5", "section": "Discussion", "chunk_idx": 0, "n_chars": 100},
        {"arxiv_id": "p6", "section": "Header", "chunk_idx": 0, "n_chars": 100},
    ]
    index = EmbeddingIndex(
        matrix=matrix,
        row_for={c["arxiv_id"]: i for i, c in enumerate(chunks)},
        model_name="fake/test",
        dims=8,
        metadata={"chunks": chunks, "max_seq_length": 12_000, "n_papers": n},
    )

    qvec = np.zeros(8, dtype=np.float32)
    qvec[0] = 1.0

    out = search_paper_semantic(index, None, qvec, k=3)
    sections = [r["section"] for r in out]
    # First three should NOT include References — it loses to the next 3.
    assert "References" not in sections
    assert sections == ["Methods", "Results", "Introduction"]


def test_search_paper_semantic_falls_back_to_junk_when_short():
    """If the corpus is mostly junk, we still return what we have rather
    than an empty list."""
    from corpus_core.corpus_index import search_paper_semantic

    chunks = [
        {"arxiv_id": "p1", "section": "References", "chunk_idx": 0, "n_chars": 100},
        {"arxiv_id": "p2", "section": "Acknowledgments", "chunk_idx": 0, "n_chars": 100},
        {"arxiv_id": "p3", "section": "Methods", "chunk_idx": 0, "n_chars": 100},
    ]
    matrix = np.zeros((3, 4), dtype=np.float32)
    matrix[0, 0] = 1.0
    matrix[1, 0] = 0.99
    matrix[2, 0] = 0.98
    index = EmbeddingIndex(
        matrix=matrix,
        row_for={c["arxiv_id"]: i for i, c in enumerate(chunks)},
        model_name="fake/test",
        dims=4,
        metadata={"chunks": chunks, "max_seq_length": 12_000, "n_papers": 3},
    )
    qvec = np.zeros(4, dtype=np.float32)
    qvec[0] = 1.0

    out = search_paper_semantic(index, None, qvec, k=5)
    # 3 results total: one clean (Methods) first, then junk.
    assert len(out) == 3
    assert out[0]["section"] == "Methods"
    assert out[1]["section"] in ("References", "Acknowledgments")


def test_search_paper_text_filters_junk():
    """Same junk filter applied to text search."""
    from corpus_core.corpus_index import search_paper_text

    chunk_texts = [
        "alpha beta gamma — references list with alpha beta",     # References (high score)
        "alpha beta — methods section using alpha beta",            # Methods
        "alpha beta — results contain alpha beta findings",         # Results
    ]
    chunk_meta = [
        {"arxiv_id": "p1", "section": "References", "chunk_idx": 0, "n_chars": 100},
        {"arxiv_id": "p2", "section": "Methods",    "chunk_idx": 0, "n_chars": 100},
        {"arxiv_id": "p3", "section": "Results",    "chunk_idx": 0, "n_chars": 100},
    ]
    out = search_paper_text(chunk_texts, chunk_meta, "alpha beta", k=2)
    sections = [r["section"] for r in out]
    assert "References" not in sections
    assert sections == ["Methods", "Results"]


# ----- incremental reindex --------------------------------------------------


class _SpyFakeEncoder:
    """Fake encoder with deterministic dim-stable output and call recording.

    Unlike _FakeEncoder, the dim is fixed so that two reindex calls always
    produce same-shape output (required for incremental concat). The text-
    derived row position is also stable, so encoding the same chunk twice
    yields identical bytes (separately verified in the
    'preserves unchanged rows byte-identical' test).
    """

    def __init__(self, dim: int = 8, model_name: str = "fake/encoder"):
        self.dim = dim
        self._model_name = model_name
        self.config = type("C", (), {"embeddings": type("E", (), {
            "batch_size": 32, "model": model_name})()})()
        self._model = type("M", (), {"max_seq_length": 512})()
        self.encoded_calls: list[list[str]] = []

    @property
    def model_name(self) -> str:
        return self._model_name

    def _ensure_loaded(self) -> None:
        pass

    def encode_passages(self, texts, show_progress=True,
                        max_seq_length=512, batch_size=None):
        self.encoded_calls.append(list(texts))
        n = len(texts)
        matrix = np.zeros((n, self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            slot = sum(ord(c) for c in t[:50]) % self.dim
            matrix[i, slot] = 1.0
        return matrix

    @property
    def total_encoded(self) -> int:
        return sum(len(c) for c in self.encoded_calls)

    @property
    def encoded_texts(self) -> list[str]:
        return [t for call in self.encoded_calls for t in call]


def _write_paper(sources_dir: Path, arxiv_id: str, body: str | None = None) -> None:
    """Create `<id>.md` + `<id>.meta.json` with reasonable defaults."""
    if body is None:
        body = (
            f"## Methods\n{arxiv_id} methods body\n\n"
            f"## Results\n{arxiv_id} results body\n"
        )
    (sources_dir / f"{arxiv_id}.md").write_text(body, encoding="utf-8")
    (sources_dir / f"{arxiv_id}.meta.json").write_text(json.dumps({
        "arxiv_id": arxiv_id, "source": "html",
        "fetch_time": "2026-05-01T00:00:00",
        "n_chars": len(body), "n_chunks_after_split": 0,
    }), encoding="utf-8")


def test_reindex_incremental_noop_when_no_changes(tmp_path: Path):
    sources = tmp_path / "sources"
    sources.mkdir()
    _write_paper(sources, "p1")

    enc = _SpyFakeEncoder(dim=8)
    reindex(tmp_path, enc)               # first build → full (no existing index)
    assert enc.total_encoded > 0
    enc.encoded_calls.clear()

    matrix_before = np.load(tmp_path / "embeddings.npy")
    reindex(tmp_path, enc)               # second pass → noop

    assert enc.total_encoded == 0, "noop reindex must not invoke the encoder"
    matrix_after = np.load(tmp_path / "embeddings.npy")
    np.testing.assert_array_equal(matrix_before, matrix_after)


def test_reindex_incremental_appends_new_paper(tmp_path: Path):
    sources = tmp_path / "sources"
    sources.mkdir()
    _write_paper(sources, "p1")

    enc = _SpyFakeEncoder(dim=8)
    reindex(tmp_path, enc)
    n_before = np.load(tmp_path / "embeddings.npy").shape[0]
    enc.encoded_calls.clear()

    _write_paper(sources, "p2")
    reindex(tmp_path, enc)

    matrix_after = np.load(tmp_path / "embeddings.npy")
    payload = json.loads((tmp_path / "index.json").read_text())

    # Only p2 chunks were encoded, not p1's.
    assert all("p2" in t for t in enc.encoded_texts), enc.encoded_texts
    assert all("p1" not in t for t in enc.encoded_texts)

    # Matrix grew by exactly p2's chunk count.
    p1_chunks = sum(1 for c in payload["chunks"] if c["arxiv_id"] == "p1")
    p2_chunks = sum(1 for c in payload["chunks"] if c["arxiv_id"] == "p2")
    assert matrix_after.shape[0] == p1_chunks + p2_chunks
    assert matrix_after.shape[0] > n_before
    # row_for[p1] preserved at row 0; p2 follows.
    assert payload["row_for"]["p1"] == 0
    assert payload["row_for"]["p2"] == p1_chunks
    assert payload["n_papers"] == 2


def test_reindex_incremental_drops_deleted_paper(tmp_path: Path):
    sources = tmp_path / "sources"
    sources.mkdir()
    _write_paper(sources, "p1")
    _write_paper(sources, "p2")

    enc = _SpyFakeEncoder(dim=8)
    reindex(tmp_path, enc)

    # Delete p2's source — meta is left behind, classification keys off .md
    # only, which is the correct contract.
    (sources / "p2.md").unlink()
    enc.encoded_calls.clear()

    reindex(tmp_path, enc)

    # No re-encode happened (only deletions).
    assert enc.total_encoded == 0

    payload = json.loads((tmp_path / "index.json").read_text())
    pids = {c["arxiv_id"] for c in payload["chunks"]}
    assert pids == {"p1"}
    assert "p2" not in payload["row_for"]
    assert payload["n_papers"] == 1


def test_reindex_incremental_replaces_changed_paper(tmp_path: Path):
    sources = tmp_path / "sources"
    sources.mkdir()
    _write_paper(sources, "p1")
    _write_paper(sources, "p2")

    enc = _SpyFakeEncoder(dim=8)
    reindex(tmp_path, enc)
    enc.encoded_calls.clear()

    # Push p1.md mtime well past its meta.indexed_at + tolerance window.
    p1_md = sources / "p1.md"
    future = time.time() + 60
    os.utime(p1_md, (future, future))

    reindex(tmp_path, enc)

    # Only p1 was re-encoded; p2 left alone.
    encoded = enc.encoded_texts
    assert encoded, "changed paper must be re-encoded"
    assert all("p1" in t for t in encoded)
    assert not any("p2" in t for t in encoded)

    payload = json.loads((tmp_path / "index.json").read_text())
    # Both still indexed; chunk count unchanged because content is unchanged.
    pids = {c["arxiv_id"] for c in payload["chunks"]}
    assert pids == {"p1", "p2"}
    assert payload["n_papers"] == 2


def test_reindex_falls_back_to_full_on_model_mismatch(tmp_path: Path):
    sources = tmp_path / "sources"
    sources.mkdir()
    _write_paper(sources, "p1")
    _write_paper(sources, "p2")

    enc = _SpyFakeEncoder(dim=8, model_name="fake/encoder")
    reindex(tmp_path, enc)

    # Stamp the on-disk index with a different model name. Next reindex
    # must detect the mismatch and re-encode every paper.
    payload = json.loads((tmp_path / "index.json").read_text())
    payload["model"] = "fake/other-model"
    (tmp_path / "index.json").write_text(json.dumps(payload, indent=1),
                                         encoding="utf-8")
    enc.encoded_calls.clear()

    reindex(tmp_path, enc)

    # Full rebuild → every paper re-encoded.
    encoded = enc.encoded_texts
    assert any("p1" in t for t in encoded)
    assert any("p2" in t for t in encoded)

    payload2 = json.loads((tmp_path / "index.json").read_text())
    assert payload2["model"] == "fake/encoder"


def test_reindex_force_full_ignores_existing_index(tmp_path: Path):
    sources = tmp_path / "sources"
    sources.mkdir()
    _write_paper(sources, "p1")
    _write_paper(sources, "p2")

    enc = _SpyFakeEncoder(dim=8)
    reindex(tmp_path, enc)
    enc.encoded_calls.clear()

    # No source changes — incremental would no-op, but force_full re-encodes.
    reindex(tmp_path, enc, incremental=False)

    encoded = enc.encoded_texts
    assert any("p1" in t for t in encoded)
    assert any("p2" in t for t in encoded)


def test_reindex_incremental_preserves_unchanged_rows_byte_identical(tmp_path: Path):
    sources = tmp_path / "sources"
    sources.mkdir()
    _write_paper(sources, "p1")

    enc = _SpyFakeEncoder(dim=8)
    reindex(tmp_path, enc)

    matrix_v1 = np.load(tmp_path / "embeddings.npy")
    payload_v1 = json.loads((tmp_path / "index.json").read_text())
    p1_n_chunks = sum(1 for c in payload_v1["chunks"] if c["arxiv_id"] == "p1")
    p1_rows_v1 = matrix_v1[:p1_n_chunks].tobytes()

    _write_paper(sources, "p2")
    enc.encoded_calls.clear()
    reindex(tmp_path, enc)

    matrix_v2 = np.load(tmp_path / "embeddings.npy")
    p1_rows_v2 = matrix_v2[:p1_n_chunks].tobytes()

    assert p1_rows_v1 == p1_rows_v2, (
        "p1's rows must be byte-identical after incremental reindex — "
        "they came from disk, not a fresh encode"
    )
