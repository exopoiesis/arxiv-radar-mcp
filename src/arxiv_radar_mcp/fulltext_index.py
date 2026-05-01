"""Fulltext index lifecycle: reindex (rebuild) + search primitives.

Reindex flow (always full, see [РЕШЕНИЕ-014]):
  1. Walk <cache_dir>/fulltext/sources/*.md
  2. For each paper: chunker.chunk_markdown(...) → list[Chunk]
  3. Concat chunk texts in document order; encode in batches with
     encoder.max_seq_length set to FULLTEXT_MAX_SEQ_LENGTH (12_288 — see
     chunker.chunk_markdown default).
  4. Persist matrix to fulltext/embeddings.npy and chunks meta to
     fulltext/index.json. row_for maps arxiv_id → first chunk row of
     that paper (used by similar_to_paper to seed the mean-of-chunks).

Search:
  * search_paper_text  — substring scan over chunk texts
  * search_paper_semantic — cosine over chunk embeddings, returns
                             {arxiv_id, section, snippet, score} payloads
  * similar_to_paper   — mean-of-chunks → cosine over chunk matrix

This module is fulltext-only; abstract semantics live in `search.py`.
"""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from arxiv_radar_mcp.chunker import Chunk, chunk_markdown
from arxiv_radar_mcp.embeddings import EmbeddingIndex, Encoder

LOG = logging.getLogger(__name__)


# Encoder seq window for fulltext chunks. Matches the chunker's default
# `max_tokens` — the chunker won't emit anything bigger than this, so we
# never need to encode at a longer window.
FULLTEXT_MAX_SEQ_LENGTH = 12_288

# Adaptive bucketing for reindex performance. Without bucketing, a 500-token
# References chunk would still pad to 12 288 (the encoder's max_seq_length),
# wasting ~96% of compute per row. Buckets group chunks by estimated token
# count so each pass uses a tighter seq window. The exact thresholds are
# heuristic — chosen so most "small" arxiv chunks (headings, captions,
# References, short paragraphs) land in the smallest bucket.
#
# Each entry: (max_tokens_inclusive, encode_seq_length, batch_size_on_12gb).
# Buckets are tried in order; chunks fall into the first whose threshold
# they fit within. Anything over the largest threshold uses the long bucket.
_REINDEX_BUCKETS = [
    # (token_threshold, encode_seq_length, batch_size)
    (   512,    512,  64),   # short: Headers, References, captions
    ( 2_048,  2_048,  16),   # medium: typical paragraphs
    (12_288, 12_288,   4),   # long: full sections
]


@dataclass
class _PaperChunks:
    arxiv_id: str
    chunks: list[Chunk]


def reindex(
    fulltext_dir: Path,
    encoder: Encoder,
    *,
    progress_cb=None,
) -> EmbeddingIndex:
    """Rebuild fulltext index from all cached source markdowns.

    `progress_cb(n_done, n_total)` is called once per paper for the jobs
    registry. Returns the freshly-built index.
    """
    sources_dir = fulltext_dir / "sources"
    if not sources_dir.exists():
        raise FileNotFoundError(f"no sources directory at {sources_dir}")

    paper_chunks = _collect_chunks(sources_dir)
    if not paper_chunks:
        raise FileNotFoundError(f"no enriched papers under {sources_dir}")

    LOG.info(f"reindex: {len(paper_chunks)} papers, "
             f"{sum(len(p.chunks) for p in paper_chunks)} chunks total")

    prior_max_seq = _get_max_seq_length(encoder)
    prior_batch = encoder.config.embeddings.batch_size

    try:
        all_chunks: list[Chunk] = []
        chunk_meta: list[dict] = []
        row_for: dict[str, int] = {}

        for pc in paper_chunks:
            row_for[pc.arxiv_id] = len(all_chunks)
            for c in pc.chunks:
                all_chunks.append(c)
                chunk_meta.append({
                    "arxiv_id": pc.arxiv_id,
                    "section": c.section,
                    "chunk_idx": c.chunk_idx,
                    "n_chars": c.n_chars,
                    "n_tokens_est": c.n_tokens_est,
                })

        t0 = time.time()
        matrix, bucket_stats = _encode_bucketed(encoder, all_chunks)
        encode_seconds = time.time() - t0

        for bucket_label, bucket_n, bucket_seconds in bucket_stats:
            if bucket_n:
                LOG.info(f"  bucket {bucket_label}: {bucket_n} chunks "
                         f"in {bucket_seconds:.1f}s "
                         f"({bucket_seconds / bucket_n:.2f}s/chunk)")

        if progress_cb is not None:
            progress_cb(len(paper_chunks), len(paper_chunks))

        # Persist (atomic).
        fulltext_dir.mkdir(parents=True, exist_ok=True)
        np.save(fulltext_dir / "embeddings.npy", matrix)
        index_payload = {
            "model": encoder.model_name,
            "dims": int(matrix.shape[1]),
            "n": int(matrix.shape[0]),
            "row_for": row_for,
            "max_seq_length": FULLTEXT_MAX_SEQ_LENGTH,
            "chunks": chunk_meta,
            "n_papers": len(paper_chunks),
            "encode_seconds": round(encode_seconds, 2),
        }
        tmp = fulltext_dir / "index.json.tmp"
        tmp.write_text(json.dumps(index_payload, indent=1), encoding="utf-8")
        tmp.replace(fulltext_dir / "index.json")

        # Backfill n_chunks_after_split into per-paper meta.json (so
        # paper_info reports it correctly without re-running chunker).
        for pc in paper_chunks:
            meta_path = sources_dir / f"{pc.arxiv_id}.meta.json"
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                meta["n_chunks_after_split"] = len(pc.chunks)
                meta["indexed_at"] = _utcnow_iso()
                tmp = meta_path.with_suffix(".json.tmp")
                tmp.write_text(json.dumps(meta, indent=1), encoding="utf-8")
                tmp.replace(meta_path)
            except (FileNotFoundError, json.JSONDecodeError):
                continue

        LOG.info(f"reindex done: {matrix.shape[0]} chunks → "
                 f"{matrix.nbytes / 1024 / 1024:.1f} MB in {encode_seconds:.1f}s")

        return EmbeddingIndex(
            matrix=matrix,
            row_for=row_for,
            model_name=encoder.model_name,
            dims=int(matrix.shape[1]),
            metadata={
                "max_seq_length": FULLTEXT_MAX_SEQ_LENGTH,
                "chunks": chunk_meta,
                "n_papers": len(paper_chunks),
            },
        )
    finally:
        _set_max_seq_length(encoder, prior_max_seq)
        encoder.config.embeddings.batch_size = prior_batch


def _collect_chunks(sources_dir: Path) -> list[_PaperChunks]:
    """Walk sources_dir, run chunker on each .md file."""
    out: list[_PaperChunks] = []
    for md_path in sorted(sources_dir.glob("*.md")):
        arxiv_id = md_path.stem
        try:
            text = md_path.read_text(encoding="utf-8")
        except OSError as e:
            LOG.warning(f"[reindex] skip {arxiv_id}: read error {e}")
            continue
        chunks = chunk_markdown(text, max_tokens=FULLTEXT_MAX_SEQ_LENGTH)
        if not chunks:
            LOG.warning(f"[reindex] skip {arxiv_id}: chunker produced nothing")
            continue
        out.append(_PaperChunks(arxiv_id=arxiv_id, chunks=chunks))
    return out


def _set_max_seq_length(encoder: Encoder, target: int) -> int:
    """Set the underlying SentenceTransformer's max_seq_length, return prior.

    Encoder lazy-loads — to keep things simple we ensure the model is loaded
    here so the attribute exists.
    """
    encoder._ensure_loaded()  # noqa: SLF001 — internal API
    prior = getattr(encoder._model, "max_seq_length", -1)
    encoder._model.max_seq_length = target
    return prior


def _get_max_seq_length(encoder: Encoder) -> int:
    """Read current max_seq_length without forcing a load if not yet loaded."""
    if encoder._model is None:  # noqa: SLF001
        return -1
    return getattr(encoder._model, "max_seq_length", -1)


def _encode_bucketed(
    encoder: Encoder, chunks: list[Chunk],
) -> tuple[np.ndarray, list[tuple[str, int, float]]]:
    """Encode chunks in length-sorted buckets so short chunks don't pay the
    cost of the longest seq window.

    Returns:
        matrix      — (N, dim) embeddings in original chunk order
        bucket_stats — [(label, n_chunks, seconds), ...] for telemetry

    The bucketing pass is correctness-neutral (same model, same prefix, same
    L2 norm — embeddings are unchanged) and reduces wasted padding-pass
    compute by 5-30× on typical arxiv-paper chunk-length distributions.
    """
    if not chunks:
        # Encode nothing to a (0, dim) array — pull dim from a one-token probe.
        probe = encoder.encode_passages(["x"], show_progress=False)
        return np.zeros((0, probe.shape[-1]), dtype=np.float32), []

    n = len(chunks)
    # bucket_idx[i] = index into _REINDEX_BUCKETS for chunk i
    bucket_assignment: list[int] = []
    for c in chunks:
        for b_idx, (threshold, _seq, _bs) in enumerate(_REINDEX_BUCKETS):
            if c.n_tokens_est <= threshold:
                bucket_assignment.append(b_idx)
                break
        else:
            # token count exceeds the largest bucket — use the largest.
            bucket_assignment.append(len(_REINDEX_BUCKETS) - 1)

    # Encode bucket-by-bucket; record output rows by their original index.
    rows: list[np.ndarray | None] = [None] * n
    stats: list[tuple[str, int, float]] = []

    for b_idx, (threshold, seq_len, batch_size) in enumerate(_REINDEX_BUCKETS):
        original_idx_in_bucket = [i for i, b in enumerate(bucket_assignment) if b == b_idx]
        label = f"≤{threshold}t"
        if not original_idx_in_bucket:
            stats.append((label, 0, 0.0))
            continue
        texts = [chunks[i].text for i in original_idx_in_bucket]

        t0 = time.time()
        bucket_matrix = encoder.encode_passages(
            texts,
            show_progress=False,
            max_seq_length=seq_len,
            batch_size=batch_size,
        )
        elapsed = time.time() - t0

        for j, orig_i in enumerate(original_idx_in_bucket):
            rows[orig_i] = bucket_matrix[j]

        stats.append((label, len(original_idx_in_bucket), elapsed))

    # Stack — by construction every slot is filled.
    assert all(r is not None for r in rows), "bucketing left a chunk un-encoded"
    matrix = np.stack(rows, axis=0).astype(np.float32, copy=False)
    return matrix, stats


def _utcnow_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Search primitives over the fulltext index
# ---------------------------------------------------------------------------


def _snippet(text: str, query: str | None = None, length: int = 240) -> str:
    """Pull a representative window from the chunk text. If query is given,
    center on the first match; else take the head."""
    if query:
        m = re.search(re.escape(query.split()[0]), text, re.IGNORECASE) if query.split() else None
        if m:
            start = max(0, m.start() - length // 3)
            end = min(len(text), start + length)
            prefix = "…" if start > 0 else ""
            suffix = "…" if end < len(text) else ""
            return prefix + text[start:end].strip() + suffix
    head = text[:length].strip()
    return head + ("…" if len(text) > length else "")


def search_paper_text(
    chunk_texts: list[str],
    chunk_meta: list[dict],
    query: str,
    k: int = 10,
) -> list[dict]:
    """Substring AND-scan over chunk texts. Title-boost not applicable here —
    chunks already carry their section as a separate field."""
    tokens = [t for t in re.split(r"\s+", query.lower().strip()) if t]
    if not tokens or not chunk_texts:
        return []

    scored: list[tuple[float, int]] = []
    for i, text in enumerate(chunk_texts):
        text_l = text.lower()
        if all(t in text_l for t in tokens):
            # Score by token-occurrence count (cheap proxy for relevance).
            score = float(sum(text_l.count(t) for t in tokens))
            scored.append((score, i))

    scored.sort(key=lambda x: x[0], reverse=True)
    out: list[dict] = []
    for score, idx in scored[:k]:
        meta = chunk_meta[idx]
        out.append({
            "arxiv_id": meta["arxiv_id"],
            "section": meta["section"],
            "chunk_idx": meta.get("chunk_idx", 0),
            "snippet": _snippet(chunk_texts[idx], query=query),
            "score": round(score, 4),
        })
    return out


def search_paper_semantic(
    index: EmbeddingIndex,
    chunk_texts: list[str] | None,
    query_vec: np.ndarray,
    k: int = 10,
) -> list[dict]:
    """Cosine over chunk embeddings; return per-chunk payloads.

    `chunk_texts` is optional — if provided we include a snippet, else
    only the meta fields. The MCP tool wires it from cached source files.
    """
    if index.metadata is None or "chunks" not in index.metadata:
        return []
    chunk_meta = index.metadata["chunks"]
    if not chunk_meta:
        return []

    sims = index.matrix @ query_vec
    top_n = min(k, len(sims))
    top = np.argpartition(-sims, top_n - 1)[:top_n]
    top_sorted = top[np.argsort(-sims[top])]

    out: list[dict] = []
    for idx in top_sorted:
        meta = chunk_meta[int(idx)]
        snippet = ""
        if chunk_texts is not None and int(idx) < len(chunk_texts):
            snippet = _snippet(chunk_texts[int(idx)])
        out.append({
            "arxiv_id": meta["arxiv_id"],
            "section": meta["section"],
            "chunk_idx": meta.get("chunk_idx", 0),
            "snippet": snippet,
            "score": round(float(sims[int(idx)]), 4),
        })
    return out


def similar_to_paper(
    index: EmbeddingIndex,
    arxiv_id: str,
    k: int = 10,
) -> list[dict]:
    """Mean-of-chunks → cosine over the chunk matrix, group results by paper.

    Returns one row per paper (best-scoring chunk wins for ranking),
    excluding the source paper itself. Useful for "show me similar papers
    based on full content, not just abstract".
    """
    rows = [i for i, _ in index.chunks_for(arxiv_id)]
    if not rows:
        return []
    mean_vec = index.matrix[rows].mean(axis=0)
    n = float(np.linalg.norm(mean_vec))
    if n == 0:
        return []
    mean_vec = (mean_vec / n).astype(np.float32)

    sims = index.matrix @ mean_vec
    chunk_meta = index.metadata.get("chunks", []) if index.metadata else []
    if not chunk_meta:
        return []

    # Group by arxiv_id, take best chunk per paper, exclude source.
    best_per_paper: dict[str, tuple[float, int]] = {}
    for i, s in enumerate(sims):
        pid = chunk_meta[i].get("arxiv_id")
        if not pid or pid == arxiv_id:
            continue
        prev = best_per_paper.get(pid)
        if prev is None or s > prev[0]:
            best_per_paper[pid] = (float(s), i)

    ranked = sorted(best_per_paper.items(), key=lambda x: x[1][0], reverse=True)
    out: list[dict] = []
    for pid, (score, row) in ranked[:k]:
        meta = chunk_meta[row]
        out.append({
            "arxiv_id": pid,
            "section": meta["section"],
            "chunk_idx": meta.get("chunk_idx", 0),
            "score": round(score, 4),
        })
    return out


def load_chunk_texts(fulltext_dir: Path, index: EmbeddingIndex) -> list[str]:
    """Re-derive chunk texts from cached source markdowns + chunker.

    The index doesn't carry chunk text bodies (would inflate index.json
    by 100×); we re-chunk on demand. Cheap because chunker is O(N) regex
    + a few string concat passes, milliseconds per paper.
    """
    if index.metadata is None or "chunks" not in index.metadata:
        return []
    chunks_meta = index.metadata["chunks"]

    # Group chunk-meta by arxiv_id to know how many we expect per paper.
    by_id: dict[str, list[dict]] = {}
    for c in chunks_meta:
        by_id.setdefault(c["arxiv_id"], []).append(c)

    # Re-chunk each source and emit texts in the order matching index rows.
    sources_dir = fulltext_dir / "sources"
    text_for_row: list[str | None] = [None] * len(chunks_meta)

    for arxiv_id, paper_meta in by_id.items():
        md_path = sources_dir / f"{arxiv_id}.md"
        try:
            text = md_path.read_text(encoding="utf-8")
        except OSError:
            continue
        rebuilt = chunk_markdown(text, max_tokens=FULLTEXT_MAX_SEQ_LENGTH)
        # Find the index range for this paper in chunk_meta order.
        # We rely on chunker being deterministic — same input produces same
        # ordered chunks in the same order it did during reindex.
        rows_for_paper = [i for i, m in enumerate(chunks_meta)
                          if m["arxiv_id"] == arxiv_id]
        for row_i, chunk in zip(rows_for_paper, rebuilt):
            text_for_row[row_i] = chunk.text

    return [t or "" for t in text_for_row]
