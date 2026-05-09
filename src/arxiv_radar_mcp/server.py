"""MCP stdio server. Wires Tools → search functions → corpus + index.

The RadarServer class holds the loaded corpus + embedding index and exposes
every tool as a sync method (testable without MCP). serve() boots the SDK,
registers the tool catalogue, and pumps the stdio transport.

Tool surface evolves in two halves:
  * abstracts (this file, fully here today): search_abstract_*, similar_to_abstract,
    paper_info, list_tags, list_domains
  * fulltext + admin (added in Phase 9): search_paper_*, similar_to_paper,
    fetch_papers, reindex, job_status, job_list, list_enriched

`search_hybrid` and `recent` were removed in 2026-05-01; see
`docs/PLAN.md` [РЕШЕНИЕ-010] and [РЕШЕНИЕ-014]. The Reranker class stays
on disk but is no longer instantiated by RadarServer.
"""
from __future__ import annotations

import asyncio
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

import httpx

from arxiv_radar_mcp.config import Config, load
from arxiv_radar_mcp.corpus import Paper, load_all
from corpus_core.embeddings import EmbeddingIndex, Encoder
from arxiv_radar_mcp.fulltext import fetch_and_save, probe_html_available
from corpus_core.corpus_index import (FULLTEXT_MAX_SEQ_LENGTH,
                                            load_chunk_texts, reindex,
                                            search_paper_semantic,
                                            search_paper_text,
                                            similar_to_paper)
from corpus_core.jobs import JobError, JobHandle, JobRegistry
from corpus_core.mcp_scaffold import (
    BackgroundTaskFactory,
    build_mcp_app,
    make_method_dispatcher,
    serve_stdio,
    serve_streamable_http,
)
from arxiv_radar_mcp.refresh import refresh_sources
from corpus_core.search import (search_semantic, search_text, similar_to)

LOG = logging.getLogger(__name__)


class RadarServer:
    """Holds the loaded corpus + embedding indexes. Tools are methods on it.

    Two indexes coexist: `abstract_index` (always present once cache built)
    and `fulltext_index` (None until the user enriches papers and reindexes).
    """

    def __init__(self, config: Config, *, encoder: Encoder | None = None) -> None:
        self.config = config
        self.papers: dict[str, Paper] = load_all(config)
        # Encoder injection point: the combined-server supervisor (in
        # lab-corpus-mcp/lab_corpus_mcp/combined.py) hands BOTH servers
        # the same Encoder instance so a single Qwen3-4B copy fits in
        # 12 GB VRAM. Plain single-server callers pass nothing and we
        # construct one ourselves (lazy weight load on first encode).
        self.encoder = encoder if encoder is not None else Encoder(config)

        self.abstract_index: EmbeddingIndex | None = None
        try:
            self.abstract_index = EmbeddingIndex.load(config.embeddings.cache_dir)
            LOG.info(f"loaded abstract index: {self.abstract_index.matrix.shape} "
                     f"({self.abstract_index.model_name})")
            if self.abstract_index.model_name != config.embeddings.model:
                LOG.warning(
                    f"index built with {self.abstract_index.model_name!r} but config "
                    f"requests {config.embeddings.model!r} — rerun "
                    f"`arxiv-radar-mcp --build-cache` to align"
                )
        except FileNotFoundError:
            LOG.warning("no abstract embedding cache yet — semantic / similar_to "
                        "tools will be unavailable until `arxiv-radar-mcp --build-cache`")

        # Fulltext index slot — populated by reindex. Stays None until the
        # user has fetched at least one paper and run reindex.
        self.fulltext_dir = config.embeddings.cache_dir.parent / "fulltext"
        self.fulltext_index: EmbeddingIndex | None = None
        try:
            self.fulltext_index = EmbeddingIndex.load(self.fulltext_dir)
            chunks_n = (self.fulltext_index.metadata or {}).get("chunks", [])
            LOG.info(f"loaded fulltext index: {self.fulltext_index.matrix.shape} "
                     f"({len(chunks_n)} chunks across "
                     f"{(self.fulltext_index.metadata or {}).get('n_papers', '?')} papers)")
        except FileNotFoundError:
            pass

        self.jobs = JobRegistry(cache_dir=config.embeddings.cache_dir.parent)

    # ----- abstracts -------------------------------------------------------

    def search_abstract_text(self, query: str, k: int = 10,
                             domain: str | None = None,
                             tag: str | None = None) -> list[dict]:
        return [_paper_payload(p, score) for p, score in
                search_text(self.papers.values(), query, k=k,
                            domain=domain, tag=tag)]

    def search_abstract_semantic(self, query: str, k: int = 10,
                                 domain: str | None = None,
                                 tag: str | None = None) -> list[dict]:
        if self.abstract_index is None:
            return [{"error": "abstract embedding cache not built. "
                              "run `arxiv-radar-mcp --build-cache`"}]
        qvec = self.encoder.encode_query(query)
        return [_paper_payload(p, score) for p, score in
                search_semantic(self.papers, qvec, self.abstract_index, k=k,
                                domain=domain, tag=tag)]

    def similar_to_abstract(self, arxiv_id: str, k: int = 10) -> list[dict]:
        if self.abstract_index is None:
            return [{"error": "abstract embedding cache not built"}]
        return [_paper_payload(p, score) for p, score in
                similar_to(self.papers, arxiv_id, self.abstract_index, k=k)]

    def paper_info(self, arxiv_id: str,
                   full_abstract: bool = False) -> dict | None:
        """Metadata + fulltext-status for one paper. None if not in corpus.

        `full_abstract=true` returns the untruncated abstract (default
        truncates to 600 chars + ellipsis to keep search-result payloads
        compact). Use when the LLM needs the complete abstract for a
        relevance decision.
        """
        p = self.papers.get(arxiv_id)
        if p is None:
            return None
        payload = _paper_payload(p, 0.0, full_abstract=full_abstract)
        payload["fulltext"] = self._fulltext_status(arxiv_id)
        return payload

    def list_tags(self, domain: str | None = None,
                  head_limit: int | None = None,
                  min_count: int = 1,
                  prefix: str | None = None) -> list[dict]:
        c: Counter[str] = Counter()
        for p in self.papers.values():
            if domain and domain not in p.domain.split(","):
                continue
            for t in p.tags:
                c[t] += 1
        # Default `most_common()` returns by frequency desc; keep that.
        rows = [(t, n) for t, n in c.most_common() if n >= min_count]
        if prefix:
            pl = prefix.lower()
            rows = [(t, n) for t, n in rows if t.lower().startswith(pl)]
        if head_limit is not None:
            rows = rows[:max(0, int(head_limit))]
        return [{"tag": t, "count": n} for t, n in rows]

    def list_domains(self) -> list[dict]:
        c: Counter[str] = Counter()
        for p in self.papers.values():
            for d in p.domain.split(","):
                if d:
                    c[d] += 1
        return [{"domain": d, "papers": n} for d, n in c.most_common()]

    # ----- fulltext-status helper (used by paper_info) ---------------------

    def _fulltext_status(self, arxiv_id: str) -> dict:
        """Inspect cache_dir/fulltext/sources/<id>.* and fulltext_index for status.

        Phase 9 will fill out indexed/n_chunks. Today returns the bare
        enriched-bit so paper_info already has the right payload shape.
        """
        cache_dir = self.config.embeddings.cache_dir.parent / "fulltext"
        source = cache_dir / "sources" / f"{arxiv_id}.md"
        if not source.exists():
            return {"enriched": False, "indexed": False, "source": None, "n_chunks": 0}

        meta_path = source.with_suffix(".meta.json")
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            meta = {}

        if self.fulltext_index is None:
            indexed = False
        else:
            chunks = getattr(self.fulltext_index, "metadata", {}).get("chunks", [])
            indexed = any(c.get("arxiv_id") == arxiv_id for c in chunks)

        return {
            "enriched": True,
            "indexed": indexed,
            "source": meta.get("source"),
            "n_chunks": meta.get("n_chunks_after_split", 0),
        }

    # ----- fulltext --------------------------------------------------------

    def search_paper_text(self, query: str, k: int = 10,
                          snippet_chars: int = 240) -> list[dict]:
        if self.fulltext_index is None:
            return [{"error": "fulltext index empty — run fetch_papers and "
                              "reindex first"}]
        chunk_texts = load_chunk_texts(self.fulltext_dir, self.fulltext_index)
        chunk_meta = (self.fulltext_index.metadata or {}).get("chunks", [])
        return search_paper_text(chunk_texts, chunk_meta, query, k=k,
                                 snippet_chars=snippet_chars)

    def search_paper_semantic(self, query: str, k: int = 10,
                              snippet_chars: int = 240) -> list[dict]:
        if self.fulltext_index is None:
            return [{"error": "fulltext index empty — run fetch_papers and "
                              "reindex first"}]
        qvec = self.encoder.encode_query(query)
        chunk_texts = load_chunk_texts(self.fulltext_dir, self.fulltext_index)
        return search_paper_semantic(self.fulltext_index, chunk_texts, qvec,
                                     k=k, snippet_chars=snippet_chars)

    def similar_to_paper(self, arxiv_id: str, k: int = 10) -> list[dict]:
        if self.fulltext_index is None:
            return [{"error": "fulltext index empty — run fetch_papers and "
                              "reindex first"}]
        return similar_to_paper(self.fulltext_index, arxiv_id, k=k)

    def list_enriched(self) -> list[str]:
        sources_dir = self.fulltext_dir / "sources"
        if not sources_dir.exists():
            return []
        return sorted(p.stem for p in sources_dir.glob("*.md"))

    def validate_arxiv_ids(self, arxiv_ids: list[str]) -> dict:
        """Pre-flight check: which ids will fetch_papers actually be able to
        enrich?

        For each id we (a) skip the probe if the paper is already cached
        (counts as ok), (b) HEAD arxiv.org/html/<id> on the throttled
        rate-limiter (1 req / 3 sec; same budget as the live fetcher).
        2xx/3xx → `ok`, 4xx/5xx → `pdf_only`. Caller can warn the user
        before submitting a long batch with known-bad ids.

        Synchronous, blocking — for typical 30-50 id batches this takes
        ≤2-3 minutes including the throttle. Skip if probing 200+ ids;
        for those just use fetch_papers and read its `failed` list.
        """
        if not arxiv_ids:
            return {"error": "arxiv_ids must be a non-empty list"}

        sources_dir = self.fulltext_dir / "sources"
        ok: list[str] = []
        pdf_only: list[str] = []
        cached: list[str] = []

        with httpx.Client(timeout=httpx.Timeout(30.0, connect=10.0),
                          headers={"User-Agent": "arxiv-radar-mcp/0.1"},
                          follow_redirects=True) as client:
            for pid in arxiv_ids:
                if (sources_dir / f"{pid}.md").exists():
                    cached.append(pid)
                    ok.append(pid)
                    continue
                if probe_html_available(pid, client=client):
                    ok.append(pid)
                else:
                    pdf_only.append(pid)

        return {
            "n_total": len(arxiv_ids),
            "n_ok": len(ok),
            "n_pdf_only": len(pdf_only),
            "n_cached": len(cached),
            "ok": ok,
            "pdf_only": pdf_only,
        }

    # ----- async admin -----------------------------------------------------

    def fetch_papers(self, arxiv_ids: list[str], force: bool = False) -> dict:
        """Submit a background fetch job. Returns {job_id} immediately.

        `force=True` re-downloads even when the source markdown is already
        cached — needed when an earlier fetch landed a stub (~1 KB body
        with only abstract+bibliography, U9 fix). Without it the cache hit
        keeps serving the bad copy.
        """
        if not arxiv_ids:
            return {"error": "arxiv_ids must be a non-empty list"}
        force_flag = bool(force)
        job_id = self.jobs.submit(
            kind="fetch_papers",
            fn=lambda h: self._do_fetch(h, arxiv_ids, force=force_flag),
            args={"arxiv_ids": arxiv_ids, "force": force_flag},
            n_total=len(arxiv_ids),
        )
        return {"job_id": job_id, "n_total": len(arxiv_ids),
                "kind": "fetch_papers", "force": force_flag}

    def refresh_abstracts(self, force_full: bool = False) -> dict:
        """Submit a background refresh job. Returns {job_id} immediately.

        Pulls the arxiv-radar-* feeds (git pull where applicable),
        diffs against the in-memory corpus, encodes new abstracts, and
        atomically swaps the abstract index. By default uses the strategy
        from radar.toml ([refresh] full_rebuild = ...). Pass force_full=true
        to override and re-encode everything.

        Refuses if a reindex is already running (encoder lock held).
        """
        if not self.jobs.acquire_reindex_lock():
            return {"error": "encoder busy (reindex/refresh in progress); "
                             "try again when it completes"}

        full = bool(force_full or self.config.refresh.full_rebuild)
        job_id = self.jobs.submit(
            kind="refresh_abstracts",
            fn=lambda h: self._do_refresh(h, full_rebuild=full),
            args={"full_rebuild": full},
            n_total=0,
        )
        return {"job_id": job_id, "kind": "refresh_abstracts",
                "strategy_planned": "full" if full else "incremental"}

    def reindex(self, force_full: bool = False) -> dict:
        """Submit a background reindex job. Returns {job_id} immediately.

        Incremental by default — only papers added or changed since the
        last index get re-encoded. Pass `force_full=true` to re-encode
        every cached paper (recovery after a corrupted swap, or after a
        manual cache surgery). Falls back to full automatically when the
        existing index was built with a different model.

        Refuses if a reindex is already running (lockfile held).
        """
        sources_dir = self.fulltext_dir / "sources"
        n_papers = sum(1 for _ in sources_dir.glob("*.md")) if sources_dir.exists() else 0
        if n_papers == 0:
            return {"error": "no enriched papers to index — run fetch_papers first"}

        if not self.jobs.acquire_reindex_lock():
            return {"error": "reindex already in progress (lockfile held)"}

        force = bool(force_full)
        job_id = self.jobs.submit(
            kind="reindex",
            fn=lambda h: self._do_reindex(h, force_full=force),
            args={"n_papers": n_papers, "force_full": force},
            n_total=n_papers,
        )
        return {"job_id": job_id, "n_total": n_papers, "kind": "reindex",
                "strategy_planned": "full" if force else "incremental"}

    def job_status(self, job_id: str) -> dict:
        info = self.jobs.get(job_id)
        if info is None:
            return {"error": f"unknown job_id: {job_id!r}"}
        return info

    def job_list(self, limit: int = 50) -> list[dict]:
        return self.jobs.list_recent(limit=limit)

    # ----- job workers (called by JobRegistry on a worker thread) ----------

    def _do_fetch(self, handle: JobHandle, arxiv_ids: list[str],
                  *, force: bool = False) -> dict:
        ok: list[dict] = []
        failed: list[dict] = []
        source_counts: Counter[str] = Counter()

        with httpx.Client(timeout=httpx.Timeout(30.0, connect=10.0),
                          headers={"User-Agent": "arxiv-radar-mcp/0.1"},
                          follow_redirects=True) as client:
            for i, pid in enumerate(arxiv_ids):
                try:
                    result = fetch_and_save(pid, self.fulltext_dir,
                                            force=force, client=client)
                except Exception as e:  # noqa: BLE001
                    failed.append({"arxiv_id": pid, "error": str(e)})
                    handle.update(n_done=i + 1)
                    continue

                if result.markdown is not None:
                    ok.append({
                        "arxiv_id": pid,
                        "source": result.source,
                        "n_chars": result.n_chars,
                    })
                    source_counts[result.source or "unknown"] += 1
                else:
                    failed.append({"arxiv_id": pid, "error": result.error or "unknown"})
                handle.update(n_done=i + 1)

        return {
            "n_total": len(arxiv_ids),
            "n_ok": len(ok),
            "n_failed": len(failed),
            "ok": ok,
            "failed": failed,
            "source_breakdown": dict(source_counts),
        }

    def _do_refresh(self, handle: JobHandle, *, full_rebuild: bool) -> dict:
        try:
            result = refresh_sources(self, full_rebuild=full_rebuild)
        except Exception as e:  # noqa: BLE001
            self.jobs.release_reindex_lock()
            raise JobError(f"refresh failed: {type(e).__name__}: {e}") from e
        else:
            self.jobs.release_reindex_lock()
        return result

    def _do_reindex(self, handle: JobHandle, *, force_full: bool = False) -> dict:
        try:
            new_index = reindex(
                self.fulltext_dir,
                self.encoder,
                incremental=not force_full,
                progress_cb=lambda done, total: handle.update(
                    n_done=done, n_total=total),
            )
        except FileNotFoundError as e:
            raise JobError(str(e)) from e
        finally:
            self.jobs.release_reindex_lock()

        # Atomic swap into the live server.
        self.fulltext_index = new_index
        n_papers = (new_index.metadata or {}).get("n_papers", 0)
        n_chunks = new_index.matrix.shape[0]
        return {
            "n_papers": n_papers,
            "n_chunks": n_chunks,
            "dims": new_index.dims,
            "model": new_index.model_name,
            "max_seq_length": FULLTEXT_MAX_SEQ_LENGTH,
        }


def _paper_payload(p: Paper, score: float, *, full_abstract: bool = False) -> dict:
    """Serialize a Paper for transport. Compact by default — abstract is
    truncated to 600 chars unless `full_abstract=True` (used by paper_info
    when the caller explicitly asks for the untruncated text)."""
    if full_abstract:
        abstract = p.abstract
    else:
        abstract = p.abstract if len(p.abstract) <= 600 else p.abstract[:600] + "…"
    return {
        "arxiv_id": p.arxiv_id,
        "title": p.title,
        "first_author": p.first_author,
        "authors": p.authors[:5],  # cap to 5 for transport
        "abstract": abstract,
        "published": p.published,
        "updated": p.updated,
        "tags": p.tags,
        "topics": p.topics,
        "domain": p.domain,
        "url": f"https://arxiv.org/abs/{p.arxiv_id}",
        "score": round(float(score), 4),
    }


# ---------------------------------------------------------------------------
# MCP tool catalogue
# ---------------------------------------------------------------------------
# Schemas are JSON Schema draft-2020-12 objects. Kept module-level so they can
# be inspected and unit-tested without instantiating the MCP transport.

_FILTER_PROPS = {
    "domain": {
        "type": "string",
        "description": "Filter by source feed name (e.g. 'chemistry'). Omit to search all.",
    },
    "tag": {
        "type": "string",
        "description": "Filter by canonical tag (e.g. 'dft', 'mlip'). Omit to search all.",
    },
}

TOOL_SPECS: list[dict[str, Any]] = [
    # ----- abstracts (6 tools) --------------------------------------------
    {
        "name": "search_abstract_text",
        "description": (
            "Substring search over title + abstract across the full corpus "
            "(multi-token AND, title boosted 3×). Cheap and deterministic; "
            "best when the query uses the exact terminology from the corpus. "
            "`tag` and `domain` are pre-search corpus filters, not embedding signals."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Free-text query."},
                "k": {"type": "integer", "default": 10, "minimum": 1, "maximum": 200},
                **_FILTER_PROPS,
            },
            "required": ["query"],
        },
    },
    {
        "name": "search_abstract_semantic",
        "description": (
            "Cosine-similarity search over abstract embeddings (Qwen3-4B-native). "
            "Robust to terminology drift (e.g. 'MLIP' ↔ 'neural network potential'). "
            "Requires the embedding cache (`arxiv-radar-mcp --build-cache`). "
            "Use this as the default first-pass tool when scanning the corpus."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Natural-language query."},
                "k": {"type": "integer", "default": 10, "minimum": 1, "maximum": 200},
                **_FILTER_PROPS,
            },
            "required": ["query"],
        },
    },
    {
        "name": "similar_to_abstract",
        "description": (
            "Nearest-neighbour papers by abstract-embedding similarity to a "
            "known arxiv_id. Self-match excluded. Useful for 'show me more "
            "like this one'."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "arxiv_id": {"type": "string", "description": "e.g. '2503.12345'"},
                "k": {"type": "integer", "default": 10, "minimum": 1, "maximum": 200},
            },
            "required": ["arxiv_id"],
        },
    },
    {
        "name": "paper_info",
        "description": (
            "Fetch the full metadata for a given arxiv_id (title, authors, "
            "abstract, tags, topics) plus its fulltext-enrichment status "
            "({enriched, indexed, source, n_chunks}). Returns null if the "
            "paper is not in the corpus. Pass `full_abstract=true` to skip "
            "the default 600-char truncation when you need the complete "
            "abstract for a relevance decision."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "arxiv_id": {"type": "string"},
                "full_abstract": {
                    "type": "boolean", "default": False,
                    "description": "If true, return the untruncated abstract.",
                },
            },
            "required": ["arxiv_id"],
        },
    },
    {
        "name": "list_tags",
        "description": (
            "Enumerate canonical tags with paper counts, sorted by frequency. "
            "Use `head_limit` to cap results (default: all tags), "
            "`min_count` to drop low-frequency tags (default: 1), and "
            "`prefix` to filter by tag prefix. `domain` restricts to one "
            "source feed."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "domain": _FILTER_PROPS["domain"],
                "head_limit": {
                    "type": "integer", "minimum": 1, "maximum": 1000,
                    "description": "Cap the number of tags returned (default: all).",
                },
                "min_count": {
                    "type": "integer", "default": 1, "minimum": 1,
                    "description": "Drop tags appearing on fewer papers than this.",
                },
                "prefix": {
                    "type": "string",
                    "description": "Only tags starting with this prefix (case-insensitive).",
                },
            },
        },
    },
    {
        "name": "list_domains",
        "description": "Enumerate configured source feeds with paper counts.",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
    # ----- fulltext (3 tools) ---------------------------------------------
    {
        "name": "search_paper_text",
        "description": (
            "Substring search over chunks of enriched paper full texts. "
            "Returns chunk-level hits {arxiv_id, section, snippet, score}. "
            "Best for exact-term queries (chemical formulas, model names, "
            "specific numeric values). Errors if no papers have been "
            "fetched + reindexed yet."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "k": {"type": "integer", "default": 10, "minimum": 1, "maximum": 100},
                "snippet_chars": {
                    "type": "integer", "default": 240, "minimum": 80, "maximum": 4000,
                    "description": (
                        "Length of returned `snippet` field. Default 240 keeps "
                        "responses compact; raise to ~600+ when extracting "
                        "recipes / numeric parameters."
                    ),
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "search_paper_semantic",
        "description": (
            "Cosine search over chunks of enriched paper full texts. "
            "Returns chunk-level hits with section and snippet so you can "
            "tell the user 'this is in Methods of paper X'. Default tool "
            "for deep dive into papers the user has fetched."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "k": {"type": "integer", "default": 10, "minimum": 1, "maximum": 100},
                "snippet_chars": {
                    "type": "integer", "default": 240, "minimum": 80, "maximum": 4000,
                    "description": (
                        "Length of returned `snippet` field. Default 240 keeps "
                        "responses compact; raise to ~600+ for recipe / "
                        "parameter extraction."
                    ),
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "similar_to_paper",
        "description": (
            "Nearest-neighbour enriched papers by mean-of-chunks embedding "
            "similarity. Like similar_to_abstract but uses full content of "
            "the source paper, not just abstract. Self-match excluded."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "arxiv_id": {"type": "string"},
                "k": {"type": "integer", "default": 10, "minimum": 1, "maximum": 100},
            },
            "required": ["arxiv_id"],
        },
    },
    # ----- async admin (5 tools) ------------------------------------------
    {
        "name": "fetch_papers",
        "description": (
            "Download + parse + cache full text for one or more arxiv_ids. "
            "Returns {job_id} immediately; poll job_status until state=done. "
            "Sources tried in order: arxiv.org/html → arxiv.org/e-print "
            "(LaTeX). Papers without HTML or LaTeX (PDF-only) fail with "
            "a clear reason. Doesn't trigger reindex — call reindex when "
            "ready. Pass `force=true` to re-download cached papers (e.g. "
            "when an earlier fetch landed a stub — short body that's only "
            "abstract + bibliography)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "arxiv_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                },
                "force": {
                    "type": "boolean", "default": False,
                    "description": (
                        "Re-fetch even if the paper is cached. Use this to "
                        "recover from a prior stub fetch."
                    ),
                },
            },
            "required": ["arxiv_ids"],
        },
    },
    {
        "name": "reindex",
        "description": (
            "Rebuild the fulltext embedding index from cached sources. "
            "Incremental by default (encode only papers added or changed "
            "since the last reindex; preserve unchanged rows byte-identical). "
            "Pass force_full=true to re-encode every paper. Returns "
            "{job_id} immediately; poll job_status. Refuses if a reindex "
            "is already running."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "force_full": {"type": "boolean", "default": False},
            },
        },
    },
    {
        "name": "refresh_abstracts",
        "description": (
            "Pull the latest abstracts from the arxiv-radar-* feeds and "
            "update the abstract embedding index. Runs in the background "
            "and returns {job_id} immediately; poll with job_status. "
            "Strategy is incremental by default (encode only new arxiv_ids); "
            "pass force_full=true to re-encode everything (slower, recovers "
            "from upstream prunes)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "force_full": {"type": "boolean", "default": False},
            },
        },
    },
    {
        "name": "job_status",
        "description": (
            "Inspect a background job (fetch_papers or reindex). State is "
            "one of: pending, running, done, failed, orphaned. When done, "
            "the `result` field carries the per-job payload (e.g. fetch "
            "source breakdown, reindex chunk counts)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "job_id": {"type": "string"},
            },
            "required": ["job_id"],
        },
    },
    {
        "name": "job_list",
        "description": (
            "List recent jobs (running first). Useful to recover a job_id "
            "you forgot or to see what's in flight."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "default": 50, "minimum": 1, "maximum": 500},
            },
        },
    },
    {
        "name": "list_enriched",
        "description": (
            "List arxiv_ids that have full text fetched and cached locally. "
            "Cheap synchronous lookup — no fulltext index required."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "validate_arxiv_ids",
        "description": (
            "Pre-flight check: classify each arxiv_id as `ok` (HTML render "
            "exists, fetch_papers will likely succeed) or `pdf_only` (no "
            "HTML — fetch_papers will fail with a clear reason). HEAD-probes "
            "arxiv.org/html/<id> at the same 1 req / 3 sec rate as the live "
            "fetcher. Already-cached papers count as ok without a probe. "
            "Use this BEFORE submitting a large fetch_papers batch so the "
            "user gets early warning about ids you can't enrich."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "arxiv_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                },
            },
            "required": ["arxiv_ids"],
        },
    },
]


def _tool_names() -> list[str]:
    return [spec["name"] for spec in TOOL_SPECS]


def _dispatch(server: RadarServer, name: str, arguments: dict[str, Any] | None) -> Any:
    """Route an MCP tool-call to the matching RadarServer method.

    Thin wrapper over `corpus_core.mcp_scaffold.make_method_dispatcher`.
    Kept under the `arxiv_radar_mcp.server` name because the test suite
    imports it directly with stub `RadarServer`-shaped objects.
    """
    return make_method_dispatcher(server, _tool_names())(name, arguments)


def _build_mcp_app(radar: RadarServer):
    """Construct the MCP app for `radar`, wired with our TOOL_SPECS."""
    return build_mcp_app(
        server_name="arxiv-radar",
        tool_specs=TOOL_SPECS,
        dispatcher=make_method_dispatcher(radar, _tool_names()),
    )


async def _warmup_encoder(radar: RadarServer) -> None:
    """Force a single dummy encode so the user's first real query doesn't
    pay the lazy-load cost (~20 sec for Qwen3-4B on RTX 4070, much longer
    if the model has to download weights from HF Hub first).

    Runs on a worker thread so the event loop stays responsive — allows
    the HTTP server to accept connections while warm-up is in progress.
    Failures are logged but never propagate; cold-start search will just
    be slow but functional.
    """
    LOG.info("encoder warm-up: starting (so first query doesn't cold-load)")
    try:
        await asyncio.to_thread(radar.encoder.encode_query, "warmup")
        LOG.info("encoder warm-up: ready")
    except Exception as e:  # noqa: BLE001
        LOG.warning(f"encoder warm-up failed (will retry on first query): {e}")


async def _refresh_loop(radar: RadarServer) -> None:
    """Background daily refresh of the abstract corpus.

    Bootstrap: if there's no cache or it's older than `interval_hours`,
    do one refresh immediately (full rebuild) so the server has data.
    Then sleep `interval_hours`, refresh, repeat.
    """
    import time

    interval_seconds = radar.config.refresh.interval_hours * 3600
    full = radar.config.refresh.full_rebuild

    if _should_refresh_at_boot(radar):
        LOG.info("refresh: bootstrap needed (no/stale abstract cache)")
        try:
            result = await asyncio.to_thread(_blocking_refresh, radar, True)
            LOG.info(f"refresh: bootstrap result: {result}")
        except Exception as e:  # noqa: BLE001
            LOG.exception(f"refresh: bootstrap failed: {e}")

    LOG.info(f"refresh loop: every {radar.config.refresh.interval_hours}h, "
             f"full_rebuild={full}")

    while True:
        try:
            await asyncio.sleep(interval_seconds)
        except asyncio.CancelledError:
            LOG.info("refresh loop: cancelled")
            return
        try:
            result = await asyncio.to_thread(_blocking_refresh, radar, full)
            LOG.info(f"refresh: result: {result}")
        except Exception as e:  # noqa: BLE001
            LOG.exception(f"refresh tick failed: {e}; retry next interval")


def _should_refresh_at_boot(radar: RadarServer) -> bool:
    """True when the abstract cache is missing or older than interval."""
    import time

    if radar.abstract_index is None:
        return True
    npy = radar.config.embeddings.cache_dir / "embeddings.npy"
    if not npy.exists():
        return True
    age_seconds = time.time() - npy.stat().st_mtime
    return age_seconds > radar.config.refresh.interval_hours * 3600


def _blocking_refresh(radar: RadarServer, full_rebuild: bool) -> dict:
    """Synchronous refresh under the encoder lock. Called via asyncio.to_thread
    from the refresh loop so the event loop stays responsive."""
    if not radar.jobs.acquire_reindex_lock():
        return {"skipped": "encoder busy"}
    try:
        return refresh_sources(radar, full_rebuild=full_rebuild)
    finally:
        radar.jobs.release_reindex_lock()


def _arxiv_background_tasks(radar: RadarServer) -> list[BackgroundTaskFactory]:
    """Background tasks the arxiv-radar shell wants alongside the MCP transport.

    Each entry is a zero-arg callable returning a fresh coroutine — the
    `corpus_core.mcp_scaffold` contract. Re-running the server thus
    produces fresh tasks instead of trying to await an already-finished
    coroutine.
    """
    factories: list[BackgroundTaskFactory] = [lambda: _warmup_encoder(radar)]
    if radar.config.refresh.enabled:
        factories.append(lambda: _refresh_loop(radar))
    return factories


async def _run_stdio(radar: RadarServer) -> None:
    """Async stdio loop. Delegates transport to corpus_core.mcp_scaffold."""
    await serve_stdio(
        server_name="arxiv-radar",
        tool_specs=TOOL_SPECS,
        dispatcher=make_method_dispatcher(radar, _tool_names()),
        background_tasks=_arxiv_background_tasks(radar),
    )


async def _run_streamable_http(radar: RadarServer, host: str, port: int) -> None:
    """Async streamable-HTTP loop. Long-running, holds RadarServer in memory.

    Bind to 127.0.0.1 by default in production deployments — perimeter
    auth comes from an SSH tunnel, NOT from this server. See README for
    the deployment topology.
    """
    await serve_streamable_http(
        server_name="arxiv-radar",
        tool_specs=TOOL_SPECS,
        dispatcher=make_method_dispatcher(radar, _tool_names()),
        host=host,
        port=port,
        background_tasks=_arxiv_background_tasks(radar),
    )


def serve(config_path: Path | None = None) -> None:
    """Entry point: stdio MCP server. Backwards-compat with single-user
    pip install path."""
    config = load(config_path)
    radar = RadarServer(config)
    asyncio.run(_run_stdio(radar))


def serve_http(host: str, port: int, config_path: Path | None = None) -> None:
    """Entry point: streamable-HTTP MCP server. Long-running backend mode."""
    config = load(config_path)
    radar = RadarServer(config)
    LOG.info(f"arxiv-radar streamable-HTTP server listening on {host}:{port}")
    LOG.info(f"  encoder model: {config.embeddings.model}")
    LOG.info(f"  cache_dir: {config.embeddings.cache_dir}")
    asyncio.run(_run_streamable_http(radar, host, port))
