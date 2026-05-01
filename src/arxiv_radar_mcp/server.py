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
from arxiv_radar_mcp.embeddings import EmbeddingIndex, Encoder
from arxiv_radar_mcp.fulltext import fetch_and_save
from arxiv_radar_mcp.fulltext_index import (FULLTEXT_MAX_SEQ_LENGTH,
                                            load_chunk_texts, reindex,
                                            search_paper_semantic,
                                            search_paper_text,
                                            similar_to_paper)
from arxiv_radar_mcp.jobs import JobError, JobHandle, JobRegistry
from arxiv_radar_mcp.search import (search_semantic, search_text, similar_to)

LOG = logging.getLogger(__name__)


class RadarServer:
    """Holds the loaded corpus + embedding indexes. Tools are methods on it.

    Two indexes coexist: `abstract_index` (always present once cache built)
    and `fulltext_index` (None until the user enriches papers and reindexes).
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.papers: dict[str, Paper] = load_all(config)
        self.encoder = Encoder(config)  # lazy — loads model on first encode

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

    def paper_info(self, arxiv_id: str) -> dict | None:
        """Metadata + fulltext-status for one paper. None if not in corpus."""
        p = self.papers.get(arxiv_id)
        if p is None:
            return None
        payload = _paper_payload(p, 0.0)
        payload["fulltext"] = self._fulltext_status(arxiv_id)
        return payload

    def list_tags(self, domain: str | None = None) -> list[dict]:
        c: Counter[str] = Counter()
        for p in self.papers.values():
            if domain and domain not in p.domain.split(","):
                continue
            for t in p.tags:
                c[t] += 1
        return [{"tag": t, "count": n} for t, n in c.most_common()]

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

    def search_paper_text(self, query: str, k: int = 10) -> list[dict]:
        if self.fulltext_index is None:
            return [{"error": "fulltext index empty — run fetch_papers and "
                              "reindex first"}]
        chunk_texts = load_chunk_texts(self.fulltext_dir, self.fulltext_index)
        chunk_meta = (self.fulltext_index.metadata or {}).get("chunks", [])
        return search_paper_text(chunk_texts, chunk_meta, query, k=k)

    def search_paper_semantic(self, query: str, k: int = 10) -> list[dict]:
        if self.fulltext_index is None:
            return [{"error": "fulltext index empty — run fetch_papers and "
                              "reindex first"}]
        qvec = self.encoder.encode_query(query)
        chunk_texts = load_chunk_texts(self.fulltext_dir, self.fulltext_index)
        return search_paper_semantic(self.fulltext_index, chunk_texts, qvec, k=k)

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

    # ----- async admin -----------------------------------------------------

    def fetch_papers(self, arxiv_ids: list[str]) -> dict:
        """Submit a background fetch job. Returns {job_id} immediately."""
        if not arxiv_ids:
            return {"error": "arxiv_ids must be a non-empty list"}
        job_id = self.jobs.submit(
            kind="fetch_papers",
            fn=lambda h: self._do_fetch(h, arxiv_ids),
            args={"arxiv_ids": arxiv_ids},
            n_total=len(arxiv_ids),
        )
        return {"job_id": job_id, "n_total": len(arxiv_ids),
                "kind": "fetch_papers"}

    def reindex(self) -> dict:
        """Submit a background reindex job. Returns {job_id} immediately.

        Refuses if a reindex is already running (lockfile held).
        """
        sources_dir = self.fulltext_dir / "sources"
        n_papers = sum(1 for _ in sources_dir.glob("*.md")) if sources_dir.exists() else 0
        if n_papers == 0:
            return {"error": "no enriched papers to index — run fetch_papers first"}

        if not self.jobs.acquire_reindex_lock():
            return {"error": "reindex already in progress (lockfile held)"}

        job_id = self.jobs.submit(
            kind="reindex",
            fn=self._do_reindex,
            args={"n_papers": n_papers},
            n_total=n_papers,
        )
        return {"job_id": job_id, "n_total": n_papers, "kind": "reindex"}

    def job_status(self, job_id: str) -> dict:
        info = self.jobs.get(job_id)
        if info is None:
            return {"error": f"unknown job_id: {job_id!r}"}
        return info

    def job_list(self, limit: int = 50) -> list[dict]:
        return self.jobs.list_recent(limit=limit)

    # ----- job workers (called by JobRegistry on a worker thread) ----------

    def _do_fetch(self, handle: JobHandle, arxiv_ids: list[str]) -> dict:
        ok: list[dict] = []
        failed: list[dict] = []
        source_counts: Counter[str] = Counter()

        with httpx.Client(timeout=httpx.Timeout(30.0, connect=10.0),
                          headers={"User-Agent": "arxiv-radar-mcp/0.1"},
                          follow_redirects=True) as client:
            for i, pid in enumerate(arxiv_ids):
                try:
                    result = fetch_and_save(pid, self.fulltext_dir, client=client)
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

    def _do_reindex(self, handle: JobHandle) -> dict:
        try:
            new_index = reindex(
                self.fulltext_dir,
                self.encoder,
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


def _paper_payload(p: Paper, score: float) -> dict:
    """Serialize a Paper for transport. Compact — abstract truncated to 600 chars."""
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
        "description": "Filter by source feed name (e.g. 'ai4chem'). Omit to search all.",
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
            "paper is not in the corpus."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "arxiv_id": {"type": "string"},
            },
            "required": ["arxiv_id"],
        },
    },
    {
        "name": "list_tags",
        "description": (
            "Enumerate canonical tags with paper counts, sorted by frequency. "
            "Optionally restrict to a single source feed."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "domain": _FILTER_PROPS["domain"],
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
            "ready."
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
    {
        "name": "reindex",
        "description": (
            "Rebuild the fulltext embedding index from all cached sources. "
            "Always full rebuild (not incremental — see [РЕШЕНИЕ-014]). "
            "Returns {job_id} immediately; poll job_status. ~1 min CPU per "
            "10 papers, much faster on GPU. Refuses if a reindex is "
            "already running."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {},
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
]


def _dispatch(server: RadarServer, name: str, arguments: dict[str, Any] | None) -> Any:
    """Route an MCP tool-call to the matching RadarServer method.

    Returns the method's return value, or an {"error": ...} dict for
    unknown tools / bad arguments. Keeping this synchronous and pure makes
    it trivial to unit-test without spinning up the SDK.
    """
    method = getattr(server, name, None)
    if method is None or name.startswith("_") or name not in {s["name"] for s in TOOL_SPECS}:
        return {"error": f"unknown tool: {name!r}"}
    try:
        return method(**(arguments or {}))
    except TypeError as e:
        return {"error": f"bad arguments for {name}: {e}"}


async def _run_stdio(radar: RadarServer) -> None:
    """Async stdio loop. Imported lazily so unit-tests don't need the SDK."""
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import TextContent, Tool

    app: Server = Server("arxiv-radar")

    @app.list_tools()
    async def _list_tools() -> list[Tool]:
        return [Tool(**spec) for spec in TOOL_SPECS]

    @app.call_tool()
    async def _call_tool(name: str, arguments: dict[str, Any] | None) -> list[TextContent]:
        result = _dispatch(radar, name, arguments)
        text = json.dumps(result, indent=1, ensure_ascii=False, default=str)
        return [TextContent(type="text", text=text)]

    async with stdio_server() as (read, write):
        await app.run(read, write, app.create_initialization_options())


def serve(config_path: Path | None = None) -> None:
    """Entry point: load corpus + index, register tools, start MCP stdio server."""
    config = load(config_path)
    radar = RadarServer(config)
    asyncio.run(_run_stdio(radar))
