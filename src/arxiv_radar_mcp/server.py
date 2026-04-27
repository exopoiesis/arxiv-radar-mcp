"""MCP stdio server. Wires Tools → search functions → corpus + index.

Skeleton: tool signatures are final, bodies wire through to search.py.
The actual MCP framework binding (mcp Python SDK) is left for next session
once the SDK API is locked in.
"""
from __future__ import annotations

import logging
from collections import Counter
from datetime import date, datetime, timedelta
from pathlib import Path

from arxiv_radar_mcp.config import Config, load
from arxiv_radar_mcp.corpus import Paper, load_all
from arxiv_radar_mcp.embeddings import EmbeddingIndex, encode_query
from arxiv_radar_mcp.search import (search_hybrid, search_semantic,
                                    search_text, similar_to)

LOG = logging.getLogger(__name__)


class RadarServer:
    """Holds the loaded corpus + embedding index. Tools are methods on it.

    Created once at startup; tools dispatch through it without reloading.
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.papers: dict[str, Paper] = load_all(config)
        self.index: EmbeddingIndex | None = None
        try:
            self.index = EmbeddingIndex.load(config.embeddings.cache_dir)
            LOG.info(f"loaded embedding index: {self.index.matrix.shape} "
                     f"({self.index.model_name})")
        except FileNotFoundError:
            LOG.warning("no embedding cache yet — semantic / hybrid / similar_to "
                        "tools will be unavailable until `arxiv-radar-mcp --build-cache`")

    # ----- tool: search_text -------------------------------------------------
    def search_text(self, query: str, k: int = 10,
                    domain: str | None = None, tag: str | None = None) -> list[dict]:
        return [_paper_payload(p, score) for p, score in
                search_text(self.papers.values(), query, k=k,
                            domain=domain, tag=tag)]

    # ----- tool: search_semantic --------------------------------------------
    def search_semantic(self, query: str, k: int = 10,
                        domain: str | None = None, tag: str | None = None) -> list[dict]:
        if self.index is None:
            return [{"error": "embedding cache not built. run `arxiv-radar-mcp --build-cache`"}]
        qvec = encode_query(query, self.config)
        return [_paper_payload(p, score) for p, score in
                search_semantic(self.papers, qvec, self.index, k=k,
                                domain=domain, tag=tag)]

    # ----- tool: search_hybrid ----------------------------------------------
    def search_hybrid(self, query: str, k: int = 10,
                      domain: str | None = None, tag: str | None = None) -> list[dict]:
        if self.index is None:
            return self.search_text(query, k=k, domain=domain, tag=tag)
        text_hits = search_text(self.papers.values(), query, k=max(k * 3, 30),
                                domain=domain, tag=tag)
        qvec = encode_query(query, self.config)
        sem_hits = search_semantic(self.papers, qvec, self.index, k=max(k * 3, 30),
                                   domain=domain, tag=tag)
        fused = search_hybrid(text_hits, sem_hits,
                              k=k, rrf_k=self.config.server.hybrid_rrf_k)
        return [_paper_payload(p, score) for p, score in fused]

    # ----- tool: similar_to -------------------------------------------------
    def similar_to(self, arxiv_id: str, k: int = 10) -> list[dict]:
        if self.index is None:
            return [{"error": "embedding cache not built"}]
        return [_paper_payload(p, score) for p, score in
                similar_to(self.papers, arxiv_id, self.index, k=k)]

    # ----- tool: recent ------------------------------------------------------
    def recent(self, days: int = 7, domain: str | None = None,
               tag: str | None = None, k: int = 50) -> list[dict]:
        cutoff = date.today() - timedelta(days=days)
        out: list[tuple[Paper, str]] = []
        for p in self.papers.values():
            if domain and domain not in p.domain.split(","):
                continue
            if tag and tag not in p.tags:
                continue
            try:
                if datetime.strptime(p.updated, "%Y-%m-%d").date() >= cutoff:
                    out.append((p, p.updated))
            except ValueError:
                continue
        out.sort(key=lambda x: x[1], reverse=True)
        return [_paper_payload(p, 0.0) for p, _ in out[:k]]

    # ----- tool: get_paper --------------------------------------------------
    def get_paper(self, arxiv_id: str) -> dict | None:
        p = self.papers.get(arxiv_id)
        return None if p is None else _paper_payload(p, 0.0)

    # ----- tool: list_tags --------------------------------------------------
    def list_tags(self, domain: str | None = None) -> list[dict]:
        c: Counter[str] = Counter()
        for p in self.papers.values():
            if domain and domain not in p.domain.split(","):
                continue
            for t in p.tags:
                c[t] += 1
        return [{"tag": t, "count": n} for t, n in c.most_common()]

    # ----- tool: list_domains -----------------------------------------------
    def list_domains(self) -> list[dict]:
        c: Counter[str] = Counter()
        for p in self.papers.values():
            for d in p.domain.split(","):
                if d:
                    c[d] += 1
        return [{"domain": d, "papers": n} for d, n in c.most_common()]


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


def serve(config_path: Path | None = None) -> None:
    """Entry point: load corpus + index, register tools, start MCP stdio server.

    SKELETON — needs binding to the `mcp` Python SDK in the next session.
    """
    config = load(config_path)
    server = RadarServer(config)

    # TODO: bind tools to mcp.Server. Pseudo-code with the official SDK:
    #
    #   from mcp.server import Server
    #   from mcp.server.stdio import stdio_server
    #   from mcp.types import Tool, TextContent
    #
    #   app = Server("arxiv-radar")
    #
    #   @app.list_tools()
    #   async def list_tools() -> list[Tool]:
    #       return [
    #           Tool(name="search_text", ..., inputSchema={...}),
    #           Tool(name="search_semantic", ...),
    #           ...
    #       ]
    #
    #   @app.call_tool()
    #   async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    #       method = getattr(server, name)
    #       result = method(**arguments)
    #       return [TextContent(type="text", text=json.dumps(result, indent=1))]
    #
    #   async with stdio_server() as (read, write):
    #       await app.run(read, write, app.create_initialization_options())
    #
    # See https://modelcontextprotocol.io/quickstart/server for current examples.
    raise NotImplementedError(
        "MCP binding TODO — see comments in serve(). Corpus + index + search "
        "logic are wired and unit-testable; only the protocol shell is missing."
    )
