# arxiv-radar-mcp

MCP server that provides semantic + text + hybrid search over the
[`daily-arxiv-*`](https://github.com/exopoiesis?tab=repositories&q=daily-arxiv) fork family — a tagged, daily-curated feed of arXiv
papers per scientific domain (`ai4chem`, future: `physics`, `polymers`, ...).

> **Status:** scaffolding stage. Architecture and decisions documented in
> [`docs/PLAN.md`](docs/PLAN.md). First working tools target Phase 6 of the
> radar plan.

---

## What it does

Each `daily-arxiv-<domain>` repo publishes:
- `data/papers-YYYY-MM.json` — monthly shards with titles, authors, abstracts, tags
- `tags/canonical.yaml` — curated tag vocabulary
- `docs/abstracts/<id>.html` — popup-rendered abstract fragments

This MCP server reads those shards (locally or via raw GitHub URLs),
keeps the corpus in memory, and exposes search / browse tools to any
MCP-compatible client (Claude Desktop, Claude Code, Cursor, ...).

## Why a separate repo

`daily-arxiv-*` is **data + Pages site**. Heavy: ~14k abstract fragments,
230 pre-rendered tag pages, monthly-pruning workflow, GitHub Pages config.

`arxiv-radar-mcp` is **code only**. Light: pip-installable, embeddings cache
generated on first run (~80 MB model, ~80 MB cache for ~100k papers
across all domains). Composes any number of `daily-arxiv-*` sources via
config — adding a new domain doesn't require any code change.

## Tools (planned)

| Tool | Purpose |
|------|---------|
| `search_text(query, k=10, domain=None, tag=None)` | substring + keyword over title+abstract |
| `search_semantic(query, k=10, domain=None, tag=None)` | cosine similarity in embedding space |
| `search_hybrid(query, k=10, domain=None, tag=None)` | reciprocal-rank fusion of the two |
| `similar_to(arxiv_id, k=10)` | nearest-neighbour by paper embedding |
| `recent(days=7, domain=None, tag=None)` | most recent in window |
| `get_paper(arxiv_id)` | full record (title, authors, abstract, tags, links) |
| `list_tags(domain=None)` | canonical tag vocabulary with paper counts |
| `list_domains()` | configured domain feeds |

## Architecture

```
[ daily-arxiv-ai4chem/data/papers-*.json ]   ┐
[ daily-arxiv-physics/data/papers-*.json ]   │ → corpus.load_all()
[ daily-arxiv-polymers/data/papers-*.json ]  ┘        │
                                                      ▼
                                              ┌───────────────┐
                                              │  in-memory    │
                                              │ paper records │
                                              └───────┬───────┘
                                                      │
                  ┌───────────────────────────────────┼──────────────────┐
                  ▼                                   ▼                  ▼
           text search                         embedding cache       tag/domain
           (substring, BM25?)                  (npy/parquet,         filter
                                                ~80 MB)
                                                      │
                                                      ▼
                                              cosine similarity
                                                   (numpy)

                                                MCP stdio server  ←  Claude / IDE
```

Embedding model: **`all-MiniLM-L6-v2`** (384 dims, 80 MB, CPU-friendly,
no GPU needed). Encodes title + abstract.

## Config

`~/.config/arxiv-radar/radar.toml` (override via `--config`):

```toml
[sources.ai4chem]
type = "github"
repo = "exopoiesis/daily-arxiv-ai4chem"
branch = "main"

[sources.physics]
type = "local"
path = "/path/to/daily-arxiv-physics"

[embeddings]
model = "sentence-transformers/all-MiniLM-L6-v2"
cache_dir = "~/.cache/arxiv-radar/embeddings"

[server]
default_k = 10
hybrid_rrf_k = 60
```

## Install (planned)

```bash
pip install arxiv-radar-mcp        # not yet on PyPI
arxiv-radar-mcp --build-cache       # ~5 min on first run (download model + embed corpus)
arxiv-radar-mcp                     # starts MCP server on stdio
```

For Claude Desktop, add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "arxiv-radar": {
      "command": "arxiv-radar-mcp"
    }
  }
}
```

## License

MIT. See [LICENSE](LICENSE).

Companion to the
[`daily-arxiv-*`](https://github.com/exopoiesis?tab=repositories&q=daily-arxiv)
fork family, which itself is forked from
[Vincentqyw/cv-arxiv-daily](https://github.com/Vincentqyw/cv-arxiv-daily).
