# arxiv-radar-mcp

MCP server providing semantic + text search over arXiv abstracts and
**on-demand full text** (HTML / LaTeX source) of papers. Reads abstracts
from the [`daily-arxiv-*`](https://github.com/exopoiesis?tab=repositories&q=daily-arxiv)
fork family (`ai4chem`, `physics`, `polymers`, …); fetches and indexes
full text per-paper at the user's request via `arxiv.org/html` and
`arxiv.org/e-print`.

> **Status:** Phase 7-10 done (2026-05-01). 14 MCP tools, two parallel
> indexes (abstracts + fulltext), 110 unit tests green, end-to-end
> validated on gomer GPU. Architecture and decisions in
> [`docs/PLAN.md`](docs/PLAN.md). Empirical embedding-stack benchmarks in
> [`docs/MODEL_BENCHMARKS.md`](docs/MODEL_BENCHMARKS.md).

---

## What it does

Two-step user flow:

1. **Scan abstracts** — Claude searches the daily-arxiv corpus
   (~14k papers in `ai4chem`, growing) via `search_abstract_*` and
   reports candidates.
2. **Drill into full text** — when the user wants depth, Claude calls
   `fetch_papers([ids])` (background job, downloads + parses), then
   `reindex` (rebuilds the fulltext embedding index), then queries it
   with `search_paper_*` to surface specific sections ("found in
   Methods of paper X").

Each `daily-arxiv-<domain>` repo publishes:
- `data/papers-YYYY-MM.json` — monthly abstract shards with titles,
  authors, abstracts, tags, topics
- `tags/canonical.yaml` — curated tag vocabulary

This server reads those shards (over GitHub raw URLs, on-disk shard
cache), keeps abstracts in memory, and adds a separate fulltext layer
fetched on demand. Full texts are cached locally per arxiv_id.

## Tool surface (14 tools)

### Abstracts (6)

| Tool | Purpose |
|------|---------|
| `search_abstract_text(query, k, domain?, tag?)` | substring AND over title+abstract, title-boost 3× |
| `search_abstract_semantic(query, k, domain?, tag?)` | cosine over abstract embeddings (Qwen3-4B-native by default) |
| `similar_to_abstract(arxiv_id, k)` | nearest-neighbour by abstract embedding |
| `paper_info(arxiv_id)` | full metadata + fulltext-enrichment status |
| `list_tags(domain?)` | canonical tag vocabulary with paper counts |
| `list_domains()` | configured source feeds |

`tag` and `domain` are pre-search corpus filters — they restrict the
ranked subset, not the embedding signal. Tags are not encoded into
embeddings (would dilute semantic signal).

### Fulltext (3)

Operates on chunks of full texts the user has explicitly enriched.

| Tool | Purpose |
|------|---------|
| `search_paper_text(query, k)` | AND-scan over chunk texts; returns `{arxiv_id, section, snippet, score}` |
| `search_paper_semantic(query, k)` | cosine over chunk embeddings; same payload shape |
| `similar_to_paper(arxiv_id, k)` | nearest-neighbour papers by mean-of-chunks embedding |

### Async admin (5)

| Tool | Purpose |
|------|---------|
| `fetch_papers([arxiv_ids])` | background job: download + parse + cache full text. Returns `{job_id}` |
| `reindex()` | background job: full rebuild of fulltext embedding index |
| `job_status(job_id)` | inspect a running/finished job |
| `job_list(limit?)` | list recent jobs |
| `list_enriched()` | sync: arxiv_ids of locally-cached full texts |

Admin operations are async because reindex on a meaningful corpus is
slow enough (5+ min CPU, 30s GPU per ~10 papers) to break the MCP
conversation if blocking. Jobs persist to disk and survive restarts.

## Full-text fetch cascade

Per arxiv_id, in order, stop at first success:

| Source | URL pattern | Coverage | Used for |
|---|---|---|---|
| arxiv-rendered HTML | `arxiv.org/html/<id>` | ~70-80% recent papers | preferred — equations preserved as inline LaTeX via `<annotation encoding="application/x-tex">` |
| LaTeX e-print | `arxiv.org/e-print/<id>` | ~85-90% all papers | fallback — pylatexenc expands macros, drops comments, leaves `$...$` math |
| (PDF — not supported) | n/a | n/a | Paper fails with clear "PDF-only on arXiv" error |

Why no PDF: `mineru[core]` adds ~2 GB of dependency to cover ~10-15%
remaining tail. Trade-off didn't make sense for this server's scope.
See [`docs/PLAN.md`](docs/PLAN.md) `[РЕШЕНИЕ-013]`.

Empirical reference (HTML+LaTeX vs PDF text quality on arXiv corpus):
HTML+LaTeX combined F1 = 0.69 vs text-from-PDF much lower
([Toward Robust URL Extraction for Open Science, arXiv 2025](https://arxiv.org/abs/2509.04759)).

## Chunking

Full-text markdown is split by `## headings`. Each section becomes one
chunk; sections >12 288 tokens (long Discussions in reviews) sub-split
by paragraph boundaries, no overlap. Each chunk row carries
`(arxiv_id, section, chunk_idx)` so search results attribute back to
paper region.

12 288 tokens chosen because: (a) it covers ~95th percentile of arxiv
section sizes without forcing chunking on most sections, (b) Qwen3-4B
bf16 + 12 288 seq length fits in 12 GB VRAM at batch_size=4.

## Architecture

```
                       daily-arxiv-* GitHub raw URLs
                                    │
                                    ▼
                          corpus.load_all()
                                    │
                                    ▼
               in-memory dict[arxiv_id, Paper]
                                    │
        ┌───────────────────────────┼──────────────────────────┐
        ▼                           ▼                          ▼
  abstract index              fulltext index               tag/domain
  cache_dir/abstracts/        cache_dir/fulltext/          filter
    embeddings.npy              embeddings.npy             (pre-search)
    index.json                  index.json
                                sources/<id>.md
                                sources/<id>.meta.json
                                                  ▲
                                                  │
                          fetch_papers / reindex jobs
                                                  │
                                       arxiv.org/html
                                       arxiv.org/e-print
                                                  │
                                       cache_dir/jobs/<id>.json

                              MCP stdio server  ←  Claude Desktop / Cursor / IDE
```

## Embedding model

**Production default: `Qwen/Qwen3-Embedding-4B`** at native 2560 dim,
bf16. Selected after benchmarking 7 caches against the 14k ai4chem
corpus on a 22-query bench (12 generic + 10 paraphrased→target). Full
results in [`docs/MODEL_BENCHMARKS.md`](docs/MODEL_BENCHMARKS.md):

| Model | dim | recall@1 | median rank | Cache (14k) |
|---|---|---|---|---|
| bge-small-en-v1.5 | 384 | 6/10 | 1 | 21 MB |
| mxbai-embed-large-v1 | 1024 | 7/10 | 1 | 56 MB |
| Qwen3-Embedding-4B (matryoshka 1024) | 1024 | 8/10 | 1 | 56 MB |
| **Qwen3-Embedding-4B (native)** | **2560** | **9/10** | **1** | **140 MB** |
| Qwen3-Embedding-8B (native) | 4096 | 9/10 | 1 | 224 MB |

The 4B native ties or beats the 8B native on this corpus and degrades
less under matryoshka truncation. Onboarding default in code is still
`mxbai-embed-large-v1` — works on CPU without a 9 GB download. Switch
to Qwen3-4B via `radar.toml` when GPU is available.

Cross-encoder reranker was evaluated and removed from the public tool
surface. On Qwen3-4B-native (1.2 avg rank) it ceilings at ~2.0 — same
as raw cosine — at the cost of 1.5s per query. The Reranker class
stays in code as a utility but isn't wired to any tool. See
`[РЕШЕНИЕ-010]` in PLAN.

## Install

### Local (CPU, abstracts-only path)

```bash
pip install -e ".[dev]"
arxiv-radar-mcp --build-cache       # ~7 min on GPU, infeasible on CPU for Qwen3-4B
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

### GPU (gomer Docker image, fulltext indexing)

Image: `exopoiesis/arxiv-radar-gpu:latest` (~10 GB pytorch + cuda12.4
+ this code). Build context = this repo only.

```bash
# one-time build on gomer
bash scripts/docker_build.sh

# one-time: write radar.toml into the named volume
bash tmp/gomer_init_volume.sh

# fetch full text for a few papers
bash scripts/docker_fetch.sh /cache/radar.toml /cache 2503.11576 2410.07073

# rebuild fulltext index on GPU
bash scripts/docker_reindex.sh /cache/radar.toml /cache

# stdio MCP server bridged to local
bash scripts/docker_serve_mcp.sh /cache/radar.toml /cache
```

The container has named volumes:
- `arxiv-radar-cache` — `/cache` (radar.toml + abstracts/ + fulltext/ + jobs/)
- `arxiv-radar-hf` — `/root/.cache/huggingface` (model weights)

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
model     = "Qwen/Qwen3-Embedding-4B"   # or "mixedbread-ai/mxbai-embed-large-v1" for CPU
cache_dir = "~/.cache/arxiv-radar/embeddings"
batch_size = 32

[reranker]
enabled = false   # not wired to any tool currently — see [РЕШЕНИЕ-010]

[server]
default_k    = 10
hybrid_rrf_k = 60
```

Full reference example: [`radar.example.toml`](radar.example.toml).

## License

MIT. See [LICENSE](LICENSE).

Reads abstracts from the
[`daily-arxiv-*`](https://github.com/exopoiesis?tab=repositories&q=daily-arxiv)
fork family, which is itself a fork of
[YuzeHao2023/daily-arxiv-ai4chem](https://github.com/YuzeHao2023/daily-arxiv-ai4chem).
