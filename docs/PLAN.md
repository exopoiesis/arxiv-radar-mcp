# Plan & decisions — arxiv-radar-mcp

Living design doc. Sibling to `daily-arxiv-*` data forks; this repo is
**code only**, the data is read from the fork family.

---

## [РЕШЕНИЕ-001] Why an MCP server (not a website / API)

The radar plan from `daily-arxiv-ai4chem` resolved Phase 6 as: a
machine-readable interface for AI assistants — Claude Desktop, Claude
Code, Cursor — over the curated corpus, not yet another HTTP API.

MCP gives one tool definition that any compliant client picks up;
search becomes a first-class capability inside the assistant the
researcher already uses.

## [РЕШЕНИЕ-002] One repo, many domains

Sources are listed in `radar.toml`. Adding `daily-arxiv-physics` is
config-only — no code change. The loader concatenates per-domain
records into one in-memory map keyed by arxiv_id and tags each Paper
with its source domain so search results can be filtered or attributed.

## [РЕШЕНИЕ-003] Embedding model

`sentence-transformers/all-MiniLM-L6-v2`:
- 384 dimensions (small, fast cosine on CPU)
- ~80 MB download, no GPU needed
- Decent on technical / scientific English; not SOTA, but the recall
  bottleneck on this corpus is terminology dispersion (MLIP /
  ML potential / NN potential / equivariant force field) which any
  half-decent semantic model handles. Bigger models would burn cache /
  install time without proportional gain.

Encoding target: `title + "\n\n" + abstract`. Title-only loses
context; full body would need chunking and we're not solving that yet.

## [РЕШЕНИЕ-004] Cache layout

Single npy + index.json sidecar:
- `<cache_dir>/embeddings.npy`  shape (N, 384) float32, L2-normalized
- `<cache_dir>/index.json`      `{model, dims, n, row_for: {arxiv_id: int}}`

L2-normalize on encode → cosine == dot product → numpy `@` is the whole
similarity step, no library needed. Works up to ~1M rows on a laptop
before we'd need to consider FAISS / hnswlib.

Embeddings are NOT in git — built at install time via
`arxiv-radar-mcp --build-cache`.

## [РЕШЕНИЕ-005] Hybrid = RRF, not weighted sum

Reciprocal Rank Fusion with `k=60` (canonical). Weighted sum requires
score normalization between text (BM25-ish counts) and semantic (cosine
0..1) which is fragile. RRF only sees ranks, blends robustly, and the
constant 60 has held up across IR literature for two decades.

## [РЕШЕНИЕ-006] Source fetch via raw.githubusercontent

`type=github` source pulls `data/papers-*.json` over HTTPS one shard at
a time, caches under `<cache_dir>/shards/<source-name>/`. Two reasons:
1. No git clone of a 14k-file repo just to read the JSONs
2. Rebuild-cache step naturally refreshes shards (we just delete the
   shard dir and re-fetch)

Rate-limiting: GitHub raw URLs are unauthenticated, generous limit. If
we hit it, switch to authenticated API or carry an `ETag` header.

## [РЕШЕНИЕ-007] Server is sync, tool dispatch is sync too

In-memory corpus + numpy is fast enough that there's no win from async.
The MCP SDK's stdio loop is async, but tool bodies stay synchronous —
they just compute and return. Avoids a parallel mental model for no
benefit.

## [РЕШЕНИЕ-008] No persistence layer

Everything is rebuilt from `data/papers-*.json` on each `--build-cache`
run. No SQLite, no full-text index file, no app state to migrate. The
data forks are the source of truth.

---

## Open questions

1. **Refresh policy** — when `arxiv-radar-mcp` runs, should it
   transparently re-fetch shards if they're older than N days?
   Default to no (manual `--build-cache`)? Or pull on first request and
   cache for 24h?
2. **Cross-domain dedup** — a paper might be tagged `chemistry` AND
   `physics` and end up in two forks. Current loader concatenates the
   `domain` field. Is that the right merge strategy or should we union
   topics + tags?
3. **BM25 upgrade** — drop-in replacement for `search_text`'s naive
   token-AND. `rank_bm25` is one extra dep, 0.5 MB, gives meaningfully
   better text relevance. Phase 2 once corpus crosses ~50k papers in
   total.
4. **Pagination** — current tools cap at k. If a researcher wants page
   2 of results, do they re-call with `offset`? Or do we just say
   "ask the LLM to ask for more if the first 10 weren't enough"?

---

## Phases

| # | Goal | Status |
|---|------|--------|
| 0 | Project scaffold (this doc, packaging, config, corpus loader, search skeletons, tests) | active |
| 1 | Wire MCP SDK in `server.serve()` — register tools, stdio loop | pending |
| 2 | Build cache end-to-end on the real ai4chem corpus, verify search quality | pending |
| 3 | First user: connect to Claude Desktop, dogfood for a week | pending |
| 4 | Add `physics` / `polymers` domain feeds as they appear | pending |
| 5 | BM25 upgrade if text relevance complaints surface | pending |
| 6 | PyPI release | pending |

---

## Code map

```
src/arxiv_radar_mcp/
├── __main__.py     # `arxiv-radar-mcp` entrypoint (--build-cache, serve)
├── config.py       # radar.toml loader + defaults
├── corpus.py       # Paper dataclass, loaders for github + local sources
├── embeddings.py   # build_cache, EmbeddingIndex, encode_query (sentence-transformers)
├── search.py       # search_text, search_semantic, search_hybrid (RRF), similar_to
└── server.py       # RadarServer holds state; tools call into search.py;
                    # MCP-SDK binding is the next thing to write
```

Tests under `tests/` cover the parts that have no model deps yet
(corpus, search_text). Embedding-dependent tests come once Phase 1 is
done and we can mock or use a tiny model.
