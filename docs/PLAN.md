# Plan & decisions — arxiv-radar-mcp

Living design doc. Sibling to `daily-arxiv-*` data forks; this repo is
**code only**, the data is read from the fork family.

---

## [РЕШЕНИЕ-001] Why an MCP server (not a website / API)

The radar plan from `arxiv-radar-chemistry` resolved Phase 6 as: a
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

## [РЕШЕНИЕ-003] Embedding model — `Qwen3-Embedding-4B` native 2560 dim (supersedes mxbai-large)

**Production default:** `Qwen/Qwen3-Embedding-4B` at native 2560 dim,
bf16, GPU-built. Selected after empirically benchmarking 7 caches against
the 14k chemistry corpus on 22 queries (12 generic + 10 paraphrased→target).
Full results in `docs/MODEL_BENCHMARKS.md`; this is just the headline.

| Model | dim | recall@1 | median | Cache (14k) |
|---|---|---|---|---|
| bge-small-en-v1.5 | 384 | 6/10 | 1 (Q5/Q6 ranks 15/13) | 21 MB |
| mxbai-embed-large-v1 | 1024 | 7/10 | 1 | 56 MB |
| Qwen3-Embedding-4B (matryoshka 1024) | 1024 | 8/10 | 1 | 56 MB |
| **Qwen3-Embedding-4B (native)** | **2560** | **9/10** | **1** | 140 MB |
| Qwen3-Embedding-8B (matryoshka 1024) | 1024 | 6/10 | 1 | 56 MB |
| Qwen3-Embedding-8B (native) | 4096 | 9/10 | 1 | 224 MB |
| harrier-oss-v1-0.6b | 1024 | 6/10 | 1 (Q10=13) | 56 MB |

Verdict: 4B native ties or beats 8B native on this corpus; the smaller
model is "denser per dim" and degrades less under matryoshka truncation
(4B 2560→1024 loses 1 recall@1, 8B 4096→1024 loses 3). Cache ~140 MB
for 14k papers, scales to ~700 MB at 70k. Apache-2.0.

**Onboarding default in code (`config.py:EmbeddingsConfig.model`):**
still `mixedbread-ai/mxbai-embed-large-v1` — works on CPU, no 9 GB
download, lets a fresh checkout build a cache without surprises. Switch
to Qwen3-4B via `radar.toml`:

```toml
[embeddings]
model = "Qwen/Qwen3-Embedding-4B"
batch_size = 32   # bf16 on a 12 GB GPU
```

**Fallback when no GPU available:** Qwen3-4B at matryoshka 1024 dim
(8/10 recall@1, 56 MB cache) beats mxbai-large and is the right
no-GPU choice once the 9 GB weights download is acceptable. Below that,
mxbai-large remains the legacy CPU baseline.

**Build cost on gomer (RTX 4070 12 GB):** ~7 min for Qwen3-4B native
2560, vs 1m24s for mxbai-large. CPU build of Qwen3-4B is infeasible
(11 min/batch on a non-MKL CPU = ~100h ETA).

Encoding target: `title + "\n\n" + abstract`. Title-only loses context;
full body would need chunking and arXiv abstracts fit one passage in
the 512-token window. **Critical Qwen3 gotcha:** ships with
`max_seq_length=32768`; must be overridden to 512 or batches pad to 32k
tokens (30-40× slower).

Models trained with explicit prefixes (Qwen3, mxbai, BGE, E5) are
auto-prefixed via `embeddings.py:_QUERY_PREFIX` / `_PASSAGE_PREFIX`.
Qwen3 prefix: `"Instruct: Given a web search query, retrieve relevant
passages that answer the query\nQuery: "`. Forgetting them silently
costs 5–15% recall.

**Three encoder gotchas baked into `Encoder._ensure_loaded`** (any
regression here costs 3-50× wall time and is hard to diagnose without
profiling):

1. **Explicit `model.to(dtype=torch.bfloat16)` after `SentenceTransformer(...)`.**
   transformers 5.7+ no longer unifies dtypes implicitly: weights load
   as bf16 (HF safetensors), activations stay fp32 by default →
   `RuntimeError: expected mat1 and mat2 to have the same dtype, but
   got: float != c10::BFloat16` on first forward through a `Linear`
   layer. CPU path stays fp32 — bf16 on CPU has no tensor-core path
   and is much slower.

2. **`max_seq_length` must be set per-call, not at load time.** Qwen3
   ships at 32 768; abstract encoding needs 512, fulltext-chunk encoding
   needs 4096 per the bucketing scheme. `Encoder.encode_passages` /
   `encode_query` accept `max_seq_length=` and apply it on the model
   before each batch. (Mutating the model attribute is fine — it's
   just a local sequence-truncation hint, not a graph rebuild.)

3. **`_ensure_loaded` must be thread-safe.** Refresh loop, warm-up task,
   and reindex jobs all live on different anyio worker threads. Without
   a lock, two of them can race the load and end up with **two**
   SentenceTransformer instances in GPU memory — 2× 8 GB on a 12 GB
   card spills to host memory and slows every subsequent encode 3-50×
   per bucket. Standard double-checked locking pattern: fast path no
   lock, lock, re-check, load, publish only when complete.

## [РЕШЕНИЕ-004] Cache layout

Single npy + index.json sidecar:
- `<cache_dir>/embeddings.npy`  shape (N, dim) float32, L2-normalized
- `<cache_dir>/index.json`      `{model, dims, n, row_for: {arxiv_id: int}}`

`dim` matches the model's encode-time output: 1024 for mxbai/bge/Qwen3
matryoshka-1024, **2560 for the production Qwen3-4B native**, 4096 for
Qwen3-8B native. Stored as float32 even when the encoder runs in bf16
on GPU — fp32 on disk simplifies cosine math and is the difference
between 70 MB and 140 MB at 14k records (acceptable).

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

## [РЕШЕНИЕ-010] No `search_hybrid`, no reranker as public tool

Both `search_hybrid` (RRF over text+semantic) and the cross-encoder
reranker (`BAAI/bge-reranker-base`) were evaluated. On Qwen3-4B-native
(our production bi-encoder, 9/10 recall@1) **both are no-ops or
slightly negative**:

| Bi-encoder | Cosine avg rank | + Rerank avg rank | Net |
|---|---|---|---|
| bge-small | 4.7 | **1.7** | **HUGE rescue** (Q5 #15→#1, Q6 #13→#2) |
| mxbai-large | 1.8 | 2.0 | slight degradation |
| **Qwen3-4B-native** | **1.2** | 2.0 | slight degradation |

Hybrid + rerank are **rescue mechanisms for weak bi-encoders**. On a
strong one they ceiling at avg rank ~2.0 — same as raw cosine. Adding
them to the public tool surface adds cognitive load (LLM must pick
between text/semantic/hybrid for every query) for zero precision gain.

**Decision:** drop `search_*_hybrid` from the MCP tool catalog
entirely. `Reranker` class stays in the codebase (it's tested, ~30 LOC,
useful if we ever swap bi-encoders down) but is not wired to any tool.
If usage data later shows that exact-term queries (chemical formulas,
model names, years) are missed by semantic ranking, we'll add hybrid
back as `search_*_hybrid` — but YAGNI now.

This supersedes the earlier "reranker default OFF" formulation: not
"off by config", just **not exposed as a tool** at all.

## [РЕШЕНИЕ-011] Multi-source: scaling roadmap

User plans 4× repos × ~14k abstracts each + a personal literature
corpus (PDFs → text). Worst-case ~100k records before any pruning.

What works as-is on 100k:
- `Config.sources` is already a list; adding a feed = config-only
- Per-source shard cache (`<cache_dir>/shards/<source>/`) means new
  source = fetch only its shards, no re-fetch of existing
- `search_text` linear scan: ~100 ms on 100k titles+abstracts. Fine
- `matrix @ vec` brute-force cosine: ~80 ms on 100k × 1024-dim float32

What needs work as we grow:
- Single monolithic `embeddings.npy` means changing the model or
  re-fetching one source forces a full re-encode. Acceptable up to
  ~50k; beyond that we should split per-source `.npy` shards and
  concat into the in-memory matrix at startup. **YAGNI until 2nd
  source exists** — premature now
- At ~500k records `matrix @ vec` crosses 500 ms, time to switch to
  FAISS-IVF or hnswlib. Not before
- Personal-literature loader (PDF → text → record) is its own
  `SourceConfig.type = "literature"` feed; deferred until we have the
  PDFs

## [РЕШЕНИЕ-012] Self-contained repo, arXiv-only scope

This repo handles **arXiv content end-to-end**: abstracts (via the
`daily-arxiv-*` fork family) and on-demand full text (via
`arxiv.org/html` and `arxiv.org/e-print`). One server, one tool
catalog, one Docker image. No dependency on, or cross-reference to,
sibling repos in the workspace.

Scope boundary: PDF-only papers (~10-15% of arXiv corpus, mostly
pre-2017 scans / non-LaTeX submissions) fail with a clear error
message. Adding a PDF parser would pull in a 2 GB MinerU dep that
isn't justified for the long-tail miss rate. If a user really needs
those papers indexed, that's outside this server's scope.

Heavy alternative platforms (Heta — AGPL-3.0 viral copyleft, 4
stateful services Postgres+Milvus+Neo4j+MinIO) were considered and
rejected: overkill for a 70k-records-max scale and incompatible with
this repo's MIT-friendly licensing.

## [РЕШЕНИЕ-013] Fulltext fetch — HTML → LaTeX cascade, no PDF

For each arxiv_id requested by the user (via `fetch_papers`), try
sources in order, stop at first success:

1. **`GET arxiv.org/html/<id>`** — arXiv-rendered HTML from author's
   LaTeX. Equations preserved as `<math><annotation
   encoding="application/x-tex">...</annotation></math>` (we extract
   the inline LaTeX). Coverage ~70-80% on recent (2020+) papers.
2. **`GET arxiv.org/e-print/<id>`** — LaTeX source tarball. Run
   `pylatexenc.LatexNodes2Text` to expand macros, drop comments,
   resolve `\input{}` mults, leave equations as inline `$...$`.
   Coverage ~85-90% of all arxiv submissions.
3. **Fail with explicit reason.** Paper is PDF-only on arXiv (~10-15%
   of corpus, mostly pre-2017 scans or non-LaTeX submissions). Error
   says "PDF parsing not supported in this server" — the user knows
   to handle that paper outside the tool.

Why this order, not PDF-first or PDF-included:
- HTML and LaTeX are **structured text**: section headings, paragraphs,
  inline equations preserved cleanly. Embedding models trained on
  arXiv-flavored corpora (Qwen3, mxbai, BGE) handle inline LaTeX math
  naturally — no preprocessing needed beyond strip/normalize.
- PDF + MinerU is heavy (~2 GB dep), slow (1-2 min/PDF on CPU), and
  loses content (equations garbled, reading order wrong on multi-col).
  At 85%+ HTML/LaTeX coverage on our recent chemistry corpus, the 10-15%
  shortfall is a worse trade than the dependency weight.
- Empirical reference: arxiv URL-extraction benchmarks show LaTeX+HTML
  combined F1=0.69 vs HTML-only 0.65 vs text-from-PDF much lower
  (Toward Robust URL Extraction for Open Science, arXiv 2025).

**Deps:** `selectolax` (HTML parsing, ~3 MB) + `pylatexenc` (~1 MB)
both required, not opt-in extras. Lightweight enough to ship as core.

**Cache:** `<cache_dir>/fulltext/sources/<arxiv_id>.md` plus
`.meta.json` with `{source: "html"|"latex", fetch_time, n_chars,
n_chunks_after_split}`. Reused by reindex without re-fetching.

If 10-15% PDF-only coverage becomes a real problem we'll add a PDF
parser as an opt-in extra. Until then: clean error and we move on.

## [РЕШЕНИЕ-014] Two indexes: abstracts and fulltext, no merge

The abstract corpus (~14k papers, growing) and the fulltext corpus
(papers the user explicitly enriched, on the order of tens) are
**different objects** and stay in separate indexes:

```
<cache_dir>/
  abstracts/
    embeddings.npy              # (N_papers, 2560)  — title + abstract, max_seq=512
    index.json                  # {model, dims, n, max_seq_length, row_for: {arxiv_id: int}}
  fulltext/
    embeddings.npy              # (N_chunks, 2560)  — heading-chunked, max_seq=4096
    index.json                  # {model, dims, n, max_seq_length, chunks: [{arxiv_id, section, chunk_idx, n_chars}, ...]}
    sources/<arxiv_id>.md       # cached full markdown (one file per enriched paper)
    sources/<arxiv_id>.meta.json
  jobs/<job_id>.json            # see [РЕШЕНИЕ-015]
```

`RadarServer` holds both:
```python
self.abstract_index = EmbeddingIndex(...)         # always present
self.fulltext_index = EmbeddingIndex(...) | None  # None until first reindex
```

**MCP tool surface (15 total):**

```
# Abstracts (6)                        — operate on abstract index
search_abstract_text(q, k, domain, tag)
search_abstract_semantic(q, k, domain, tag)
similar_to_abstract(arxiv_id, k)
paper_info(arxiv_id)                   # metadata + fulltext-status (was get_paper)
list_tags()
list_domains()

# Fulltext (3)                          — operate on fulltext index
search_paper_text(q, k)                # returns [{arxiv_id, section, snippet, score}]
search_paper_semantic(q, k)
similar_to_paper(arxiv_id, k)          # mean-of-chunks per paper

# Admin (6)
fetch_papers([arxiv_ids])              → {job_id}    # bg: download+parse+chunk+save
reindex()                              → {job_id}    # bg: full re-encode of all sources
refresh_abstracts(force_full=false)    → {job_id}    # bg: pull shards + refresh abstracts
job_status(job_id)                     → {state, progress, result?, error?}
job_list()                             → [{job_id, kind, state, ...}]
list_enriched()                        → [arxiv_id, ...]
```

Removed from earlier scaffolds: `recent` (no good standalone use case
without temporal filtering), `*_hybrid` (see [РЕШЕНИЕ-010]),
`paper_status` (subsumed by `paper_info`'s extended payload).

When `fulltext_index is None` (no reindex yet), `search_paper_*` and
`similar_to_paper` return an explicit error: "fulltext index empty,
run fetch_papers + reindex first" — not an empty list (which would
mislead the LLM into "no results").

`tag` and `domain` filters on abstract searches are pre-search corpus
restriction, not embedding-level signal. Tags are not encoded into
embeddings (would dilute semantic signal); they're a sub-corpus
selector applied before ranking.

**Chunking:** split fulltext markdown by `## headings`, `max_tokens=4096`,
paragraph-aligned overlap of ~12% (≈ 500 tokens carry-tail between
adjacent sub-chunks of the same section). Each chunk row carries
`(arxiv_id, section_name, chunk_idx)` so search results attribute
"found in Methods of paper X". See `docs/MODEL_BENCHMARKS.md` for the
empirical run that fixed `max_tokens=4096` (the original 12 288 made
long-bucket encoding 51 s/chunk on RTX 4070 — 96% of reindex wall time
on 12% of chunks).

**Do not raise `max_tokens` back up.** The 12 288 trial was kept long
enough to hit a real benchmark, and 4096 is strictly better on:
encode wall time (3.3× faster), retrieval granularity (top-k now points
at sub-section level, not a section-as-a-whole average), and overlap
captures boundary answers anyway. A larger window only helps if a
single query needs >4k tokens of context to be answered correctly,
which we have not observed.

**Junk section filter** ([fulltext_index.is_junk_section]):
`References`, `Acknowledgments`, `Bibliography`, `Data availability`,
`Author contributions`, `Funding`, `Conflict of interest`,
`Supplementary [Material|Information]`, `Appendix [A-Z]` are demoted
in `search_paper_*`. They have high keyword density (cited papers,
authors, affiliations) but rarely match user intent. `Header` deliberately
stays clean — it carries title + abstract, often the right hit. Filter
oversamples top-k×4 candidates, returns clean first; junk only fills
the tail when filtering would leave fewer than k results.

## [РЕШЕНИЕ-015] Async jobs for fetch_papers and reindex

Synchronous MCP tool calls block the LLM. `fetch_papers([10 ids])` is
~30 seconds (HTTP + parse), `reindex()` after 20 enriched papers is
~5 minutes CPU / 30 seconds GPU. Blocking the conversation that long
breaks the UX.

Both return immediately with `{job_id}`. The LLM polls `job_status`,
reports progress to the user, or moves on and checks back. Stateful
across server restarts: jobs persist to `<cache_dir>/jobs/<job_id>.json`,
get reloaded on startup (anything left in `running` is marked
`orphaned` — the server didn't survive its own job).

**Job state machine:**
```
pending → running → done | failed | orphaned
```

**Lockfile:** `<cache_dir>/fulltext/.reindex.lock` prevents two
concurrent reindex jobs from corrupting `embeddings.npy`. Second
reindex queued (or rejected — TBD during impl).

**Cleanup:** jobs older than 7 days deleted on server start.

**Implementation:** `concurrent.futures.ThreadPoolExecutor(max_workers=2)`
— enough for one fetch + one reindex in parallel. Long-term, if heavy
GPU work piles up, swap for an external worker (Celery/RQ) — but YAGNI.

## [РЕШЕНИЕ-016] Daily refresh — git pull + diff + atomic update

The `daily-arxiv-*` repos publish a fresh `papers-YYYY-MM.json` shard
each morning (GitHub Actions cron 0 0 * * *). Without an in-server
refresh loop, the abstract corpus drifts more stale every day. Manual
`--build-cache` puts that work on the user.

`refresh.py` runs inside the long-running backend on a configurable
schedule (`[refresh] interval_hours = 24`). One round:

1. **`git pull`** for sources whose `path` is a git working tree
   (sparse-checkout or full clone). The recommended setup on GPU hosts
   is sparse-clone of `data/` + `tags/` only — ~50 MB instead of ~500 MB
   that the full repo (with rendered docs/abstracts/) would weigh.
2. **Reload corpus** via existing `corpus.load_all()`.
3. **Diff** against `radar.papers`: emit `added` and `deleted` sets.
4. **Strategy decision:**
   - `full_rebuild=True` OR any deletions → re-encode all abstracts.
     Server / GPU mode: nightly full rebuild (~7 min on RTX 4070
     for 14k papers). Robust to upstream prunes.
   - else (incremental) → encode only `added` arxiv_ids, `np.concatenate`
     to existing matrix, merge into `row_for`. Local / CPU mode:
     ~10 sec per 50 new papers. Drifts when papers are pruned upstream
     — user runs `--build-cache` periodically to resync.
5. **Atomic write**: `embeddings.npy.tmp` + `index.json.tmp` → rename.
   Concurrent searches see either the old or new state, never torn.
6. **Hot-swap** `radar.abstract_index` and `radar.papers` in-place.

Encoder lock (`acquire_reindex_lock` from `JobRegistry`) is shared with
`reindex` — a single encoder instance can't serve two passes at once.
Refresh that collides with reindex is silently skipped this tick;
manual `refresh_abstracts` tool returns `{error: "encoder busy"}`.

Bootstrap behaviour at backend startup:
  * `abstract_index` exists and `embeddings.npy` younger than interval →
    sleep until next tick.
  * Otherwise → kick off an immediate full-rebuild refresh, then enter
    the regular loop.

Manual triggers:
  * **MCP tool `refresh_abstracts(force_full=false)`** — returns a job_id;
    Claude can poll with `job_status`.
  * **CLI `arxiv-radar-mcp --build-cache`** — synchronous full rebuild,
    same effect as `force_full=true` minus the job wrapper. Useful for
    initial provisioning.

Rejected designs:
  * Per-source incremental with deletion tracking — too fragile, every
    edge case (revisions, archive moves, fork resyncs) needs explicit
    handling. Full rebuild is dumb-and-cheap.
  * Cron inside the container — couples lifecycle of the backend to
    cron daemon presence. asyncio task in the same process is simpler.

## [РЕШЕНИЕ-009] Repo-local `tmp/` for throw-away scripts

Bash helper scripts (venv setup, smoke runs, ad-hoc one-shots) live in
`./tmp/` *inside this repo* — not in the parent project's shared `tmp/`.
The repo is a self-contained sibling to `daily-arxiv-*`, and stays
isolated: no scripts, paths, or state escape upward. Single exception:
the cross-project radar plan at `knowledge/arxiv_radar_plan.md` in the
parent — that's the rolling log across all the radar repos and is the
right place to record cross-cutting decisions.

`tmp/` is gitignored. If a script becomes load-bearing (re-run regularly,
referenced from docs), promote it out of `tmp/` into a real path
(`scripts/`, a Makefile target, or a `tools/` module).

---

## Outstanding work — next session pickup

The 2026-05-02 session ended with backend production-ready on gomer
(image `a0a8a85d1a41`, 14k abstracts indexed, refresh loop active,
HTTP transport via SSH tunnel working, 181 tests green, 5 commits
pushed to origin/main). Nothing below blocks Phase 3 dogfood — they're
quality-of-life, scale-validation, or hardening items for after the
first user runs.

Priority order (top first):

| # | Item | Size | Trigger / when |
|---|------|------|----------------|
| 1 | **Stress test reindex on 50-100 fetched papers** — extrapolation said ~30 min, want to confirm no OOM, no torn-write, no encoder lock starvation; capture real bucket distribution on a real corpus | 30 min run + diagnose | before Phase 3 dogfood — sets user expectations |
| 2 | **GitHub Actions CI** — pytest matrix on push/PR, py 3.11 + 3.12 | ~30 LOC YAML | before public traffic |
| 3 | **PyPI release** (Phase 6) | ~1 day — version policy, license review, README rendering | after dogfood feedback |
| 4 | **Multi-source feeds** (Phase 4) — add `daily-arxiv-physics` etc. as the forks materialize | config-only, 5 min per source | when fork repos exist |
| 5 | **BM25 upgrade for `search_*_text`** (Phase 5) — `rank_bm25` is 0.5 MB extra dep | ~0.5 day | only if real users complain about text-search relevance |
| 6 | **Operational hardening** — log rotation inside container (uvicorn → docker logs unbounded), disk monitoring on named volumes, backend health endpoint (`GET /healthz`), rolling-update path for backend restart without dropping live SSH-tunneled MCP sessions | ~1 day | when this graduates from personal lab to multi-user service |

Completed since this pickup:

* 2026-05-03 — README now has a 5-minute CPU quickstart, local Claude
  Desktop config examples, a troubleshooting section, and the tool count
  synchronized to the 15-tool `TOOL_SPECS` surface (`refresh_abstracts`
  included).

Minor cleanup also pending:

* `tmp/` accumulated 50+ scripts during the perf-tuning sessions. Most
  are one-shot probes; the load-bearing Docker setup scripts have been
  promoted to `scripts/` (`docker_init_volume.sh`,
  `docker_setup_source.sh`).
  Periodic prune: keep what's referenced from docs or scripts/, delete
  the rest. .gitignored so cleanup is local.
* Encoder warm-up only primes the **query** path (`encode_query`).
  First chunk encode after a cold restart still pays a couple-second
  bucket-load cost. Minor, but a `_warmup_encoder` extension to also
  encode one short chunk would close the gap.
* Two short HTML-parsing edge cases left to sample: the parser handles
  Pixtral-style `\part` wrappers (fix from 9c798a4) and SmolDocling-style
  flat sections; we have not stress-tested it on more exotic LaTeXML
  outputs (review papers with chapters, conference notes, books).
  Spot-check ~50 random papers when next pulling fresh shards.

Track follow-ups inline here rather than in a separate TODO file —
this section is the canonical "what's next" so a future session opens
docs/PLAN.md and sees the punch list.

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
| 0 | Project scaffold (this doc, packaging, config, corpus loader, search skeletons, tests) | done |
| 1 | Wire MCP SDK in `server.serve()` — register tools, stdio loop | done (2026-04-27) |
| 2 | Build cache end-to-end on the real chemistry corpus, verify search quality | done (2026-04-27) — 7 caches built on gomer; Qwen3-4B-native chosen ([РЕШЕНИЕ-003]); benchmarks in `docs/MODEL_BENCHMARKS.md` |
| 7 | Tool rename + remove hybrid + paper_info (was get_paper) — public API freeze for fulltext expansion | done (2026-05-01) |
| 8 | Fulltext fetcher (HTML+LaTeX cascade, [РЕШЕНИЕ-013]), chunker by headings ([РЕШЕНИЕ-014]) | done (2026-05-01) |
| 9 | Async jobs registry ([РЕШЕНИЕ-015]) + new MCP tools (search_paper_*, fetch_papers, reindex, job_status, job_list) | done (2026-05-01) |
| 10 | Build GPU Docker image on gomer, run scenario tests for fulltext search | done (2026-05-01) — 22-chunk reindex passes, all 3 search modes return relevant hits |
| 11 | Production deploy: streamable-HTTP transport + SSH-tunnel proxy + long-running backend | done (2026-05-01) — `--remote user@host`, container `--restart unless-stopped`, perimeter via SSH keys |
| 12 | Daily auto-refresh of abstract corpus ([РЕШЕНИЕ-016]) — `refresh.py`, scheduler loop, MCP tool refresh_abstracts | done (2026-05-01) |
| 13 | Production polish from W1 e2e: junk-section filter, chunker max_tokens 12 288→4 096 + paragraph overlap, encoder warm-up, bf16 cast, thread-lock | done (2026-05-02) — reindex 954 s → 287 s on 16 papers; junk hits 7/15 → 0/15. See `docs/MODEL_BENCHMARKS.md` |
| 3 | First user: connect to Claude Desktop, dogfood for a week | pending — gated on 7-13 |
| 4 | Add `physics` / `polymers` domain feeds as they appear | pending |
| 5 | BM25 upgrade if text relevance complaints surface | pending |
| 6 | PyPI release | pending |
| — | Non-arXiv content (PDFs without arxiv_id, video, books) | out of scope — this server handles arxiv only ([РЕШЕНИЕ-012]) |

---

## Code map

```
src/arxiv_radar_mcp/
├── __main__.py        # `arxiv-radar-mcp` entrypoint (--build-cache, serve)
├── config.py          # radar.toml loader + defaults
├── corpus.py          # Paper dataclass, loaders for github + local sources
├── embeddings.py      # Encoder (lazy bi-encoder + prefixes), EmbeddingIndex, build_cache (abstracts)
├── reranker.py        # Reranker class (kept, not wired to tools — see РЕШЕНИЕ-010)
├── search.py          # search_abstract_text/semantic, similar_to_abstract (abstract index ops)
├── fulltext.py        # HTML/LaTeX fetcher (selectolax + pylatexenc); source-cascade per arxiv_id
├── chunker.py         # split markdown by ## headings; sub-split sections >max_seq_length
├── fulltext_index.py  # reindex (full rebuild); search_paper_text/semantic; similar_to_paper
├── jobs.py            # JobRegistry: ThreadPoolExecutor + persistent jobs/<id>.json + lockfile
├── refresh.py         # daily refresh: git pull → diff → encode new → atomic swap ([РЕШЕНИЕ-016])
├── proxy.py           # local stdio→remote-HTTP proxy with SSH tunnel (--remote mode)
├── fulltext_cli.py    # `python -m arxiv_radar_mcp.fulltext_cli` — fetch helper for in-container use
├── reindex_cli.py     # `python -m arxiv_radar_mcp.reindex_cli` — reindex helper for in-container use
└── server.py          # RadarServer holds {abstract_index, fulltext_index, jobs};
                       # TOOL_SPECS (15 tools); serve() stdio + serve_http() streamable-HTTP;
                       # _refresh_loop() asyncio background task

Dockerfile             # GPU image: pytorch/cuda12.4 + this code, ~10 GB
.dockerignore          # strips tests/tmp/docs/.venv from build context
scripts/
├── docker_build.sh         # build image on gomer (`docker --context gomer build`)
├── docker_serve_mcp.sh     # stdio MCP bridged to local for Claude Desktop
├── docker_fetch.sh         # one-shot enrich: fetch_papers via CLI, no MCP
├── docker_reindex.sh       # one-shot rebuild fulltext index on GPU
└── docker_entrypoint.sh    # in-container dispatcher: mcp / build-cache / fetch / reindex
```

Tests under `tests/` cover what doesn't require the live encoder:
corpus loader, text search, TOOL_SPECS shape, dispatcher routing,
chunker, fulltext-source cascade with mocked httpx, jobs lifecycle +
persistence, fulltext_index search primitives with a fake encoder,
RadarServer end-to-end against a synthetic 3-paper corpus. 110 tests
green as of 2026-05-01.

Encoder-dependent paths (live semantic search, similar_to_*, real
reindex on GPU) are covered by gomer scenario scripts in
`tmp/gomer_scenario.sh` rather than mocked unit tests — the gap between
a mocked encoder and the real one is exactly what bites at integration
time. Full results in `docs/MODEL_BENCHMARKS.md`.
