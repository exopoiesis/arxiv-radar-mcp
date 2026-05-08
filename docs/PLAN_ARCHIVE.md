# Plan archive — arxiv-radar-mcp

> Historical record of decisions, completed phases, and resolved UX issues.
> Active plan lives in `docs/PLAN.md`. Items here are kept for context — why
> the codebase looks the way it does, what was tried and rejected, what was
> a real bug versus invented overcaution.
>
> Move items here when they're **fully implemented + tests cover them + no
> follow-up actions remain**. Anything still actionable (even if the design
> is decided) stays in `PLAN.md`.

---

## Resolved UX issues

| # | Title | Where | Resolution |
|---|-------|---|---|
| U1 | `job_status` returns stale `running 0%` long after completion | `jobs.py:JobRegistry.get` | done 2026-05-08 — `get()` re-reads the persisted `jobs/<id>.json` when in-memory state is `pending`/`running` but disk has progressed to a terminal state. Cheap (small JSON) and only fires on the path where memory could be lagging; terminal in-memory states stay authoritative. Tests: `test_get_rereads_disk_when_inmemory_state_is_running`, `test_get_does_not_reread_disk_for_terminal_inmemory_state`, `test_get_disk_reread_handles_corrupted_json`. |
| U2 | No pre-flight validation for fetch_papers — PDF-only IDs only fail after launch | `server.py:validate_arxiv_ids` + `fulltext.py:probe_html_available` | done 2026-05-08 — new MCP tool `validate_arxiv_ids(arxiv_ids)` HEAD-probes `arxiv.org/html/<id>` on the same throttled rate limiter as the live fetcher and returns `{ok, pdf_only, n_total, n_cached}`. Already-cached papers count as ok without a probe so dogfood batches don't burn ToS budget on already-enriched IDs. Tests in `test_server.py` + `test_fulltext.py::probe_*`. |
| U3 | HTML "ok" with <5000 chars is silently a stub | `fulltext.py:_fetch_html` | done 2026-05-08 — root cause was U10, see below. The "<5000 chars + magic phrase" sniff is now superseded by the structural echo detector. |
| U5 | `list_tags()` dumps 200+ tags, no pagination / sort filter | `server.py:list_tags` | done 2026-05-08 — added `head_limit`, `min_count`, `prefix` args to the tool. Default behaviour unchanged. Tests in `test_server.py`. |
| U6 | Rate limit / retry on 429-503 | `fulltext.py` | done 2026-05-08 — see [РЕШЕНИЕ-018]. Patch applied + hot-deployed to gomer container 13:42 UTC. To be baked in next `docker_build.sh` rebuild. |
| U8 | Local proxy does not survive backend container restart | `proxy.py:_run_bridge` + new `_bridge_loop` | done 2026-05-08 (Option B from plan) — stdio is now opened **once** for the lifetime of the proxy; the bridge runs in a reconnect loop that opens a fresh streamable-HTTP session, runs the pipes until disconnect, sleeps with exponential backoff, and retries. After `max_consecutive_failures` of pure connection failures (default 10), the proxy exits non-zero so Claude Code's MCP supervisor respawns us (= falls through to Option A). The successful-handshake path resets the counter so a flaky-but-recovering backend doesn't burn through the budget. Events arrive over the **new** SSE channel after each reconnect (server-initiated notifications during the disconnect window are lost — only ones after reconnect flow). Tests: `test_bridge_loop_exits_after_max_consecutive_connection_failures`, `test_bridge_loop_resets_counter_on_successful_session`, `test_bridge_loop_caps_backoff_at_max`. |
| U9 | `fetch_papers` has no `force=True` to re-fetch a stub | `server.py` + tool spec | done 2026-05-08 — `fetch_papers(arxiv_ids, force=False)` surfaces the existing `fetch_and_save(force=True)` capability so a stub fetch (e.g. `2411.12261` 987-char body) can be recovered without shell access. Tests in `test_server.py`. |
| U10 | arXiv HTML "skeleton-only" render: section headings without bodies | `fulltext.py:_fetch_html` + new `_looks_like_echo_skeleton()` + `_normalize_heading_for_compare()` | done 2026-05-08 (~2 hours actual). **Status:** patch applied + hot-deployed + verified in production at 14:24 UTC 2026-05-08. After clearing the 4 cached stubs and re-fetching: Myers FeSe `2411.12261` html 987 → **latex 29978** chars (30×); Cu-vac `2510.26991` 720 → **latex 35783** (50×); Fe2S2 `2604.21613` 1796 → **latex 56939** (32×); Yang Hubbard `2512.16803` 6433 → latex 63866 (was already partial OK). Iteration story: first detector pass missed 3/4 because real arXiv `_html_to_markdown` output uses numbered/Roman section labels (`## 1 Introduction`, `## I Methods`, `### Appendix A Foo`) that didn't text-equal the body's plain `Introduction` echo; added `_normalize_heading_for_compare()` (strip leading `[0-9]+\\.?\\s+`, `[ivxlcdm]+\\.?\\s+`, `appendix [a-z]+\\s+` prefixes, applied twice for `Appendix A → A → empty`); 6/6 smoke tests pass on 4 real stubs + 2 negative controls. **Discovered 2026-05-08 by `docker exec cat <id>.md` on the 4 stub papers**. Pattern (4/4 stubs identical): `arxiv.org/html/<id>` returns a multi-kilobyte HTML body with proper structure — title, authors, affiliations, all `\section{}` headings present — but **every section body is just the heading echoed** (`## Abstract\n\nAbstract\n\n## 1 Introduction\n\nIntroduction\n\n...`). Body of every section blank. **Cause:** authors use LaTeX with separate files (`\input{intro.tex}` `\input{methods.tex}`) and arXiv's LaTeXML rendering pipeline appears not to resolve those `\input{}` references — outer skeleton renders, content stays empty. **Fix:** detect "echo skeleton" — when post-conversion markdown has more `## H` headings than non-trivial paragraphs, OR `body_chars / heading_count < 50`, treat as failed render → fall through to e-print path. The e-print tarball includes the `\input{}`'d subfiles and `pylatexenc` resolves them locally, so e-print should succeed where HTML failed for exactly this pattern. Test: `tests/test_fulltext_echo_skeleton.py`. |
| U10b | `selectolax.traverse(include_text=True)` crosses sibling boundaries; heading-text echoes on decompose | `fulltext.py:_iter_descendants` (new) replacing direct `.traverse()` calls | done 2026-05-08 — root-cause discovered while writing U13 tests. The selectolax `Node.traverse()` call from `_node_to_markdown` was **not** limited to descendants of the starting node — `div.ltx_abstract.traverse()` happily yielded the next `<section>` as well, leaking the whole article into the abstract render. Compounding: when consumer called `n.decompose()` mid-walk, the descendants of `n` were still iterated, so heading text leaked as a body echo (`## Methods\n\nMethods\n\n...`). Fix: replaced with a manual DFS via `.iter(include_text=True)` that snapshots descendants up front, then on each yield re-checks the parent chain (`==`, NOT `is` — selectolax wraps the same DOM node in fresh Python objects on each `.parent` access; the C extension implements `__eq__` via the underlying node). The U10 echo-skeleton heuristic now sees clean structural output and stops false-positive-failing on properly-rendered papers. |
| U11 | `search_paper_*` snippets truncated at ~250-350 chars | `fulltext_index.py:_snippet` + tool specs | done 2026-05-08 — added `snippet_chars` parameter (default 240) to `search_paper_text` and `search_paper_semantic`; threaded through the underlying `_snippet(length=...)` call. Researchers can request longer snippets when extracting recipes/numeric parameters. Tests in `test_fulltext_index.py`. |
| U12 | `paper_info` abstract truncated to 600 chars | `server.py:paper_info` + `_paper_payload` | done 2026-05-08 — added `full_abstract: bool = False` parameter. Default behaviour unchanged (compact 600-char + ellipsis); explicit `full_abstract=true` returns the untruncated string. Tests in `test_server.py`. |
| U13 | HTML→markdown loses `<a href="...">` URLs | `fulltext.py:_node_to_markdown` | done 2026-05-08 — anchor handling added: external links emit `[text](url)`, bare URLs emit `<url>`, internal fragments (`#fig-1`) and `javascript:` are stripped to plain text only. DOI / repo / dataset URLs now survive HTML→markdown→chunker. Tests in `test_fulltext.py`. |

---

## Completed phases

| # | Goal | Done |
|---|------|------|
| 0 | Project scaffold (this doc, packaging, config, corpus loader, search skeletons, tests) | done |
| 1 | Wire MCP SDK in `server.serve()` — register tools, stdio loop | 2026-04-27 |
| 2 | Build cache end-to-end on the real chemistry corpus, verify search quality | 2026-04-27 — 7 caches built on gomer; Qwen3-4B-native chosen ([РЕШЕНИЕ-003]); benchmarks in `docs/MODEL_BENCHMARKS.md` |
| 4 | Add `chemical_engineering` / `electrochemistry` / `physics` / `polymer` / `sulfide_materials` domain feeds | 2026-05-06 |
| 7 | Tool rename + remove hybrid + paper_info (was get_paper) — public API freeze for fulltext expansion | 2026-05-01 |
| 8 | Fulltext fetcher (HTML+LaTeX cascade, [РЕШЕНИЕ-013]), chunker by headings ([РЕШЕНИЕ-014]) | 2026-05-01 |
| 9 | Async jobs registry ([РЕШЕНИЕ-015]) + new MCP tools (search_paper_*, fetch_papers, reindex, job_status, job_list) | 2026-05-01 |
| 10 | Build GPU Docker image on gomer, run scenario tests for fulltext search | 2026-05-01 — 22-chunk reindex passes, all 3 search modes return relevant hits |
| 11 | Production deploy: streamable-HTTP transport + SSH-tunnel proxy + long-running backend | 2026-05-01 — `--remote user@host`, container `--restart unless-stopped`, perimeter via SSH keys |
| 12 | Daily auto-refresh of abstract corpus ([РЕШЕНИЕ-016]) — `refresh.py`, scheduler loop, MCP tool refresh_abstracts | 2026-05-01 |
| 13 | Production polish from W1 e2e: junk-section filter, chunker max_tokens 12 288→4 096 + paragraph overlap, encoder warm-up, bf16 cast, thread-lock | 2026-05-02 — reindex 954 s → 287 s on 16 papers; junk hits 7/15 → 0/15. See `docs/MODEL_BENCHMARKS.md` |
| 14 | Incremental fulltext reindex ([РЕШЕНИЕ-017]) — classify new/changed/deleted/unchanged, encode only the delta, fall back to full on model mismatch or `force_full=True` | 2026-05-08 — 7 new tests, 190 total green |

---

## Resolved decisions (РЕШЕНИЯ-001..017)

All implemented and stable in code. Кept here for design rationale —
"why does the codebase look this way" answers.

### [РЕШЕНИЕ-001] Why an MCP server (not a website / API)

The radar plan from `arxiv-radar-chemistry` resolved Phase 6 as: a
machine-readable interface for AI assistants — Claude Desktop, Claude
Code, Cursor — over the curated corpus, not yet another HTTP API.

MCP gives one tool definition that any compliant client picks up;
search becomes a first-class capability inside the assistant the
researcher already uses.

### [РЕШЕНИЕ-002] One repo, many domains

Sources are listed in `radar.toml`. Adding another `arxiv-radar-*`
source is config-only — no code change. The loader concatenates per-domain
records into one in-memory map keyed by arxiv_id and tags each Paper
with its source domain so search results can be filtered or attributed.

### [РЕШЕНИЕ-003] Embedding model — `Qwen3-Embedding-4B` native 2560 dim (supersedes mxbai-large)

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

### [РЕШЕНИЕ-004] Cache layout

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

### [РЕШЕНИЕ-005] Hybrid = RRF, not weighted sum

Reciprocal Rank Fusion with `k=60` (canonical). Weighted sum requires
score normalization between text (BM25-ish counts) and semantic (cosine
0..1) which is fragile. RRF only sees ranks, blends robustly, and the
constant 60 has held up across IR literature for two decades.

### [РЕШЕНИЕ-006] Source fetch via raw.githubusercontent

`type=github` source pulls `data/papers-*.json` over HTTPS one shard at
a time, caches under `<cache_dir>/shards/<source-name>/`. Two reasons:
1. No git clone of a 14k-file repo just to read the JSONs
2. Rebuild-cache step naturally refreshes shards (we just delete the
   shard dir and re-fetch)

Rate-limiting: GitHub raw URLs are unauthenticated, generous limit. If
we hit it, switch to authenticated API or carry an `ETag` header.

### [РЕШЕНИЕ-007] Server is sync, tool dispatch is sync too

In-memory corpus + numpy is fast enough that there's no win from async.
The MCP SDK's stdio loop is async, but tool bodies stay synchronous —
they just compute and return. Avoids a parallel mental model for no
benefit.

### [РЕШЕНИЕ-008] No persistence layer

Everything is rebuilt from `data/papers-*.json` on each `--build-cache`
run. No SQLite, no full-text index file, no app state to migrate. The
data forks are the source of truth.

### [РЕШЕНИЕ-009] Repo-local `tmp/` for throw-away scripts

Bash helper scripts (venv setup, smoke runs, ad-hoc one-shots) live in
`./tmp/` *inside this repo* — not in the parent project's shared `tmp/`.
The repo is a self-contained sibling to `arxiv-radar-*`, and stays
isolated: no scripts, paths, or state escape upward. Single exception:
the cross-project radar plan at `knowledge/arxiv_radar_plan.md` in the
parent — that's the rolling log across all the radar repos and is the
right place to record cross-cutting decisions.

`tmp/` is gitignored. If a script becomes load-bearing (re-run regularly,
referenced from docs), promote it out of `tmp/` into a real path
(`scripts/`, a Makefile target, or a `tools/` module).

### [РЕШЕНИЕ-010] No `search_hybrid`, no reranker as public tool

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

### [РЕШЕНИЕ-011] Multi-source: scaling roadmap

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

### [РЕШЕНИЕ-012] Self-contained repo, arXiv-only scope

This repo handles **arXiv content end-to-end**: abstracts (via the
`arxiv-radar-*` fork family) and on-demand full text (via
`arxiv.org/html` and `arxiv.org/e-print`). One server, one tool
catalog, one Docker image. No dependency on, or cross-reference to,
sibling repos in the workspace.

Scope boundary: PDF-only papers (~10-15% of arXiv corpus, mostly
pre-2017 scans / non-LaTeX submissions) fail with a clear error
message. Adding a PDF parser would pull in a 2 GB MinerU dep that
isn't justified for the long-tail miss rate. If a user really needs
those papers indexed, that's outside this server's scope.

(Note 2026-05-08 dogfood: empirical PDF-only rate on a 45-paper batch was 26%, not 10-15%. See active PLAN.md U7 about reconsidering this scope choice.)

Heavy alternative platforms (Heta — AGPL-3.0 viral copyleft, 4
stateful services Postgres+Milvus+Neo4j+MinIO) were considered and
rejected: overkill for a 70k-records-max scale and incompatible with
this repo's MIT-friendly licensing.

### [РЕШЕНИЕ-013] Fulltext fetch — HTML → LaTeX cascade, no PDF

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

### [РЕШЕНИЕ-014] Two indexes: abstracts and fulltext, no merge

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

### [РЕШЕНИЕ-015] Async jobs for fetch_papers and reindex

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

### [РЕШЕНИЕ-016] Daily refresh — git pull + diff + atomic update

The `arxiv-radar-*` repos publish a fresh `papers-YYYY-MM.json` shard
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

### [РЕШЕНИЕ-017] Incremental fulltext reindex by default

`reindex()` was full-rebuild only through 2026-05-07: every call re-chunked
and re-encoded all enriched papers, even ones unchanged since the last
index. On a 50-paper corpus that's 5-30 minutes; after a single
`fetch_papers([new_id])` the user paid that cost just to add ~8 chunks.
65 minutes was empirically observed on the 36k-abstract sister index
(separate code path, but same problem class).

The new dispatcher in `fulltext_index.reindex(*, incremental=True)`:

1. Loads the existing `fulltext/index.json` (None if missing).
2. Classifies every paper into **new / changed / deleted / unchanged**:
   - new = `<id>.md` exists, no chunks for it in index
   - changed = `<id>.md.mtime > meta.indexed_at + 1.0s` tolerance
   - deleted = chunks for it in index, no `<id>.md` on disk
   - unchanged = the rest
3. Falls back to full rebuild when: no existing index, model name
   mismatch (`index.model != encoder.model_name`), or
   `incremental=False` was passed explicitly.
4. Otherwise:
   - **noop** path (no changes at all) — re-loads from disk, returns;
     no encode call, no write.
   - **append-only** path (only new) — encodes new chunks, concatenates
     to the existing matrix, atomic write.
   - **mixed** path (deletions and/or changes) — drops affected rows
     via numpy fancy-indexing (which copies — releases the mmap),
     encodes new+changed, concat, atomic write.
5. `_stamp_meta` only touches `<id>.meta.json` for re-encoded papers,
   so unchanged papers keep their existing `indexed_at`.

Properties verified by tests:

* Unchanged paper rows are byte-identical across reindexes (read from
  disk via mmap, copied through fancy-indexing, written back unchanged).
* No-op reindex makes zero `Encoder.encode_passages` calls.
* Force-full path (`incremental=False`) re-encodes every paper even
  when no source changed — recovery hatch for corrupted indexes.

Trade-offs:

* Mtime-based change detection misses the case where a `.md` file is
  modified but its mtime is rewound below `indexed_at`. We tolerate this:
  `force_full=True` is the recovery hatch and the case is rare.
* Papers with no `meta.indexed_at` (legacy enriched corpora from before
  this change) are classified as **changed** so they're re-encoded once,
  after which their meta gets stamped and they settle.
* The mixed path holds the existing matrix in RAM (`np.array(... [survive_rows])`)
  before write — for our scale (tens of thousands of chunks × 2560 dim
  × 4 bytes ≈ 100 MB) this is fine. If we ever index millions of chunks
  we'll need a streaming concat path instead.

The `reindex` MCP tool exposes `force_full: bool = false`. Default
behaviour is incremental. `RadarServer.reindex(force_full=False)`
returns `{job_id, n_total, kind, strategy_planned}` so the LLM can tell
the user "incremental" vs "full" upfront.

This supersedes the earlier "Always full rebuild" note in
[РЕШЕНИЕ-014]; the two-index layout in that decision still stands, but
the encode strategy of the fulltext index is now incremental-by-default.
