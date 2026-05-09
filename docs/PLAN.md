# Plan & decisions — arxiv-radar-mcp (active)

Living design doc. Sibling to `arxiv-radar-*` data forks; this repo is
**code only**, the data is read from the fork family.

> **Historical record** (resolved decisions Р-001..Р-017, completed phases,
> fixed UX issues U3/U6/U10) → `docs/PLAN_ARCHIVE.md`. This file holds only
> what's still actionable.

> **Phase 3 done (2026-05-09):** `corpus_core` extracted to its own repo
> (see `docs/PLAN_CORE_EXTRACTION.md`). arxiv-radar-mcp now declares
> `corpus-core>=0.1.0` as a regular dependency; the standalone
> `arxiv-radar-gpu` Dockerfile uses parent build context to COPY the
> sibling repo. Production deploy of 2026-05-09 → 10 promoted the
> running `arxiv-radar-backend` container into the lab-corpus-mcp
> combined image (`exopoiesis/lab-corpus-gpu`) — 34,627 abstract
> embeddings + 466 fulltext chunks (51 papers) migrated from the
> `arxiv-radar-cache` volume into `/srv/arxiv-radar/cache/`. Single
> Qwen3-4B in VRAM shared between both backends; refresh loop runs
> nightly with `full_rebuild=false` so existing embeddings are
> preserved. See `lab-corpus-mcp/docs/DEPLOY.md` for the migration
> recipe and current operational layout.

---

## [РЕШЕНИЕ-018] arXiv rate-limit + retry on 429/503 (applied 2026-05-08)

`fetch_papers` was firing GETs back-to-back from `_do_fetch` in
`server.py:285-305`: a shared `httpx.Client` for connection reuse, but
no inter-request delay. Empirically a batch of 38 IDs ran in 65 sec
(~1.7 sec/paper end-to-end, peak ~1 req/0.85 sec when HTML+LaTeX
fallback both fire) — 3.5× faster than arXiv's published policy of
1 request / 3 sec. We did not catch a 429 on the 2026-05-08 dogfood
batches, but a) those were 38 + 7 IDs (small), b) arXiv banhammer
triggers on sustained ≥1k req/h, not on 76-request bursts. A 200+ ID
batch would almost certainly trip it.

Patch in `fulltext.py`:

1. Module-global throttle:
   ```python
   _RATE_LIMIT_S = 3.0
   _last_request_at = 0.0
   _rate_lock = threading.Lock()

   def _throttle() -> None:
       global _last_request_at
       with _rate_lock:
           now = time.monotonic()
           wait = _RATE_LIMIT_S - (now - _last_request_at)
           if wait > 0:
               time.sleep(wait)
           _last_request_at = time.monotonic()
   ```
   Module-level (not per-`Client`) so HTML→LaTeX fallback within one
   paper AND consecutive papers in a batch all see the rate limit.
   Threading lock because `JobRegistry` uses `ThreadPoolExecutor` and
   refresh-loop / fetch / reindex can race outbound GETs.

2. `_request_with_retry(client, url, max_attempts=3)` — calls
   `_throttle()` then `client.get()`, returns immediately on non-{429,
   503}, otherwise sleeps `Retry-After` (or exponential backoff
   3→6→12s) and retries. Final attempt's response is returned even
   if still 429 — caller decides what to do.

3. `_fetch_html` and `_fetch_eprint` switched their bare
   `r = client.get(url)` to `r = _request_with_retry(client, url)`.

4. `probe_html_available` (added for U2) reuses `_throttle()` so
   `validate_arxiv_ids` shares the same 1 req / 3 sec budget as live
   fetches.

Cache hits are unaffected: `fetch_and_save` checks the on-disk cache
before calling `fetch_paper`, so already-enriched IDs in a batch don't
pay the 3-sec wait.

Cost: a 200-ID batch goes from a worst-case 200 sec (current,
ban-risk) to ~600 sec (10 min, ToS-compliant). For our typical
30-50 ID batches we add ~90-150 sec, an acceptable trade for
not getting our gomer IP rate-limited.

Reference: `arxiv-radar-chemistry/tools/daily_arxiv.py:122` already
uses the canonical pattern via the `arxiv` python lib:
```python
client = arxiv.Client(page_size=100, delay_seconds=3.0, num_retries=5)
```
Our `fetch_papers` now matches that posture for the HTML/e-print
pulls.

Hot-deployed to gomer's `arxiv-radar-backend` container at 13:42 UTC
2026-05-08 via `tmp/deploy_arxiv_radar_throttle_patch.sh` — `docker cp`
of the patched `fulltext.py` + `docker restart`. Image was not
rebuilt; `pip install -e .` makes `src/` live so a Python-process
restart picks up the new code. Verified `_RATE_LIMIT_S = 3.0` present
in the running container, fulltext index reloaded (434 chunks, 48
papers).

**Persistence-risk follow-up (still pending — git+image):**
1. ~~`git -C arxiv-radar-mcp status` → confirm modifications still present~~ — done.
2. `git commit` (split into Р-018 throttle + U10 echo-skeleton + 2026-05-08 UX wave: U1/U2/U5/U8/U9/U11/U12/U13 + U10b selectolax DFS fix).
3. `git push` to `exopoiesis/arxiv-radar-mcp`.
4. `bash scripts/docker_build.sh` on gomer to bake patches into image.
5. `bash scripts/docker_serve_backend.sh` to restart container with the new image.

Until step 4-5 the image (= fresh container after any `docker rm`) reverts
to pre-patch code.

---

## Active UX issues — pending fix

Captured during a real research workflow on the Third Matter project on
2026-05-08 (10 question topics, 50 papers fetched across 5 batches, two
patch deployments, ten `search_paper_semantic` queries). Resolved items
moved to `PLAN_ARCHIVE.md`. Items below are still actionable, ordered by
severity.

| # | Title | Severity | Where | Fix size | Notes |
|---|-------|:---:|---|:---:|---|
| U4 | **Domain feed coverage gap for our use case (Fe-S sulfide chemistry)** | HIGH (project-specific) | `radar.toml` source feed selection upstream in `arxiv-radar-*` forks | external repo | Searched for known important authors/keywords: `Sundararaman` 0 hits, `Marx` 0 hits, `Roldan iron sulfide` 0 hits, `mackinawite` 0 hits, `Beyazay` 0 hits, `Muchowska` 0 hits. All these have arXiv preprints — they just aren't in the `arxiv-radar-*` filter results. Specifically: `mackinawite` is a centerpiece mineral of the Third Matter project but doesn't show up at all. **Two complementary fixes (in the *fork* not this server):** (a) keyword-trigger inclusion: any paper whose abstract contains a curated mineral term (mackinawite/pentlandite/greigite/pyrrhotite/troilite/chalcopyrite) auto-enters `sulfide_materials`; (b) author-whitelist feed: a small list of high-signal authors (Sundararaman, Marx, Roldan, Santos-Carballal, Behler, Csányi, Reuter, Andreussi) gets every preprint pulled regardless of category match. |
| U7 | **Project-level: PDF-only papers (12 in dogfood batch) need an out-of-band pipeline** | MEDIUM | scope choice (see [РЕШЕНИЕ-013] in archive — PDF intentionally not supported) | architectural | The dogfood session caught real research papers that are PDF-only on arXiv (e.g. `2512.14129` Yin pyrrhotite, the highest-value hit for the user's T11 test). [РЕШЕНИЕ-013] said PDF is out of scope because the failure rate is "10-15%" — but our session saw 26% (12/45). At project-specific corpora the rate may be higher than the arXiv average. **Options to consider:** (a) opt-in PDF parser via extra (`pip install arxiv-radar-mcp[pdf]` → MinerU or marker), gated by `radar.toml [fulltext.pdf] enabled = false`; (b) a thin `fetch_papers_pdf` companion tool that just downloads PDFs to `<cache_dir>/fulltext/pdf_pending/<id>.pdf` for the user to process out-of-band. Decision deferred — list of 12 papers from this session is in the user's project at `knowledge/lit_pdf_only_pending_2026-05-08.md`. The U2 `validate_arxiv_ids` tool now lets the LLM warn the user up front about pdf-only IDs in a batch, partial mitigation. |

---

## Outstanding work — pending pickup

Quality-of-life, scale-validation, or hardening items. Priority order
(top first):

| # | Item | Size | Trigger / when |
|---|------|------|----------------|
| 1 | **GitHub Actions CI** — pytest matrix on push/PR, py 3.11 + 3.12 | ~30 LOC YAML | before public traffic |
| 2 | **PyPI release** (Phase 6) | ~1 day — version policy, license review, README rendering | after dogfood feedback |
| 3 | **Additional source feeds** — add new science areas as the forks materialize | config-only, 5 min per source | when fork repos exist |
| 4 | **BM25 upgrade for `search_*_text`** (Phase 5) — `rank_bm25` is 0.5 MB extra dep | ~0.5 day | only if real users complain about text-search relevance |
| 5 | **Operational hardening** — log rotation inside container (uvicorn → docker logs unbounded), disk monitoring on named volumes, backend health endpoint (`GET /healthz`), rolling-update path for backend restart without dropping live SSH-tunneled MCP sessions | ~1 day | when this graduates from personal lab to multi-user service |
| 6 | **Per-source-per-month abstract shards** — replace single monolithic `abstracts/embeddings.npy` with shard tree (`<source>/<YYYY-MM>/...`), so daily refresh only touches current-month shards. Foundation for 100k+ corpus and adding 7th+ domain without full re-encode. | ~1-2 days | when corpus crosses ~50k papers or 7th source is added |

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

## Pending phases

Completed phases moved to `docs/PLAN_ARCHIVE.md`. Still pending:

| # | Goal | Status |
|---|------|--------|
| 3 | First user: connect to Claude Desktop, dogfood for a week | partial — 2026-05-08 dogfood session via Claude Code (this repo). Claude Desktop integration not yet exercised. |
| 5 | BM25 upgrade if text relevance complaints surface | pending |
| 6 | PyPI release | pending |
| — | Non-arXiv content (PDFs without arxiv_id, video, books) | scope choice — see U7 (PDF rate ended up higher than [Р-013] предположил) |

---

## Code map

```
src/
└── arxiv_radar_mcp/     # arxiv-radar-mcp shell (arxiv-specific only).
                         # corpus_core lives in its own repo since
                         # Phase 3 (2026-05-09) — install via
                         # `pip install -e ../corpus-core` for dev.
    ├── __main__.py      # `arxiv-radar-mcp` entrypoint (--build-cache, serve)
    ├── config.py        # radar.toml loader + defaults
    ├── corpus.py        # Paper dataclass, loaders for github + local sources
    ├── build_cache.py   # `--build-cache` CLI orchestrator (uses corpus_core.Encoder)
    ├── fulltext.py      # HTML/LaTeX fetcher (selectolax + pylatexenc); source-cascade per arxiv_id;
    │                    # _throttle + _request_with_retry (Р-018);
    │                    # _looks_like_echo_skeleton + _normalize_heading_for_compare (U10);
    │                    # probe_html_available (U2);
    │                    # _iter_descendants — DFS replacement for selectolax.traverse (U10b);
    │                    # anchor-href preservation in _node_to_markdown (U13)
    ├── refresh.py       # daily refresh: git pull → diff → encode new → atomic swap (Р-016)
    ├── fulltext_cli.py  # `python -m arxiv_radar_mcp.fulltext_cli` — fetch helper for in-container use
    ├── reindex_cli.py   # `python -m arxiv_radar_mcp.reindex_cli` — reindex helper for in-container use
    └── server.py        # RadarServer holds {abstract_index, fulltext_index, jobs};
                         # TOOL_SPECS (16 tools — incl. validate_arxiv_ids U2);
                         # list_tags filters (U5), fetch_papers force flag (U9),
                         # search_paper_* snippet_chars (U11), paper_info full_abstract (U12);
                         # serve() stdio + serve_http() streamable-HTTP;
                         # _refresh_loop() asyncio background task;
                         # imports infrastructure from corpus_core

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
RadarServer end-to-end against a synthetic 3-paper corpus, U10
echo-skeleton detector regression (`test_fulltext_echo_skeleton.py`).
After Phase-1 extraction these tests double as `corpus_core` tests
(see `docs/PLAN_CORE_EXTRACTION.md`); they'll move with the modules
when Phase 3 ships corpus-core to its own repo.

Encoder-dependent paths (live semantic search, similar_to_*, real
reindex on GPU) are covered by gomer scenario scripts in
`tmp/gomer_scenario.sh` rather than mocked unit tests — the gap between
a mocked encoder and the real one is exactly what bites at integration
time. Full results in `docs/MODEL_BENCHMARKS.md`.
