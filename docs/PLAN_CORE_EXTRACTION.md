# Plan: extract shared `corpus-core` for arxiv-radar-mcp + lab-corpus-mcp

> Active plan for splitting shared infrastructure out of `arxiv-radar-mcp`
> into a peer package called **`corpus-core`** (decided 2026-05-09) that
> both `arxiv-radar-mcp` (arxiv-only topical radar) and `lab-corpus-mcp`
> (multi-source PDF/video/PubMed/Scholar personal corpus, next session)
> will depend on. Avoids the circular "PDF processing as ETL plugin" trap
> by introducing a third tier instead of letting the two mcp servers
> depend on each other.
>
> The Python package name is `corpus_core` (PEP 8: underscore in module
> path); the project / PyPI name is `corpus-core` (PEP 503: hyphen in
> distribution name).

## Goal

- `arxiv-radar-mcp` stays focused on **arXiv ingestion** (HTML/LaTeX
  cascade, fork-family loader, daily refresh from github).
- `lab-corpus-mcp` covers **everything else**: PDF parsing, YouTube /
  generic video → slides + transcript, PubMed, Scholar, lab-internal
  documents.
- Shared `corpus-core` package owns **indexing + search + MCP server +
  job orchestration**. Both projects depend on it; neither plugs into
  the other; "skills" (improvements / fixes) propagate to both via
  `corpus-core` upgrades.

## Architecture

```
                corpus-core  (shared package, eventually pip-installable)
                ────────────
                Encoder + EmbeddingIndex + bucket encoding
                Chunker (markdown→chunks with sections)
                search primitives: semantic / text / similar_to
                corpus_index (renamed from fulltext_index)
                JobRegistry (async, persistent, U1 disk-truth)
                MCP server scaffold: build_mcp_app, serve_stdio/http, dispatch
                proxy reconnect-loop (U8 Option B)
                junk-section filter (References / Acks / Appendix etc)
                has_domain_or_method gate
                author whitelist mechanism (generalised over schema)

      ↑                                            ↑
      │                                            │
   depends on                                  depends on

arxiv-radar-mcp                             lab-corpus-mcp
─────────────────                           ─────────────────
arxiv search (OR-set query)                 PDF parser (MinerU/marker)
HTML/LaTeX fetcher cascade                  Video → slide+transcript (Whisper)
arxiv-radar-* fork loader                   PubMed / Scholar fetchers
relevance_filter (chem/phys/polymer/...)    DOI/PMID/lab-id schema
canonical tags / authors.yaml               lab-specific topical filters
daily git-pull refresh                      per-user corpora
echo-skeleton detector (U10)                PDF stub detector
selectolax DFS workaround (U10b)            format-specific quirks
probe_html_available (U2)
fetch_papers force (U9)
anchor URL preservation (U13)
```

## What goes into `corpus-core`

Shared infrastructure (~60 % of current `arxiv-radar-mcp` codebase):

- `embeddings.py` — Encoder + EmbeddingIndex + adaptive bucket encoding
- `chunker.py` — markdown→Chunk dataclass, section split + paragraph overlap
- `search.py` — semantic / text / similar_to primitives
- `fulltext_index.py` → renamed to `corpus_index.py` (generic chunk index)
- `jobs.py` — JobRegistry with U1 disk-truth fallback in `get()`
- `proxy.py` — stdio↔HTTP bridge with U8 reconnect-loop
- `mcp_scaffold.py` (new file) — extracted `_build_mcp_app`, `serve_stdio`,
  `serve_streamable_http`, `_dispatch` — generic dispatch with
  per-server tool catalogue.
- helpers: junk-section regex, snippet truncation, has_domain_or_method,
  author-whitelist matcher.

## What stays in `arxiv-radar-mcp`

arxiv-specific ingestion + topical-radar features:

- `fulltext.py` — HTML/LaTeX cascade (arxiv-specific):
  - HTML stub detection
  - echo-skeleton detector (U10) + heading-normalise (U10)
  - `_iter_descendants` selectolax DFS fix (U10b)
  - anchor URL preservation (U13)
  - `probe_html_available` (U2)
- `corpus.py` — Paper dataclass + `arxiv-radar-*` fork loader
- `refresh.py` — daily `git pull` of `/cache/sources/<name>/` clones
- `config.py` — radar.toml schema for arxiv sources
- `TOOL_SPECS` for arxiv-specific tools:
  `paper_info`, `list_tags`, `list_domains`, `fetch_papers` (with U9 force),
  `validate_arxiv_ids`, `search_abstract_*`, `search_paper_*`,
  `similar_to_*`, `reindex`, `refresh_abstracts`, `job_status`,
  `job_list`, `list_enriched`
- `relevance_filter` config schema (direct/method/domain patterns)
- `tags/canonical.yaml` + `tags/authors.yaml` loader (calls
  `corpus-core` mechanism)

## What goes into `lab-corpus-mcp` (next session)

Multi-source ingestion + lab-corpus features:

- `fetchers/pdf.py` — PDF→markdown via MinerU or marker
- `fetchers/video.py` — YouTube / generic video → slide capture +
  Whisper transcript → markdown
- `fetchers/pubmed.py` — PubMed API search + abstract pull
- `fetchers/scholar.py` — Google Scholar scrape (if legally tractable)
- `corpus.py` — multi-source paper schema:
  `paper_id` ∈ {DOI, PMID, arxiv_id, sha256-of-pdf, url-hash}
- `TOOL_SPECS` lab tools: `ingest_pdf`, `ingest_video`, `search_pubmed`,
  `ingest_local_dir`, `search_paper_*` (delegates to `corpus-core`)
- lab-specific config schema (per-user corpora, lab-internal categories)

## U7 (PDF for arxiv) lands cleanly here

Currently U7 is "deferred — architectural" in arxiv-radar-mcp's
`docs/PLAN.md`: 26 % of arxiv papers are PDF-only, can't be ingested.
Resolution path:

- PDF parsing lives as **optional extra** of `corpus-core`:
  `corpus-core[pdf]` pulls in MinerU or marker (heavy: ~2 GB / ~500 MB).
- `arxiv-radar-mcp` users can opt in:
  `pip install arxiv-radar-mcp[pdf]` →
  `fulltext.py` cascade gains a third tier (HTML → e-print → PDF).
- `lab-corpus-mcp` always ships with `[pdf]` extra (PDF is its primary
  path).

No circular dependency: PDF is a *feature of the shared core*, not a
plugin between the two mcp servers.

## Skills reuse — concrete propagation table

| Improvement | Lives in | Benefits |
|-------------|----------|----------|
| U1 job_status disk-truth fallback | `corpus-core/jobs.py` | both |
| U8 proxy reconnect loop | `corpus-core/proxy.py` | both |
| U11 snippet_chars param | `corpus-core/search.py` | both |
| U12 full_abstract param | `corpus-core/_paper_payload` (parameterised) | both (if abstract field present) |
| Adaptive bucket encoding | `corpus-core/embeddings.py` | both |
| has_domain_or_method gate | `corpus-core/relevance.py` | both (per-repo configs feed it) |
| Author whitelist with `via:author-whitelist:<note>` topic | `corpus-core/whitelist.py` | both (with per-repo `tags/authors.yaml`) |
| Bare-word query disambiguation pattern (RAW: + AND-context) | `corpus-core` doc | both (architectural guidance) |
| ENVIRON-style probe pattern | `corpus-core/probe.py` (testing helper) | both |
| **U14 fetch-by-URL** (2026-05-13): `fetch_url` / `fetch_arxiv_pdf` + arxiv `Throttle` singleton | `corpus-core/http_fetch.py` | both — arxiv-radar's `_fetch_html` / `_fetch_eprint` and lab-corpus's `fetch_and_ingest` (powers `ingest_url` / `ingest_arxiv_pdf`) all share one process-wide arXiv 1 req / 3 sec budget |
| U2 `validate_arxiv_ids` | `arxiv-radar-mcp/fulltext.py` | arxiv only (HEAD probe to arxiv.org) |
| U9 `fetch_papers force=True` | `arxiv-radar-mcp/server.py` | arxiv only |
| U10 echo-skeleton detector | `arxiv-radar-mcp/fulltext.py` | arxiv only (LaTeXML quirk) |
| U10b selectolax DFS workaround | `arxiv-radar-mcp/fulltext.py` | arxiv only (HTML parser quirk; lab-corpus uses different parsers) |
| U13 anchor URL preservation | `arxiv-radar-mcp/fulltext.py` | arxiv only (LaTeXML→HTML quirk) |
| PDF stub detector (when written) | `corpus-core/pdf.py` (in `[pdf]` extra) | both |
| Whisper-chunk-by-slide pattern | `corpus-core/video.py` (in `[video]` extra) | both |
| RAW: AND-context query design | `corpus-core` doc | both (config authors learn from radar precedent) |

## Phased rollout

### Phase 1 — In-place extract (1 session in `arxiv-radar-mcp`) — DONE 2026-05-09

Name decided: `corpus-core` (PyPI distribution name) /
`corpus_core` (Python module). See "Naming decision" section.

1. ✅ Create `src/corpus_core/` subpackage **inside current
   `arxiv-radar-mcp` repo** (Phase 1 lives here; Phase 3 extracts to a
   separate repo). [commit cab7f3b]
2. ✅ Move shared modules into `src/corpus_core/`:
   - `embeddings.py`, `chunker.py`, `search.py`, `jobs.py`, `proxy.py`,
     `reranker.py` [commit cab7f3b]
   - `fulltext_index.py` → `corpus_index.py` (rename: it indexes any
     chunked corpus, not just arxiv fulltext) [commit cab7f3b]
   - **Phase 1.5 (2026-05-09):** extract `mcp_scaffold.py` from current
     `server.py` (the `_build_mcp_app`, `serve_stdio`,
     `serve_streamable_http`, `_dispatch` machinery, parameterised
     over per-server `server_name` + `tool_specs` + `dispatcher` +
     optional `background_tasks`). Generic `make_method_dispatcher`
     builds the dispatcher from any handler + tool-name allowlist.
     `arxiv-radar-mcp/server.py` keeps thin wrappers `_dispatch` /
     `_build_mcp_app` / `_run_stdio` / `_run_streamable_http` that
     delegate to the scaffold (preserves test imports). The
     arxiv-specific background tasks (`_warmup_encoder`, `_refresh_loop`)
     stay in the radar shell and are passed in as factories.
3. ✅ Update `arxiv-radar-mcp` imports:
   `from corpus_core import Encoder, EmbeddingIndex, JobRegistry, ...`.
   Phase 1.5 added `make_method_dispatcher`, `build_mcp_app`,
   `serve_stdio`, `serve_streamable_http`.
4. ✅ Re-run the full test suite — 230/230 green after Phase 1.5
   (focused: 41/41 server + http + cli wiring). Tests double as
   `corpus_core` tests for the moment.
5. ✅ Sub-README at `src/corpus_core/README.md` lists the public API
   surface and downstream invariants. Updated 2026-05-09 with the
   `mcp_scaffold` module row + dropped the "still inside server.py"
   caveat.
6. ✅ Top-level `arxiv-radar-mcp/docs/PLAN.md` code-map reflects the
   new layout (`corpus_core/` subpackage + arxiv shell).

**Do NOT publish to PyPI yet.** Let the API stabilise through Phase 2.

### Phase 2 — `lab-corpus-mcp` scaffold — DONE 2026-05-09 → 10

1. ✅ Created `exopoiesis/lab-corpus-mcp` repo (Phase 2A initial
   skeleton, 2026-05-09; published to GitHub 2026-05-10).
2. ✅ `pyproject.toml` declares `corpus-core>=0.1.0` and
   `arxiv-radar-mcp>=0.0.1` as deps (Phase 3 final wiring).
3. ✅ MinerU PDF ingest implemented (Phase 2B-1). Default backend
   `pipeline` (Phase 2B+, 2026-05-10) — `vlm-transformers` wedges
   on a 12 GB GPU when sharing VRAM with our embedding Qwen, so
   we forward `-b pipeline` to the mineru CLI by default; users
   with 24 GB+ headroom can override per-call. Marker bench still
   open, but pipeline produced clean markdown + figures on
   `2512.14129` in 86 sec, so the urgency is gone.
4. ✅ Multi-source `paper_id` schema in `lab_corpus_mcp.corpus`:
   filename arxiv-id pattern → sha256 prefix → `user_supplied`
   override, with `paper_id_kind` distinguishing them.
5. ✅ All shared infra (`JobRegistry`, `EmbeddingIndex`,
   `mcp_scaffold`, search primitives, chunker) imported from
   `corpus_core`. No glue duplication between siblings.
6. ✅ Tool catalogue: 11 tools — `corpus_stats`, `list_corpus`,
   `paper_info`, `job_status`, `job_list`, `ingest_pdf`,
   `ingest_local_dir`, `rebuild_index`, `search_paper_text`,
   `search_paper_semantic`, `similar_to_paper`. Each gains a
   `backend` parameter where MinerU runs are involved.
7. ✅ End-to-end verified on `arxiv:2512.14129` (Yin et al.,
   (Cr,Fe)S pyrrhotite — the paper from arxiv-radar's U7
   dogfood batch that triggered the whole effort): 86 sec parse,
   16 chunks indexed, `search_paper_text("pyrrhotite")` returns
   correct hits, `search_paper_semantic("ferrimagnetic
   compensation temperature")` ranks Keywords > Header > Abstract
   semantically. A second smoke (`arxiv:2511.18000`) confirmed
   cross-paper search isolation.

### Phase 3 — Extract `corpus_core` to its own repo — DONE 2026-05-09 → 10

1. ✅ New repo `git/corpus-core/` (initial commit `d7e189b`).
   Published as [exopoiesis/corpus-core](https://github.com/exopoiesis/corpus-core)
   on 2026-05-10 (public). PyPI publication deferred — for now
   downstream installs path-style via `pip install -e ../corpus-core`.
2. ✅ Moved `src/corpus_core/` from arxiv-radar-mcp to corpus-core.
   Pre-extraction cleanup of dep-leaks (corpus_core was importing
   `arxiv_radar_mcp.config.RerankerConfig` and
   `arxiv_radar_mcp.corpus.Paper` — both replaced with local
   dataclass + Protocol declarations so the package is genuinely
   standalone).
3. ✅ Standalone packaging: pyproject.toml + LICENSE + README +
   .gitignore + tests/ (97 tests, 57% coverage; pure-corpus_core
   surface fully covered, the heavy reindex pipeline deferred to
   host-project integration tests).
4. ⏸ PyPI publication — verify name availability + tag v0.1.0
   before shipping. Deferred to a focused publication session.
5. ✅ arxiv-radar-mcp dropped `src/corpus_core/`; pyproject lists
   `corpus-core>=0.1.0` as dependency. 230/230 tests still green.
6. ✅ lab-corpus-mcp pyproject lists `corpus-core>=0.1.0` and the
   Dockerfile installs all three siblings (`corpus-core`,
   `arxiv-radar-mcp`, `lab-corpus-mcp`) editable in order. Tests
   green. Dockerfile combined-mode builds + serves both backends
   on one Qwen.
7. ⏸ Tag versions for compatibility tracking — done after first
   PyPI release.

The "test_no_host_project_imports" smoke test in corpus-core/tests
locks in the architectural invariant: corpus-core must never
auto-import any host-project module. Any future regression turns
red there before it reaches downstream.

## Naming decision (2026-05-09)

Settled on **`corpus-core`** because:

- Honest: the package is the core of corpus operations (indexing, search,
  MCP server). Both `arxiv-radar-mcp` and `lab-corpus-mcp` work on
  corpora — naming the shared layer after that shared work is direct.
- SEO-clean: nothing collides on PyPI / GitHub at the time of decision
  (verify availability before Phase 3 ships).
- Branding-consistent: the family is `arxiv-radar-mcp` (radar layer
  over arxiv corpus), `lab-corpus-mcp` (lab corpus), `corpus-core` (the
  thing that powers corpus search). Each name says what each thing does.

PEP-503 distribution name (`pyproject.toml`, PyPI): `corpus-core`.
PEP-8 Python module name (`import corpus_core`): `corpus_core`.

Considered and dropped: `codex-core` (OpenAI Codex name clash and
ML-loaded), `folio-core` (too scholarly-niche), `scribe-mcp` ("mcp"
suffix belongs to end servers, not the core), `paperforge` (lab-corpus
has video so paper-only branding is wrong), `indexworks`, `semindex`
(dry, no character).

## Open questions

- Publish to PyPI under `exopoiesis-corpus-core` namespace or just
  `corpus-core` flat? Verify name availability before Phase 3 ships.
- Whether to use a separate `mcp_scaffold` module or keep transport
  bits in `server.py` of each downstream project (less code in core,
  but more duplicated server boilerplate downstream).
- MinerU vs marker for PDF in `corpus-core[pdf]` extra. Bench needed
  in Phase 2.
