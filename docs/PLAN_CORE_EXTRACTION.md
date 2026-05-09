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
| U2 `validate_arxiv_ids` | `arxiv-radar-mcp/fulltext.py` | arxiv only (HEAD probe to arxiv.org) |
| U9 `fetch_papers force=True` | `arxiv-radar-mcp/server.py` | arxiv only |
| U10 echo-skeleton detector | `arxiv-radar-mcp/fulltext.py` | arxiv only (LaTeXML quirk) |
| U10b selectolax DFS workaround | `arxiv-radar-mcp/fulltext.py` | arxiv only (HTML parser quirk; lab-corpus uses different parsers) |
| U13 anchor URL preservation | `arxiv-radar-mcp/fulltext.py` | arxiv only (LaTeXML→HTML quirk) |
| PDF stub detector (when written) | `corpus-core/pdf.py` (in `[pdf]` extra) | both |
| Whisper-chunk-by-slide pattern | `corpus-core/video.py` (in `[video]` extra) | both |
| RAW: AND-context query design | `corpus-core` doc | both (config authors learn from radar precedent) |

## Phased rollout

### Phase 1 — In-place extract (1 session in `arxiv-radar-mcp`)

Name decided: `corpus-core` (PyPI distribution name) /
`corpus_core` (Python module). See "Naming decision" section.

1. Create `src/corpus_core/` subpackage **inside current
   `arxiv-radar-mcp` repo** (Phase 1 lives here; Phase 3 extracts to a
   separate repo).
2. Move shared modules into `src/corpus_core/`:
   - `embeddings.py`, `chunker.py`, `search.py`, `jobs.py`, `proxy.py`
   - `fulltext_index.py` → `corpus_index.py` (rename: it indexes any
     chunked corpus, not just arxiv fulltext)
   - extract `mcp_scaffold.py` from current `server.py` (the
     `_build_mcp_app`, `serve_stdio`, `serve_streamable_http`,
     `_dispatch` machinery, parameterised over the per-server
     `TOOL_SPECS` and dispatcher).
3. Update `arxiv-radar-mcp` imports:
   `from corpus_core import Encoder, EmbeddingIndex, JobRegistry, ...`
4. Re-run the full test suite — no behaviour changes expected. Tests
   double as `corpus_core` tests for the moment.
5. Add a sub-README under `src/corpus_core/README.md` listing the
   public API surface and the few invariants downstream packages must
   honour (e.g. paper schema fields used by `_paper_payload`).
6. Update top-level `arxiv-radar-mcp/docs/PLAN.md` code-map section to
   reflect the new layout.

**Do NOT publish to PyPI yet.** Let the API stabilise through Phase 2.

### Phase 2 — `lab-corpus-mcp` scaffold (1–2 sessions in new repo)

1. Create `exopoiesis/lab-corpus-mcp` repo.
2. `pyproject.toml`: depend on `arxiv-radar-mcp[corpus_core-only]`
   (extras_require pointing to local subpackage), or path-install for
   dev: `pip install -e ../arxiv-radar-mcp` and import `corpus_core` from there.
3. Build PDF fetcher first (most-needed feature, U7 trigger). Use
   MinerU; benchmark vs marker.
4. Multi-source paper_id schema in `corpus.py`.
5. Reuse `JobRegistry`, `EmbeddingIndex`, MCP scaffold from `corpus_core`.
6. Tool catalogue: `ingest_pdf`, `ingest_local_dir`, `search_paper_*`,
   `paper_info`, `job_status`, `job_list`.
7. Test against a real lab-corpus mini-batch (10–20 PDFs).

### Phase 3 — Extract `corpus_core` to its own repo (when both stabilise)

1. New repo `exopoiesis/corpus_core`.
2. Move `src/corpus_core/` from arxiv-radar-mcp to new repo.
3. Add proper packaging (pyproject.toml, README, tests).
4. Publish to PyPI.
5. arxiv-radar-mcp + lab-corpus-mcp drop in-repo subpackage and depend
   on PyPI'd `corpus_core` instead.
6. Tag versions for compatibility tracking.

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
