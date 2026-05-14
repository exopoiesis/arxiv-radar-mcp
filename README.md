# arxiv-radar-mcp

MCP server providing semantic + text search over arXiv abstracts and
**on-demand full text** (HTML / LaTeX source) of papers. Reads abstracts
from the [`arxiv-radar-*`](https://github.com/exopoiesis?tab=repositories&q=arxiv-radar)
source family, grouped by science area; fetches and indexes full text
per-paper at the user's request via `arxiv.org/html` and `arxiv.org/e-print`.

> **Status:** Phase 7-13 done (2026-05-02). 15 MCP tools, two parallel
> indexes (abstracts + fulltext), unit tests green, end-to-end
> validated on gomer GPU. Architecture and decisions in
> [`docs/PLAN.md`](docs/PLAN.md). Empirical embedding-stack benchmarks in
> [`docs/MODEL_BENCHMARKS.md`](docs/MODEL_BENCHMARKS.md).

---

## What it does

Two-step user flow:

1. **Scan abstracts** — Claude searches the multi-domain arxiv-radar
   corpus via `search_abstract_*` and reports candidates. Use `domain`
   to restrict search to a science area.
2. **Drill into full text** — when the user wants depth, Claude calls
   `fetch_papers([ids])` (background job, downloads + parses), then
   `reindex` (rebuilds the fulltext embedding index), then queries it
   with `search_paper_*` to surface specific sections ("found in
   Methods of paper X").

Each `arxiv-radar-<domain>` repo publishes:
- `data/papers-YYYY-MM.json` — monthly abstract shards with titles,
  authors, abstracts, tags, topics
- `tags/canonical.yaml` — curated tag vocabulary

This server reads those shards (over GitHub raw URLs, on-disk shard
cache), keeps abstracts in memory, and adds a separate fulltext layer
fetched on demand. Full texts are cached locally per arxiv_id.

Default science-area feeds:

| Domain filter | Source repo |
|---|---|
| `chemistry` | `exopoiesis/arxiv-radar-chemistry` |
| `chemical_engineering` | `exopoiesis/arxiv-radar-chem-eng` |
| `electrochemistry` | `exopoiesis/arxiv-radar-electrochemistry` |
| `physics` | `exopoiesis/arxiv-radar-physics` |
| `polymer` | `exopoiesis/arxiv-radar-polymer` |
| `sulfide_materials` | `exopoiesis/arxiv-radar-sulfide-materials` |

Current source shards contain about 36.6k domain assignments before
cross-domain dedup: `chemistry` 14.4k, `physics` 17.7k, `polymer`
3.2k, `chemical_engineering` 1.1k, `electrochemistry` 108, and
`sulfide_materials` 99. A paper can belong to more than one domain; in
that case its `domain` field is a comma-separated list.

## Tool surface (15 tools)

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

Common domain-filtered calls:

```json
{"query": "graph neural network potentials for phase transitions", "domain": "physics", "k": 10}
{"query": "polymer electrolyte ion transport", "domain": "polymer", "k": 10}
{"query": "surrogate models for distillation columns", "domain": "chemical_engineering", "k": 10}
{"query": "solid electrolyte interphase impedance lithium metal", "domain": "electrochemistry", "k": 10}
{"query": "argyrodite sulfide solid electrolyte grain boundary", "domain": "sulfide_materials", "k": 10}
```

Call `list_domains()` to see the active source feeds and paper counts in
the running server. Call `list_tags({"domain": "physics"})` or the same
tool with another domain to inspect that area's canonical vocabulary.

### Fulltext (3)

Operates on chunks of full texts the user has explicitly enriched.

| Tool | Purpose |
|------|---------|
| `search_paper_text(query, k)` | AND-scan over chunk texts; returns `{arxiv_id, section, snippet, score}` |
| `search_paper_semantic(query, k)` | cosine over chunk embeddings; same payload shape |
| `similar_to_paper(arxiv_id, k)` | nearest-neighbour papers by mean-of-chunks embedding |

### Async admin (6)

| Tool | Purpose |
|------|---------|
| `fetch_papers([arxiv_ids])` | background job: download + parse + cache full text. Returns `{job_id}` |
| `reindex()` | background job: full rebuild of fulltext embedding index |
| `refresh_abstracts(force_full?)` | background job: pull/update abstract shards and refresh the abstract index |
| `job_status(job_id)` | inspect a running/finished job |
| `job_list(limit?)` | list recent jobs |
| `list_enriched()` | sync: arxiv_ids of locally-cached full texts |

Admin operations are async because refresh/reindex on a meaningful
corpus is slow enough to break the MCP conversation if blocking. Jobs
persist to disk and survive restarts.

There are two refresh modes for abstract sources:

- `type = "github"` reads `data/papers-*.json` from GitHub raw URLs and
  caches shards under `<cache_dir>/shards/<domain>`.
- `type = "local"` reads a sparse clone or local checkout. If that path
  is a git repo, `refresh_abstracts` runs `git pull --ff-only` before
  reloading the corpus.

For long-running GPU backends, `scripts/docker_setup_source.sh` creates
local sparse clones for all default science areas and rewrites
`/cache/radar.toml` to use `type = "local"`.

## Full-text fetch cascade

Per arxiv_id, in order, stop at first success:

| Source | URL pattern | Coverage | Used for |
|---|---|---|---|
| arxiv-rendered HTML | `arxiv.org/html/<id>` | ~70-80% recent papers | preferred — equations preserved as inline LaTeX via `<annotation encoding="application/x-tex">` |
| LaTeX e-print | `arxiv.org/e-print/<id>` | ~85-90% all papers | fallback — pylatexenc expands macros, drops comments, leaves `$...$` math |
| (PDF — not supported) | n/a | n/a | Paper fails with clear "PDF-only on arXiv" error |

Why no PDF: `mineru[core]` adds ~2 GB of dependency to cover ~10-15%
remaining tail. Trade-off didn't make sense for this server's scope.
See [`docs/PLAN.md`](docs/PLAN.md) `[РЕШЕНИЕ-013]`. For PDF-only
arxiv papers, use the sibling [lab-corpus-mcp](https://github.com/exopoiesis/lab-corpus-mcp)
server's `ingest_arxiv_pdf(arxiv_id)` tool — it shares the same
arxiv 1 req / 3 sec throttle via `corpus_core.http_fetch` (U14, 2026-05-13).

Empirical reference (HTML+LaTeX vs PDF text quality on arXiv corpus):
HTML+LaTeX combined F1 = 0.69 vs text-from-PDF much lower
([Toward Robust URL Extraction for Open Science, arXiv 2025](https://arxiv.org/abs/2509.04759)).

## Chunking

Full-text markdown is split by `## headings`. Each section becomes one
chunk; sections >4 096 estimated tokens sub-split by paragraph
boundaries with ~12% paragraph-aligned overlap. Each chunk row carries
`(arxiv_id, section, chunk_idx)` so search results attribute back to
paper region.

4 096 tokens is the current production balance: it keeps section-level
context, gives finer retrieval attribution than whole long Methods
sections, and avoids the 12k-token long-bucket cost that dominated
reindex time on real papers.

## Architecture

```
                       arxiv-radar-* GitHub raw URLs
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
bf16. Selected after benchmarking 7 caches against the 14k chemistry
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

## Deployment

Start with the CPU quickstart for a local smoke test. After that, pick
one of the two supported operating topologies: local stdio or remote
GPU backend over SSH.

### 5-minute CPU quickstart

This path needs no GPU and uses the code default
`mixedbread-ai/mxbai-embed-large-v1`. The first cache build downloads
the embedding model and the current abstract shards; later runs reuse
the local cache.

```bash
git clone https://github.com/exopoiesis/arxiv-radar-mcp.git
cd arxiv-radar-mcp
python -m venv .venv
source .venv/bin/activate        # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -e .
arxiv-radar-mcp --build-cache
```

For CPU laptops, keep abstract refresh incremental so the daily
background update only encodes new papers:

```toml
# ~/.config/arxiv-radar/radar.toml
[sources.chemistry]
type = "github"
repo = "exopoiesis/arxiv-radar-chemistry"
branch = "main"

[sources.chemical_engineering]
type = "github"
repo = "exopoiesis/arxiv-radar-chem-eng"
branch = "main"

[sources.electrochemistry]
type = "github"
repo = "exopoiesis/arxiv-radar-electrochemistry"
branch = "main"

[sources.physics]
type = "github"
repo = "exopoiesis/arxiv-radar-physics"
branch = "main"

[sources.polymer]
type = "github"
repo = "exopoiesis/arxiv-radar-polymer"
branch = "main"

[sources.sulfide_materials]
type = "github"
repo = "exopoiesis/arxiv-radar-sulfide-materials"
branch = "main"

[embeddings]
model = "mixedbread-ai/mxbai-embed-large-v1"
batch_size = 64

[refresh]
enabled = true
interval_hours = 24
full_rebuild = false
```

Claude Desktop local config should point at the venv executable, because
desktop apps often do not inherit your shell `PATH`:

```json
{
  "mcpServers": {
    "arxiv-radar": {
      "command": "/absolute/path/to/arxiv-radar-mcp/.venv/bin/arxiv-radar-mcp"
    }
  }
}
```

On Windows, use the `.exe` path, for example:

```json
{
  "mcpServers": {
    "arxiv-radar": {
      "command": "D:\\home\\me\\arxiv-radar-mcp\\.venv\\Scripts\\arxiv-radar-mcp.exe"
    }
  }
}
```

Restart Claude Desktop after editing the config, then ask it to call
`search_abstract_semantic` for a query such as "machine learned
interatomic potentials for ionic diffusion".

The two supported long-running topologies are:

### A. Local (CPU, no GPU required)

The whole server runs on the user's machine. Easiest setup, suitable
for casual use and CPU-friendly models (mxbai-embed-large-v1,
bge-large-en-v1.5). Full-text fetch + chunk + encode all stay local.

```bash
pip install arxiv-radar-mcp                  # not yet on PyPI; pip install -e . for source
arxiv-radar-mcp --build-cache                # build abstract index once
arxiv-radar-mcp                              # stdio MCP server
```

`claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "arxiv-radar": {
      "command": "arxiv-radar-mcp"
    }
  }
}
```

What runs on CPU comfortably:
- abstract `search_abstract_*` over tens of thousands of papers — sub-second
- `fetch_papers` (HTTP/LaTeX, no model) — seconds per paper
- `reindex` over a few enriched papers with mxbai — minutes
- Heavy Qwen3-4B reindex on CPU — **infeasible** (~100 hours / 100 papers).
  Use mxbai or bge for CPU-only.

### B. Remote backend over SSH (GPU host)

Heavy work runs in a long-running container on a GPU host (gomer, Vast,
any sshd-reachable Linux box with `--gpus all`). The user's laptop runs
a thin stdio→HTTP proxy. SSH provides perimeter auth — no Bearer tokens,
no TLS certs to manage. The backend binds loopback only inside the
container; the only way in is your existing SSH key.

```
Claude Desktop ── stdio ──▶ arxiv-radar-mcp --remote user@gomer
                                │
                                │ subprocess: ssh -L LOCAL:127.0.0.1:8765 user@gomer
                                ▼
                          [SSH tunnel]
                                │
                                ▼
                  arxiv-radar-mcp container (long-running)
                  Qwen3-4B in memory, jobs persist across requests
```

**One-time backend setup on the GPU host:**

```bash
# build (or pull) the GPU image
bash scripts/docker_build.sh

# write radar.toml into the named volume
bash scripts/docker_init_volume.sh

# optional: sparse-clone all default science-area sources into the cache volume
bash scripts/docker_setup_source.sh

# start the long-running backend (--restart unless-stopped)
bash scripts/docker_serve_backend.sh
```

The backend container exposes 8765 on host loopback only
(`-p 127.0.0.1:8765:8765`). It survives reboots; jobs persist in the
named volume.

**Local proxy on your laptop** — ensure `ssh` is on PATH (built into
Windows 10+ / macOS / Linux), then:

```json
{
  "mcpServers": {
    "arxiv-radar": {
      "command": "arxiv-radar-mcp",
      "args": ["--remote", "user@gomer.lan"]
    }
  }
}
```

The proxy opens an SSH tunnel each time Claude Desktop launches it,
forwards MCP traffic to the remote backend, and tears the tunnel down
on exit. Subsequent MCP calls have no cold-start cost — Qwen3-4B is
already loaded on the GPU.

**Backend management:**

```bash
bash scripts/docker_logs_backend.sh   # tail -f the container
bash scripts/docker_stop_backend.sh   # remove it
```

Named volumes used by the backend:
- `arxiv-radar-cache` → `/cache` (radar.toml + abstracts/ + fulltext/ + jobs/)
- `arxiv-radar-hf` → `/root/.cache/huggingface` (Qwen3-4B weights)

### Choice between A and B

| | Local (A) | Remote SSH (B) |
|---|---|---|
| Hardware | any laptop | sshd-reachable host with NVIDIA GPU |
| Embedding model | mxbai / bge (CPU-friendly) | Qwen3-4B native (production winner) |
| Reindex 50 papers | ~30 min on a recent CPU | ~5 min on RTX 4070 |
| Cold start per Claude session | ~5 sec | <100 ms (backend stays loaded) |
| Setup steps | `pip install` + 1 line config | container + ssh keys + 1 line config |
| Auth | n/a | SSH keys you already have |

## Troubleshooting

### Claude Desktop does not show the server

Use an absolute `command` path in `claude_desktop_config.json`. Desktop
apps often launch without your shell profile, so `arxiv-radar-mcp` may
not be on `PATH` even though it works in a terminal. On Windows the
command should usually end in `.venv\\Scripts\\arxiv-radar-mcp.exe`;
on macOS/Linux it should usually end in `.venv/bin/arxiv-radar-mcp`.

Check the server from a terminal first:

```bash
arxiv-radar-mcp --build-cache
arxiv-radar-mcp --log-level DEBUG
```

The second command is a stdio MCP process and will wait for client
messages; stop it with Ctrl-C after confirming it starts cleanly.

### `--build-cache` is slow or downloads a large model

The CPU quickstart uses `mixedbread-ai/mxbai-embed-large-v1`, not
`Qwen/Qwen3-Embedding-4B`. If you copied `radar.example.toml`, change
`[embeddings].model` back to `mixedbread-ai/mxbai-embed-large-v1` for
CPU-only use. Qwen3-4B is the production-quality GPU model; on CPU it
is not practical.

### Semantic tools say the abstract cache is missing

Run:

```bash
arxiv-radar-mcp --build-cache
```

Make sure the same config file is used by both the cache build and the
server. If you pass `--config ./radar.toml` during build, also pass it
in the MCP config `args`, or set `ARXIV_RADAR_CONFIG` consistently.

### Full-text search returns `fulltext index empty`

Full-text search is opt-in per paper. The intended flow is:

1. Find candidates with `search_abstract_semantic` or
   `search_abstract_text`.
2. Call `fetch_papers(["2501.01234", "..."])` and poll `job_status`.
3. Call `reindex()` and poll `job_status`.
4. Use `search_paper_semantic` or `search_paper_text`.

`fetch_papers` only downloads and parses sources. `reindex` is the step
that embeds the chunks and makes `search_paper_*` available.

### A paper fails during `fetch_papers`

The fetch cascade only supports arXiv-rendered HTML and LaTeX e-print
sources. PDF-only submissions fail with an explicit error because PDF
parsing is outside this server's default dependency budget. Recent
LaTeX-authored papers usually work through HTML or e-print.

### SSH remote mode cannot connect

First verify plain SSH from the same machine:

```bash
ssh user@gomer.lan
```

Then verify the backend is running on the GPU host and bound to loopback
port 8765:

```bash
bash scripts/docker_logs_backend.sh
```

The local proxy uses `ssh -L <local>:127.0.0.1:8765 user@host`. It does
not add Bearer auth or TLS; the security boundary is your SSH key and
the backend binding to `127.0.0.1`.

### Backend hangs on first query

The first semantic query may be loading or downloading the embedding
model. On the GPU backend, startup runs an encoder warm-up task, but a
fresh host still needs the Hugging Face model files in the
`arxiv-radar-hf` Docker volume. Watch backend logs; after warm-up,
later queries reuse the in-memory model.

### Reindex is slower than expected

Reindex time scales with the number and length of enriched paper
chunks. The current production chunker caps chunks at 4 096 estimated
tokens and encodes by length bucket. On an RTX 4070, the measured
reference is about 5 minutes for 16 papers and roughly 15-30 minutes
for 50-100 papers, depending on long-section density. On CPU, keep the
embedding model to mxbai/BGE and expect minutes for small batches, not
interactive latency.

### Refresh or reindex says the encoder is busy

`refresh_abstracts` and `reindex` share a lock because both use the
same encoder and write embedding indexes. Wait for the running job to
finish, or inspect recent jobs with `job_list()` and `job_status()`.

### Named Docker volumes fill disk

The backend keeps model weights in `arxiv-radar-hf` and indexes, jobs,
full texts, and `radar.toml` in `arxiv-radar-cache`. If the host runs
out of disk, prune old Docker images first and inspect those two volumes
before deleting anything; deleting `arxiv-radar-cache` removes fetched
full texts and built indexes.

## Config

`~/.config/arxiv-radar/radar.toml` (override via `--config`):

```toml
[sources.chemistry]
type = "github"
repo = "exopoiesis/arxiv-radar-chemistry"
branch = "main"

[sources.chemical_engineering]
type = "github"
repo = "exopoiesis/arxiv-radar-chem-eng"
branch = "main"

[sources.electrochemistry]
type = "github"
repo = "exopoiesis/arxiv-radar-electrochemistry"
branch = "main"

[sources.physics]
type = "github"
repo = "exopoiesis/arxiv-radar-physics"
branch = "main"

[sources.polymer]
type = "github"
repo = "exopoiesis/arxiv-radar-polymer"
branch = "main"

[sources.sulfide_materials]
type = "github"
repo = "exopoiesis/arxiv-radar-sulfide-materials"
branch = "main"

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
[`arxiv-radar-*`](https://github.com/exopoiesis?tab=repositories&q=arxiv-radar)
source family.
