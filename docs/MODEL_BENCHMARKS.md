# Embedding Model Benchmarks

Long-running record of every bi-encoder we've evaluated against the
`exopoiesis/daily-arxiv-ai4chem` corpus (and successors). Append rows for
new models as they ship — never edit historical numbers; they're the only
honest baseline when re-comparing.

## Test corpus snapshot

| Date | Sources | Records | Untagged | Notes |
|------|---------|---------|----------|-------|
| 2026-04-27 | ai4chem only | 14 234 | 2 137 (15.0 %) | Pre-multi-domain, pre-literature. Future runs will be on a larger corpus — record separately. |

Future expected scale (per project plan): ~50–70k records once polymers /
materials / X feeds and the personal literature corpus (~600 PDFs) ship.

## Test queries

Two banks, both stable across runs:

* **12 generic queries** — terminology drift, multi-concept, indirect
  phrasing. Used for pairwise overlap@k between models.
* **10 paraphrased queries** — each maps to ONE specific arxiv_id picked
  from the untagged subset (random seed=42, see `tmp/sample_untagged.py`).
  Used for the **rank-of-target** metric: where does the right paper land
  in each model's ranked list?

Full text in `tmp/queries.txt` (commit-fixed) and the (query, target_id)
pairs in `tmp/compare_4way.py:TARGETED_QUERIES`.

## Scoring metrics

| Metric | What it captures |
|--------|------------------|
| `overlap@15`     | Fraction of top-15 that two models share. Pairwise. Symmetric. Higher = the models agree. Sensitive only to the top of the list, not to deep recall. |
| `target rank`    | Position of the paraphrase target in the model's full corpus ranking. 1 = top hit. Big number = model couldn't find it. |
| `recall@k`       | Number of targeted queries (out of 10) that landed in top-k. Reported for k ∈ {1, 3, 5, 10, 15, 50, 100}. |
| `Δrank` on shared | When two models both surface the same paper in their top-15, by how many positions does its rank shift. Indicates ranking instability. |

Reproducer: `bash tmp/run_compare.sh` (loads each cache + its precomputed
query_vectors; lazy-encodes locally if a cache lacks them).

## Models evaluated

### Specs (static — model card facts)

| Model | Params | Native dim | License | Pooling | Query prefix? | Notes |
|-------|--------|------------|---------|---------|---------------|-------|
| `BAAI/bge-small-en-v1.5` | 33 M | 384 | MIT | mean | yes ("Represent…") | Lightweight baseline |
| `mixedbread-ai/mxbai-embed-large-v1` | 335 M | 1024 | Apache 2.0 | mean | yes ("Represent…") | First production default |
| `BAAI/bge-large-en-v1.5` | 335 M | 1024 | MIT | mean | yes | Peer of mxbai |
| `intfloat/e5-large-v2` | 335 M | 1024 | MIT | mean | "query: "/"passage: " | Symmetric prefixes |
| `microsoft/harrier-oss-v1-0.6b` | 0.6 B | 1024 | MIT | last-token | yes (instruction) | _planned_ |
| `jinaai/jina-embeddings-v5-text-small` | 677 M | 1024 (Matryoshka 32-1024) | CC BY-NC 4.0 | last-token | task= API | _planned_, non-commercial |
| `Qwen/Qwen3-Embedding-4B` | 4 B | 2560 (Matryoshka 32-2560) | Apache 2.0 | last-token | "Instruct: …\nQuery: " | |
| `Qwen/Qwen3-Embedding-8B` | 8 B | 4096 (Matryoshka 32-4096) | Apache 2.0 | last-token | "Instruct: …\nQuery: " | |
| `microsoft/harrier-oss-v1-27b` | 27 B | 5376 | MIT | last-token | yes (instruction) | _planned_, needs ≥80 GB VRAM |
| `Qwen/Qwen3-Embedding-0.6B` | 0.6 B | 1024 | Apache 2.0 | last-token | "Instruct: …\nQuery: " | _planned, sibling of 4B/8B_ |

### Reported MTEB (third-party, for triangulation)

| Model | MTEB v2 (English) | MTEB v2 (Multilingual) | Source |
|-------|--------------------|--------------------------|--------|
| bge-small-en-v1.5 | ~62 | — | published |
| mxbai-embed-large-v1 | ~64.7 | — | published |
| harrier-oss-v1-0.6b | — | 69.0 | model card |
| jina-v5-text-small | 71.7 | 67.7 | model card |
| Qwen3-Embedding-4B | 74.6 | 69.4 | model card |
| Qwen3-Embedding-8B | 75.2 | 70.6 | model card |
| harrier-oss-v1-27b | — | 74.3 | model card |

These are **independent of our corpus** — included to anchor expectations,
not as our own measurements.

### Build cost

| Model | dtype | Hardware | Build time | Encode batch | Cache size (14k) | Date |
|-------|-------|----------|------------|--------------|------------------|------|
| bge-small-en-v1.5 | fp32 | local CPU (Ryzen) | ~20 min (slow CPU) | 64 | 21 MB | 2026-04-27 |
| mxbai-embed-large-v1 | fp32 | gomer RTX 4070 | 1m24s | 64 | 56 MB | 2026-04-27 |
| Qwen3-Embedding-4B | bf16 | gomer RTX 4070 | ~7 min (matryoshka 1024) | 32 | 56 MB | 2026-04-27 |
| Qwen3-Embedding-8B | int8 | gomer RTX 4070 | ~30 min (matryoshka 1024) | 16 | 56 MB | 2026-04-27 |
| Qwen3-Embedding-8B | bf16 | vast RTX 3090 | _running…_ (native 4096) | 32 | 224 MB | 2026-04-27 |
| Qwen3-Embedding-4B | bf16 | gomer RTX 4070 | _running…_ (native 2560) | 32 | 144 MB | 2026-04-27 |

**Notable build gotchas** (saving so future me doesn't re-discover):
- Qwen3 family ships with `max_seq_length=32768` — must set to 512 for our
  short abstracts, otherwise batches pad to 32k tokens and slow down 30-40×.
- Qwen3 in fp32 on a 12 GB GPU OOMs / falls back to swap (4B = 16 GB
  weights). bf16 is required. int8 (bitsandbytes) for 8B on 12 GB.
- mxbai-embed-large-v1 (335 M) on a non-MKL CPU was ~11 min/batch (= ~43 h
  total). Use a GPU.
- MSYS path conv on Windows breaks `docker cp local skypilot:/path`; needs
  `MSYS_NO_PATHCONV=1` and Windows-style absolute paths.
- Output dim is just final projection — `target_dim` truncation is free
  (same compute as native dim).

## Run results

> Each subsection below records one comparison run. Append, never edit.

### Run 2026-04-27 — 7 caches

Caches evaluated:

| Model | dim | Hardware | Date | Cache size |
|-------|-----|----------|------|------------|
| `BAAI/bge-small-en-v1.5` | 384 | local CPU | 2026-04-27 18:43 | 21 MB |
| `mixedbread-ai/mxbai-embed-large-v1` | 1024 | gomer RTX 4070 (fp32) | 2026-04-27 18:58 | 56 MB |
| `Qwen/Qwen3-Embedding-4B` (matryoshka 1024) | 1024 | gomer RTX 4070 bf16 | 2026-04-27 21:17 | 56 MB |
| `Qwen/Qwen3-Embedding-4B` (native) | 2560 | gomer RTX 4070 bf16 | 2026-04-27 22:46 | 140 MB |
| `Qwen/Qwen3-Embedding-8B` (matryoshka 1024) | 1024 | gomer RTX 4070 int8 | 2026-04-27 22:21 | 56 MB |
| `Qwen/Qwen3-Embedding-8B` (native) | 4096 | vast RTX 3090 bf16 | 2026-04-27 22:51 | 223 MB |
| `microsoft/harrier-oss-v1-0.6b` | 1024 | vast RTX 3090 bf16 | 2026-04-27 23:04 | 56 MB |

Caches NOT evaluated (and why):
- `harrier-oss-v1-27b` — 54 GB bf16 download wouldn't fit vast 35704028's
  17 GB free disk after Qwen3-8B + harrier-0.6b cached. GGUF Q4 (~16 GB)
  + llama-cpp-python is the realistic path; deferred.
- `jina-v5-text-small` — needs `trust_remote_code=True` + custom encode
  API (`task=` arg); separate code branch. Deferred.

#### Target recall — `recall@k` and median rank

10 paraphrased queries, each targeting one specific untagged paper. Lower
median rank = better. recall@k counts how many of the 10 targets land in
the top-k of that model's ranking.

| Model | r@1 | r@3 | r@5 | r@10 | median | failures (rank > 5) |
|-------|-----|-----|-----|------|--------|---------------------|
| bge-small | 6 | 6 | **7** | 8 | 1 | Q5=15, Q6=13, Q10=9 |
| mxbai-large | 7 | 9 | 9 | 10 | 1 | Q5=6 |
| qwen3-4B-1024 | 8 | 9 | 9 | 10 | 1 | Q10=6 |
| **qwen3-4B-2560** | **9** | **10** | **10** | 10 | **1** | (none — Q7=3 max) |
| qwen3-8B-1024 | 6 | 8 | 10 | 10 | 1 | Q5=5 |
| qwen3-8B-4096 | 9 | 9 | 9 | 10 | 1 | Q5=6 |
| harrier-0.6b | 6 | 9 | 9 | 9 | 1 | Q10=13 |

All models eventually find every target by rank 15 except harrier-0.6b
(Q10 at rank 13 — multimodal protein query, edge of training distribution).

#### Pairwise overlap@15 on 12 generic queries

(symmetric — % of top-15 shared between two models on the same query,
averaged across all 12 generic queries)

| | bge | mxbai | 4B-1024 | 4B-2560 | 8B-1024 | 8B-4096 | harrier |
|---|---|---|---|---|---|---|---|
| bge-small      | 100% | 47% | 39% | 37% | 37% | 39% | 45% |
| mxbai-large    | 47%  | 100% | 43% | 45% | 38% | 44% | 47% |
| qwen3-4B-1024  | 39%  | 43% | 100% | **86%** | 58% | 61% | 59% |
| qwen3-4B-2560  | 37%  | 45% | **86%** | 100% | 56% | 62% | 55% |
| qwen3-8B-1024  | 37%  | 38% | 58% | 56% | 100% | **78%** | 51% |
| qwen3-8B-4096  | 39%  | 44% | 61% | 62% | **78%** | 100% | 57% |
| harrier-0.6b   | 45%  | 47% | 59% | 55% | 51% | 57% | 100% |

#### Key observations

1. **Qwen3-Embedding-4B at native 2560 dim is the strongest model on this
   corpus.** 9/10 recall@1, 10/10 recall@3, only one query out of place
   (Q7 at rank 3, an exoplanet/atmospheric science topic at the edge of
   ai4chem's domain). It ties OR beats Qwen3-Embedding-8B-native despite
   being 2× smaller. **For 14k chemistry/AI corpus, 4B is enough.**

2. **Matryoshka truncation is not free**, especially on the 8B:
   - 4B-2560 → 4B-1024: -1 recall@1 (9 → 8)
   - 8B-4096 → 8B-1024: **-3 recall@1 (9 → 6)** ← painful
   The 8B model packs more semantic resolution into the high-dim tail,
   so truncation hurts more. The 4B's 2560-dim native is "denser" per
   dimension and degrades less under truncation.

3. **mxbai-large is the best legacy option** if you can't run a 4B model:
   7/10 recall@1, 9/10 recall@3 — usable, but two recall@1 points behind
   Qwen3-4B-native.

4. **bge-small is unsuitable for this workload.** 6/10 recall@1 with two
   total failures (rank 13 and 15 on a 14k corpus = "didn't find"). Only
   useful as a sanity baseline.

5. **harrier-oss-v1-0.6b underwhelms on niche topics.** 6/10 recall@1,
   complete miss on Q10 (multimodal protein generation, rank 13).
   Surprising given its 69.0 MTEB rating — suggests our paraphrase queries
   on the chemistry-domain corpus aren't well-aligned with what its
   instruction template was trained on. Could re-evaluate with different
   instruction prompts (sts_query, bitext_query) — deferred.

6. **Pairwise overlap structure** confirms the model-family hypothesis:
   - 86% / 78% within Qwen3 family across matryoshka dims (same model,
     same training, just different tail length)
   - 56-62% across Qwen3 sizes (4B vs 8B)
   - 37-47% from bge/mxbai to Qwen3 family (different architectures)
   - harrier-0.6b sits roughly equidistant from everyone (45-59%) —
     different training data, different decoder.

#### Recommendation for production (as of 2026-04-27)

**Default for ai4chem corpus today: Qwen3-Embedding-4B at native 2560
dim, bf16.** ~9 GB VRAM, ~140 MB cache for 14k papers, scales to ~700 MB
at 70k. Top empirical recall on our tests, no reranker needed for the
majority of queries.

**Fallback if no GPU / VRAM-constrained:** Qwen3-Embedding-4B at
matryoshka 1024 — 8/10 recall@1, 56 MB cache, beats mxbai-large.

**Reject for production:** bge-small (too lossy on paraphrase), bge-tier
(insufficient capacity), mxbai-large (clearly outperformed by Qwen3 family).

**Open question:** harrier-oss-v1-27b at GGUF Q4 (~16 GB) — would it
beat 4B-native? We're below the disk budget for the int4 BnB path on the
current vast instance; needs separate llama-cpp-python pipeline or a
larger-disk rental. Tracked as task #23.

---

## Fulltext reindex performance (Phase 9-10 introduction)

When the `fulltext_index` flow landed (2026-05-01) — chunker by
headings + per-chunk embedding through Qwen3-4B at `max_seq_length =
12 288` — we measured the cold-start cost of reindex on gomer
(RTX 4070, 12 GB VRAM, bf16).

### Run 2026-05-01 — first end-to-end on gomer

| Step | Outcome |
|------|---------|
| `fetch_papers` (3 papers via HTML) | 3 × HTML downloads + parse, 5 sec wall time |
| `chunker.chunk_markdown` on 3 papers | 22 chunks total (avg 7.3 sections per paper) |
| Qwen3-4B model load on GPU (cold) | 23 sec (download + load to VRAM) |
| Encode 22 chunks at seq=12 288, batch=4 | **317 sec ≈ 14.4 sec/chunk** |
| Persist (`embeddings.npy` + `index.json`) | <1 sec, 0.2 MB total |
| **Total wall** | **345 sec (5.8 min)** for 3 papers cold-start |

### Adaptive bucketing: re-run 2026-05-01 (same 3 papers, same 22 chunks)

Followed up the cold baseline above with `_REINDEX_BUCKETS` in
`fulltext_index.py`: chunks are sorted into three length buckets
before encode, each bucket gets its own (`max_seq_length`,
`batch_size`) pair. Same model, same prefix, same L2 norm — embeddings
are bit-identical to the baseline; only the wasted padding-pass
compute changes.

Bucket configuration:

| Bucket | token threshold | encode `max_seq_length` | batch_size on 12 GB |
|---|---|---|---|
| short | ≤ 512 | 512 | 64 |
| medium | ≤ 2 048 | 2 048 | 16 |
| long | ≤ 12 288 | 12 288 | 4 |

Re-run results:

| Bucket | n chunks | wall | per-chunk |
|---|---|---|---|
| ≤512t | 19 | 2.3 s | **0.12 s/chunk** |
| ≤12288t | 3 | 224.5 s | 74.8 s/chunk |
| **total encode** | 22 | **226.8 s** | — |
| reindex total (incl. cold model load) | — | **255.4 s (4.3 min)** | — |

Compared to the baseline:

| Metric | Cold baseline | + bucketing | Speedup |
|---|---|---|---|
| encode wall time | 317 s | **227 s** | **-29%** |
| total reindex wall | 345 s | **255 s** | -26% |
| short-chunk per-cost | 14 s | **0.12 s** | **117×** |
| long-chunk per-cost | 14 s (avg) | 75 s | (concentrated, expected) |

The 117× short-chunk speedup confirms the diagnosis: ~85% of baseline
compute was wasted padding short chunks to 12 k. Bucketing recovers it.

Long-chunk per-cost is now isolated to actual full-section encoding —
that's the genuine work, can't be cheated. Distribution on this
3-paper corpus is 19/3 short/long; on a typical 100-paper corpus the
ratio runs ~70/25/5 short/medium/long, so the new total cost
extrapolates to **~85 min for 100 papers** on RTX 4070 cold
(vs ~3 hours pre-bucketing). Within reach of "user adds 50 papers, gets
a coffee" interactive flow.

### Search quality smoke (same run)

After reindex, ran 4 search queries against the 3-paper corpus:

| Query type | Query | Top hit | Notes |
|---|---|---|---|
| text | `"Pixtral"` | 2410.07073 / Abstract (39 freq) | exact-token match works |
| semantic broad | `"multimodal vision language model architecture"` | Pixtral (0.622), SmolDocling (0.604) | both VLM papers ranked first; expected |
| semantic precise | `"tokenizer training data"` | URL paper / "3 Dataset" (0.396) | section-level attribution working |
| semantic cross-section | `"evaluation benchmark accuracy"` | Pixtral / Abstract (0.406) | reasonable, abstract dominates due to topic density |

`similar_to_paper` returns mean-of-chunks neighbours:
- Pixtral → SmolDocling (0.705), URL paper (0.591) — both other VLMs
- SmolDocling → URL paper / Dataset (0.769), Pixtral / Appendix (0.681)

Both pairings are semantically defensible. Section-level snippets in
the search payloads make Claude's response naturally cite "found in
Methods of paper X" — exactly the UX the fulltext layer was designed to
enable.

---

## Production rebalance — 2026-05-02 (W1 verification e2e)

After the first full e2e test on real chemistry papers (D:/home/ignat/
project-third-matter/knowledge/W1_VERIFICATION_2026-05-01.md context — 3
queries about proton transfer / MSD / VDOS, fetched 13/15 papers via
HTML or LaTeX) we hit two production issues that the synthetic 3-paper
benchmark above didn't surface:

  * Top-k of `search_paper_*` was 7/15 junk chunks (References,
    Acknowledgments, Data availability, Appendix [A-Z]). High keyword
    density → high cosine, but useless to the user.
  * Reindex on a real 16-paper corpus took 16 minutes (954 s).
    Long-bucket chunks at `max_seq=12288` cost 51 s/chunk vs ≈0.2 s
    for short — and a real corpus has a non-trivial fraction of long
    sections (whole Methods on a 5-page paper).

Five orthogonal fixes shipped together (commit `26e1a7d`):

### 1. Junk section filter (search_paper_*)

`is_junk_section(name)` strips numbering/Roman prefixes and matches
against a regex of citations/admin section titles (References,
Acknowledgments, Bibliography, Data availability, Author contributions,
Funding, Conflict of interest, Supplementary [Material|Information],
Appendix [A-Z]). `search_paper_text` and `search_paper_semantic`
oversample top-k×4 candidates, push junk to a fallback list, return
clean hits first; junk only fills the tail when the corpus is so junk-
heavy that filtering would leave fewer than k clean hits. `Header`
deliberately stays clean — it carries title + abstract, often the right
hit. Empirical: W1 e2e top-k went from 7/15 junk → **0/15 junk**.

### 2. Chunker rebalance: max_tokens 12 288 → 4 096, +12% paragraph overlap

The previous 12 288 chunk size meant entire Methods sections of a
5-page paper landed in **one** embedding row — the cosine averaged
intro/details/results into one vector, losing sub-section granularity.
Plus the encode cost was disproportionate (51 s/chunk on long bucket).

New defaults:
  * `chunker.max_tokens = 4096` — covers ~90th percentile of arxiv
    sections without splitting; longer sections sub-split with paragraph-
    aligned overlap (carry trailing paragraphs of chunk N into start of
    chunk N+1, up to ~12% of max_tokens ≈ 500 tokens).
  * `_REINDEX_BUCKETS` updated to (≤512, ≤2048, ≤4096) — long bucket
    encodes at seq=4096, batch=8.
  * Overlap is paragraph-aligned (no mid-word splits) and bounded
    (`split_long_section` carry-tail loop guarantees `n_chunks ≤ n_paragraphs`).

**Do not raise `max_tokens` back up** without re-measuring. The reasons
to keep it at 4096:

  | Cost dimension | At max=12288 | At max=4096 |
  |---|---:|---:|
  | Long-bucket per-chunk encode | 51 s | 9 s |
  | Reindex on 16-paper corpus | 954 s | 287 s |
  | Index size on same corpus | 152 chunks | 166 chunks (+9% rows) |
  | Section retrieval granularity | "Methods as a whole" | "Methods sub-section X" |
  | Boundary-answer recall | depends on luck | overlap captures it |

Going back to 12288 only makes sense if a real query consistently needs
>4k tokens of context to be answered correctly — we have not seen that
case. Long-bucket encode quadratic-attention cost is the main reason
to keep it at 4096.

### 3. Encoder warm-up at backend startup

`_warmup_encoder` runs one dummy `encode_query("warmup")` on a worker
thread at backend startup. Before this, the user's first real query
paid the ~25-30 sec SentenceTransformer cold-load. Failure path: log
warning and continue — the first query will pay the cost itself, not a
hard fail. W1 e2e first paper-search query: 29 sec → 0.5 sec.

### 4. Explicit `model.to(dtype=torch.bfloat16)` after load

transformers 5.7 stopped doing partial dtype unification on Qwen3:
weights load as bf16 but activations stay fp32, causing
`RuntimeError: expected mat1 and mat2 to have the same dtype, but got
float != c10::BFloat16` inside Linear layers on first forward. Explicit
cast unifies the pipeline. CPU path stays fp32 (bf16 on CPU is
catastrophically slow, no tensor-core acceleration).

### 5. Thread-lock for `_ensure_loaded` (race fix)

Refresh loop, warm-up task, and reindex jobs all run on different anyio
worker threads. Without a lock, two of them could each see
`self._model is None` and call `SentenceTransformer(...)` concurrently
— **two model instances** in GPU memory. Two × Qwen3-4B bf16 = 16 GB on
a 12 GB card → host-memory spillover → 3-50× slower encoding per
bucket. Standard double-checked locking (fast path no lock, lock,
re-check, load, publish only when complete) fixes it.

This race actually fired between warm-up and bootstrap-refresh on first
restart and slowed the first reindex from expected ~290 s to 727 s.
**The thread-lock is the single most important fix in the bundle** in
absolute wall-time terms — without it, items 1-3 don't show their
expected speedup.

### Run 2026-05-02 — W1 e2e full loop

16-paper corpus (3 from previous synthetic run + 13 fetched fresh
during W1 e2e), 166 chunks total, RTX 4070 with bf16 cast, thread-lock
fix in place:

| Bucket | n chunks | wall | per-chunk |
|---|---|---|---|
| ≤512t | 131 | 31.6 s | 0.24 s/chunk |
| ≤2048t | 9 | 13.0 s | 1.44 s/chunk |
| ≤4096t | 26 | 242.5 s | 9.33 s/chunk |
| **total encode** | 166 | **287 s** | — |

End-to-end loop time on the 3 W1 verification queries:

| Stage | 2026-05-01 baseline | 2026-05-02 polish | Speedup |
|---|---:|---:|---:|
| Stage 1 — 3× search_abstract_semantic | 0.5 s | 0.5 s | — |
| Stage 2 — fetch_papers (15 ids) | 12 s | 12 s | — |
| Stage 3 — reindex (16 papers / 166 chunks) | **954 s** | **287 s** | **3.3×** |
| Stage 4 — 3× search_paper_semantic | 29 s (cold) | 0.5 s (warm) | **58×** |
| **Total e2e** | **~16.5 min** | **~5 min** | **3.3×** |

Search quality on the W1 corpus (post-fix):
  * **Q1 "Zundel Eigen mechanism"** top-2 → S10 chunk with the
    Landau-Zener probability formula `$$P_{ij}^{LZ}=\exp(-π/2ℏ\sqrt{...})$$`
    — sub-section granularity working, semantic cosine pulled the
    formula chunk above the abstract.
  * **Q2 "MSD diffusion AIMD water"** top-3 → Hydroxide Mobility paper
    (Header) — directly relevant.
  * **Q3 "VDOS O-H stretch"** top-2 → "Table summarizes fitting results.
    The incoherent VDOS is dominated by H2O" — concrete result-table
    chunk, not averaged Methods.
  * Junk hits: 0/15 across all three queries (was 7/15 pre-filter).

### Extrapolation to scale

| Corpus | Estimate (RTX 4070 cold-start) |
|---|---|
| 16 papers (this run) | 287 s = 5 min |
| 50 papers (typical "research week") | ~15 min |
| 100 papers (small literature project) | ~30 min |
| 300 papers (paper-deep dive) | ~90 min |

Linear scaling holds because no bucket is GPU-bound at our batch sizes
— we're seq-length-bound. The thread-lock fix unlocks this scaling;
without it the GPU spillover would make 100-paper reindex effectively
infeasible.

---

## Model wishlist for future evaluation

Recorded for posterity — recheck quarterly as new models ship.

* **harrier-oss-v1-27b** — Microsoft, 27 B, MTEB 74.3. **Fits in 24 GB
  via int4 (nf4)** quantization through BitsAndBytes — weights ~13.5 GB +
  activations ~3 GB at batch=8, total ~17 GB on a 24 GB card. No need for
  A100 80 GB. Driver: `tmp/run_harrier27b_int4_vast.sh`. Quality cost vs
  bf16 is typically 1-3 MTEB points. (Pre-quantized GGUF exists at
  Abiray/harrier-oss-v1-27b-GGUF for llama.cpp users, but our pipeline
  uses sentence-transformers + BitsAndBytes on-the-fly quantization.)
* **Qwen3-Embedding-0.6B** — sibling of 4B/8B, sub-1B Apache-2.0. Useful
  budget reference vs harrier-0.6b and jina-v5-small.
* **Anything new at the top of MTEB v2 leaderboard** — recheck listing
  every quarter; if a model beats Qwen3-8B by ≥3 points on EN MTEB and is
  open-weights and fits ≤24 GB int8, eval it.

## How to add a new model to this doc

1. Add row to **Specs** with model card facts.
2. Add row to **Reported MTEB** with the numbers from the model card.
3. Add to `_QUERY_PREFIX` / `_PASSAGE_PREFIX` registry in
   `src/arxiv_radar_mcp/embeddings.py` (or special-case in
   `tmp/vast_remote_build.sh` if it needs custom load like jina-v5).
4. Build cache: pick the right `tmp/run_*.sh` driver (or add a new one).
   Record build time + hardware + cache size in **Build cost**.
5. Add the cache_dir to `CACHES` in `tmp/compare_4way.py`.
6. Run `bash tmp/run_compare.sh`.
7. Append a new **Run YYYY-MM-DD** subsection with numbers — don't edit
   prior runs.
