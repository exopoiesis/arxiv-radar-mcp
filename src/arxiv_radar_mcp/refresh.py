"""Daily refresh of the abstract corpus from the arxiv-radar-* feeds.

The flow:
  1. For each `source` whose `path` is a git working tree, run
     `git -C path pull`. New shards get pulled in, archived shards
     drop out — sparse-checkout users see the upstream prune cleanly.
  2. Reload the corpus from disk / re-fetch shards from github.
  3. Diff against the in-memory `radar.papers`. Emit added / deleted
     sets.
  4. Decide rebuild strategy:
       * full_rebuild=True OR any deletions detected → re-encode all
       * else → encode only new arxiv_ids, append to embeddings.npy
  5. Atomic swap of `<cache_dir>/embeddings.npy` and `index.json`
     (write to `.tmp` then rename) so concurrent searches see either
     the old or new state, never a torn read.
  6. Hot-swap `radar.abstract_index` and `radar.papers` in-place.

Concurrent-write protection: the encoder lock from JobRegistry is shared
with reindex (one encoder, can't be entered twice). A refresh that runs
into a reindex just fails fast with JobError, the scheduler retries
next interval.
"""
from __future__ import annotations

import io
import json
import logging
import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from corpus_core.embeddings import EmbeddingIndex
from corpus_core.jobs import JobError

if TYPE_CHECKING:
    from arxiv_radar_mcp.server import RadarServer

LOG = logging.getLogger(__name__)


def is_git_repo(path: Path) -> bool:
    """True if `path/.git` exists (file or directory — sparse-checkout uses dir)."""
    return (path / ".git").exists()


def git_pull(path: Path) -> None:
    """Run `git -C path pull --ff-only`. Logs result. Doesn't raise on
    network errors — silent retry next refresh."""
    if shutil.which("git") is None:
        LOG.warning(f"git binary not found on PATH; skipping pull of {path}")
        return
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "pull", "--ff-only"],
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode == 0:
            head = (result.stdout or "").strip().splitlines()
            LOG.info(f"git pull {path}: {head[-1] if head else 'ok'}")
        else:
            LOG.warning(f"git pull {path} failed (rc={result.returncode}): "
                        f"{(result.stderr or '').strip()[:200]}")
    except subprocess.TimeoutExpired:
        LOG.warning(f"git pull {path}: timeout, skipping")
    except OSError as e:
        LOG.warning(f"git pull {path}: {e}")


def refresh_sources(
    radar: "RadarServer",
    *,
    full_rebuild: bool = False,
) -> dict:
    """One-shot refresh. Returns a summary dict for logging / job result.

    Caller is responsible for the encoder lock — wrap in JobRegistry.submit
    so reindex/refresh serialize via .reindex.lock.
    """
    from arxiv_radar_mcp.corpus import load_all

    # 1. git pull on local-path sources that are git repos.
    for src in radar.config.sources:
        if src.type == "local" and src.path and is_git_repo(src.path):
            git_pull(src.path)

    # 2. Reload corpus.
    new_papers = load_all(radar.config)
    old_ids = set(radar.papers.keys())
    new_ids = set(new_papers.keys())
    added = sorted(new_ids - old_ids)
    deleted = old_ids - new_ids

    LOG.info(f"refresh: {len(old_ids)} → {len(new_ids)} papers "
             f"(+{len(added)} -{len(deleted)})")

    # 3. Decide strategy.
    has_index = radar.abstract_index is not None
    do_full = full_rebuild or bool(deleted) or not has_index

    if do_full:
        strategy = "full"
        paper_list = sorted(new_papers.values(), key=lambda p: p.arxiv_id)
        texts = [p.search_text for p in paper_list]
        LOG.info(f"refresh: full rebuild, encoding {len(texts)} abstracts")
        matrix = radar.encoder.encode_passages(texts, show_progress=False)
        row_for = {p.arxiv_id: i for i, p in enumerate(paper_list)}
    else:
        if not added:
            radar.papers = new_papers
            return {
                "strategy": "noop", "added": 0, "deleted": 0,
                "total": len(new_papers),
            }
        strategy = "incremental"
        added_papers = [new_papers[pid] for pid in added]
        texts = [p.search_text for p in added_papers]
        LOG.info(f"refresh: incremental, encoding {len(texts)} new abstracts")
        new_matrix = radar.encoder.encode_passages(texts, show_progress=False)
        old_matrix = np.asarray(radar.abstract_index.matrix)
        matrix = np.concatenate([old_matrix, new_matrix], axis=0)
        row_for = dict(radar.abstract_index.row_for)
        offset = old_matrix.shape[0]
        for i, pid in enumerate(added):
            row_for[pid] = offset + i

    # 4. Atomic persist.
    cache_dir = radar.config.embeddings.cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    _atomic_save(cache_dir, matrix, row_for, model_name=radar.encoder.model_name)

    # 5. Hot-swap in-memory refs.
    radar.abstract_index = EmbeddingIndex.load(cache_dir)
    radar.papers = new_papers

    return {
        "strategy": strategy,
        "added": len(added),
        "deleted": len(deleted),
        "total": len(new_papers),
        "dims": int(matrix.shape[1]),
    }


def _atomic_save(
    cache_dir: Path,
    matrix: np.ndarray,
    row_for: dict[str, int],
    *,
    model_name: str,
) -> None:
    """Write embeddings.npy + index.json atomically (write .tmp → rename).

    `np.save` would normally append `.npy` to a filename without it; we
    open the file by handle so the .tmp name is respected.
    """
    npy_final = cache_dir / "embeddings.npy"
    json_final = cache_dir / "index.json"
    npy_tmp = cache_dir / "embeddings.npy.tmp"
    json_tmp = cache_dir / "index.json.tmp"

    with open(npy_tmp, "wb") as f:
        np.save(f, matrix)

    payload = {
        "model": model_name,
        "dims": int(matrix.shape[1]),
        "n": int(matrix.shape[0]),
        "row_for": row_for,
    }
    json_tmp.write_text(json.dumps(payload, indent=1), encoding="utf-8")

    npy_tmp.replace(npy_final)
    json_tmp.replace(json_final)
