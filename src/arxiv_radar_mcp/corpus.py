"""Corpus loader: ingests papers-*.json shards from one or more daily-arxiv-* sources.

A `Paper` keeps the upstream schema verbatim plus a `domain` field marking
which source feed it came from (so search results can be filtered or
attributed without losing the original record).

The loader supports two source types:
  * type="github" — fetches raw .json files from the repo's tree via the
    GitHub API; one HTTP request per shard. Cached on disk under
    embeddings.cache_dir / "shards".
  * type="local"  — reads <path>/data/papers-*.json directly. Use during
    development of a sibling fork before it's pushed.
"""
from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import httpx

from arxiv_radar_mcp.config import Config, SourceConfig

LOG = logging.getLogger(__name__)


@dataclass
class Paper:
    """One paper record, schema mirrors data/papers-*.json with a domain tag."""
    arxiv_id: str
    title: str
    first_author: str
    authors: list[str]
    abstract: str
    primary_category: str
    categories: list[str]
    published: str  # YYYY-MM-DD
    updated: str    # YYYY-MM-DD
    pdf_url: str
    topics: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    domain: str = ""  # added by loader: name of the source feed

    @property
    def search_text(self) -> str:
        """Concatenation used for both substring search and embedding."""
        return f"{self.title}\n\n{self.abstract}"


def load_all(config: Config) -> dict[str, Paper]:
    """Load every paper from every configured source. Keyed by arxiv_id.

    If the same arxiv_id shows up in multiple domains (rare — only if a
    paper is genuinely cross-disciplinary and tagged in two feeds), the
    later domain wins for the dict but both domains are kept in
    paper.domain as a comma-separated string.
    """
    by_id: dict[str, Paper] = {}
    for src in config.sources:
        for paper in _load_source(src, config):
            existing = by_id.get(paper.arxiv_id)
            if existing:
                paper.domain = f"{existing.domain},{paper.domain}"
            by_id[paper.arxiv_id] = paper
    LOG.info(f"loaded {len(by_id)} papers from {len(config.sources)} source(s)")
    return by_id


def _load_source(src: SourceConfig, config: Config) -> Iterable[Paper]:
    if src.type == "local":
        yield from _load_local(src)
    elif src.type == "github":
        yield from _load_github(src, config)
    else:
        raise ValueError(f"unknown source type: {src.type!r}")


def _load_local(src: SourceConfig) -> Iterable[Paper]:
    if src.path is None:
        raise ValueError(f"source '{src.name}' is type=local but has no path")
    data_dir = src.path / "data"
    if not data_dir.exists():
        LOG.warning(f"  {src.name}: {data_dir} does not exist; skipping")
        return
    n = 0
    for shard in sorted(data_dir.glob("papers-*.json")):
        with open(shard, encoding="utf-8") as f:
            records = json.load(f)
        for arxiv_id, rec in records.items():
            yield _record_to_paper(arxiv_id, rec, src.name)
            n += 1
    LOG.info(f"  {src.name} (local): {n} papers")


def _load_github(src: SourceConfig, config: Config) -> Iterable[Paper]:
    if src.repo is None:
        raise ValueError(f"source '{src.name}' is type=github but has no repo")
    cache = config.embeddings.cache_dir / "shards" / src.name
    _prepare_github_cache(cache, repo=src.repo, branch=src.branch)

    shard_names = _list_shards_via_api(src.repo, src.branch)
    n = 0
    for shard_name in shard_names:
        local = cache / shard_name
        if not local.exists():
            url = f"https://raw.githubusercontent.com/{src.repo}/{src.branch}/data/{shard_name}"
            r = httpx.get(url, timeout=30.0)
            r.raise_for_status()
            local.write_bytes(r.content)
        with open(local, encoding="utf-8") as f:
            records = json.load(f)
        for arxiv_id, rec in records.items():
            yield _record_to_paper(arxiv_id, rec, src.name)
            n += 1
    LOG.info(f"  {src.name} (github:{src.repo}): {n} papers")


def _prepare_github_cache(cache: Path, *, repo: str, branch: str) -> None:
    """Ensure a shard cache belongs to the requested upstream repo/branch.

    The cache path is keyed by source name for backward compatibility
    (`shards/chemistry`). When the source repo changes but shard filenames
    stay the same, reusing old files would silently keep serving the old
    corpus. A tiny marker lets us invalidate that cache on repo/branch
    migrations.
    """
    marker = cache / ".source.json"
    expected = {"repo": repo, "branch": branch}

    if cache.exists():
        try:
            current = json.loads(marker.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            current = None
        if current != expected:
            shutil.rmtree(cache)

    cache.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps(expected, indent=1), encoding="utf-8")


def _list_shards_via_api(repo: str, branch: str) -> list[str]:
    """Use GitHub's contents API to list /data/papers-*.json filenames."""
    url = f"https://api.github.com/repos/{repo}/contents/data?ref={branch}"
    r = httpx.get(url, timeout=30.0)
    r.raise_for_status()
    return [item["name"] for item in r.json()
            if item["type"] == "file"
            and item["name"].startswith("papers-")
            and item["name"].endswith(".json")]


def _record_to_paper(arxiv_id: str, rec: dict, domain: str) -> Paper:
    return Paper(
        arxiv_id=arxiv_id,
        title=rec.get("title", "").strip(),
        first_author=rec.get("first_author", ""),
        authors=list(rec.get("authors", [])),
        abstract=rec.get("abstract", ""),
        primary_category=rec.get("primary_category", ""),
        categories=list(rec.get("categories", [])),
        published=rec.get("published", ""),
        updated=rec.get("updated", ""),
        pdf_url=rec.get("pdf_url", ""),
        topics=list(rec.get("topics", []) or []),
        tags=list(rec.get("tags", []) or []),
        domain=domain,
    )
