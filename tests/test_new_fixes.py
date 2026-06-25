"""Tests for the architectural fixes in arxiv-radar-mcp (Blocks F, G, H).

Covers:
  * build_cache uses EmbeddingIndex.save (atomic, no leftover .tmp)
  * refresh._atomic_save delegates to EmbeddingIndex.save
  * _load_github: GITHUB_TOKEN sent as Authorization header
  * _load_github: fail-soft on 429/network error when cache present
  * _load_github: cold start with no cache re-raises on error
  * fulltext.fetch_and_save writes parse_quality to meta.json
  * _parse_quality fields are correct for html/latex branches
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from arxiv_radar_mcp import corpus as corpus_mod
from arxiv_radar_mcp.corpus import _github_headers, _list_shards_via_api
from arxiv_radar_mcp.fulltext import FetchResult, _parse_quality
from arxiv_radar_mcp.refresh import _atomic_save
from corpus_core.embeddings import EmbeddingIndex


# ---------------------------------------------------------------------------
# Block F -- build_cache / _atomic_save use EmbeddingIndex.save
# ---------------------------------------------------------------------------


def test_atomic_save_no_tmp_files_left(tmp_path: Path):
    """_atomic_save (wrapper around EmbeddingIndex.save) must leave no .tmp files."""
    matrix = np.zeros((3, 4), dtype=np.float32)
    row_for = {"p1": 0, "p2": 1, "p3": 2}
    _atomic_save(tmp_path, matrix, row_for, model_name="test/model")

    assert (tmp_path / "embeddings.npy").exists()
    assert (tmp_path / "index.json").exists()
    assert not (tmp_path / "embeddings.npy.tmp").exists()
    assert not (tmp_path / "index.json.tmp").exists()


def test_atomic_save_index_json_content(tmp_path: Path):
    matrix = np.zeros((2, 8), dtype=np.float32)
    row_for = {"a": 0, "b": 1}
    _atomic_save(tmp_path, matrix, row_for, model_name="my/model")

    payload = json.loads((tmp_path / "index.json").read_text(encoding="utf-8"))
    assert payload["model"] == "my/model"
    assert payload["dims"] == 8
    assert payload["n"] == 2
    assert payload["row_for"] == row_for


def test_atomic_save_loadable_by_embedding_index(tmp_path: Path):
    """The file pair produced by _atomic_save must be loadable by EmbeddingIndex.load."""
    matrix = np.eye(4, dtype=np.float32)
    row_for = {f"id{i}": i for i in range(4)}
    _atomic_save(tmp_path, matrix, row_for, model_name="test/enc")

    idx = EmbeddingIndex.load(tmp_path)
    assert idx.model_name == "test/enc"
    assert idx.row_for == row_for
    np.testing.assert_array_equal(idx.matrix, matrix)


# ---------------------------------------------------------------------------
# Block G -- GITHUB_TOKEN header + fail-soft
# ---------------------------------------------------------------------------


def test_github_headers_empty_when_no_token(monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    h = _github_headers()
    assert "Authorization" not in h


def test_github_headers_bearer_when_token_set(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_testtoken123")
    h = _github_headers()
    assert h.get("Authorization") == "Bearer ghp_testtoken123"


def test_github_headers_trims_whitespace(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "  token_with_spaces  ")
    h = _github_headers()
    assert h["Authorization"] == "Bearer token_with_spaces"


def test_list_shards_via_api_sends_auth_header(monkeypatch):
    """When GITHUB_TOKEN is set, the GET to the API includes Authorization."""

    captured_headers: dict = {}

    class _FakeResp:
        def raise_for_status(self):
            pass
        def json(self):
            return [
                {"type": "file", "name": "papers-2026-01.json"},
                {"type": "file", "name": "README.md"},  # should be ignored
            ]

    def _fake_get(url, *, timeout, headers=None):
        captured_headers.update(headers or {})
        return _FakeResp()

    monkeypatch.setenv("GITHUB_TOKEN", "tok123")
    monkeypatch.setattr(corpus_mod.httpx, "get", _fake_get)

    names = _list_shards_via_api("exopoiesis/test-repo", "main")
    assert names == ["papers-2026-01.json"]
    assert captured_headers.get("Authorization") == "Bearer tok123"


def test_load_github_failsoft_uses_cache_on_429(monkeypatch, tmp_path: Path):
    """When the GitHub API returns 429, _load_github should fall back to
    locally cached shards and log a WARNING rather than raising."""
    import httpx

    from arxiv_radar_mcp.config import EmbeddingsConfig, SourceConfig

    # Pre-populate a cached shard.
    src = SourceConfig(
        name="chemistry", type="github",
        repo="exopoiesis/arxiv-radar-chemistry", branch="main",
    )
    cfg_embeddings = EmbeddingsConfig(cache_dir=tmp_path / "cache")
    cache = tmp_path / "cache" / "shards" / "chemistry"
    cache.mkdir(parents=True)
    (cache / ".source.json").write_text(
        json.dumps({"repo": "exopoiesis/arxiv-radar-chemistry", "branch": "main"}),
        encoding="utf-8",
    )
    shard_data = {
        "2601.00001": {
            "title": "cached paper", "abstract": "abs",
            "first_author": "A", "authors": ["A"],
            "primary_category": "cs.LG", "categories": ["cs.LG"],
            "published": "2026-01-01", "updated": "2026-01-01",
            "pdf_url": "", "topics": [], "tags": [],
        }
    }
    (cache / "papers-2026-01.json").write_text(
        json.dumps(shard_data), encoding="utf-8"
    )

    # Make the API call raise HTTPStatusError (like 429).
    def _fail_list(*args, **kwargs):
        raise httpx.HTTPStatusError(
            "429 Too Many Requests",
            request=MagicMock(), response=MagicMock(status_code=429),
        )

    monkeypatch.setattr(corpus_mod, "_list_shards_via_api", _fail_list)

    papers_list = list(corpus_mod._load_github(src, _FakeCfg(cfg_embeddings)))

    assert len(papers_list) == 1
    assert papers_list[0].arxiv_id == "2601.00001"


def test_load_github_cold_start_reraises_on_api_error(monkeypatch, tmp_path: Path):
    """On cold start (no cached shards), a network error must propagate."""
    import httpx

    from arxiv_radar_mcp.config import EmbeddingsConfig, SourceConfig

    src = SourceConfig(
        name="chemistry", type="github",
        repo="exopoiesis/arxiv-radar-chemistry", branch="main",
    )
    cfg_embeddings = EmbeddingsConfig(cache_dir=tmp_path / "cache")
    # Cache dir exists but has no shard files.
    cache = tmp_path / "cache" / "shards" / "chemistry"
    cache.mkdir(parents=True)
    (cache / ".source.json").write_text(
        json.dumps({"repo": "exopoiesis/arxiv-radar-chemistry", "branch": "main"}),
        encoding="utf-8",
    )

    def _fail_list(*args, **kwargs):
        raise httpx.HTTPStatusError(
            "429 Too Many Requests",
            request=MagicMock(), response=MagicMock(status_code=429),
        )

    monkeypatch.setattr(corpus_mod, "_list_shards_via_api", _fail_list)

    with pytest.raises(httpx.HTTPStatusError):
        list(corpus_mod._load_github(src, _FakeCfg(cfg_embeddings)))


class _FakeCfg:
    """Minimal config duck-type for _load_github."""
    def __init__(self, embeddings):
        self.embeddings = embeddings


# ---------------------------------------------------------------------------
# Block H -- parse_quality in meta.json
# ---------------------------------------------------------------------------


def test_parse_quality_html_branch():
    md = "## Abstract\n\nSome text.\n\n## Introduction\n\nIntro body here.\n"
    result = FetchResult(
        arxiv_id="2601.00001", source="html",
        markdown=md, n_chars=len(md), error=None,
    )
    quality = _parse_quality(result)

    assert quality["branch"] == "html"
    assert quality["n_headings"] == 2
    assert quality["avg_body_len"] > 0
    assert isinstance(quality["echo_skeleton"], bool)


def test_parse_quality_latex_branch():
    md = "## Methods\n\nSome methods text.\n"
    result = FetchResult(
        arxiv_id="2601.00002", source="latex",
        markdown=md, n_chars=len(md), error=None,
    )
    quality = _parse_quality(result)

    assert quality["branch"] == "latex"
    # latex branch never triggers echo_skeleton detector.
    assert quality["echo_skeleton"] is False


def test_parse_quality_none_on_failure():
    result = FetchResult(
        arxiv_id="2601.00003", source=None,
        markdown=None, n_chars=0, error="pdf-only",
    )
    quality = _parse_quality(result)

    assert quality["branch"] is None
    assert quality["n_headings"] == 0
    assert quality["avg_body_len"] == 0.0
    assert quality["echo_skeleton"] is False


def test_parse_quality_no_headings():
    """Markdown with no ## headings: avg_body_len = total length."""
    md = "Just some text without headings. " * 5
    result = FetchResult(
        arxiv_id="2601.00004", source="latex",
        markdown=md, n_chars=len(md), error=None,
    )
    quality = _parse_quality(result)

    assert quality["n_headings"] == 0
    assert quality["avg_body_len"] == pytest.approx(float(len(md)))


def test_fetch_and_save_writes_parse_quality(tmp_path: Path, monkeypatch):
    """fetch_and_save must include parse_quality in the written meta.json."""
    from arxiv_radar_mcp import fulltext as ft_mod
    from arxiv_radar_mcp.fulltext import fetch_and_save

    md_text = "## Methods\n\nSome method text.\n\n## Results\n\nSome results.\n"
    fake_result = FetchResult(
        arxiv_id="2601.99999", source="html",
        markdown=md_text, n_chars=len(md_text), error=None,
        images=[],
    )

    monkeypatch.setattr(ft_mod, "fetch_paper", lambda *a, **kw: fake_result)
    monkeypatch.setattr(ft_mod, "_download_images", lambda imgs, dest, client: [])

    fulltext_dir = tmp_path / "fulltext"
    fetch_and_save("2601.99999", fulltext_dir)

    meta_path = fulltext_dir / "sources" / "2601.99999.meta.json"
    assert meta_path.exists()
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    assert "parse_quality" in meta
    pq = meta["parse_quality"]
    assert pq["branch"] == "html"
    assert pq["n_headings"] == 2
    assert isinstance(pq["avg_body_len"], float)
    assert isinstance(pq["echo_skeleton"], bool)
