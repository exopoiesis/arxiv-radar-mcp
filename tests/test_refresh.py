"""Tests for refresh.py — git pull + diff + atomic abstract index update."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from arxiv_radar_mcp import refresh as refresh_mod
from arxiv_radar_mcp.refresh import (_atomic_save, git_pull, is_git_repo,
                                     refresh_sources)


# ----- is_git_repo -----------------------------------------------------------


def test_is_git_repo_detects_dot_git_dir(tmp_path: Path):
    (tmp_path / ".git").mkdir()
    assert is_git_repo(tmp_path) is True


def test_is_git_repo_detects_dot_git_file(tmp_path: Path):
    """sparse-checkout / submodules can use a .git file pointing elsewhere."""
    (tmp_path / ".git").write_text("gitdir: /elsewhere")
    assert is_git_repo(tmp_path) is True


def test_is_git_repo_returns_false_when_no_dot_git(tmp_path: Path):
    assert is_git_repo(tmp_path) is False


# ----- git_pull --------------------------------------------------------------


def test_git_pull_invokes_git_with_correct_args(monkeypatch, tmp_path: Path):
    captured = {}

    class _FakeResult:
        returncode = 0
        stdout = "Already up to date."
        stderr = ""

    def _fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return _FakeResult()

    monkeypatch.setattr(refresh_mod.subprocess, "run", _fake_run)
    monkeypatch.setattr(refresh_mod.shutil, "which", lambda _: "/usr/bin/git")
    git_pull(tmp_path)

    assert captured["cmd"][0] == "git"
    assert captured["cmd"][1] == "-C"
    assert captured["cmd"][2] == str(tmp_path)
    assert captured["cmd"][3] == "pull"
    assert "--ff-only" in captured["cmd"]
    assert captured["kwargs"]["timeout"] == 60


def test_git_pull_silent_on_failure(monkeypatch, tmp_path: Path, caplog):
    monkeypatch.setattr(refresh_mod.shutil, "which", lambda _: "/usr/bin/git")

    class _FailedResult:
        returncode = 1
        stdout = ""
        stderr = "fatal: not a git repository"

    monkeypatch.setattr(refresh_mod.subprocess, "run", lambda *a, **k: _FailedResult())
    git_pull(tmp_path)  # must not raise


def test_git_pull_skips_when_git_missing(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(refresh_mod.shutil, "which", lambda _: None)
    # Should be a no-op, not raise.
    git_pull(tmp_path)


def test_git_pull_handles_timeout(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(refresh_mod.shutil, "which", lambda _: "/usr/bin/git")

    def _timeout(*_a, **_kw):
        raise subprocess.TimeoutExpired("git", 60)
    monkeypatch.setattr(refresh_mod.subprocess, "run", _timeout)
    git_pull(tmp_path)  # silent


# ----- _atomic_save ----------------------------------------------------------


def test_atomic_save_writes_npy_and_json(tmp_path: Path):
    matrix = np.zeros((3, 4), dtype=np.float32)
    row_for = {"p1": 0, "p2": 1, "p3": 2}
    _atomic_save(tmp_path, matrix, row_for, model_name="test/model")

    assert (tmp_path / "embeddings.npy").exists()
    assert (tmp_path / "index.json").exists()
    assert not (tmp_path / "embeddings.npy.tmp").exists()
    assert not (tmp_path / "index.json.tmp").exists()

    payload = json.loads((tmp_path / "index.json").read_text())
    assert payload["model"] == "test/model"
    assert payload["dims"] == 4
    assert payload["n"] == 3
    assert payload["row_for"] == row_for


def test_atomic_save_overwrites_existing(tmp_path: Path):
    """Second save must replace the first cleanly."""
    m1 = np.zeros((2, 4), dtype=np.float32)
    _atomic_save(tmp_path, m1, {"a": 0, "b": 1}, model_name="x")

    m2 = np.zeros((5, 4), dtype=np.float32)
    _atomic_save(tmp_path, m2, {"x": 0, "y": 1, "z": 2, "w": 3, "v": 4},
                 model_name="x")

    payload = json.loads((tmp_path / "index.json").read_text())
    assert payload["n"] == 5
    loaded = np.load(tmp_path / "embeddings.npy")
    assert loaded.shape == (5, 4)


# ----- refresh_sources high-level dispatch ----------------------------------


class _FakeEncoder:
    """Drop-in for corpus_core.embeddings.Encoder."""

    def __init__(self, dim: int = 4):
        self.dim = dim
        self.calls: list[list[str]] = []
        self.config = type("C", (), {"embeddings": type("E", (), {
            "batch_size": 32, "model": "fake/encoder"})()})()

    @property
    def model_name(self):
        return "fake/encoder"

    def _ensure_loaded(self):
        pass

    def encode_passages(self, texts, show_progress=True,
                        max_seq_length=512, batch_size=None):
        self.calls.append(list(texts))
        return np.ones((len(texts), self.dim), dtype=np.float32)


class _FakeRadar:
    """Subset of RadarServer needed by refresh_sources."""

    def __init__(self, papers, encoder, cache_dir, sources):
        self.papers = dict(papers)
        self.encoder = encoder
        self.config = type("Cfg", (), {})()
        self.config.sources = sources
        self.config.embeddings = type("E", (), {"cache_dir": cache_dir})()
        self.abstract_index = None


def _make_paper(arxiv_id: str, title: str = "t"):
    from arxiv_radar_mcp.corpus import Paper
    return Paper(
        arxiv_id=arxiv_id, title=title, first_author="x", authors=["x"],
        abstract=f"abs for {title}", primary_category="cs.LG",
        categories=[], published="2025-01-01", updated="2025-01-01",
        pdf_url="", topics=[], tags=[], domain="chemistry",
    )


def test_refresh_full_rebuild_encodes_all(monkeypatch, tmp_path: Path):
    """When full_rebuild=True, every paper goes through the encoder once."""
    sources = [type("S", (), {"name": "x", "type": "local",
                              "path": tmp_path / "src"})()]

    fresh_papers = {f"p{i}": _make_paper(f"p{i}") for i in range(3)}
    monkeypatch.setattr(refresh_mod, "load_all", lambda _cfg: fresh_papers,
                        raising=False)
    # load_all is imported inside refresh_sources, so patch the source too.
    monkeypatch.setattr("arxiv_radar_mcp.corpus.load_all",
                        lambda _cfg: fresh_papers)

    enc = _FakeEncoder()
    radar = _FakeRadar(papers={}, encoder=enc, cache_dir=tmp_path, sources=sources)

    result = refresh_sources(radar, full_rebuild=True)

    assert result["strategy"] == "full"
    assert result["added"] == 3
    assert result["deleted"] == 0
    assert result["total"] == 3
    assert (tmp_path / "embeddings.npy").exists()
    # All three abstracts encoded in one call.
    assert len(enc.calls) == 1
    assert len(enc.calls[0]) == 3


def test_refresh_incremental_only_encodes_added(monkeypatch, tmp_path: Path):
    sources = [type("S", (), {"name": "x", "type": "github", "path": None})()]

    initial = {"p1": _make_paper("p1"), "p2": _make_paper("p2")}
    fresh_papers = {**initial, "p3": _make_paper("p3"), "p4": _make_paper("p4")}
    monkeypatch.setattr("arxiv_radar_mcp.corpus.load_all",
                        lambda _cfg: fresh_papers)

    enc = _FakeEncoder()
    radar = _FakeRadar(papers=initial, encoder=enc, cache_dir=tmp_path, sources=sources)

    # Pre-seed an existing abstract index so the incremental path can find it.
    from corpus_core.embeddings import EmbeddingIndex
    existing_matrix = np.ones((2, enc.dim), dtype=np.float32)
    radar.abstract_index = EmbeddingIndex(
        matrix=existing_matrix,
        row_for={"p1": 0, "p2": 1},
        model_name="fake/encoder",
        dims=enc.dim,
    )

    result = refresh_sources(radar, full_rebuild=False)

    assert result["strategy"] == "incremental"
    assert result["added"] == 2
    assert result["deleted"] == 0
    assert result["total"] == 4
    # Only the two new papers got encoded.
    assert len(enc.calls) == 1
    assert len(enc.calls[0]) == 2


def test_refresh_forces_full_when_papers_deleted(monkeypatch, tmp_path: Path):
    """A paper missing from the new corpus → must full-rebuild (incremental
    can't drop rows from embeddings.npy without index re-mapping)."""
    sources = [type("S", (), {"name": "x", "type": "github", "path": None})()]

    initial = {f"p{i}": _make_paper(f"p{i}") for i in range(5)}
    fresh = {f"p{i}": _make_paper(f"p{i}") for i in range(3)}  # 2 dropped
    monkeypatch.setattr("arxiv_radar_mcp.corpus.load_all", lambda _cfg: fresh)

    enc = _FakeEncoder()
    radar = _FakeRadar(papers=initial, encoder=enc, cache_dir=tmp_path, sources=sources)
    from corpus_core.embeddings import EmbeddingIndex
    radar.abstract_index = EmbeddingIndex(
        matrix=np.ones((5, enc.dim), dtype=np.float32),
        row_for={f"p{i}": i for i in range(5)},
        model_name="fake/encoder",
        dims=enc.dim,
    )

    result = refresh_sources(radar, full_rebuild=False)

    assert result["strategy"] == "full"
    assert result["deleted"] == 2


def test_refresh_noop_when_nothing_changed(monkeypatch, tmp_path: Path):
    sources = [type("S", (), {"name": "x", "type": "github", "path": None})()]
    same = {"p1": _make_paper("p1"), "p2": _make_paper("p2")}
    monkeypatch.setattr("arxiv_radar_mcp.corpus.load_all", lambda _cfg: same)

    enc = _FakeEncoder()
    radar = _FakeRadar(papers=same, encoder=enc, cache_dir=tmp_path, sources=sources)
    from corpus_core.embeddings import EmbeddingIndex
    radar.abstract_index = EmbeddingIndex(
        matrix=np.ones((2, enc.dim), dtype=np.float32),
        row_for={"p1": 0, "p2": 1},
        model_name="fake/encoder",
        dims=enc.dim,
    )

    result = refresh_sources(radar, full_rebuild=False)

    assert result["strategy"] == "noop"
    assert result["added"] == 0
    assert result["deleted"] == 0
    assert len(enc.calls) == 0


def test_refresh_invokes_git_pull_for_local_git_sources(monkeypatch, tmp_path: Path):
    """When source.type=='local' and path is a git repo, refresh runs git pull."""
    git_path = tmp_path / "src"
    git_path.mkdir()
    (git_path / ".git").mkdir()  # mark as repo

    pulled: list[Path] = []

    def _record_pull(p: Path) -> None:
        pulled.append(p)
    monkeypatch.setattr(refresh_mod, "git_pull", _record_pull)

    sources = [type("S", (), {"name": "x", "type": "local", "path": git_path})()]
    monkeypatch.setattr("arxiv_radar_mcp.corpus.load_all", lambda _cfg: {})

    enc = _FakeEncoder()
    radar = _FakeRadar(papers={}, encoder=enc, cache_dir=tmp_path, sources=sources)
    refresh_sources(radar, full_rebuild=True)

    assert pulled == [git_path]


def test_refresh_skips_git_pull_for_non_git_local_path(monkeypatch, tmp_path: Path):
    plain_path = tmp_path / "src"
    plain_path.mkdir()
    pulled: list[Path] = []
    monkeypatch.setattr(refresh_mod, "git_pull", lambda p: pulled.append(p))
    sources = [type("S", (), {"name": "x", "type": "local", "path": plain_path})()]
    monkeypatch.setattr("arxiv_radar_mcp.corpus.load_all", lambda _cfg: {})

    enc = _FakeEncoder()
    radar = _FakeRadar(papers={}, encoder=enc, cache_dir=tmp_path, sources=sources)
    refresh_sources(radar, full_rebuild=True)
    assert pulled == []
