"""Tests for corpus loader."""
from arxiv_radar_mcp.corpus import _prepare_github_cache, load_all


def test_load_local_yields_all_papers(local_config):
    papers = load_all(local_config)
    assert set(papers.keys()) == {"2503.00001", "2504.00001", "2504.00002"}


def test_paper_carries_domain_tag(local_config):
    papers = load_all(local_config)
    for p in papers.values():
        assert p.domain == "chemistry"


def test_paper_search_text_combines_title_and_abstract(local_config):
    papers = load_all(local_config)
    p = papers["2503.00001"]
    assert "mackinawite" in p.search_text
    assert "density functional theory" in p.search_text


def test_prepare_github_cache_invalidates_when_repo_changes(tmp_path):
    cache = tmp_path / "shards" / "chemistry"
    cache.mkdir(parents=True)
    (cache / "papers-2026-05.json").write_text("{}", encoding="utf-8")

    _prepare_github_cache(
        cache,
        repo="exopoiesis/arxiv-radar-chemistry",
        branch="main",
    )

    assert not (cache / "papers-2026-05.json").exists()
    assert (cache / ".source.json").exists()


def test_prepare_github_cache_keeps_matching_repo_cache(tmp_path):
    cache = tmp_path / "shards" / "chemistry"
    _prepare_github_cache(
        cache,
        repo="exopoiesis/arxiv-radar-chemistry",
        branch="main",
    )
    shard = cache / "papers-2026-05.json"
    shard.write_text("{}", encoding="utf-8")

    _prepare_github_cache(
        cache,
        repo="exopoiesis/arxiv-radar-chemistry",
        branch="main",
    )

    assert shard.exists()
