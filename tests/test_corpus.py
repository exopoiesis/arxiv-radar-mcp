"""Tests for corpus loader."""
from arxiv_radar_mcp.corpus import load_all


def test_load_local_yields_all_papers(local_config):
    papers = load_all(local_config)
    assert set(papers.keys()) == {"2503.00001", "2504.00001", "2504.00002"}


def test_paper_carries_domain_tag(local_config):
    papers = load_all(local_config)
    for p in papers.values():
        assert p.domain == "ai4chem"


def test_paper_search_text_combines_title_and_abstract(local_config):
    papers = load_all(local_config)
    p = papers["2503.00001"]
    assert "mackinawite" in p.search_text
    assert "density functional theory" in p.search_text
