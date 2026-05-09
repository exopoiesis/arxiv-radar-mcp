"""Text search has no model deps — testable end-to-end."""
from arxiv_radar_mcp.corpus import load_all
from corpus_core.search import search_text


def test_search_text_finds_keyword_in_abstract(local_config):
    papers = load_all(local_config)
    hits = search_text(papers.values(), "diffusion catalysts", k=5)
    assert hits
    assert hits[0][0].arxiv_id == "2504.00002"


def test_search_text_title_boosts_score(local_config):
    """A query that hits the title should score above one that only hits abstract."""
    papers = load_all(local_config)
    title_hits = search_text(papers.values(), "MLIP", k=5)
    abstract_hits = search_text(papers.values(), "QM9", k=5)
    assert title_hits and abstract_hits
    assert title_hits[0][1] > abstract_hits[0][1]


def test_search_text_filters_by_tag(local_config):
    papers = load_all(local_config)
    hits = search_text(papers.values(), "model", k=5, tag="generative-model")
    assert {p.arxiv_id for p, _ in hits} == {"2504.00002"}
