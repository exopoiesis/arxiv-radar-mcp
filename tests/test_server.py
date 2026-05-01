"""Tool catalogue + dispatcher tests.

These exercise the parts of server.py that are pure / sync — no MCP SDK
runtime, no stdio, no encoder-loading. The async stdio loop is left to
integration testing once we dogfood the server against a real client.

Tools added in 2026-05-01 pivot ([РЕШЕНИЕ-014]):
  * search_abstract_text/semantic, similar_to_abstract — renames
  * paper_info — renamed from get_paper, with extended payload
  * search_paper_text/semantic, similar_to_paper — fulltext family
  * fetch_papers, reindex, job_status, job_list — async admin
  * list_enriched — sync listing
Tools removed: search_*_hybrid, recent, paper_status, get_paper.
"""
from __future__ import annotations

import inspect

import pytest

from arxiv_radar_mcp.server import (TOOL_SPECS, RadarServer, _dispatch,
                                    _paper_payload)


# ----- TOOL_SPECS shape ------------------------------------------------------

EXPECTED_TOOLS = {
    # abstracts (6)
    "search_abstract_text", "search_abstract_semantic", "similar_to_abstract",
    "paper_info", "list_tags", "list_domains",
    # fulltext (3)
    "search_paper_text", "search_paper_semantic", "similar_to_paper",
    # async admin (5)
    "fetch_papers", "reindex", "job_status", "job_list", "list_enriched",
}


def test_tool_specs_cover_all_expected_tools():
    listed = {s["name"] for s in TOOL_SPECS}
    assert listed == EXPECTED_TOOLS, (
        f"missing: {EXPECTED_TOOLS - listed}, extra: {listed - EXPECTED_TOOLS}")


def test_tool_specs_match_method_signatures():
    """Every tool's required-args must be actual parameters on RadarServer."""
    for spec in TOOL_SPECS:
        method = getattr(RadarServer, spec["name"])
        sig = inspect.signature(method)
        params = set(sig.parameters) - {"self"}
        declared = set(spec["inputSchema"].get("properties", {}))
        assert declared <= params, (
            f"{spec['name']}: schema declares {declared - params} "
            f"that aren't method params {params}")
        required = set(spec["inputSchema"].get("required", []))
        assert required <= params, f"{spec['name']}: missing required {required - params}"


def test_tool_specs_have_descriptions_and_object_schema():
    for spec in TOOL_SPECS:
        assert spec["description"].strip(), f"{spec['name']}: empty description"
        assert spec["inputSchema"]["type"] == "object"


def test_no_removed_tools_resurrected():
    """Guard against accidental re-add of search_hybrid / recent / get_paper."""
    listed = {s["name"] for s in TOOL_SPECS}
    forbidden = {"search_hybrid", "search_abstract_hybrid", "search_paper_hybrid",
                 "recent", "get_paper", "paper_status", "search_text",
                 "search_semantic", "similar_to"}
    assert listed.isdisjoint(forbidden), (
        f"forbidden tools resurrected: {listed & forbidden}")


# ----- _dispatch -------------------------------------------------------------

class _StubRadar:
    """Mimics the subset of RadarServer surface that _dispatch routes to."""

    def search_abstract_text(self, query, k=10, domain=None, tag=None):
        return [{"called": "search_abstract_text", "query": query, "k": k}]

    def paper_info(self, arxiv_id):
        return {"arxiv_id": arxiv_id}

    def list_domains(self):
        return [{"domain": "ai4chem", "papers": 1}]

    def list_enriched(self):
        return ["2503.00001", "2504.00002"]


def test_dispatch_routes_to_method():
    out = _dispatch(_StubRadar(), "search_abstract_text",
                    {"query": "dft", "k": 3})
    assert out == [{"called": "search_abstract_text", "query": "dft", "k": 3}]


def test_dispatch_handles_none_arguments():
    out = _dispatch(_StubRadar(), "list_domains", None)
    assert out == [{"domain": "ai4chem", "papers": 1}]


def test_dispatch_unknown_tool_returns_error():
    out = _dispatch(_StubRadar(), "drop_database", {})
    assert "error" in out and "drop_database" in out["error"]


def test_dispatch_rejects_dunder_or_private_names():
    out = _dispatch(_StubRadar(), "__init__", {})
    assert "error" in out


def test_dispatch_bad_arguments_returns_error():
    out = _dispatch(_StubRadar(), "paper_info", {"wrong_kw": "x"})
    assert "error" in out and "paper_info" in out["error"]


def test_dispatch_rejects_old_tool_names():
    """Even though the methods used to exist, after the rename they're gone."""
    for old_name in ("search_text", "search_hybrid", "get_paper", "recent"):
        out = _dispatch(_StubRadar(), old_name, {})
        assert "error" in out, f"{old_name} unexpectedly accepted"


# ----- _paper_payload --------------------------------------------------------

def test_paper_payload_truncates_long_abstract(local_config):
    from arxiv_radar_mcp.corpus import load_all
    papers = load_all(local_config)
    p = next(iter(papers.values()))
    p.abstract = "x" * 800
    out = _paper_payload(p, 0.5)
    assert len(out["abstract"]) == 601
    assert out["abstract"].endswith("…")
    assert out["score"] == 0.5
    assert out["url"] == f"https://arxiv.org/abs/{p.arxiv_id}"


def test_paper_payload_caps_authors():
    from arxiv_radar_mcp.corpus import Paper
    p = Paper(
        arxiv_id="2503.99999", title="t", first_author="a",
        authors=[f"author_{i}" for i in range(20)],
        abstract="a", primary_category="x", categories=[],
        published="2025-03-01", updated="2025-03-01", pdf_url="",
    )
    out = _paper_payload(p, 0.0)
    assert len(out["authors"]) == 5


# ----- RadarServer end-to-end ------------------------------------------------

def test_radar_server_search_abstract_text_via_dispatch(local_config):
    radar = RadarServer(local_config)
    try:
        out = _dispatch(radar, "search_abstract_text",
                        {"query": "diffusion catalysts", "k": 3})
        assert out and out[0]["arxiv_id"] == "2504.00002"
    finally:
        radar.jobs.shutdown()


def test_radar_server_list_domains(local_config):
    radar = RadarServer(local_config)
    try:
        out = _dispatch(radar, "list_domains", {})
        assert out == [{"domain": "ai4chem", "papers": 3}]
    finally:
        radar.jobs.shutdown()


def test_radar_server_paper_info_returns_none_for_missing(local_config):
    radar = RadarServer(local_config)
    try:
        out = _dispatch(radar, "paper_info", {"arxiv_id": "0000.00000"})
        assert out is None
    finally:
        radar.jobs.shutdown()


def test_radar_server_paper_info_carries_fulltext_status(local_config):
    """paper_info on a known paper should include fulltext={enriched: false, ...}."""
    radar = RadarServer(local_config)
    try:
        out = _dispatch(radar, "paper_info", {"arxiv_id": "2503.00001"})
        assert out is not None
        assert "fulltext" in out
        assert out["fulltext"]["enriched"] is False
    finally:
        radar.jobs.shutdown()


def test_semantic_tools_return_error_when_no_index(local_config):
    radar = RadarServer(local_config)  # no embedding cache built
    try:
        out = _dispatch(radar, "search_abstract_semantic", {"query": "dft"})
        assert out and "error" in out[0]
        out2 = _dispatch(radar, "similar_to_abstract", {"arxiv_id": "2503.00001"})
        assert out2 and "error" in out2[0]
    finally:
        radar.jobs.shutdown()


def test_fulltext_tools_return_error_when_no_index(local_config):
    radar = RadarServer(local_config)
    try:
        out = _dispatch(radar, "search_paper_text", {"query": "dft"})
        assert out and "error" in out[0]
        assert "fetch_papers" in out[0]["error"]
        out2 = _dispatch(radar, "search_paper_semantic", {"query": "dft"})
        assert out2 and "error" in out2[0]
        out3 = _dispatch(radar, "similar_to_paper", {"arxiv_id": "2503.00001"})
        assert out3 and "error" in out3[0]
    finally:
        radar.jobs.shutdown()


def test_list_enriched_empty_when_nothing_fetched(local_config):
    radar = RadarServer(local_config)
    try:
        out = _dispatch(radar, "list_enriched", {})
        assert out == []
    finally:
        radar.jobs.shutdown()


def test_reindex_with_no_sources_returns_error(local_config):
    radar = RadarServer(local_config)
    try:
        out = _dispatch(radar, "reindex", {})
        assert "error" in out
        assert "fetch_papers" in out["error"]
    finally:
        radar.jobs.shutdown()


def test_fetch_papers_validates_input(local_config):
    radar = RadarServer(local_config)
    try:
        out = _dispatch(radar, "fetch_papers", {"arxiv_ids": []})
        assert "error" in out
    finally:
        radar.jobs.shutdown()


def test_job_status_unknown_id_returns_error(local_config):
    radar = RadarServer(local_config)
    try:
        out = _dispatch(radar, "job_status", {"job_id": "nonexistent"})
        assert "error" in out
    finally:
        radar.jobs.shutdown()
