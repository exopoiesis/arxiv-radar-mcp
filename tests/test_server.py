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
    # async admin (6)
    "fetch_papers", "reindex", "refresh_abstracts",
    "job_status", "job_list", "list_enriched",
    # pre-flight (1)
    "validate_arxiv_ids",
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
        return [{"domain": "chemistry", "papers": 1}]

    def list_enriched(self):
        return ["2503.00001", "2504.00002"]


def test_dispatch_routes_to_method():
    out = _dispatch(_StubRadar(), "search_abstract_text",
                    {"query": "dft", "k": 3})
    assert out == [{"called": "search_abstract_text", "query": "dft", "k": 3}]


def test_dispatch_handles_none_arguments():
    out = _dispatch(_StubRadar(), "list_domains", None)
    assert out == [{"domain": "chemistry", "papers": 1}]


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
        assert out == [{"domain": "chemistry", "papers": 3}]
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


def test_paper_info_full_abstract_returns_untruncated(local_config):
    """U12: full_abstract=true skips the 600-char truncation."""
    radar = RadarServer(local_config)
    try:
        # Tamper a corpus entry to have a long abstract.
        p = radar.papers["2503.00001"]
        p.abstract = "A" * 1500

        truncated = _dispatch(radar, "paper_info", {"arxiv_id": "2503.00001"})
        full = _dispatch(radar, "paper_info",
                         {"arxiv_id": "2503.00001", "full_abstract": True})

        assert truncated["abstract"].endswith("…")
        assert len(truncated["abstract"]) == 601
        assert len(full["abstract"]) == 1500
        assert not full["abstract"].endswith("…")
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


def test_fetch_papers_accepts_force_arg(local_config):
    """U9: force=true must be a documented MCP parameter that round-trips
    through to fetch_and_save."""
    radar = RadarServer(local_config)
    try:
        out = _dispatch(radar, "fetch_papers",
                        {"arxiv_ids": ["2503.12345"], "force": True})
        assert "job_id" in out
        assert out["force"] is True
    finally:
        radar.jobs.shutdown()


def test_fetch_papers_force_threaded_to_fetch_and_save(local_config, monkeypatch):
    """When force=True, the worker must pass force=True to fetch_and_save."""
    import time as _time
    radar = RadarServer(local_config)
    try:
        seen: list[bool] = []

        def fake_fetch_and_save(arxiv_id, fulltext_dir, *, force=False, client=None):
            seen.append(force)
            from arxiv_radar_mcp.fulltext import FetchResult
            return FetchResult(arxiv_id=arxiv_id, source="html",
                               markdown="x", n_chars=1, error=None)

        monkeypatch.setattr("arxiv_radar_mcp.server.fetch_and_save",
                            fake_fetch_and_save)

        out = _dispatch(radar, "fetch_papers",
                        {"arxiv_ids": ["2503.12345"], "force": True})
        # Wait for the job to finish.
        deadline = _time.time() + 5
        while _time.time() < deadline:
            info = radar.jobs.get(out["job_id"])
            if info and info["state"] in ("done", "failed"):
                break
            _time.sleep(0.05)
        assert seen == [True]
    finally:
        radar.jobs.shutdown()


def test_job_status_unknown_id_returns_error(local_config):
    radar = RadarServer(local_config)
    try:
        out = _dispatch(radar, "job_status", {"job_id": "nonexistent"})
        assert "error" in out
    finally:
        radar.jobs.shutdown()


# ----- U5: list_tags filtering -----------------------------------------------


def test_list_tags_default_unfiltered_returns_all(local_config):
    """Without args, list_tags returns every tag, sorted by frequency.
    Same as before U5 — the fixture corpus has only a few tags so the
    default head_limit doesn't truncate it."""
    radar = RadarServer(local_config)
    try:
        out = _dispatch(radar, "list_tags", {})
        names = {row["tag"] for row in out}
        assert {"dft", "ab-initio", "mlip", "gnn",
                "generative-model", "catalysis"} <= names
    finally:
        radar.jobs.shutdown()


def test_list_tags_head_limit_caps_output(local_config):
    radar = RadarServer(local_config)
    try:
        out = _dispatch(radar, "list_tags", {"head_limit": 2})
        assert len(out) == 2
        # Highest-frequency tags first; in the fixture all tags have count 1
        # so we just check we didn't return everything.
    finally:
        radar.jobs.shutdown()


def test_list_tags_min_count_filters(local_config):
    """min_count=2 drops singleton tags. Fixture has all-singletons → empty."""
    radar = RadarServer(local_config)
    try:
        out = _dispatch(radar, "list_tags", {"min_count": 2})
        assert out == []
    finally:
        radar.jobs.shutdown()


def test_list_tags_prefix_filter(local_config):
    radar = RadarServer(local_config)
    try:
        out = _dispatch(radar, "list_tags", {"prefix": "d"})
        names = {row["tag"] for row in out}
        assert "dft" in names
        assert "mlip" not in names
    finally:
        radar.jobs.shutdown()


# ----- U2: validate_arxiv_ids -------------------------------------------------


def test_validate_arxiv_ids_partitions_ok_and_pdf_only(local_config, monkeypatch):
    """HEAD-probe every id, partition into {ok, pdf_only}. Nothing fancy:
    if probe returns True the id has an HTML render (or a stub — caller can
    still try); if False arxiv has no HTML for it (PDF-only)."""
    radar = RadarServer(local_config)
    try:
        # Stub the probe so we don't hit network. Even-suffix → ok, odd → pdf.
        def fake_probe(arxiv_id, *, client=None):
            return int(arxiv_id.replace(".", "")) % 2 == 0
        monkeypatch.setattr("arxiv_radar_mcp.server.probe_html_available",
                            fake_probe)

        out = _dispatch(radar, "validate_arxiv_ids", {
            "arxiv_ids": ["2503.00002", "2503.00003", "2504.00004", "2504.00005"],
        })
        assert out["n_total"] == 4
        assert sorted(out["ok"]) == ["2503.00002", "2504.00004"]
        assert sorted(out["pdf_only"]) == ["2503.00003", "2504.00005"]
    finally:
        radar.jobs.shutdown()


def test_validate_arxiv_ids_skips_probe_for_cached(local_config, monkeypatch, tmp_path):
    """Already-enriched papers don't need a probe — count as ok and don't
    burn arXiv ToS budget."""
    radar = RadarServer(local_config)
    try:
        sources = radar.fulltext_dir / "sources"
        sources.mkdir(parents=True, exist_ok=True)
        (sources / "2503.99999.md").write_text("# cached\nbody", encoding="utf-8")
        (sources / "2503.99999.meta.json").write_text(
            '{"arxiv_id": "2503.99999", "source": "html"}', encoding="utf-8")

        called: list[str] = []
        def fake_probe(arxiv_id, *, client=None):
            called.append(arxiv_id)
            return False
        monkeypatch.setattr("arxiv_radar_mcp.server.probe_html_available",
                            fake_probe)

        out = _dispatch(radar, "validate_arxiv_ids", {
            "arxiv_ids": ["2503.99999", "2504.00000"],
        })
        assert "2503.99999" in out["ok"]
        assert called == ["2504.00000"]  # only the uncached id was probed
    finally:
        radar.jobs.shutdown()


def test_validate_arxiv_ids_empty_list_errors(local_config):
    radar = RadarServer(local_config)
    try:
        out = _dispatch(radar, "validate_arxiv_ids", {"arxiv_ids": []})
        assert "error" in out
    finally:
        radar.jobs.shutdown()


# ----- VRAM release after refresh / reindex / bootstrap ---------------------

def test_release_encoder_vram_proxies_to_encoder_unload(local_config):
    radar = RadarServer(local_config)
    try:
        calls = {"n": 0}

        def _fake_unload():
            calls["n"] += 1
            return True

        radar.encoder.unload = _fake_unload  # type: ignore[method-assign]
        radar._release_encoder_vram()
        assert calls["n"] == 1
    finally:
        radar.jobs.shutdown()


def test_release_encoder_vram_swallows_exceptions(local_config, caplog):
    radar = RadarServer(local_config)
    try:
        def _boom():
            raise RuntimeError("simulated cuda issue")
        radar.encoder.unload = _boom  # type: ignore[method-assign]
        with caplog.at_level("WARNING"):
            radar._release_encoder_vram()  # must not raise
        assert any("encoder.unload() failed" in rec.message
                   for rec in caplog.records)
    finally:
        radar.jobs.shutdown()


def test_blocking_refresh_releases_vram_on_success(monkeypatch, local_config):
    """Bootstrap + nightly tick path: _blocking_refresh must call unload
    after refresh_sources returns, regardless of refresh outcome."""
    from arxiv_radar_mcp.server import _blocking_refresh

    radar = RadarServer(local_config)
    try:
        calls = {"n": 0}
        radar.encoder.unload = lambda: (  # type: ignore[method-assign]
            calls.__setitem__("n", calls["n"] + 1) or True)
        monkeypatch.setattr(
            "arxiv_radar_mcp.server.refresh_sources",
            lambda r, full_rebuild=False: {"strategy": "noop", "total": 0,
                                            "added": 0, "deleted": 0},
        )
        result = _blocking_refresh(radar, full_rebuild=True)
        assert result["strategy"] == "noop"
        assert calls["n"] == 1, "unload must fire after successful refresh"
    finally:
        radar.jobs.shutdown()


def test_blocking_refresh_releases_vram_on_failure(monkeypatch, local_config):
    """Even if refresh_sources blows up, the finally must release VRAM."""
    from arxiv_radar_mcp.server import _blocking_refresh

    radar = RadarServer(local_config)
    try:
        calls = {"n": 0}
        radar.encoder.unload = lambda: (  # type: ignore[method-assign]
            calls.__setitem__("n", calls["n"] + 1) or True)

        def _boom(r, full_rebuild=False):
            raise RuntimeError("simulated refresh failure")
        monkeypatch.setattr("arxiv_radar_mcp.server.refresh_sources", _boom)

        with pytest.raises(RuntimeError, match="simulated refresh failure"):
            _blocking_refresh(radar, full_rebuild=True)
        assert calls["n"] == 1, "unload must fire even on refresh failure"
    finally:
        radar.jobs.shutdown()


def test_blocking_refresh_skipped_when_lock_held(local_config):
    """If the reindex lock is already taken, _blocking_refresh returns
    skipped immediately — no encoder load, no unload needed."""
    from arxiv_radar_mcp.server import _blocking_refresh

    radar = RadarServer(local_config)
    try:
        # Steal the lock.
        assert radar.jobs.acquire_reindex_lock()
        try:
            calls = {"n": 0}
            radar.encoder.unload = lambda: (  # type: ignore[method-assign]
                calls.__setitem__("n", calls["n"] + 1) or True)
            result = _blocking_refresh(radar, full_rebuild=False)
            assert result == {"skipped": "encoder busy"}
            # Lock-busy path returns BEFORE acquiring → no unload fire.
            assert calls["n"] == 0
        finally:
            radar.jobs.release_reindex_lock()
    finally:
        radar.jobs.shutdown()
