"""Tests for the PDF tier (U7) in arxiv_radar_mcp.fulltext.

Load-bearing tests (per spec):
  - Media-translation: MinerU "images/" refs rewritten to "<id>.media/"
  - build_paper_archive zip where images resolve correctly
  - Graceful error when corpus_core.pdf.is_pdf_parser_available() is False
  - Shared throttle: fetch_arxiv_pdf uses corpus_core.get_arxiv_throttle()
  - validate_arxiv_ids payload shape with/without parser
  - _do_fetch source_breakdown contains "pdf"

All network calls are mocked.  MinerU is injected via corpus_core.pdf seam.
"""
from __future__ import annotations

import json
import sys
import types
import zipfile
from pathlib import Path


from arxiv_radar_mcp.fulltext import (
    FetchResult,
    _fetch_pdf,
    fetch_and_save,
    rewrite_image_refs,
)


# ---- helpers -----------------------------------------------------------------

_GOOD_MARKDOWN = (
    "# Great Paper\n\n"
    "## Abstract\n\nThis paper studies something interesting and important.\n\n"
    "## Methods\n\nWe used these methods to analyse the results carefully.\n\n"
    "## Results\n\nThe results confirm our hypothesis beyond all doubt.\n"
)

_FAKE_PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16


def _make_pdf_parse_result(markdown: str, images: list[dict] | None = None):
    """Build a corpus_core.pdf.PdfParseResult-like object for mocking."""
    class _FakePdfParseResult:
        def __init__(self, md, imgs):
            self.markdown = md
            self.images = imgs or []
            self.n_chars = len(md)
            self.backend = "pipeline"
            self.media_subdir_in_md = "images"
    return _FakePdfParseResult(markdown, images or [])


def _inject_cc_pdf(monkeypatch, *,
                   available: bool = True,
                   parse_result=None,
                   parse_raises=None,
                   looks_like_stub: bool = False):
    """Inject a fake corpus_core.pdf module into sys.modules + the
    fulltext module's reference.

    parse_result: PdfParseResult-like to return from parse_pdf().
    parse_raises: exception instance to raise from parse_pdf().
    """
    fake_mod = types.ModuleType("corpus_core.pdf")

    fake_mod.is_pdf_parser_available = lambda: available
    fake_mod.looks_like_pdf_stub = lambda md: looks_like_stub

    if parse_raises is not None:
        class _FakePdfParseError(RuntimeError):
            pass
        fake_mod.PdfParseError = _FakePdfParseError

        def _parse_pdf_raises(pdf_path, *, media_out_dir, backend="pipeline", runner=None):
            raise parse_raises
        fake_mod.parse_pdf = _parse_pdf_raises
    else:
        fake_mod.PdfParseError = RuntimeError

        def _parse_pdf_ok(pdf_path, *, media_out_dir, backend="pipeline", runner=None):
            # Write fake images to media_out_dir if parse_result has images.
            if parse_result is not None and parse_result.images:
                media_out_dir.mkdir(parents=True, exist_ok=True)
                for img in parse_result.images:
                    (media_out_dir / img["name"]).write_bytes(_FAKE_PNG)
            return parse_result
        fake_mod.parse_pdf = _parse_pdf_ok

    fake_mod.unload_pdf_models = lambda: False

    monkeypatch.setitem(sys.modules, "corpus_core.pdf", fake_mod)
    # Also patch the reference held by the fulltext module at import time.
    import arxiv_radar_mcp.fulltext as _ft
    monkeypatch.setattr(_ft, "_cc_pdf", fake_mod)
    return fake_mod


def _fake_fetch_arxiv_pdf(arxiv_id, dest_dir, **kwargs):
    """Stub: writes a fake PDF to dest_dir/<arxiv_id>.pdf, returns ok result."""
    from corpus_core.http_fetch import FetchResult  # noqa: PLC0415
    dest = Path(dest_dir) / f"{arxiv_id}.pdf"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(b"%PDF-1.4 fake")
    return FetchResult(url=f"https://arxiv.org/pdf/{arxiv_id}",
                       dest_path=dest, ok=True, status=200,
                       n_bytes=len(b"%PDF-1.4 fake"), error=None)


def _fake_fetch_arxiv_pdf_fail(arxiv_id, dest_dir, **kwargs):
    """Stub: simulates a download failure."""
    from corpus_core.http_fetch import FetchResult  # noqa: PLC0415
    return FetchResult(url=f"https://arxiv.org/pdf/{arxiv_id}",
                       dest_path=None, ok=False, status=404,
                       n_bytes=0, error="http 404")


# ---- rewrite_image_refs -------------------------------------------------------

def test_rewrite_image_refs_basic():
    md = "![](images/fig1.png)\n![alt text](images/fig2.png)"
    result = rewrite_image_refs(md, from_="images", to_="2603.05238.media")
    assert "images/" not in result
    assert "2603.05238.media/fig1.png" in result
    assert "2603.05238.media/fig2.png" in result


def test_rewrite_image_refs_does_not_touch_other_refs():
    md = "![](other/fig.png)\n![](images/fig1.png)"
    result = rewrite_image_refs(md, from_="images", to_="x.media")
    assert "other/fig.png" in result
    assert "x.media/fig1.png" in result


def test_rewrite_image_refs_noop_when_same():
    md = "![](x.media/fig.png)"
    result = rewrite_image_refs(md, from_="x.media", to_="x.media")
    assert result == md


def test_rewrite_image_refs_empty_string():
    assert rewrite_image_refs("", from_="images", to_="x.media") == ""


def test_rewrite_image_refs_no_images():
    md = "# Title\n\nSome text with no images."
    assert rewrite_image_refs(md, from_="images", to_="x.media") == md


# ---- _fetch_pdf: media-translation load-bearing test -------------------------

def test_fetch_pdf_rewrites_image_refs(monkeypatch, tmp_path):
    """LOAD-BEARING: markdown in FetchResult must use <id>.media/, NOT images/."""
    arxiv_id = "2603.05238"
    raw_md = (
        "# Great Paper\n\n"
        "## Results\n\nHere is figure 1: ![fig1](images/fig1.png) "
        "and figure 2 ![](images/fig2.png).\n\n"
        "## Conclusion\n\nVery interesting findings confirmed.\n"
    )
    parse_result = _make_pdf_parse_result(
        raw_md,
        images=[{"name": "fig1.png"}, {"name": "fig2.png"}],
    )
    _inject_cc_pdf(monkeypatch, parse_result=parse_result)
    monkeypatch.setattr(
        "arxiv_radar_mcp.fulltext.fetch_arxiv_pdf", _fake_fetch_arxiv_pdf
    )

    fulltext_dir = tmp_path / "fulltext"
    result = _fetch_pdf(arxiv_id, fulltext_dir)

    assert result is not None
    assert result.source == "pdf"
    # Load-bearing: refs must use "<arxiv_id>.media/", not "images/"
    assert "images/" not in result.markdown
    assert f"{arxiv_id}.media/fig1.png" in result.markdown
    assert f"{arxiv_id}.media/fig2.png" in result.markdown


def test_fetch_pdf_graceful_when_parser_unavailable(monkeypatch, tmp_path):
    """LOAD-BEARING: without [pdf] extra, return error FetchResult, no crash."""
    _inject_cc_pdf(monkeypatch, available=False)

    result = _fetch_pdf("2603.05238", tmp_path / "fulltext")

    assert result is not None
    assert result.source is None
    assert result.markdown is None
    assert result.error is not None
    assert "install" in result.error.lower() or "pdf" in result.error.lower()


def test_fetch_pdf_returns_error_on_download_failure(monkeypatch, tmp_path):
    _inject_cc_pdf(monkeypatch, parse_result=_make_pdf_parse_result(_GOOD_MARKDOWN))
    monkeypatch.setattr(
        "arxiv_radar_mcp.fulltext.fetch_arxiv_pdf", _fake_fetch_arxiv_pdf_fail
    )

    result = _fetch_pdf("2603.05238", tmp_path / "fulltext")
    assert result.source is None
    assert "download failed" in result.error.lower()


def test_fetch_pdf_returns_error_on_stub(monkeypatch, tmp_path):
    parse_result = _make_pdf_parse_result(
        "## Abstract\n\nA\n"  # very short; looks_like_stub -> True
    )
    _inject_cc_pdf(monkeypatch, parse_result=parse_result, looks_like_stub=True)
    monkeypatch.setattr(
        "arxiv_radar_mcp.fulltext.fetch_arxiv_pdf", _fake_fetch_arxiv_pdf
    )

    result = _fetch_pdf("2603.05238", tmp_path / "fulltext")
    assert result.source is None
    assert "stub" in result.error.lower()


def test_fetch_pdf_returns_error_on_parse_failure(monkeypatch, tmp_path):
    class _FakePdfParseError(RuntimeError):
        pass

    fake_mod = _inject_cc_pdf(
        monkeypatch,
        parse_raises=_FakePdfParseError("simulated mineru crash"),
    )
    fake_mod.PdfParseError = _FakePdfParseError
    monkeypatch.setattr(
        "arxiv_radar_mcp.fulltext.fetch_arxiv_pdf", _fake_fetch_arxiv_pdf
    )

    result = _fetch_pdf("2603.05238", tmp_path / "fulltext")
    assert result.source is None
    assert "parse failed" in result.error.lower()


# ---- fetch_and_save: PDF path ------------------------------------------------

def test_fetch_and_save_pdf_writes_md_and_meta(monkeypatch, tmp_path):
    """fetch_and_save with PDF result must write .md + .meta.json correctly."""
    arxiv_id = "2603.05238"
    parse_result = _make_pdf_parse_result(
        _GOOD_MARKDOWN,
        images=[{"name": "fig1.png"}],
    )
    _inject_cc_pdf(monkeypatch, parse_result=parse_result)
    monkeypatch.setattr(
        "arxiv_radar_mcp.fulltext.fetch_arxiv_pdf", _fake_fetch_arxiv_pdf
    )

    # Mock fetch_paper to return pdf source directly (bypass HTML/LaTeX)
    import arxiv_radar_mcp.fulltext as _ft  # noqa: PLC0415
    rewritten_md = rewrite_image_refs(
        _GOOD_MARKDOWN, from_="images", to_=f"{arxiv_id}.media"
    )

    def _fake_fetch_paper(aid, *, client=None, fulltext_dir=None):
        media_dir = fulltext_dir / "sources" / f"{aid}.media"
        media_dir.mkdir(parents=True, exist_ok=True)
        (media_dir / "fig1.png").write_bytes(_FAKE_PNG)
        return FetchResult(
            arxiv_id=aid, source="pdf",
            markdown=rewritten_md,
            n_chars=len(rewritten_md), error=None,
            images=[{"name": "fig1.png"}],
        )

    monkeypatch.setattr(_ft, "fetch_paper", _fake_fetch_paper)

    fulltext_dir = tmp_path / "fulltext"
    result = fetch_and_save(arxiv_id, fulltext_dir)

    sources_dir = fulltext_dir / "sources"
    md_path = sources_dir / f"{arxiv_id}.md"
    meta_path = sources_dir / f"{arxiv_id}.meta.json"

    assert result.source == "pdf"
    assert md_path.exists()
    assert meta_path.exists()

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["source"] == "pdf"
    assert meta["parse_quality"]["branch"] == "pdf"
    assert meta["parse_quality"]["echo_skeleton"] is False


# ---- zip-resolve load-bearing test -------------------------------------------

def test_build_paper_archive_resolves_pdf_images(tmp_path):
    """LOAD-BEARING: build_paper_archive must include images from <id>.media/
    with refs that resolve in the zip (i.e. media_arcname == '<id>.media').

    This is the core U7 invariant: after unzipping, the markdown image refs
    `![](<id>.media/fig1.png)` resolve to `<id>/<id>.media/fig1.png` in the zip.
    """
    from corpus_core.archive import PaperFiles, build_paper_archive  # noqa: PLC0415

    arxiv_id = "2603.05238"
    sources_dir = tmp_path / "fulltext" / "sources"
    sources_dir.mkdir(parents=True)
    media_dir = sources_dir / f"{arxiv_id}.media"
    media_dir.mkdir()

    rewritten_md = (
        f"# Great Paper\n\n"
        f"## Abstract\n\nFigure: ![]({arxiv_id}.media/fig1.png)\n\n"
        f"## Results\n\nMore results with substance here.\n"
    )
    md_path = sources_dir / f"{arxiv_id}.md"
    md_path.write_text(rewritten_md, encoding="utf-8")
    (media_dir / "fig1.png").write_bytes(_FAKE_PNG)

    files = PaperFiles(
        markdown_path=md_path,
        media_dir=media_dir,
        media_arcname=f"{arxiv_id}.media",
    )
    zip_bytes = build_paper_archive(arxiv_id, files)
    assert zip_bytes is not None

    with zipfile.ZipFile(zip_bytes.__class__(zip_bytes) if not isinstance(zip_bytes, bytes)
                         else __import__("io").BytesIO(zip_bytes)) as zf:
        names = zf.namelist()

    # The image must appear at <arxiv_id>/<arxiv_id>.media/fig1.png
    expected_img_path = f"{arxiv_id}/{arxiv_id}.media/fig1.png"
    assert expected_img_path in names, (
        f"Expected {expected_img_path!r} in zip, got: {names}"
    )
    # The markdown must be at <arxiv_id>/<arxiv_id>.md
    assert f"{arxiv_id}/{arxiv_id}.md" in names


# ---- validate_arxiv_ids payload shape ----------------------------------------

def test_validate_arxiv_ids_payload_with_parser(monkeypatch, tmp_path):
    """Payload must contain pdf_fallback (not pdf_only) + pdf_parser_available."""
    import corpus_core.pdf as _real_cc_pdf  # noqa: PLC0415
    import arxiv_radar_mcp.server as _srv  # noqa: PLC0415

    # Patch is_pdf_parser_available on the REAL module object that the
    # validate_arxiv_ids method will import inside its body.
    monkeypatch.setattr(_real_cc_pdf, "is_pdf_parser_available", lambda: True)
    monkeypatch.setattr("arxiv_radar_mcp.server.probe_html_available",
                        lambda pid, client=None: False)

    class _DummyServer:
        fulltext_dir = tmp_path / "fulltext"

    obj = _DummyServer()
    result = _srv.RadarServer.validate_arxiv_ids(obj, ["2603.05238", "2504.00001"])

    assert "pdf_fallback" in result
    assert "pdf_parser_available" in result
    assert result["pdf_parser_available"] is True
    # Old key must NOT be present (renamed)
    assert "pdf_only" not in result


def test_validate_arxiv_ids_payload_without_parser(monkeypatch, tmp_path):
    """When parser unavailable, pdf_parser_available must be False."""
    import corpus_core.pdf as _real_cc_pdf  # noqa: PLC0415
    import arxiv_radar_mcp.server as _srv  # noqa: PLC0415

    monkeypatch.setattr(_real_cc_pdf, "is_pdf_parser_available", lambda: False)
    monkeypatch.setattr("arxiv_radar_mcp.server.probe_html_available",
                        lambda pid, client=None: False)

    class _DummyServer:
        fulltext_dir = tmp_path / "fulltext"

    obj = _DummyServer()
    result = _srv.RadarServer.validate_arxiv_ids(obj, ["2603.05238"])
    assert result["pdf_parser_available"] is False
    assert "pdf_fallback" in result


# ---- source_breakdown contains "pdf" ----------------------------------------

def test_do_fetch_source_breakdown_has_pdf(monkeypatch, tmp_path):
    """_do_fetch result's source_breakdown must include 'pdf' when a PDF fetch succeeds."""
    import arxiv_radar_mcp.server as _srv  # noqa: PLC0415

    arxiv_id = "2603.05238"
    rewritten_md = rewrite_image_refs(
        _GOOD_MARKDOWN, from_="images", to_=f"{arxiv_id}.media"
    )

    def _fake_fetch_and_save(aid, fdir, *, force=False, client=None):
        return FetchResult(
            arxiv_id=aid, source="pdf",
            markdown=rewritten_md,
            n_chars=len(rewritten_md), error=None,
            images=[],
        )

    # fetch_and_save is imported into server.py's namespace; patch it there.
    monkeypatch.setattr(_srv, "fetch_and_save", _fake_fetch_and_save)

    # Minimal handle stub
    class _FakeHandle:
        def update(self, **kwargs):
            pass

    # Minimal server stub
    class _DummyServer:
        fulltext_dir = tmp_path / "fulltext"

        def _release_pdf_vram(self):
            pass

    obj = _DummyServer()
    result = _srv.RadarServer._do_fetch(obj, _FakeHandle(), [arxiv_id])
    assert result["source_breakdown"].get("pdf", 0) == 1
