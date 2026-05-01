"""Tests for fulltext.py — HTML/LaTeX cascade fetcher.

Network is mocked via httpx.MockTransport. We don't actually hit
arxiv.org from unit tests.
"""
from __future__ import annotations

import gzip
import io
import json
import tarfile
from pathlib import Path

import httpx
import pytest

from arxiv_radar_mcp.fulltext import (FetchResult, fetch_and_save, fetch_paper,
                                      _add_markdown_headings, _expand_inputs,
                                      _extract_main_tex, _pick_main_tex)


# ----- helpers ---------------------------------------------------------------


_FAKE_HTML = """<!DOCTYPE html>
<html>
<head><title>Test paper</title></head>
<body>
<article>
<h1 class="ltx_title_document">A test paper</h1>
<div class="ltx_authors">Alice and Bob</div>
<div class="ltx_abstract">
  <h6 class="ltx_title">Abstract</h6>
  We test the fetcher.
</div>
<section class="ltx_section">
  <h2 class="ltx_title">Methods</h2>
  We do <math display="inline"><annotation encoding="application/x-tex">E = mc^2</annotation></math> things.
</section>
<section class="ltx_section">
  <h2 class="ltx_title">Results</h2>
  Stuff worked.
</section>
</article>
</body>
</html>
"""


_FAKE_LATEX_SOURCE = r"""\documentclass{article}
\title{Test paper}
\begin{document}
\maketitle

% comment to drop

\section{Methods}
We use $\alpha = 1$ throughout.

\section{Results}
We observed strong correlations.

\end{document}
"""


def _make_eprint_tarball(tex_content: str, name: str = "main.tex") -> bytes:
    """Build a gzipped tar like arxiv serves for /e-print."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        info = tarfile.TarInfo(name=name)
        encoded = tex_content.encode("utf-8")
        info.size = len(encoded)
        tar.addfile(info, io.BytesIO(encoded))
    return buf.getvalue()


def _client_with_handler(handler):
    """Build an httpx.Client with a custom request handler."""
    transport = httpx.MockTransport(handler)
    return httpx.Client(transport=transport)


# ----- HTML path -------------------------------------------------------------


def test_fetch_paper_html_success():
    def handler(request):
        if "/html/" in str(request.url):
            return httpx.Response(200, text=_FAKE_HTML)
        return httpx.Response(404)

    with _client_with_handler(handler) as client:
        result = fetch_paper("2503.99999", client=client)

    assert result.source == "html"
    assert result.markdown is not None
    assert "A test paper" in result.markdown
    assert "Methods" in result.markdown
    # Inline math should be preserved as $...$.
    assert "E = mc^2" in result.markdown
    assert result.error is None


def test_fetch_paper_html_falls_back_to_latex_on_404():
    eprint_tarball = _make_eprint_tarball(_FAKE_LATEX_SOURCE)

    def handler(request):
        url = str(request.url)
        if "/html/" in url:
            return httpx.Response(404)
        if "/e-print/" in url:
            return httpx.Response(200, content=eprint_tarball,
                                  headers={"content-type": "application/x-eprint-tar"})
        return httpx.Response(404)

    with _client_with_handler(handler) as client:
        result = fetch_paper("2503.99999", client=client)

    assert result.source == "latex"
    assert result.markdown is not None
    assert "Methods" in result.markdown


def test_fetch_paper_html_stub_falls_through():
    """arxiv sometimes returns 200 with 'no HTML available' — we should fall through."""
    eprint_tarball = _make_eprint_tarball(_FAKE_LATEX_SOURCE)
    stub_html = "<html><body>Conversion is not available for this paper</body></html>"

    def handler(request):
        url = str(request.url)
        if "/html/" in url:
            return httpx.Response(200, text=stub_html)
        if "/e-print/" in url:
            return httpx.Response(200, content=eprint_tarball)
        return httpx.Response(404)

    with _client_with_handler(handler) as client:
        result = fetch_paper("2503.99999", client=client)

    assert result.source == "latex"  # fell through HTML stub


def test_fetch_paper_both_fail_returns_error():
    def handler(request):
        return httpx.Response(404)

    with _client_with_handler(handler) as client:
        result = fetch_paper("2503.99999", client=client)

    assert result.source is None
    assert result.markdown is None
    assert result.error is not None
    assert "PDF-only" in result.error or "no HTML" in result.error
    assert "lab-corpus" not in result.error  # this repo is self-contained


# ----- LaTeX path ------------------------------------------------------------


def test_extract_main_tex_picks_documentclass():
    tarball = _make_eprint_tarball(_FAKE_LATEX_SOURCE)
    out = _extract_main_tex(tarball)
    assert out is not None
    assert "\\documentclass" in out
    assert "Methods" in out


def test_extract_main_tex_handles_plain_gzip():
    """Some old submissions are plain gzip of a single .tex."""
    plain = gzip.compress(_FAKE_LATEX_SOURCE.encode("utf-8"))
    out = _extract_main_tex(plain)
    assert out is not None
    assert "\\documentclass" in out


def test_extract_main_tex_returns_none_on_garbage():
    assert _extract_main_tex(b"not a tarball, not gzip, not tex") is None


def test_pick_main_tex_prefers_documentclass_largest():
    candidates = [
        ("aux.tex", "% just aux\n"),
        ("main.tex", "\\documentclass{article}\n" + "x" * 1000),
        ("intro.tex", "\\documentclass{article}\n" + "y" * 100),
    ]
    out = _pick_main_tex(candidates)
    assert out and out.startswith("\\documentclass")
    assert "x" * 100 in out  # the larger one


def test_expand_inputs_resolves_basename_match():
    main = "\\input{methods}\nmain stuff"
    all_tex = {"methods.tex": "method body"}
    out = _expand_inputs(main, all_tex)
    assert "method body" in out
    assert "main stuff" in out


def test_expand_inputs_leaves_unresolved_inputs_alone():
    main = "\\input{not_there}\nmain stuff"
    out = _expand_inputs(main, {})
    assert "\\input{not_there}" in out


def test_add_markdown_headings_injects_section_markers():
    tex = r"""
\section{Methods}
\section{Results}
"""
    plain = "Methods\n\nbody1\n\nResults\n\nbody2"
    out = _add_markdown_headings(tex, plain)
    assert out is not None
    assert "## Methods" in out
    assert "## Results" in out


def test_add_markdown_headings_returns_none_when_no_sections():
    out = _add_markdown_headings("just text, no sections", "just text")
    assert out is None


# ----- fetch_and_save (cache layer) ------------------------------------------


def test_fetch_and_save_writes_md_and_meta(tmp_path: Path):
    def handler(request):
        if "/html/" in str(request.url):
            return httpx.Response(200, text=_FAKE_HTML)
        return httpx.Response(404)

    with _client_with_handler(handler) as client:
        result = fetch_and_save("2503.99999", tmp_path, client=client)

    assert result.markdown is not None
    md_path = tmp_path / "sources" / "2503.99999.md"
    meta_path = tmp_path / "sources" / "2503.99999.meta.json"
    assert md_path.exists()
    assert meta_path.exists()

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["arxiv_id"] == "2503.99999"
    assert meta["source"] == "html"
    assert meta["n_chars"] > 0
    assert "fetch_time" in meta


def test_fetch_and_save_idempotent_returns_cached(tmp_path: Path):
    """Second call should not re-fetch."""
    call_count = {"n": 0}

    def handler(request):
        call_count["n"] += 1
        if "/html/" in str(request.url):
            return httpx.Response(200, text=_FAKE_HTML)
        return httpx.Response(404)

    with _client_with_handler(handler) as client:
        first = fetch_and_save("2503.99999", tmp_path, client=client)
        second = fetch_and_save("2503.99999", tmp_path, client=client)

    assert first.markdown == second.markdown
    # Only one HTTP call (the first one); second was served from cache.
    assert call_count["n"] == 1


def test_fetch_and_save_force_refetches(tmp_path: Path):
    call_count = {"n": 0}

    def handler(request):
        call_count["n"] += 1
        return httpx.Response(200, text=_FAKE_HTML)

    with _client_with_handler(handler) as client:
        fetch_and_save("2503.99999", tmp_path, client=client)
        fetch_and_save("2503.99999", tmp_path, client=client, force=True)

    assert call_count["n"] == 2  # forced re-fetch
