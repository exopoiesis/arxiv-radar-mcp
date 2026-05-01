"""arXiv full-text fetcher: HTML → LaTeX cascade.

For one arxiv_id, try in order:
  1. arxiv.org/html/<id>     — author-LaTeX rendered to HTML by arXiv. Math
                                preserved as <math><annotation encoding=
                                "application/x-tex">...</annotation>; we
                                extract the inline LaTeX. Coverage 70-80%
                                of recent submissions (2020+).
  2. arxiv.org/e-print/<id>  — gzipped tar of LaTeX source (or single .tex).
                                Run pylatexenc to expand macros, drop
                                comments, leave equations as inline `$...$`.
                                Coverage ~85-90% of all arxiv submissions.
  3. fail with explicit reason ("PDF-only on arXiv, full text not extractable
     by this server").

Output: markdown saved at <fulltext_dir>/sources/<arxiv_id>.md plus
<fulltext_dir>/sources/<arxiv_id>.meta.json with `{source, fetch_time,
n_chars, n_chunks_after_split}`. Idempotent — already-cached papers are
skipped unless `force=True`.

This module knows nothing about embeddings or chunking. It just turns an
arxiv_id into clean markdown.
"""
from __future__ import annotations

import gzip
import io
import json
import logging
import re
import tarfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import httpx

LOG = logging.getLogger(__name__)

_HTML_URL = "https://arxiv.org/html/{id}"
_EPRINT_URL = "https://arxiv.org/e-print/{id}"
_USER_AGENT = "arxiv-radar-mcp/0.1 (+https://github.com/exopoiesis/arxiv-radar-mcp)"
_TIMEOUT = httpx.Timeout(30.0, connect=10.0)


@dataclass
class FetchResult:
    arxiv_id: str
    source: str | None       # "html" | "latex" | None on failure
    markdown: str | None     # None on failure
    n_chars: int             # 0 on failure
    error: str | None        # human-readable reason on failure


def fetch_paper(arxiv_id: str, *, client: httpx.Client | None = None) -> FetchResult:
    """Try HTML, then LaTeX. Return whichever succeeded first.

    Pass a shared `httpx.Client` when fetching a batch — connection reuse
    cuts wall time on 5+ papers significantly.
    """
    own_client = client is None
    if own_client:
        client = httpx.Client(
            timeout=_TIMEOUT,
            headers={"User-Agent": _USER_AGENT},
            follow_redirects=True,
        )
    try:
        # 1. HTML
        try:
            md = _fetch_html(arxiv_id, client)
            if md:
                return FetchResult(arxiv_id=arxiv_id, source="html",
                                   markdown=md, n_chars=len(md), error=None)
        except _FetchError as e:
            LOG.info(f"[{arxiv_id}] html unavailable: {e}")

        # 2. LaTeX e-print
        try:
            md = _fetch_eprint(arxiv_id, client)
            if md:
                return FetchResult(arxiv_id=arxiv_id, source="latex",
                                   markdown=md, n_chars=len(md), error=None)
        except _FetchError as e:
            LOG.info(f"[{arxiv_id}] e-print unavailable: {e}")

        return FetchResult(
            arxiv_id=arxiv_id, source=None, markdown=None, n_chars=0,
            error=("no HTML or LaTeX source on arXiv — paper is PDF-only. "
                   "PDF parsing is not supported in this server."),
        )
    finally:
        if own_client:
            client.close()


def fetch_and_save(
    arxiv_id: str,
    fulltext_dir: Path,
    *,
    force: bool = False,
    client: httpx.Client | None = None,
) -> FetchResult:
    """Fetch one paper and persist to <fulltext_dir>/sources/<id>.md (+ meta.json).

    Idempotent: returns cached result if .md already exists and `force=False`.
    The meta.json's `n_chunks_after_split` is left as 0 here — the chunker
    fills it in during reindex.
    """
    sources_dir = fulltext_dir / "sources"
    sources_dir.mkdir(parents=True, exist_ok=True)
    md_path = sources_dir / f"{arxiv_id}.md"
    meta_path = sources_dir / f"{arxiv_id}.meta.json"

    if md_path.exists() and meta_path.exists() and not force:
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            return FetchResult(
                arxiv_id=arxiv_id, source=meta.get("source"),
                markdown=md_path.read_text(encoding="utf-8"),
                n_chars=meta.get("n_chars", 0), error=None,
            )
        except (json.JSONDecodeError, OSError) as e:
            LOG.warning(f"[{arxiv_id}] cached meta unreadable, re-fetching: {e}")

    result = fetch_paper(arxiv_id, client=client)
    if result.markdown is not None:
        md_path.write_text(result.markdown, encoding="utf-8")
        meta_path.write_text(json.dumps({
            "arxiv_id": arxiv_id,
            "source": result.source,
            "fetch_time": datetime.now(timezone.utc).isoformat(),
            "n_chars": result.n_chars,
            "n_chunks_after_split": 0,   # filled in by reindex
        }, indent=1), encoding="utf-8")
    return result


# ---------------------------------------------------------------------------
# HTML path
# ---------------------------------------------------------------------------


class _FetchError(Exception):
    pass


def _fetch_html(arxiv_id: str, client: httpx.Client) -> str | None:
    """GET arxiv.org/html/<id>, return clean markdown or None.

    Returns None when the response is the "no HTML available" stub page that
    arXiv serves (status 200 with a fallback message). Raises _FetchError
    on non-200 responses.
    """
    url = _HTML_URL.format(id=arxiv_id)
    r = client.get(url)
    if r.status_code != 200:
        raise _FetchError(f"http {r.status_code}")

    # arxiv serves a fallback HTML on missing renders — usually a tiny page.
    # Sniff before we pay for full parse.
    body = r.text
    if len(body) < 5_000 and ("Conversion is not available" in body
                              or "no HTML available" in body.lower()):
        return None

    try:
        from selectolax.parser import HTMLParser
    except ImportError as e:
        raise _FetchError(f"selectolax not installed: {e}") from e

    return _html_to_markdown(body, parser_cls=HTMLParser)


def _html_to_markdown(html: str, *, parser_cls) -> str:
    """Walk the arxiv HTML structure and emit markdown.

    arxiv.org/html/<id> is LaTeXML-rendered — fairly stable top-level
    structure:
      - <h1 class="ltx_title_document"> paper title
      - <div class="ltx_abstract"> with its own h6 for the heading
      - sections delimited by <h2 class="ltx_title_section"> (top-level)
      - subsections by <h3 class="ltx_title_subsection"> etc.

    Strategy: render the whole article subtree to markdown via _node_to_markdown,
    promoting h2/h3/h4 to ##/###/####. Title and abstract are handled as
    special cases for cleanliness.
    """
    tree = parser_cls(html)
    parts: list[str] = []

    # Title
    title_node = tree.css_first("h1.ltx_title_document, h1.ltx_title")
    if title_node:
        title = _clean_text(title_node)
        if title:
            parts.append(f"# {title}")
        title_node.decompose()

    # Abstract (lives outside ltx_section)
    abstract = tree.css_first("div.ltx_abstract")
    if abstract:
        body = _node_to_markdown(abstract, skip_first_heading=True)
        if body:
            parts.append("## Abstract")
            parts.append(body)
        abstract.decompose()

    # The rest of the article body — h2/h3 become markdown headings.
    article = (tree.css_first("article")
               or tree.css_first("div.ltx_page_main")
               or tree.body)
    if article:
        body = _node_to_markdown(article, skip_first_heading=False)
        if body:
            parts.append(body)

    text = "\n\n".join(p for p in parts if p.strip())
    return _normalize_whitespace(text)


def _node_to_markdown(node, *, skip_first_heading: bool) -> str:
    """Recursively render a node subtree to text/markdown.

    Heading promotion in arxiv (LaTeXML-rendered) HTML is **driven by CSS
    class, not by tag name**. Different paper templates use different h
    levels for the same logical level:

      `\\section{}`     → h2.ltx_title_section  (most common)
                       OR h3.ltx_title_section  (when paper has \\part{})
      `\\subsection{}`  → h3.ltx_title_subsection or h4.ltx_title_subsection
      `\\part{}`        → h2.ltx_title_part     (rare; we skip it — usually
                                                 just labels "Main" vs "Appendix")
      `\\chapter{}`     → h2.ltx_title_chapter  (very rare in arxiv papers)

    So we look at the title class, not the tag, to decide markdown level.

    Other behaviours:
      - <math> nodes become inline `$...$` (or `$$...$$` for display) using
        the LaTeX from <annotation encoding="application/x-tex">.
      - The first <h*> child can be skipped if `skip_first_heading=True`
        (we emitted that one separately as title or abstract heading).
      - <script>/<style>/<nav>/pagination divs are dropped.

    Walks all descendants in document order via selectolax's traverse().
    """
    if node is None:
        return ""

    pieces: list[str] = []
    skipped = not skip_first_heading

    skip_classes = ("ltx_pagination", "ltx_navbar", "ltx_break")
    skip_tags = ("script", "style", "nav")

    for n in node.traverse(include_text=True):
        if n.tag == "-text":
            txt = n.text() or ""
            if txt.strip():
                pieces.append(txt)
            continue

        if n.tag in skip_tags:
            n.decompose()
            continue

        cls = n.attributes.get("class") or ""
        if any(c in cls for c in skip_classes):
            n.decompose()
            continue

        if not skipped and n.tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            skipped = True
            n.decompose()
            continue

        if n.tag == "math":
            tex = _extract_latex_from_math(n)
            if tex:
                is_display = (n.attributes.get("display") == "block"
                              or "ltx_displaymath" in cls)
                pieces.append(f"\n$$\n{tex}\n$$\n" if is_display else f"${tex}$")
            n.decompose()
            continue

        # Heading promotion by CSS class — robust across LaTeXML's level
        # variations (different papers use h2/h3/h4 for the same logical role).
        if n.tag in ("h1", "h2", "h3", "h4", "h5", "h6") and "ltx_title" in cls:
            md_level = _heading_md_level(cls)
            if md_level is None:
                # Unrecognized title class — drop to avoid duplicating text
                # which would otherwise be emitted by the descendant walker.
                n.decompose()
                continue
            heading_text = _clean_text(n)
            if heading_text:
                if pieces and not pieces[-1].endswith("\n"):
                    pieces.append("\n\n")
                pieces.append(f"{'#' * md_level} {heading_text}\n\n")
            n.decompose()
            continue

        # Block elements — insert paragraph break.
        if n.tag in ("p", "div", "section", "article", "li", "tr"):
            if pieces and not pieces[-1].endswith("\n"):
                pieces.append("\n\n")
            continue

    out = "".join(pieces)
    return _normalize_whitespace(out)


def _heading_md_level(cls: str) -> int | None:
    """Map LaTeXML `ltx_title_*` class → markdown heading level (#-count).

    Returns None for class flavours we want to suppress (e.g. ltx_title_part
    in dual-part papers like Pixtral, where the only `ltx_title_part`
    headings are "Main" / "Appendix" wrappers that don't add real content).
    """
    if "ltx_title_document" in cls:
        return 1
    if "ltx_title_chapter" in cls:
        return 2
    if "ltx_title_part" in cls:
        # Skip: usually a meta-grouping label ("Main", "Appendix"), not a
        # real section. Real sections inside have their own ltx_title_section.
        return None
    if "ltx_title_section" in cls:
        return 2
    if "ltx_title_subsection" in cls:
        return 3
    if "ltx_title_subsubsection" in cls:
        return 4
    if "ltx_title_paragraph" in cls:
        return 5
    # Generic ltx_title without a level qualifier — treat as h3 (typical for
    # abstract heading, figure captions promoted to titles, etc.).
    if "ltx_title" in cls:
        return 3
    return None


def _clean_text(node) -> str:
    """Get a node's full text with whitespace normalized to single spaces."""
    if node is None:
        return ""
    text = node.text(deep=True) or ""
    return re.sub(r"\s+", " ", text).strip()


def _extract_latex_from_math(math_node) -> str:
    """Pull the original LaTeX out of MathML's <annotation encoding='application/x-tex'>."""
    annot = math_node.css_first("annotation")
    if annot and annot.attributes.get("encoding") == "application/x-tex":
        return (annot.text() or "").strip()
    # Fallback: MathML text content (lossy but readable).
    return (math_node.text() or "").strip()


def _normalize_whitespace(text: str) -> str:
    """Collapse 3+ blank lines → 2; strip trailing whitespace per line."""
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# LaTeX e-print path
# ---------------------------------------------------------------------------


def _fetch_eprint(arxiv_id: str, client: httpx.Client) -> str | None:
    """GET arxiv.org/e-print/<id>, decode tarball, run pylatexenc."""
    url = _EPRINT_URL.format(id=arxiv_id)
    r = client.get(url)
    if r.status_code != 200:
        raise _FetchError(f"http {r.status_code}")

    payload = r.content
    if not payload:
        raise _FetchError("empty body")

    tex_source = _extract_main_tex(payload)
    if not tex_source:
        raise _FetchError("no .tex file in e-print archive")

    try:
        from pylatexenc.latex2text import LatexNodes2Text
    except ImportError as e:
        raise _FetchError(f"pylatexenc not installed: {e}") from e

    converter = LatexNodes2Text(
        math_mode="verbatim",   # keep $...$ as-is — embedding model handles it
        keep_comments=False,
        strict_latex_spaces=False,
    )
    # pylatexenc crashes on some malformed/custom-macro source files
    # (IndexError in macro arg parsing). Treat any internal exception as
    # "this LaTeX path failed" so we report the failure cleanly instead
    # of the user seeing a stack trace.
    try:
        plain = converter.latex_to_text(tex_source)
    except Exception as e:  # noqa: BLE001 — pylatexenc raises IndexError, KeyError, ...
        raise _FetchError(f"pylatexenc parse error: {type(e).__name__}: {e}") from e
    plain = _normalize_whitespace(plain)
    if not plain:
        return None

    # pylatexenc strips section markers — re-inject markdown headings around
    # \section{...} text we kept in the source. The section_to_md pre-pass
    # below mutates tex_source first, then we re-run the conversion.
    md = _add_markdown_headings(tex_source, plain)
    return md or plain


def _extract_main_tex(payload: bytes) -> str | None:
    """Pull the main .tex content from an arxiv e-print payload.

    arxiv e-prints come in three shapes:
      1. gzipped tar archive (most common, multi-file projects)
      2. plain gzip of a single .tex file (older single-file submissions)
      3. plain text .tex (rare)
    We try each in turn and return the longest .tex we find.
    """
    # Try gzipped tar first.
    try:
        with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as tar:
            tex_files: list[tuple[str, str]] = []
            for member in tar.getmembers():
                if not member.isfile() or not member.name.endswith(".tex"):
                    continue
                f = tar.extractfile(member)
                if f is None:
                    continue
                try:
                    content = f.read().decode("utf-8", errors="replace")
                except UnicodeDecodeError:
                    continue
                tex_files.append((member.name, content))

            if not tex_files:
                return None

            main = _pick_main_tex(tex_files)
            if main is None:
                return None
            return _expand_inputs(main, dict(tex_files))
    except (tarfile.ReadError, OSError):
        pass

    # Try single gzipped .tex.
    try:
        decompressed = gzip.decompress(payload)
        text = decompressed.decode("utf-8", errors="replace")
        if "\\documentclass" in text or "\\begin{document}" in text:
            return text
    except (gzip.BadGzipFile, OSError, UnicodeDecodeError):
        pass

    # Plain text fallback.
    try:
        text = payload.decode("utf-8", errors="replace")
        if "\\documentclass" in text or "\\begin{document}" in text:
            return text
    except UnicodeDecodeError:
        pass

    return None


def _pick_main_tex(tex_files: list[tuple[str, str]]) -> str | None:
    """Pick the .tex file with `\\documentclass` and the most content."""
    candidates = [(n, c) for n, c in tex_files if "\\documentclass" in c]
    if not candidates:
        # No documentclass anywhere — pick the largest .tex.
        candidates = tex_files
    if not candidates:
        return None
    candidates.sort(key=lambda x: len(x[1]), reverse=True)
    return candidates[0][1]


_INPUT_RE = re.compile(r"\\(?:input|include)\{([^}]+)\}")


def _expand_inputs(main_tex: str, all_tex: dict[str, str], depth: int = 0) -> str:
    """Inline `\\input{foo}` and `\\include{foo}` once.

    Limited recursion (depth=2) to avoid pathological loops. Unresolved
    references stay as-is and are stripped by pylatexenc later.
    """
    if depth > 2:
        return main_tex

    def _resolve(match: re.Match[str]) -> str:
        ref = match.group(1).strip()
        for ext in ("", ".tex"):
            for name, content in all_tex.items():
                # Match by basename or by exact match suffix.
                if name.endswith(ref + ext) or name == ref + ext:
                    return _expand_inputs(content, all_tex, depth=depth + 1)
        return match.group(0)

    return _INPUT_RE.sub(_resolve, main_tex)


_SECTION_RE = re.compile(
    r"\\(section|subsection|subsubsection)\*?\{([^}]+)\}"
)


def _add_markdown_headings(tex_source: str, plain: str) -> str | None:
    """Re-inject `## Heading` lines into the pylatexenc-converted plain text.

    pylatexenc renders `\\section{Foo}` as `§ FOO` (section glyph +
    uppercased title) — we replace these with `## Foo` (markdown ATX
    heading + original case from the tex source) so the chunker can split
    on them. Match is case-insensitive and tolerates whitespace/glyph
    differences.

    Best-effort: if a title can't be found in the plain output we skip it.
    """
    headings = [m.group(2).strip() for m in _SECTION_RE.finditer(tex_source)]
    if not headings:
        return None

    out = plain
    cursor = 0
    for h in headings:
        norm = re.sub(r"\s+", " ", h)
        # Match: optional § glyph + whitespace + heading title (case-insensitive,
        # whitespace-tolerant). Limit to the region after our cursor so we
        # walk in tex-source order.
        title_pattern = r"\s*".join(re.escape(w) for w in norm.split())
        full_pattern = re.compile(r"§?\s*" + title_pattern,
                                  flags=re.IGNORECASE)
        m = full_pattern.search(out, cursor)
        if not m:
            continue
        start, end = m.span()
        # Idempotent — skip if we already inserted a marker just before.
        if start >= 3 and out[start - 3:start] == "## ":
            cursor = end
            continue
        out = out[:start] + "## " + h + out[end:]
        cursor = start + 3 + len(h)

    return out
