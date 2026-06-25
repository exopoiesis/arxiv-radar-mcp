"""arXiv full-text fetcher: HTML -> LaTeX -> PDF cascade.

For one arxiv_id, try in order:
  1. arxiv.org/html/<id>     -- author-LaTeX rendered to HTML by arXiv. Math
                                preserved as <math><annotation encoding=
                                "application/x-tex">...</annotation>; we
                                extract the inline LaTeX. Coverage 70-80%
                                of recent submissions (2020+).
  2. arxiv.org/e-print/<id>  -- gzipped tar of LaTeX source (or single .tex).
                                Run pylatexenc to expand macros, drop
                                comments, leave equations as inline `$...$`.
                                Coverage ~85-90% of all arxiv submissions.
  3. arxiv.org/pdf/<id>      -- PDF download + MinerU parse (tier 3, U7).
                                Available when `arxiv-radar-mcp[pdf]` /
                                `corpus-core[pdf]` is installed.  Covers the
                                remaining ~10-25% of PDF-only submissions.
                                Install: pip install arxiv-radar-mcp[pdf]
                                Without the extra: graceful error, no crash.
  4. fail with explicit reason when all three tiers are exhausted.

Output: markdown saved at <fulltext_dir>/sources/<arxiv_id>.md plus
<fulltext_dir>/sources/<arxiv_id>.meta.json with `{source, fetch_time,
n_chars, n_chunks_after_split, images}`. Figure images referenced in the
HTML render are downloaded into <fulltext_dir>/sources/<arxiv_id>.media/
and the markdown carries `![caption](<id>.media/<name>)` refs.
For the PDF tier, images are parsed by MinerU and placed in the same
<arxiv_id>.media/ dir with refs rewritten from MinerU's "images/" to
"<arxiv_id>.media/".
Idempotent -- already-cached papers are skipped unless `force=True`.

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
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin

import httpx

from corpus_core.archive import PaperFiles
from corpus_core.archive import build_paper_archive as _core_build_archive

from corpus_core.http_fetch import (
    ARXIV_RATE_LIMIT_S,
    fetch_arxiv_pdf,
    get_arxiv_throttle,
    request_with_retry,
)

# corpus_core.pdf is imported lazily inside _fetch_pdf() so that
# `import arxiv_radar_mcp.fulltext` stays cheap on hosts without the
# corpus-core[pdf] extra.  The module-level name below is just a sentinel.
import corpus_core.pdf as _cc_pdf

LOG = logging.getLogger(__name__)

_HTML_URL = "https://arxiv.org/html/{id}"
_EPRINT_URL = "https://arxiv.org/e-print/{id}"
_USER_AGENT = "arxiv-radar-mcp/0.1 (+https://github.com/exopoiesis/arxiv-radar-mcp)"
_TIMEOUT = httpx.Timeout(30.0, connect=10.0)


# Backwards-compatible aliases. Throttle state and retry logic live in
# `corpus_core.http_fetch` so the arxiv-radar fetcher and the lab-corpus
# `ingest_arxiv_pdf` tool share **one** module-global rate limiter when
# they run together in the combined image.
_RATE_LIMIT_S = ARXIV_RATE_LIMIT_S


def _throttle() -> None:
    """Block until the next arxiv.org GET respects the rate limit."""
    get_arxiv_throttle().wait()


def _request_with_retry(
    client: httpx.Client, url: str, *, max_attempts: int = 3,
) -> httpx.Response:
    """GET with arXiv rate-limit + exponential backoff on 429/503.

    Thin wrapper over ``corpus_core.http_fetch.request_with_retry`` that
    pins the arXiv throttle and the legacy backoff seed (``_RATE_LIMIT_S``).
    """
    return request_with_retry(
        client, url,
        throttle=get_arxiv_throttle(),
        max_attempts=max_attempts,
        backoff_seed_s=_RATE_LIMIT_S,
    )


@dataclass
class FetchResult:
    arxiv_id: str
    source: str | None       # "html" | "latex" | None on failure
    markdown: str | None     # None on failure
    n_chars: int             # 0 on failure
    error: str | None        # human-readable reason on failure
    # Figure images discovered in the HTML render, as {"name": local filename,
    # "url": absolute source URL}. Empty for the latex path and on failure.
    # `fetch_and_save` downloads these into <id>.media/ and records the ones
    # that landed on disk.
    images: list[dict] = field(default_factory=list)


def probe_html_available(
    arxiv_id: str, *, client: httpx.Client | None = None,
) -> bool:
    """HEAD arxiv.org/html/<id>; True iff the path exists (2xx/3xx).

    Lightweight pre-flight for `validate_arxiv_ids`: a 404 means PDF-only
    on arXiv (no HTML render), a 200 means the HTML path exists. We do
    NOT inspect the body, so echo-skeleton / 'no HTML available' stub
    pages still count as ok=True. Caller then knows to expect at least
    one of HTML / e-print to work for those IDs.

    Throttled by `_throttle()` on the same module-global lock as
    `_request_with_retry`, so this and active fetches share arXiv's
    1 req / 3 sec budget.
    """
    own_client = client is None
    if own_client:
        client = httpx.Client(
            timeout=_TIMEOUT,
            headers={"User-Agent": _USER_AGENT},
            follow_redirects=True,
        )
    try:
        url = _HTML_URL.format(id=arxiv_id)
        _throttle()
        try:
            r = client.head(url)
        except httpx.HTTPError as e:
            LOG.info(f"probe {arxiv_id}: transport error {type(e).__name__}: {e}")
            return False
        return 200 <= r.status_code < 400
    finally:
        if own_client:
            client.close()


def rewrite_image_refs(markdown: str, *, from_: str, to_: str) -> str:
    """Rewrite image reference directory in markdown.

    Replaces `![](<from_>/name)` and `![alt](<from_>/name)` with
    `![](<to_>/name)` (and same with alt).  Used to translate MinerU's
    native "images/" subdir refs to the arxiv-radar public "<id>.media/"
    convention required by the download zip and frontend consumers.

    Only matches refs where the directory prefix is exactly `from_`
    followed by `/` -- does not touch refs that already use `to_` or
    unrelated refs.

    Parameters
    ----------
    markdown : str
        Input markdown text.
    from_ : str
        Source directory prefix (e.g. "images").
    to_ : str
        Target directory prefix (e.g. "2603.05238.media").

    Returns
    -------
    str
        Markdown with all matching refs rewritten.
    """
    if not from_ or not to_ or from_ == to_:
        return markdown
    # Match ![anything](<from_>/filename) -- captures optional alt text
    # and the filename.  Uses a non-greedy match for the alt text so
    # multiple images on the same line are handled correctly.
    pattern = re.compile(
        r"(!\[[^\]]*\]\()" + re.escape(from_) + r"/"
    )
    return pattern.sub(r"\g<1>" + to_ + "/", markdown)


def fetch_paper(
    arxiv_id: str,
    *,
    client: httpx.Client | None = None,
    fulltext_dir: Path | None = None,
) -> FetchResult:
    """Try HTML, then LaTeX, then PDF (tier 3, U7). Return first success.

    Pass a shared `httpx.Client` when fetching a batch -- connection reuse
    cuts wall time on 5+ papers significantly.

    `fulltext_dir` is required for the PDF tier (MinerU writes images to
    `<fulltext_dir>/sources/<id>.media/`).  When None, the PDF tier is
    skipped even if MinerU is available.
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
            fetched = _fetch_html(arxiv_id, client)
            if fetched:
                md, images = fetched
                return FetchResult(arxiv_id=arxiv_id, source="html",
                                   markdown=md, n_chars=len(md), error=None,
                                   images=images)
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

        # 3. PDF tier (U7) -- only when fulltext_dir provided and MinerU available
        if fulltext_dir is not None:
            pdf_result = _fetch_pdf(arxiv_id, fulltext_dir)
            if pdf_result is not None:
                return pdf_result

        pdf_tier_hint = (
            " Install corpus-core[pdf] (pip install arxiv-radar-mcp[pdf]) "
            "to enable PDF-tier parsing for this paper."
            if not _cc_pdf.is_pdf_parser_available()
            else ""
        )
        return FetchResult(
            arxiv_id=arxiv_id, source=None, markdown=None, n_chars=0,
            error=(
                "no HTML or LaTeX source on arXiv -- paper is PDF-only."
                + pdf_tier_hint
            ),
        )
    finally:
        if own_client:
            client.close()


def _fetch_pdf(arxiv_id: str, fulltext_dir: Path) -> FetchResult | None:
    """Tier-3 PDF fetch + MinerU parse.  Returns FetchResult or None on error.

    None is returned when:
      - MinerU is not installed (graceful degradation).
      - PDF download fails.
      - MinerU parse fails.
      - Parsed markdown looks like a stub (scan-only PDF with no OCR layer).

    In all failure cases a human-readable FetchResult(error=...) is
    returned (NOT None) so the caller can surface the reason.  None is
    returned only when the PDF tier is silently unavailable.

    Image files are placed in <fulltext_dir>/sources/<arxiv_id>.media/.
    Markdown refs are rewritten from MinerU's native "images/" to
    "<arxiv_id>.media/" so they resolve correctly in the download zip.
    """
    if not _cc_pdf.is_pdf_parser_available():
        return FetchResult(
            arxiv_id=arxiv_id, source=None, markdown=None, n_chars=0,
            error=(
                "PDF-only paper on arXiv; install arxiv-radar-mcp[pdf] "
                "to enable PDF-tier parsing."
            ),
        )

    sources_dir = fulltext_dir / "sources"
    inbox_dir = fulltext_dir / "inbox"
    inbox_dir.mkdir(parents=True, exist_ok=True)

    # Download PDF -- uses the shared arxiv throttle from corpus_core.
    fetch_res = fetch_arxiv_pdf(arxiv_id, inbox_dir, overwrite=False)
    if not fetch_res.ok:
        return FetchResult(
            arxiv_id=arxiv_id, source=None, markdown=None, n_chars=0,
            error=f"PDF download failed: {fetch_res.error or f'http {fetch_res.status}'}",
        )

    pdf_path = fetch_res.dest_path
    assert pdf_path is not None  # ok=True guarantees dest_path is set

    # MinerU parse -- images go directly to <id>.media/ (our public dir).
    media_dir = sources_dir / f"{arxiv_id}.media"
    try:
        parse_res = _cc_pdf.parse_pdf(
            pdf_path,
            media_out_dir=media_dir,
        )
    except _cc_pdf.PdfParseError as exc:
        return FetchResult(
            arxiv_id=arxiv_id, source=None, markdown=None, n_chars=0,
            error=f"PDF parse failed: {exc}",
        )

    # Rewrite MinerU's "images/" refs -> "<arxiv_id>.media/" so the
    # download zip resolves correctly.
    md = rewrite_image_refs(
        parse_res.markdown,
        from_=parse_res.media_subdir_in_md,
        to_=f"{arxiv_id}.media",
    )

    if _cc_pdf.looks_like_pdf_stub(md):
        return FetchResult(
            arxiv_id=arxiv_id, source=None, markdown=None, n_chars=0,
            error="PDF parse produced stub (scan-only PDF or no text layer)",
        )

    images = [{"name": img["name"]} for img in parse_res.images]
    return FetchResult(
        arxiv_id=arxiv_id,
        source="pdf",
        markdown=md,
        n_chars=len(md),
        error=None,
        images=images,
    )


def fetch_and_save(
    arxiv_id: str,
    fulltext_dir: Path,
    *,
    force: bool = False,
    client: httpx.Client | None = None,
) -> FetchResult:
    """Fetch one paper and persist to <fulltext_dir>/sources/<id>.md (+ meta.json).

    Idempotent: returns cached result if .md already exists and `force=False`.
    The meta.json's `n_chunks_after_split` is left as 0 here -- the chunker
    fills it in during reindex.

    U7: passes `fulltext_dir` to `fetch_paper` so the PDF tier (tier 3)
    can place its images in `sources/<id>.media/`.  For source="pdf" we do
    NOT call _download_images (images are already on disk from MinerU).
    """
    sources_dir = fulltext_dir / "sources"
    sources_dir.mkdir(parents=True, exist_ok=True)
    md_path = sources_dir / f"{arxiv_id}.md"
    meta_path = sources_dir / f"{arxiv_id}.meta.json"
    media_dir = sources_dir / f"{arxiv_id}.media"

    if md_path.exists() and meta_path.exists() and not force:
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            return FetchResult(
                arxiv_id=arxiv_id, source=meta.get("source"),
                markdown=md_path.read_text(encoding="utf-8"),
                n_chars=meta.get("n_chars", 0), error=None,
                images=meta.get("images", []),
            )
        except (json.JSONDecodeError, OSError) as e:
            LOG.warning(f"[{arxiv_id}] cached meta unreadable, re-fetching: {e}")

    # Use a shared client for the page fetch AND the image downloads so they
    # ride the same connection pool + arXiv rate limiter.
    own_client = client is None
    if own_client:
        client = httpx.Client(
            timeout=_TIMEOUT,
            headers={"User-Agent": _USER_AGENT},
            follow_redirects=True,
        )
    try:
        # Pass fulltext_dir so fetch_paper can activate the PDF tier (U7).
        result = fetch_paper(arxiv_id, client=client, fulltext_dir=fulltext_dir)
        if result.markdown is not None:
            md_path.write_text(result.markdown, encoding="utf-8")

            if result.source == "pdf":
                # PDF tier: images already on disk from MinerU parse.
                # result.images already carries {name} dicts from _fetch_pdf.
                # Enrich with n_bytes from disk for meta consistency.
                saved = []
                for img in result.images:
                    name = img.get("name")
                    if not name:
                        continue
                    p = media_dir / name
                    if p.exists():
                        saved.append({"name": name, "n_bytes": p.stat().st_size})
                result.images = saved
            else:
                saved = _download_images(result.images, media_dir, client=client)
                result.images = saved

            meta_path.write_text(json.dumps({
                "arxiv_id": arxiv_id,
                "source": result.source,
                "fetch_time": datetime.now(timezone.utc).isoformat(),
                "n_chars": result.n_chars,
                "n_chunks_after_split": 0,   # filled in by reindex
                "images": result.images,
                "parse_quality": _parse_quality(result),
            }, indent=1), encoding="utf-8")
        return result
    finally:
        if own_client:
            client.close()


# ---------------------------------------------------------------------------
# Parse quality observability
# ---------------------------------------------------------------------------


def _parse_quality(result: "FetchResult") -> dict:
    """Build a parse_quality summary dict for meta.json.

    Records which branch of the fetch cascade produced the markdown,
    basic structural statistics, and whether the echo-skeleton detector
    was triggered (it always fires during _fetch_html; its result is
    what determines whether we fell through to e-print). The function
    NEVER changes any parsing logic or thresholds -- it only observes.

    Fields:
      branch          "html" | "latex" | "pdf" | None (same as FetchResult.source)
      n_headings      int  -- number of ## headings in the final markdown
      avg_body_len    float -- mean chars per inter-heading span
      echo_skeleton   bool -- True iff _looks_like_echo_skeleton fired on
                              the html render (always False for latex/pdf branch,
                              since we only call the detector in _fetch_html)
    """
    if result.markdown is None:
        return {
            "branch": None,
            "n_headings": 0,
            "avg_body_len": 0.0,
            "echo_skeleton": False,
        }

    lines = result.markdown.split("\n")
    heading_idxs = [i for i, ln in enumerate(lines) if ln.startswith("## ")]
    n_headings = len(heading_idxs)

    if n_headings == 0:
        avg_body_len = float(len(result.markdown))
    else:
        body_lengths: list[int] = []
        for k, hi in enumerate(heading_idxs):
            end = heading_idxs[k + 1] if k + 1 < n_headings else len(lines)
            span = "\n".join(lines[hi + 1:end]).strip()
            body_lengths.append(len(span))
        avg_body_len = sum(body_lengths) / len(body_lengths) if body_lengths else 0.0

    # echo_skeleton: only relevant for the html branch.
    # For latex and pdf, the detector is not consulted.
    echo_skeleton = False
    if result.source == "html" and result.markdown:
        echo_skeleton = _looks_like_echo_skeleton(result.markdown)

    return {
        "branch": result.source,
        "n_headings": n_headings,
        "avg_body_len": round(avg_body_len, 1),
        "echo_skeleton": echo_skeleton,
    }


# ---------------------------------------------------------------------------
# Archive (download side-channel)
# ---------------------------------------------------------------------------


def paper_files(fulltext_dir: Path, arxiv_id: str) -> PaperFiles:
    """Locate a fetched paper's pieces for the download archive.

    arxiv-radar keeps figures in `sources/<id>.media/` and the markdown
    refs them as `![](<id>.media/<name>)`, so the in-archive subdir name
    matches the source dir name.
    """
    sources_dir = fulltext_dir / "sources"
    return PaperFiles(
        markdown_path=sources_dir / f"{arxiv_id}.md",
        media_dir=sources_dir / f"{arxiv_id}.media",
        media_arcname=f"{arxiv_id}.media",
        meta_path=sources_dir / f"{arxiv_id}.meta.json",
    )


def build_paper_archive(fulltext_dir: Path, arxiv_id: str) -> bytes | None:
    """Zip a fetched paper (markdown + figures + meta) into one `<id>/`
    folder. Thin adapter over `corpus_core.build_paper_archive` — see there
    for the layout. Returns None when the paper has not been fetched yet."""
    return _core_build_archive(arxiv_id, paper_files(fulltext_dir, arxiv_id))


# ---------------------------------------------------------------------------
# HTML path
# ---------------------------------------------------------------------------


class _FetchError(Exception):
    pass


def _fetch_html(
    arxiv_id: str, client: httpx.Client,
) -> tuple[str, list[dict]] | None:
    """GET arxiv.org/html/<id>, return (clean markdown, image manifest) or None.

    The image manifest is a list of {"name", "url"} for every <img> in the
    render, with `url` resolved against the (post-redirect) page URL so the
    caller can download them. Markdown carries `![alt](<id>.media/<name>)`
    refs to the same names.

    Returns None when the response is the "no HTML available" stub page that
    arXiv serves (status 200 with a fallback message), OR when the body
    parses to a "skeleton-only" render — section headings present but
    every body is empty/echoes the heading (the `\\input{...}`-not-resolved
    failure mode). In both cases callers fall through to the e-print path.

    Raises _FetchError on non-200 responses.
    """
    url = _HTML_URL.format(id=arxiv_id)
    r = _request_with_retry(client, url)
    if r.status_code != 200:
        raise _FetchError(f"http {r.status_code}")

    # arxiv serves a fallback HTML on missing renders — usually a tiny page.
    # Sniff before we pay for full parse. Older stub: "no HTML available"
    # (<5 KB); newer stub: a ~11 KB shell whose <title> is empty and body
    # says "No HTML" — catch both.
    body = r.text
    if len(body) < 5_000 and ("Conversion is not available" in body
                              or "no HTML available" in body.lower()):
        return None
    if len(body) < 15_000 and "<title> | arXiv e-print repository</title>" in body:
        return None

    try:
        from selectolax.parser import HTMLParser
    except ImportError as e:
        raise _FetchError(f"selectolax not installed: {e}") from e

    images: list[dict] = []
    md = _html_to_markdown(
        body, parser_cls=HTMLParser,
        base_url=str(r.url), image_dir=f"{arxiv_id}.media", images=images,
    )
    if md and _looks_like_echo_skeleton(md):
        LOG.info(f"[{arxiv_id}] html render is echo-skeleton "
                 f"(headings present, bodies empty); falling through to e-print")
        return None
    return md, images


_NUM_PREFIX_RE = re.compile(
    r"^(?:appendix\s+[a-z]+\s+|[0-9]+\.?\s+|[ivxlcdm]+\.?\s+|[a-z]\.\s+)",
    re.IGNORECASE,
)


def _normalize_heading_for_compare(text: str) -> str:
    """Strip leading 'numbering' so '1 Introduction' compares equal to
    'Introduction' when both are echo-skeleton renders. Handles:
        '1 Introduction'     → 'introduction'
        '1. Introduction'    → 'introduction'
        'I Introduction'     → 'introduction'      (Roman)
        'IV Methods'         → 'methods'
        'A. Foo'             → 'foo'
        'Appendix A Foo'     → 'foo'
    """
    out = text.strip().lower()
    # repeat once — 'Appendix A Foo' → 'A Foo' → 'foo'
    for _ in range(2):
        new = _NUM_PREFIX_RE.sub("", out)
        if new == out:
            break
        out = new
    return out.strip()


def _looks_like_echo_skeleton(md: str) -> bool:
    """True iff the markdown looks like arXiv HTML's `\\input{}`-not-resolved
    failure mode: every section heading is present, but the body of each
    section is empty or just repeats the heading text.

    Examples (real, observed 2026-05-08 in dogfood batch):
        ## Abstract\n\nAbstract\n\n## 1 Introduction\n\nIntroduction\n\n...

    Heuristic (tuned on 4 known-bad papers — 2411.12261, 2510.26991,
    2604.21613, 2512.16803):
      - need at least 3 ##/### headings to have a meaningful denominator
      - for each (heading_i, heading_{i+1}) span, normalize the heading
        text (drop "1 " / "I " / "Appendix A " prefixes) and treat the
        body as pure-echo when it equals the normalized heading text
      - skeleton if more than 70% of spans are <10 chars OR average body
        length is <50 chars

    Returns False for genuinely short papers (<3 headings) — those flow
    through to e-print only on other failures.
    """
    lines = md.split("\n")
    heading_idxs = [i for i, ln in enumerate(lines) if ln.startswith("#")]
    if len(heading_idxs) < 3:
        return False

    body_lengths: list[int] = []
    for k, hi in enumerate(heading_idxs):
        end = heading_idxs[k + 1] if k + 1 < len(heading_idxs) else len(lines)
        raw_heading = lines[hi].lstrip("#").strip()
        heading_norm = _normalize_heading_for_compare(raw_heading)
        body_text = "\n".join(lines[hi + 1:end]).strip()
        body_norm = _normalize_heading_for_compare(body_text)
        # Pure echo: body normalises to the heading
        if body_norm == heading_norm:
            body_lengths.append(0)
            continue
        # Body starts with the heading word and adds little after
        stripped = body_text
        if stripped.lower().startswith(raw_heading.lower()):
            stripped = stripped[len(raw_heading):].strip()
        elif body_norm.startswith(heading_norm) and heading_norm:
            # length-conservative strip via lowercase index
            idx = body_text.lower().find(heading_norm)
            if idx != -1:
                stripped = body_text[idx + len(heading_norm):].strip()
        body_lengths.append(len(stripped))

    n = len(body_lengths)
    near_empty = sum(1 for s in body_lengths if s < 10)
    avg = sum(body_lengths) / n
    return (near_empty / n) > 0.70 or avg < 50


def _html_to_markdown(
    html: str, *, parser_cls,
    base_url: str = "", image_dir: str = "", images: list[dict] | None = None,
) -> str:
    """Walk the arxiv HTML structure and emit markdown.

    When `images` (a mutable list) is supplied, every <img> is recorded as
    {"name", "url"} (url resolved against `base_url`) and rendered inline as
    `![alt](image_dir/name)`. Omit `images` to skip figure handling entirely
    (back-compat: the bare two-arg call still returns plain markdown).

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
        body = _node_to_markdown(
            article, skip_first_heading=False,
            base_url=base_url, image_dir=image_dir, images=images,
        )
        if body:
            parts.append(body)

    text = "\n\n".join(p for p in parts if p.strip())
    return _normalize_whitespace(text)


def _iter_descendants(root):
    """DFS pre-order over `root`'s descendants (text + element nodes).

    Worked around two `selectolax` gotchas observed in the wild:

      * `.traverse(include_text=True)` does not stay within the starting
        node — it crosses out into siblings/ancestors, so calling it on
        `div.ltx_abstract` happily yields the next `<section>` as well.
        That leaked the whole article into our abstract render and made
        the U10 echo-skeleton heuristic fire on perfectly good papers.
      * When the consumer calls `n.decompose()` mid-walk, descendants
        of `n` should also disappear from the iteration. With this DFS
        we honour that by re-checking on each yield whether `n` is still
        attached to `root`; orphans are silently skipped.
    """
    if root is None:
        return

    snapshot: list = []

    def _push(parent) -> None:
        for c in parent.iter(include_text=True):
            snapshot.append(c)
            if c.tag != "-text":
                _push(c)

    _push(root)

    # Selectolax wraps the same DOM node in fresh Python objects on each
    # access, so `is` is unreliable for parent-chain comparisons; the C
    # extension implements `__eq__` to compare underlying nodes — use that.
    for n in snapshot:
        cur = n.parent
        attached = False
        while cur is not None:
            if cur == root:
                attached = True
                break
            cur = cur.parent
        if attached:
            yield n


def _node_to_markdown(
    node, *, skip_first_heading: bool,
    base_url: str = "", image_dir: str = "", images: list[dict] | None = None,
) -> str:
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

    Walks descendants in pre-order via `_iter_descendants` — see that
    helper for why we don't use `selectolax.traverse()`.
    """
    if node is None:
        return ""

    pieces: list[str] = []
    skipped = not skip_first_heading

    skip_classes = ("ltx_pagination", "ltx_navbar", "ltx_break")
    skip_tags = ("script", "style", "nav")

    for n in _iter_descendants(node):
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

        # Figure images. Only when a manifest list is provided (the ingest
        # path). Resolve the src against the page URL, record it for download,
        # and emit a markdown image ref pointing at the local media dir. The
        # arXiv `alt` is almost always the placeholder "Refer to caption", so
        # we drop it — the real caption follows as figcaption text anyway.
        if n.tag == "img" and images is not None:
            src = (n.attributes.get("src") or "").strip()
            if src:
                name = _safe_image_name(src)
                url = urljoin(base_url, src) if base_url else src
                if not any(im["name"] == name for im in images):
                    images.append({"name": name, "url": url})
                ref = f"{image_dir}/{name}" if image_dir else name
                if pieces and not pieces[-1].endswith("\n"):
                    pieces.append("\n\n")
                pieces.append(f"![]({ref})\n\n")
            n.decompose()
            continue

        # External anchor links — preserve the URL as inline markdown so
        # DOI / repo / dataset links survive HTML→markdown→chunker. arxiv
        # `\href{X}{Y}` lands as `<a href="X">Y</a>`. Internal fragments
        # (`#fig-1`) are emitted as plain text without the URL — they're
        # navigation hints, not citable resources.
        if n.tag == "a":
            href = (n.attributes.get("href") or "").strip()
            link_text = _clean_text(n)
            if not href or href.startswith("#") or href.startswith("javascript:"):
                if link_text:
                    pieces.append(link_text)
            else:
                if link_text:
                    pieces.append(f"[{link_text}]({href})")
                else:
                    pieces.append(f"<{href}>")
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


_IMG_NAME_SAFE_RE = re.compile(r"[^A-Za-z0-9._-]")


def _safe_image_name(src: str) -> str:
    """Derive a safe local filename from an <img> src.

    Drops any directory components and query/fragment, then sanitises the
    basename to [A-Za-z0-9._-] so it can't escape the media dir. Falls back
    to 'figure.png' when nothing usable remains.

        '2603.05238v2/x1.png'        -> 'x1.png'
        'extracted/3/fig%201.png?x'  -> 'fig_201.png'
    """
    base = src.split("?")[0].split("#")[0].rstrip("/")
    base = base.replace("\\", "/").split("/")[-1]
    base = _IMG_NAME_SAFE_RE.sub("_", base).lstrip(".")
    return base or "figure.png"


def _download_images(
    images: list[dict], dest_dir: Path, *, client: httpx.Client,
) -> list[dict]:
    """Download each {"name", "url"} image into dest_dir. Best-effort.

    Returns the subset that landed on disk (with bytes recorded under
    "n_bytes"). Individual failures are logged and skipped — a missing
    figure must never fail the whole paper fetch. Already-present files are
    kept (idempotent re-fetch).
    """
    if not images:
        return []
    dest_dir.mkdir(parents=True, exist_ok=True)
    saved: list[dict] = []
    for im in images:
        name, url = im.get("name"), im.get("url")
        if not name or not url:
            continue
        path = dest_dir / name
        if path.exists() and path.stat().st_size > 0:
            saved.append({"name": name, "url": url, "n_bytes": path.stat().st_size})
            continue
        try:
            r = _request_with_retry(client, url)
            if r.status_code != 200 or not r.content:
                LOG.info(f"image {url}: http {r.status_code}, skipping")
                continue
            path.write_bytes(r.content)
            saved.append({"name": name, "url": url, "n_bytes": len(r.content)})
        except (httpx.HTTPError, OSError) as e:
            LOG.info(f"image {url}: {type(e).__name__}: {e}, skipping")
    return saved


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
    r = _request_with_retry(client, url)
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
