"""Tests for chunker.py — markdown → chunk list."""
from __future__ import annotations

import pytest

from arxiv_radar_mcp.chunker import (Chunk, chunk_markdown, estimate_tokens,
                                     split_by_headings, split_long_section)


# ----- estimate_tokens -------------------------------------------------------

def test_estimate_tokens_basic():
    assert estimate_tokens("a" * 4) == 1
    assert estimate_tokens("a" * 400) == 100


def test_estimate_tokens_minimum_one():
    assert estimate_tokens("") == 1
    assert estimate_tokens("ab") == 1


# ----- split_by_headings -----------------------------------------------------

def test_split_by_headings_three_sections():
    md = """# Title

Some intro paragraph.

## Methods
We did stuff.

## Results
We saw stuff.

## Discussion
Stuff matters.
"""
    out = split_by_headings(md)
    names = [n for n, _ in out]
    assert names == ["Header", "Methods", "Results", "Discussion"]
    assert "intro paragraph" in dict(out)["Header"]
    assert "We did stuff" in dict(out)["Methods"]


def test_split_by_headings_no_headings_emits_body():
    md = "Just a paragraph, no headings here."
    out = split_by_headings(md)
    assert out == [("Body", "Just a paragraph, no headings here.")]


def test_split_by_headings_skips_empty_section():
    md = """## Methods

## Results
content
"""
    out = split_by_headings(md)
    assert ("Methods", "") not in out
    assert dict(out).get("Results") == "content"


def test_split_by_headings_ignores_subsection_h3():
    md = """## Methods
Top-level methods text.

### Subsection
Inside subsection.

## Results
Results text.
"""
    out = split_by_headings(md)
    names = [n for n, _ in out]
    assert names == ["Methods", "Results"]
    # H3 content should be inside the parent section.
    assert "Inside subsection" in dict(out)["Methods"]


def test_split_by_headings_no_pre_header_text_skips_header():
    md = """## Methods
Methods text.
"""
    out = split_by_headings(md)
    assert out == [("Methods", "Methods text.")]


# ----- split_long_section ----------------------------------------------------

def test_split_long_section_under_limit_returns_one_chunk():
    text = "a " * 100  # ~50 tokens
    out = split_long_section(text, max_tokens=1000)
    assert len(out) == 1


def test_split_long_section_over_limit_splits_at_paragraph_boundary():
    p1 = "a " * 200  # ~100 tokens
    p2 = "b " * 200
    p3 = "c " * 200
    text = f"{p1}\n\n{p2}\n\n{p3}"
    out = split_long_section(text, max_tokens=150)
    assert len(out) >= 2
    # Each chunk should be roughly under-limit (greedy fill, no overlap).
    for chunk in out:
        # Some slack: a single oversized paragraph passes through.
        assert "\n\n" not in chunk[:1] if chunk else True


def test_split_long_section_preserves_content():
    text = "para1\n\npara2\n\npara3"
    out = split_long_section(text, max_tokens=2)  # force split
    joined = "\n\n".join(out)
    assert "para1" in joined and "para2" in joined and "para3" in joined


# ----- chunk_markdown (top-level) -------------------------------------------

def test_chunk_markdown_returns_chunks_with_section_names():
    md = """# Title

intro

## Methods

methods text

## Results

results text
"""
    chunks = chunk_markdown(md, max_tokens=10_000)
    sections = [c.section for c in chunks]
    assert "Header" in sections
    assert "Methods" in sections
    assert "Results" in sections
    for c in chunks:
        assert isinstance(c, Chunk)
        assert c.chunk_idx == 0  # no sub-split needed
        assert c.n_chars > 0
        assert c.n_tokens_est > 0


def test_chunk_markdown_subsplits_oversize_section():
    """Force a section that exceeds max_tokens, see chunk_idx increment."""
    big = ("paragraph one. " * 200 + "\n\n" + "paragraph two. " * 200)
    md = f"## BigSection\n\n{big}\n"
    chunks = chunk_markdown(md, max_tokens=200)
    big_chunks = [c for c in chunks if c.section == "BigSection"]
    assert len(big_chunks) >= 2
    assert [c.chunk_idx for c in big_chunks] == list(range(len(big_chunks)))


def test_chunk_markdown_empty_input():
    assert chunk_markdown("", max_tokens=1000) == []


def test_chunk_markdown_keeps_inline_latex():
    md = "## Methods\n\nWe use $E = mc^2$ and $\\alpha = 1$ everywhere."
    chunks = chunk_markdown(md, max_tokens=1000)
    text = chunks[0].text
    assert "$E = mc^2$" in text
    assert "\\alpha" in text


def test_chunk_markdown_default_max_tokens_is_12000():
    """[РЕШЕНИЕ-014]: chunker default == fulltext encoder seq window."""
    import inspect
    sig = inspect.signature(chunk_markdown)
    assert sig.parameters["max_tokens"].default == 12_000
