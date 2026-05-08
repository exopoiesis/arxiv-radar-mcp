"""Regression tests for the HTML "echo-skeleton" detector in fulltext.py.

Background. arXiv's HTML render (`arxiv.org/html/<id>`) sometimes returns a
multi-kilobyte body that *looks* fine — title, authors, all section headings
present — but every section body is empty or just echoes the heading text:

    ## Abstract\n\nAbstract\n\n## 1 Introduction\n\nIntroduction\n\n...

This happens when the LaTeX source uses `\\input{intro.tex}`-style subfile
inclusion and arXiv's LaTeXML pipeline does not resolve those `\\input{}`
references. The skeleton sails past `_fetch_html`'s "Conversion is not
available" sniff because the markup IS valid, just empty.

`_looks_like_echo_skeleton` catches this so `_fetch_html` returns None and
`fetch_paper` falls through to the e-print path (which has the LaTeX tarball
including the `\\input{}`'d subfiles, and pylatexenc resolves them locally).

The four positive-control fixtures below are *literal copies* of cached `.md`
files for arxiv IDs 2411.12261 / 2510.26991 / 2604.21613 / 2512.16803 as they
were produced by the BROKEN HTML render on 2026-05-08 (see
`docs/PLAN.md` U10). After the fix landed, all four re-fetched cleanly via
e-print (29-57 kB latex). The two negative-control fixtures are a short comment
paper and a typical full paper — both real-content shapes that must NOT trigger
the detector.

If arXiv changes their fallback render format and starts emitting some new
"empty section" pattern that this heuristic misses, that's a regression — add
a fixture here from `docker exec cat <id>.md` and tighten the heuristic.
"""
from __future__ import annotations

import pytest

from arxiv_radar_mcp.fulltext import (
    _looks_like_echo_skeleton,
    _normalize_heading_for_compare,
)


# ---------------------------------------------------------------------------
# Positive controls — real arXiv `\input{}`-broken HTML renders observed
# 2026-05-08. All four MUST be detected as skeleton (return True).
# ---------------------------------------------------------------------------

ECHO_2411_12261 = """\
# Stripe Antiferromagnetic Ground-State Configuration of FeSe Revealed by Density Functional Theory

\\addbibresource

references.bib

Luke Allen Myers
Department of Materials Science and Engineering, The Pennsylvania State University, ...
## Abstract

Abstract

## 1 Introduction

Introduction

## 2 Computational Methods

Computational Methods

## 3 Results and Discussion

Results and Discussion

## 4 Summary

Summary

## 5 Acknowledgments

Acknowledgments

### Appendix A Appendix

Appendix
"""

ECHO_2510_26991 = """\
# Atomistic Simulations of H–Cu Vacancy Cosegregation and H Diffusion in Cu Grain Boundary

## Abstract

Abstract

## 1 Introduction & Background

Introduction & Background

## 2 Calculation Methods

Calculation Methods

## 3 Results

Results

## 4 Discussion and Conclusions

Discussion and Conclusions

## Acknowledgements

Acknowledgements

### References

References
"""

ECHO_2604_21613 = """\
# Emergence of a non-bulk hexagonal Fe2S2 single layer via phase transformation

## Abstract

Abstract

Affan Safeer
II. Physikalisches Institut, Universität zu Köln, ...
## I Introduction

Introduction

## II Results and discussion

Results and discussion

## III Summary

Summary

## IV Methods

Methods
"""

# 2512.16803 Yang Hubbard — partial salvage (5 echo + 1 real-content METHODS
# section + 2 sub-section echos). We INTENTIONALLY want this flagged True so
# the caller falls through to e-print and gets the FULL paper rather than a
# 6.4 kB stub.
ECHO_2512_16803_PARTIAL = """\
# Comparing Hubbard parameters from linear-response theory and Hartree-Fock-based approach

## Abstract

Abstract

## INTRODUCTION

INTRODUCTION

## RESULTS

RESULTS

## DISCUSSION

DISCUSSION

## METHODS

METHODS

Density-functional theory (DFT) underpins modern first-principles simulations
in physics, chemistry, and materials science. Despite its remarkable success,
progress hinges on the development of increasingly accurate exchange-correlation
(xc) functionals. Standard approximations, such as the local spin-density
approximation (LSDA) and spin-polarized generalized-gradient approximation.
While semi-empirical fitting of Hubbard parameters can be effective, it
becomes impractical in materials discovery where experimental data are
unavailable or when multiple target properties must be simultaneously reproduced,
sometimes requiring advanced optimization schemes such as Bayesian approaches.

A natural alternative is to compute Hubbard parameters from first principles
using methods such as constrained DFT, Hartree-Fock-based schemes, or the
constrained random-phase approximation. However, predicted values can differ
substantially among these approaches, and systematic comparisons of their
theoretical foundations remain scarce.

Here, we present a comparative study of two such first-principles approaches.
The first is based on cDFT formulated within linear-response theory which
determines on-site U.

### The DFT+U+V approach.

The DFT+U+V approach.

### Hubbard parameters from LRT.

Hubbard parameters from LRT.
"""


# ---------------------------------------------------------------------------
# Negative controls — real-content shapes that must NOT trigger.
# ---------------------------------------------------------------------------

GENUINE_SHORT = """\
# A Short Comment on Foo

## 1 Introduction

We respond to a recent claim by Bar et al. that the proposed mechanism is
incorrect. In what follows, we show by direct calculation that their analysis
overlooks the contribution of the cross term. Section 2 derives the corrected
expression; Section 3 discusses implications.

## 2 Derivation

Starting from equation (3) of Bar et al., one obtains by straightforward
manipulation that the leading correction is of order alpha squared, not alpha.
Including this term yields the corrected formula presented below.

## 3 Discussion

The corrected expression resolves the discrepancy with experiment reported
in Ref. 14. We thank the authors of Bar et al. for the original analysis
that motivated this comment.
"""

GENUINE_FULL = """\
# Title

## Abstract

This paper presents a comprehensive analysis of foo bar baz. We propose a new
method that achieves state-of-the-art results on three benchmarks, with
significant improvements over prior work.

## 1 Introduction

The study of foo has a long history dating back to the 1950s. Early work
focused on bar, but more recent advances have shifted attention to baz.
Here we revisit the problem from a new angle, leveraging recent advances
in qux.

## 2 Methods

We use the standard quux protocol with the following modifications. First,
we replace the corge step with a more accurate grault routine. Second, we
introduce a regularization term that penalizes garply configurations. Third,
we tune hyperparameters via Bayesian optimization on a held-out validation set.

## 3 Results

Results on benchmark A show a 12.3 percent improvement over the previous
state of the art. Benchmark B shows similar gains. Benchmark C, the hardest,
shows a more modest 4.7 percent improvement, but still statistically significant.

## 4 Discussion

These results suggest that our approach generalizes across benchmarks. The
key insight is that combining grault with garply regularization yields a
more robust optimizer than either alone.
"""


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("label,md", [
    ("2411.12261 Myers FeSe",        ECHO_2411_12261),
    ("2510.26991 Cu-vac",            ECHO_2510_26991),
    ("2604.21613 Fe2S2",             ECHO_2604_21613),
    ("2512.16803 Yang (partial)",    ECHO_2512_16803_PARTIAL),
])
def test_detects_echo_skeleton(label: str, md: str) -> None:
    """Real broken HTML renders observed 2026-05-08 must trigger the detector."""
    assert _looks_like_echo_skeleton(md), (
        f"{label}: detector failed to flag a real `\\input`-broken render. "
        "If arXiv changed their stub format, capture the new shape via "
        "`docker exec arxiv-radar-backend cat /cache/fulltext/sources/<id>.md` "
        "and add it as a fixture before tightening the heuristic."
    )


@pytest.mark.parametrize("label,md", [
    ("genuine short comment", GENUINE_SHORT),
    ("genuine full paper",    GENUINE_FULL),
])
def test_does_not_flag_real_content(label: str, md: str) -> None:
    """Real-content papers (short or full) must NOT trigger the detector,
    or every fetch_papers call would needlessly fall through to e-print."""
    assert not _looks_like_echo_skeleton(md), (
        f"{label}: false positive — real content treated as echo skeleton. "
        "Tighten the threshold in `_looks_like_echo_skeleton` (currently "
        "70% near-empty OR avg body < 50 chars)."
    )


def test_short_paper_under_three_headings_not_flagged() -> None:
    """Heuristic explicitly bails out for <3 headings — too few to judge."""
    md = "# Title\n\n## Abstract\n\nReal abstract content here.\n"
    assert not _looks_like_echo_skeleton(md)


# ---------------------------------------------------------------------------
# Heading normaliser — independent unit test
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ("Introduction",          "introduction"),
    ("1 Introduction",        "introduction"),
    ("1. Introduction",       "introduction"),
    ("I Introduction",        "introduction"),    # Roman
    ("IV Methods",            "methods"),
    ("A. Foo",                "foo"),
    ("Appendix A Foo",        "foo"),
    ("INTRODUCTION",          "introduction"),     # case-insensitive
    ("",                      ""),
])
def test_normalize_heading_for_compare(raw: str, expected: str) -> None:
    assert _normalize_heading_for_compare(raw) == expected
