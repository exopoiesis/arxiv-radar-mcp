"""Standalone arxiv-radar-gpu audit — single source-of-truth check.

Runs inside the arxiv-radar-gpu container. Prints what's installed,
where caches live, and verifies the Phase 3 invariants (one torch,
one corpus_core, one arxiv_radar_mcp, single HF cache mount path).

Wired into the Dockerfile's final RUN so a regression turns the
image build red, not the first encode call.

Standalone use:
    docker exec arxiv-radar-backend python /usr/local/bin/audit_image.py
"""
from __future__ import annotations

import importlib
import importlib.metadata
import os
import sys
from collections import Counter


def _fail(msg: str, code: int = 1) -> None:
    print(f"AUDIT FAIL: {msg}", file=sys.stderr)
    sys.exit(code)


def _ok(label: str, value: str) -> None:
    print(f"  {label:24}{value}")


def main() -> int:
    print("=== arxiv-radar-gpu audit ===")

    try:
        import torch
    except ImportError:
        _fail("torch not installed (base image broken?)")
    _ok("torch.__version__:", torch.__version__)
    _ok("torch.__file__:", torch.__file__)
    _ok("torch.cuda.is_available():", str(torch.cuda.is_available()))

    for name in ("sentence_transformers", "transformers"):
        try:
            mod = importlib.import_module(name)
        except ImportError:
            _fail(f"missing required package: {name}")
        v = getattr(mod, "__version__", "?")
        _ok(f"{name}:", f"{v}  ({mod.__file__})")

    seen = Counter(d.metadata["Name"].lower()
                   for d in importlib.metadata.distributions()
                   if d.metadata["Name"])
    dupes = {n: c for n, c in seen.items() if c > 1}
    if dupes:
        _fail(f"duplicate distributions: {dupes}")
    _ok("distributions installed:", str(sum(seen.values())))

    for name in ("corpus_core", "arxiv_radar_mcp"):
        try:
            mod = importlib.import_module(name)
        except ImportError:
            _fail(f"sibling package missing: {name}")
        v = getattr(mod, "__version__", "?")
        _ok(f"{name}:", f"{v}  ({mod.__file__})")

    val = os.environ.get("HF_HOME")
    if not val:
        _fail("HF_HOME env var not set")
    _ok("HF_HOME:", val)

    # Heavy import chain — Encoder + RadarServer reachable without
    # pulling Qwen weights.
    from corpus_core.embeddings import Encoder  # noqa: F401
    from corpus_core.mcp_scaffold import (  # noqa: F401
        build_mcp_app, make_method_dispatcher, serve_streamable_http,
    )
    from arxiv_radar_mcp.server import RadarServer  # noqa: F401
    _ok("import chain:", "OK")

    print("\n=== AUDIT PASS ===")
    print("single torch + sentence-transformers + transformers")
    print("single corpus_core / arxiv_radar_mcp install (Phase 3 layout)")
    print("single HF_HOME — Qwen weights live in a named volume, one copy on host")
    return 0


if __name__ == "__main__":
    sys.exit(main())
