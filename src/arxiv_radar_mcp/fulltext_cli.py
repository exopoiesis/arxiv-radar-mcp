"""CLI helper: fetch full text for one or more arxiv_ids and cache them.

Used inside the GPU container for scenario testing without going through
the MCP transport. For interactive use, the MCP `fetch_papers` tool
returns a job_id; this CLI is synchronous and prints a summary.

Usage:
    python -m arxiv_radar_mcp.fulltext_cli 2503.11576 2410.07073 [...]
    python -m arxiv_radar_mcp.fulltext_cli --config /data/radar.toml <ids>
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

import httpx

from arxiv_radar_mcp.config import load
from arxiv_radar_mcp.fulltext import fetch_and_save


def main() -> int:
    parser = argparse.ArgumentParser(prog="arxiv-radar-fetch")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--force", action="store_true",
                        help="re-fetch even if cached")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("arxiv_ids", nargs="+")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level), stream=sys.stderr,
                        format="[%(asctime)s %(levelname)s %(name)s] %(message)s",
                        datefmt="%H:%M:%S")

    config = load(args.config)
    fulltext_dir = config.embeddings.cache_dir.parent / "fulltext"
    fulltext_dir.mkdir(parents=True, exist_ok=True)

    ok: list[dict] = []
    failed: list[dict] = []
    sources: Counter[str] = Counter()

    with httpx.Client(timeout=httpx.Timeout(30.0, connect=10.0),
                      headers={"User-Agent": "arxiv-radar-mcp/0.1"},
                      follow_redirects=True) as client:
        for pid in args.arxiv_ids:
            r = fetch_and_save(pid, fulltext_dir, force=args.force, client=client)
            if r.markdown is not None:
                ok.append({"arxiv_id": pid, "source": r.source,
                           "n_chars": r.n_chars})
                sources[r.source or "unknown"] += 1
                print(f"[ok] {pid}  source={r.source}  n_chars={r.n_chars}")
            else:
                failed.append({"arxiv_id": pid, "error": r.error})
                print(f"[fail] {pid}  {r.error}")

    summary = {
        "n_total": len(args.arxiv_ids),
        "n_ok": len(ok),
        "n_failed": len(failed),
        "source_breakdown": dict(sources),
        "ok": ok,
        "failed": failed,
    }
    print()
    print(json.dumps(summary, indent=2))
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
