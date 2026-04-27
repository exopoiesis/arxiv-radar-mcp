"""CLI entrypoint: `arxiv-radar-mcp`. Starts the MCP server on stdio."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(prog="arxiv-radar-mcp")
    parser.add_argument("--config", type=Path, default=None,
                        help="path to radar.toml (default: ~/.config/arxiv-radar/radar.toml)")
    parser.add_argument("--build-cache", action="store_true",
                        help="(re)build the embedding cache and exit")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        # MCP uses stdout for protocol messages; logs go to stderr.
        stream=sys.stderr,
        format="[%(asctime)s %(levelname)s %(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.build_cache:
        from arxiv_radar_mcp.embeddings import build_cache
        build_cache(config_path=args.config)
        return 0

    from arxiv_radar_mcp.server import serve
    serve(config_path=args.config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
