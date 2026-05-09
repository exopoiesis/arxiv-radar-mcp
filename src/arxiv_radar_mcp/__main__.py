"""CLI entrypoint: `arxiv-radar-mcp`.

Three modes of operation, all through the same binary:

  Local stdio (default — laptop CPU, single user):
      arxiv-radar-mcp [--config PATH]

  Remote HTTP backend (long-running on GPU host):
      arxiv-radar-mcp --transport http [--bind HOST] [--port PORT]
                      [--config PATH]

  Local stdio→remote-HTTP proxy (for Claude Desktop pointing at a remote backend):
      arxiv-radar-mcp --remote user@host [--remote-port 8765]

  Cache build (one-shot):
      arxiv-radar-mcp --build-cache [--config PATH]
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="arxiv-radar-mcp",
        description="MCP server for arXiv abstract + fulltext semantic search.",
    )
    parser.add_argument("--config", type=Path, default=None,
                        help="path to radar.toml (default: platform user-config)")
    parser.add_argument("--build-cache", action="store_true",
                        help="(re)build the abstract embedding cache and exit")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    transport_group = parser.add_argument_group("transport")
    transport_group.add_argument(
        "--transport", choices=["stdio", "http"], default="stdio",
        help="MCP transport (default: stdio for direct Claude Desktop use; "
             "use 'http' for a long-running backend on a GPU host)",
    )
    transport_group.add_argument(
        "--bind", default="127.0.0.1",
        help="host to bind for --transport=http (default: 127.0.0.1 — "
             "expose only via SSH tunnel, see README)",
    )
    transport_group.add_argument(
        "--port", type=int, default=8765,
        help="port for --transport=http (default: 8765)",
    )

    remote_group = parser.add_argument_group("remote-proxy mode")
    remote_group.add_argument(
        "--remote", default=None, metavar="USER@HOST",
        help="run as stdio→HTTP proxy: open SSH tunnel to USER@HOST and "
             "forward MCP traffic to the backend. Mutually exclusive with --transport.",
    )
    remote_group.add_argument(
        "--remote-port", type=int, default=8765,
        help="remote backend port for SSH tunnel (default: 8765)",
    )
    remote_group.add_argument(
        "--ssh-binary", default="ssh",
        help="path to ssh binary (default: 'ssh' on PATH)",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        stream=sys.stderr,
        format="[%(asctime)s %(levelname)s %(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # Mode selection — at most one of {build-cache, transport=http, remote}.
    # build-cache always wins; remote and transport=http are mutually exclusive.
    if args.build_cache:
        from arxiv_radar_mcp.build_cache import build_cache
        build_cache(config_path=args.config)
        return 0

    if args.remote and args.transport != "stdio":
        parser.error("--remote and --transport=http are mutually exclusive — "
                     "the proxy itself runs on stdio")

    if args.remote:
        from corpus_core.proxy import run_proxy
        return run_proxy(
            target=args.remote,
            remote_port=args.remote_port,
            ssh_binary=args.ssh_binary,
        )

    if args.transport == "http":
        from arxiv_radar_mcp.server import serve_http
        serve_http(host=args.bind, port=args.port, config_path=args.config)
        return 0

    # Default: stdio MCP server.
    from arxiv_radar_mcp.server import serve
    serve(config_path=args.config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
