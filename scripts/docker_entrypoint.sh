#!/usr/bin/env bash
# Container dispatcher. Routes the first arg to a sub-command.
# Unknown first arg → exec verbatim (handy for debugging shells, python -c, etc.).
set -e

case "${1:-mcp-http}" in
    mcp-http)
        # Long-running streamable-HTTP MCP backend (the production mode).
        # Listens on 0.0.0.0 inside the container; users tunnel in via SSH
        # so the host port mapping is loopback-only (-p 127.0.0.1:8765:8765).
        shift
        exec python -m arxiv_radar_mcp \
            --transport http --bind 0.0.0.0 --port 8765 "$@"
        ;;
    mcp)
        # Stdio mode — for legacy single-shot usage.
        exec python -m arxiv_radar_mcp "${@:2}"
        ;;
    build-cache)
        exec python -m arxiv_radar_mcp --build-cache "${@:2}"
        ;;
    fetch)
        # In-container helper: arxiv_radar_mcp.fulltext.fetch_and_save
        # for ad-hoc enrich runs without going through the MCP transport.
        # Usage: docker run ... fetch <arxiv_id> [<arxiv_id> ...]
        shift
        exec python -m arxiv_radar_mcp.fulltext_cli "$@"
        ;;
    reindex)
        # In-container helper: full rebuild of fulltext index.
        # Usage: docker run ... reindex
        shift
        exec python -m arxiv_radar_mcp.reindex_cli "$@"
        ;;
    *)
        exec "$@"
        ;;
esac
