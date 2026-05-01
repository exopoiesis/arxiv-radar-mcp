#!/usr/bin/env bash
# Container dispatcher. Routes the first arg to a sub-command.
# Unknown first arg → exec verbatim (handy for debugging shells, python -c, etc.).
set -e

case "${1:-mcp}" in
    mcp)
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
