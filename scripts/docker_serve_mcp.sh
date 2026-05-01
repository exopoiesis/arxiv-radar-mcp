#!/usr/bin/env bash
# Run the arxiv-radar MCP server on gomer, bridging stdio to the local
# shell. Wire into a Claude Desktop / MCP-client config:
#
#   {
#     "mcpServers": {
#       "arxiv-radar": {
#         "command": "bash",
#         "args": [
#           "<absolute path>/scripts/docker_serve_mcp.sh",
#           "/srv/arxiv-radar/radar.toml",
#           "/srv/arxiv-radar/cache"
#         ]
#       }
#     }
#   }
#
# Args (paths are on gomer, not local):
#   $1  radar.toml path
#   $2  cache dir (must contain abstracts/embeddings.npy if you want
#                  search_abstract_*; fulltext/ is created on demand)
set -e
export MSYS_NO_PATHCONV=1
export MSYS2_ARG_CONV_EXCL='*'

CONFIG="${1:?radar.toml path on gomer required}"
CACHE_DIR="${2:?cache dir on gomer required}"
TAG="${3:-exopoiesis/arxiv-radar-gpu:latest}"

# `exec` so signals from the client (Ctrl+C, SIGTERM) propagate cleanly.
# `-i` keeps stdin open for MCP transport. NO `-t` — clean binary stream.
exec docker --context gomer run --rm -i \
    --gpus all \
    -v "arxiv-radar-hf:/root/.cache/huggingface" \
    -v "$CONFIG:/data/radar.toml:ro" \
    -v "$CACHE_DIR:/cache" \
    "$TAG" mcp --config /data/radar.toml
