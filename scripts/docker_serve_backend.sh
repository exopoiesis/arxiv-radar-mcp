#!/usr/bin/env bash
# Start the long-running arxiv-radar HTTP backend on gomer (or any ssh-able
# Docker host with --gpus all support).
#
# Idempotent: if `arxiv-radar-backend` already exists it gets restarted
# with the latest image; new container picks up new code.
#
# Args (positional, paths on the gomer host):
#   $1   radar.toml path (mounted read-only at /cache/radar.toml)
#         OR pass an existing named-volume path like /cache/radar.toml when
#         the radar.toml lives inside the cache volume itself.
#
# Topology:
#   * Container EXPOSE 8765 → published to host loopback only
#     (-p 127.0.0.1:8765:8765) — the only way in is via SSH tunnel.
#   * Named volumes:
#       arxiv-radar-cache  →  /cache  (radar.toml, abstracts/, fulltext/, jobs/)
#       arxiv-radar-hf     →  /root/.cache/huggingface  (Qwen3-4B weights)
set -e
export MSYS_NO_PATHCONV=1
export MSYS2_ARG_CONV_EXCL='*'

CONFIG_PATH="${1:-/cache/radar.toml}"
TAG="${2:-exopoiesis/arxiv-radar-gpu:latest}"
CONTAINER="${3:-arxiv-radar-backend}"

echo "[serve-backend] container=$CONTAINER  image=$TAG  config=$CONFIG_PATH"

# Stop any prior instance (idempotent restart).
if docker --context gomer ps -a --format '{{.Names}}' | grep -q "^${CONTAINER}$"; then
    echo "[serve-backend] removing prior container"
    docker --context gomer rm -f "$CONTAINER" >/dev/null
fi

docker --context gomer run -d \
    --name "$CONTAINER" \
    --restart unless-stopped \
    --gpus all \
    -p 127.0.0.1:8765:8765 \
    -v "arxiv-radar-cache:/cache" \
    -v "arxiv-radar-hf:/root/.cache/huggingface" \
    "$TAG" \
    mcp-http --config "$CONFIG_PATH"

echo "[serve-backend] container started. logs:"
echo "  bash scripts/docker_logs_backend.sh"
echo "  bash scripts/docker_stop_backend.sh"
echo
echo "[serve-backend] tunnel from your laptop:"
echo "  arxiv-radar-mcp --remote user@gomer.lan"
