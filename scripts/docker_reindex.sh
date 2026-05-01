#!/usr/bin/env bash
# Rebuild the fulltext embedding index on gomer GPU.
#
# Args:
#   $1  radar.toml on gomer
#   $2  cache dir on gomer  (reads fulltext/sources/, writes fulltext/embeddings.npy + index.json)
set -e
export MSYS_NO_PATHCONV=1
export MSYS2_ARG_CONV_EXCL='*'

CONFIG="${1:?radar.toml required}"
CACHE_DIR="${2:?cache dir required}"
TAG="${3:-exopoiesis/arxiv-radar-gpu:latest}"

docker --context gomer run --rm \
    --gpus all \
    -v "arxiv-radar-hf:/root/.cache/huggingface" \
    -v "$CONFIG:/data/radar.toml:ro" \
    -v "$CACHE_DIR:/cache" \
    "$TAG" reindex --config /data/radar.toml
