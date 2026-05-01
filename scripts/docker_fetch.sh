#!/usr/bin/env bash
# In-container fetch helper: download + parse arxiv full text, cache to
# fulltext/sources/<id>.md. No GPU needed, but uses the bundled image
# for dependency consistency.
#
# Args:
#   $1     radar.toml path on gomer
#   $2     cache dir on gomer
#   $3..N  arxiv_ids to fetch
#
# Run:  bash scripts/docker_fetch.sh /srv/.../radar.toml /srv/.../cache 2503.11576 2410.07073
set -e
export MSYS_NO_PATHCONV=1
export MSYS2_ARG_CONV_EXCL='*'

CONFIG="${1:?radar.toml required}"; shift
CACHE_DIR="${1:?cache dir required}"; shift
[ "$#" -gt 0 ] || { echo "no arxiv_ids given"; exit 1; }
TAG="exopoiesis/arxiv-radar-gpu:latest"

docker --context gomer run --rm \
    -v "$CONFIG:/data/radar.toml:ro" \
    -v "$CACHE_DIR:/cache" \
    "$TAG" fetch --config /data/radar.toml "$@"
