#!/usr/bin/env bash
# One-shot: initialize the arxiv-radar-cache named volume with radar.toml.
# Idempotent — re-running just rewrites the file.
set -e
export MSYS_NO_PATHCONV=1
export MSYS2_ARG_CONV_EXCL='*'

TAG="${1:-exopoiesis/arxiv-radar-gpu:latest}"
VOL="${2:-arxiv-radar-cache}"

docker --context gomer run --rm \
    -v "$VOL:/cache" \
    "$TAG" sh -c '
mkdir -p /cache /cache/abstracts /cache/fulltext/sources /cache/jobs
cat > /cache/radar.toml <<EOF
# arxiv-radar config for GPU backend.
[sources.chemistry]
type   = "github"
repo   = "exopoiesis/arxiv-radar-chemistry"
branch = "main"

[embeddings]
model      = "Qwen/Qwen3-Embedding-4B"
cache_dir  = "/cache/abstracts"
batch_size = 32

[reranker]
enabled = false

[refresh]
enabled        = true
interval_hours = 24
full_rebuild   = true
EOF
echo "radar.toml written:"
cat /cache/radar.toml
echo
echo "structure:"
find /cache -maxdepth 3 -type d
'
