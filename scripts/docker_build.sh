#!/usr/bin/env bash
# Build the arxiv-radar standalone GPU image on gomer.
#
# Context = PARENT directory containing both sibling repos:
#   <parent>/
#     ├── corpus-core/
#     └── arxiv-radar-mcp/
#
# The Dockerfile COPYs from each sibling. .dockerignore strips
# tests/, tmp/, docs/, .venv to keep upload tiny.
#
# Run: bash scripts/docker_build.sh
set -e
export MSYS_NO_PATHCONV=1
export MSYS2_ARG_CONV_EXCL='*'

PARENT_WIN="D:/home/ignat/project-third-matter/git"
DOCKERFILE="$PARENT_WIN/arxiv-radar-mcp/Dockerfile"
TAG="${1:-exopoiesis/arxiv-radar-gpu:latest}"

echo "[build] tag=$TAG  context=$PARENT_WIN  dockerfile=$DOCKERFILE"

docker --context gomer build \
    --tag "$TAG" \
    --file "$DOCKERFILE" \
    "$PARENT_WIN"

echo
docker --context gomer images "$TAG"
echo
echo "[build] done."
