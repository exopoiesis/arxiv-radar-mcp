#!/usr/bin/env bash
# Build the arxiv-radar GPU image on gomer.
#
# Context = THIS repo (no parent imports — repo is self-contained).
# .dockerignore strips tests/, tmp/, docs/, .venv to keep upload tiny.
#
# Run: bash scripts/docker_build.sh
set -e
export MSYS_NO_PATHCONV=1
export MSYS2_ARG_CONV_EXCL='*'

REPO_WIN="D:/home/ignat/project-third-matter/git/arxiv-radar-mcp"
DOCKERFILE="$REPO_WIN/Dockerfile"
TAG="${1:-exopoiesis/arxiv-radar-gpu:latest}"

echo "[build] tag=$TAG  context=$REPO_WIN"

docker --context gomer build \
    --tag "$TAG" \
    --file "$DOCKERFILE" \
    "$REPO_WIN"

echo
docker --context gomer images "$TAG"
echo
echo "[build] done."
