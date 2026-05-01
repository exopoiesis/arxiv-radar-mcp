#!/usr/bin/env bash
# Stop and remove the long-running backend container.
set -e
export MSYS_NO_PATHCONV=1
export MSYS2_ARG_CONV_EXCL='*'

CONTAINER="${1:-arxiv-radar-backend}"
docker --context gomer rm -f "$CONTAINER"
