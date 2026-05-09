# arxiv-radar-mcp standalone GPU image — for fast Qwen3-4B embedding
# builds and reindex when you DON'T need MinerU + lab-corpus on the
# same host.
#
# After Phase 3 (corpus-core extracted to its own repo), the build
# context is the PARENT directory containing the two sibling repos so
# both can be COPY'd in one go:
#
#   <parent>/
#     ├── corpus-core/           (shared Encoder, JobRegistry, MCP scaffold)
#     └── arxiv-radar-mcp/       (this repo — RadarServer + arxiv shards)
#
# Build:
#   docker build -f arxiv-radar-mcp/Dockerfile -t exopoiesis/arxiv-radar-gpu:latest .
#
# (See scripts/docker_build.sh — it sets the right context.)
#
# For the COMBINED arxiv-radar + lab-corpus + MinerU bundle (one Qwen
# in 12 GB VRAM), use lab-corpus-mcp/Dockerfile instead.
#
# Base: torch 2.7.1 + CUDA 12.6 (Python 3.11). Bumped from 2.5.1 to
# stay at-or-above MinerU 3.x's `torch>=2.6,<3` floor — that way we
# don't pay for a separate ~800 MB pip-side torch reinstall when the
# combined sibling adds MinerU. We deliberately don't pin torch in
# our pyprojects: MinerU's transitive constraint already fences the
# acceptable range, and adding a second pin would just create drift.
FROM pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime

LABEL org.opencontainers.image.title="arxiv-radar-gpu" \
      org.opencontainers.image.description="MCP server for arXiv abstract + fulltext semantic search" \
      org.opencontainers.image.source="https://github.com/exopoiesis/arxiv-radar-mcp"

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/root/.cache/huggingface

# System deps — ca-certificates for HTTPS (HF + arxiv), git for any pip
# installs that pull from VCS.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

# Sibling 1: corpus-core. Phase 3 extracted the shared infrastructure
# from arxiv-radar-mcp into its own repo; arxiv-radar-mcp now declares
# `corpus-core>=0.1.0` as a hard dependency. Path-installed editable
# inside the image so a corpus-core source edit re-flows without a
# re-publish to PyPI.
COPY corpus-core/pyproject.toml /opt/corpus-core/
COPY corpus-core/README.md     /opt/corpus-core/
COPY corpus-core/src           /opt/corpus-core/src
RUN pip install -e /opt/corpus-core

# This repo: arxiv-radar shell on top.
COPY arxiv-radar-mcp/pyproject.toml /opt/arxiv-radar-mcp/
COPY arxiv-radar-mcp/README.md     /opt/arxiv-radar-mcp/
COPY arxiv-radar-mcp/src           /opt/arxiv-radar-mcp/src
RUN pip install -e /opt/arxiv-radar-mcp

# Tiny dispatcher routes by first CMD arg (mcp / build-cache / fetch / reindex).
COPY arxiv-radar-mcp/scripts/docker_entrypoint.sh /usr/local/bin/arxiv-radar-entrypoint
RUN chmod +x /usr/local/bin/arxiv-radar-entrypoint

# Build-time audit — fail fast if pip somehow installed two torches or
# either of the two siblings is missing.
COPY arxiv-radar-mcp/scripts/audit_image.py /usr/local/bin/audit_image.py
RUN python /usr/local/bin/audit_image.py

RUN mkdir -p /data /cache /workspace
WORKDIR /workspace

# Persisted state lives in named volumes / bind-mounts:
#   /root/.cache/huggingface — sentence-transformers / transformers / Qwen weights
#   /cache                   — embeddings (abstracts/, fulltext/), jobs/, fulltext/sources/
#   /data                    — corpus shards, radar.toml
#
# Tip: if you also run lab-corpus-gpu on the same host, point both
# images at the SAME named volume `lab-corpus-hf:/root/.cache/huggingface`
# so the Qwen weights are downloaded only once per machine.
VOLUME ["/root/.cache/huggingface", "/cache", "/data"]

EXPOSE 8765

ENTRYPOINT ["arxiv-radar-entrypoint"]
CMD ["mcp-http"]
