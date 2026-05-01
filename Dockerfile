# arxiv-radar-mcp GPU image — for fast Qwen3-4B embedding builds and reindex.
#
# Built on gomer via `bash scripts/docker_build.sh`. Image is self-contained:
# context is THIS repo only (no sibling imports). To rebuild after src/
# changes, just re-run docker_build.sh — pip install -e picks up the new
# code, layer cache keeps the heavy pytorch base.
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

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

# Install the package. Two-step COPY so dep resolution layer-caches
# independently of source-code edits.
WORKDIR /opt/arxiv-radar-mcp
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --upgrade pip && pip install -e .

# Tiny dispatcher routes by first CMD arg (mcp / build-cache / fetch / reindex).
COPY scripts/docker_entrypoint.sh /usr/local/bin/arxiv-radar-entrypoint
RUN chmod +x /usr/local/bin/arxiv-radar-entrypoint

RUN mkdir -p /data /cache /workspace
WORKDIR /workspace

# Persisted state lives in named volumes / bind-mounts:
#   /root/.cache/huggingface — sentence-transformers / transformers / Qwen weights
#   /cache                   — embeddings (abstracts/, fulltext/), jobs/, fulltext/sources/
#   /data                    — corpus shards, radar.toml
VOLUME ["/root/.cache/huggingface", "/cache", "/data"]

EXPOSE 8765

ENTRYPOINT ["arxiv-radar-entrypoint"]
CMD ["mcp-http"]
