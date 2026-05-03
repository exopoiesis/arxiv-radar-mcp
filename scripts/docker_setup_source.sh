#!/usr/bin/env bash
# Sparse-clone the chemistry abstracts repo into the backend cache volume.
#
# Only data/ and tags/ are materialized. After cloning, /cache/radar.toml is
# switched to type=local so daily refresh can use git pull instead of GitHub
# raw downloads.
set -e
export MSYS_NO_PATHCONV=1
export MSYS2_ARG_CONV_EXCL='*'

cd "$(dirname "$0")/.."

CONTAINER="${1:-arxiv-radar-backend}"
SOURCE_REPO="${2:-https://github.com/exopoiesis/arxiv-radar-chemistry}"
SOURCE_NAME="${3:-chemistry}"

echo "[setup-source] sparse-clone $SOURCE_REPO into named volume via $CONTAINER"

docker --context gomer exec "$CONTAINER" bash -c "
set -e
mkdir -p /cache/sources
cd /cache/sources

if [ -d $SOURCE_NAME/.git ]; then
    echo '  already cloned, updating remote and running git pull'
    git -C $SOURCE_NAME remote set-url origin '$SOURCE_REPO'
    git -C $SOURCE_NAME sparse-checkout set data tags
    git -C $SOURCE_NAME pull --ff-only
else
    echo '  cloning fresh'
    git clone --filter=blob:none --sparse '$SOURCE_REPO' $SOURCE_NAME
    cd $SOURCE_NAME
    git sparse-checkout init --cone
    git sparse-checkout set data tags
fi

echo
echo '  contents:'
ls /cache/sources/$SOURCE_NAME/data/ | head -5
echo '  ... ($(ls /cache/sources/$SOURCE_NAME/data/ | wc -l) shards)'
"

echo
echo "[setup-source] updating /cache/radar.toml to use type=local"
docker --context gomer exec "$CONTAINER" bash -c "
cat > /cache/radar.toml <<'EOF'
# arxiv-radar config — daily refresh from local sparse-clone.
[sources.chemistry]
type = \"local\"
path = \"/cache/sources/chemistry\"

[embeddings]
model      = \"Qwen/Qwen3-Embedding-4B\"
cache_dir  = \"/cache/abstracts\"
batch_size = 32

[reranker]
enabled = false

[refresh]
enabled        = true
interval_hours = 24
full_rebuild   = true
EOF
echo '  written:'
cat /cache/radar.toml
"
