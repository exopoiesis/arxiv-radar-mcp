#!/usr/bin/env bash
# Sparse-clone the abstract source repos into the backend cache volume.
#
# Only data/ and tags/ are materialized. After cloning, /cache/radar.toml is
# switched to type=local so daily refresh can use git pull instead of GitHub
# raw downloads. With no repo/name arguments this installs the default
# science-area feeds: chemistry, chemical_engineering, physics, polymer.
set -e
export MSYS_NO_PATHCONV=1
export MSYS2_ARG_CONV_EXCL='*'

cd "$(dirname "$0")/.."

CONTAINER="${1:-arxiv-radar-backend}"

if [ "$#" -ge 3 ]; then
    SOURCE_REPOS=("$2")
    SOURCE_NAMES=("$3")
else
    SOURCE_REPOS=(
        "https://github.com/exopoiesis/arxiv-radar-chemistry"
        "https://github.com/exopoiesis/arxiv-radar-chem-eng"
        "https://github.com/exopoiesis/arxiv-radar-physics"
        "https://github.com/exopoiesis/arxiv-radar-polymer"
    )
    SOURCE_NAMES=("chemistry" "chemical_engineering" "physics" "polymer")
fi

for i in "${!SOURCE_REPOS[@]}"; do
    SOURCE_REPO="${SOURCE_REPOS[$i]}"
    SOURCE_NAME="${SOURCE_NAMES[$i]}"
    echo "[setup-source] sparse-clone $SOURCE_REPO as $SOURCE_NAME via $CONTAINER"

    docker --context gomer exec "$CONTAINER" bash -s -- "$SOURCE_REPO" "$SOURCE_NAME" <<'REMOTE'
set -e
SOURCE_REPO="$1"
SOURCE_NAME="$2"
mkdir -p /cache/sources
cd /cache/sources

if [ -d "$SOURCE_NAME/.git" ]; then
    echo '  already cloned, updating remote and running git pull'
    git -C "$SOURCE_NAME" remote set-url origin "$SOURCE_REPO"
    git -C "$SOURCE_NAME" sparse-checkout set data tags
    git -C "$SOURCE_NAME" pull --ff-only
else
    echo '  cloning fresh'
    git clone --filter=blob:none --sparse "$SOURCE_REPO" "$SOURCE_NAME"
    cd "$SOURCE_NAME"
    git sparse-checkout init --cone
    git sparse-checkout set data tags
fi

echo
echo '  contents:'
ls "/cache/sources/$SOURCE_NAME/data/" | head -5
count=$(find "/cache/sources/$SOURCE_NAME/data" -maxdepth 1 -name 'papers-*.json' | wc -l)
echo "  ... ($count shards)"
REMOTE
done

echo
echo "[setup-source] updating /cache/radar.toml to use type=local"
if [ "$#" -ge 3 ]; then
    docker --context gomer exec "$CONTAINER" bash -s -- "${SOURCE_NAMES[0]}" <<'REMOTE'
set -e
SOURCE_NAME="$1"
cat > /cache/radar.toml <<EOF
# arxiv-radar config — daily refresh from local sparse-clone.
[sources.$SOURCE_NAME]
type = "local"
path = "/cache/sources/$SOURCE_NAME"

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
echo '  written:'
cat /cache/radar.toml
REMOTE
else
    docker --context gomer exec "$CONTAINER" bash -c "
cat > /cache/radar.toml <<'EOF'
# arxiv-radar config — daily refresh from local sparse-clones.
[sources.chemistry]
type = \"local\"
path = \"/cache/sources/chemistry\"

[sources.chemical_engineering]
type = \"local\"
path = \"/cache/sources/chemical_engineering\"

[sources.physics]
type = \"local\"
path = \"/cache/sources/physics\"

[sources.polymer]
type = \"local\"
path = \"/cache/sources/polymer\"

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
fi
