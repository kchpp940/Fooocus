#!/bin/bash
set -e

APP_DIR=/app
DATA_DIR="${DATADIR:-/data}"
MANIFEST_CHECK="${MANIFEST_CHECK:-1}"
MANIFEST_STRICT="${MANIFEST_STRICT:-0}"
MANIFEST_CHECK_HASH="${MANIFEST_CHECK_HASH:-0}"
MANIFEST_PATH="${MANIFEST_PATH:-}"

echo "=========================================="
echo " Fooocus Docker Entrypoint"
echo "=========================================="
echo "App directory:  $APP_DIR"
echo "Data directory: $DATA_DIR"
echo "Manifest check: $MANIFEST_CHECK"
echo "=========================================="

mkdir -p "$DATA_DIR"
mkdir -p "$DATA_DIR/models"
mkdir -p "$DATA_DIR/cache"
mkdir -p "$DATA_DIR/outputs"
mkdir -p "$DATA_DIR/config"
mkdir -p "$DATA_DIR/user_presets"

function import_dir() {
    local import_src="/import/$1"
    local dest_dir="$DATA_DIR/$1"
    if [ -d "$import_src" ]; then
        echo "[Import] Copying $import_src -> $dest_dir"
        mkdir -p "$dest_dir"
        cp -Rpn "$import_src"/. "$dest_dir"/
    fi
}

import_dir "models"
import_dir "outputs"
import_dir "user_presets"

if [ -f "/import/manifest.json" ] && [ ! -f "$DATA_DIR/manifest.json" ]; then
    echo "[Import] Copying manifest.json"
    cp -p "/import/manifest.json" "$DATA_DIR/manifest.json"
fi

export FOOOCUS_DATA_DIR="$DATA_DIR"
export FOOOCUS_USER_DATA_DIR="$DATA_DIR"

if [ -z "$config_path" ]; then
    export config_path="$DATA_DIR/config.txt"
fi
if [ -z "$config_example_path" ]; then
    export config_example_path="$DATA_DIR/config_modification_tutorial.txt"
fi
if [ -z "$sorted_styles_path" ]; then
    export sorted_styles_path="$DATA_DIR/sorted_styles.json"
fi
if [ -z "$temp_path" ]; then
    export temp_path="$DATA_DIR/cache/temp"
fi

cd "$APP_DIR"

if [ "$MANIFEST_CHECK" = "1" ]; then
    echo ""
    echo "[Manifest] Running pre-launch manifest check..."
    MANIFEST_ARG=""
    if [ -n "$MANIFEST_PATH" ]; then
        MANIFEST_ARG="--manifest-path $MANIFEST_PATH"
    fi
    if [ "$MANIFEST_STRICT" = "1" ]; then
        MANIFEST_ARG="$MANIFEST_ARG --manifest-strict"
    fi
    if [ "$MANIFEST_CHECK_HASH" = "1" ]; then
        MANIFEST_ARG="$MANIFEST_ARG --manifest-check-hash"
    fi
    python -c "
import sys
sys.path.insert(0, '$APP_DIR')
from modules.manifest import (
    load_manifest, check_manifest_resources, print_manifest_report,
    build_default_manifest, get_manifest_path
)
import modules.config as config

manifest_path = '$MANIFEST_PATH' if '$MANIFEST_PATH' else get_manifest_path(config.get_data_dir())
manifest = load_manifest(manifest_path)

if not manifest.resources:
    print('[Manifest] No manifest file found, generating default manifest from config...')
    from modules.config import (
        checkpoint_downloads, lora_downloads, embeddings_downloads, vae_downloads
    )
    vae_approx_filenames = []
    fooocus_expansion_url = 'https://huggingface.co/lllyasviel/misc/resolve/main/fooocus_expansion.bin'
    manifest = build_default_manifest(
        checkpoint_downloads, lora_downloads, embeddings_downloads, vae_downloads,
        vae_approx_filenames, fooocus_expansion_url,
        fooocus_version='docker'
    )

models_root = config.get_models_dir()
check_hash = '$MANIFEST_CHECK_HASH' == '1'
result = check_manifest_resources(manifest, models_root, check_hash=check_hash)
ok = print_manifest_report(result, manifest)

if not ok and '$MANIFEST_STRICT' == '1':
    print()
    print('[ERROR] Manifest check failed in strict mode. Aborting launch.')
    print('[ERROR] Please ensure all required models are present in ' + models_root)
    sys.exit(1)
elif not ok:
    print()
    print('[WARNING] Some required resources are missing.')
    print('[WARNING] Fooocus will attempt to download them at startup.')
    print('[WARNING] If you are in an offline environment, set MANIFEST_STRICT=1 to fail fast.')
"
fi

echo ""
echo "[Entrypoint] Starting Fooocus..."
echo ""

exec python launch.py "$@"
