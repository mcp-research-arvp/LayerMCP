#!/bin/bash
set -euo pipefail

if [ -z "${SCRATCH:-}" ]; then
  echo "SCRATCH is not set. Run this on Narval where SCRATCH is available." >&2
  exit 1
fi

TARGET_DIR="$SCRATCH/LayerMCP/checkpoints/qwen-3.6"

mkdir -p "$TARGET_DIR"

hf download Qwen/Qwen3.6-27B \
  --local-dir "$TARGET_DIR"

echo "Downloaded Qwen checkpoint to: $TARGET_DIR"

