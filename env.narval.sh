#!/usr/bin/env bash

# Source this file before running LayerMCP on Alliance Canada Narval.
# Example:
#   source env.narval.sh

if [ -z "${SCRATCH:-}" ]; then
  echo "SCRATCH is not set. On Narval, load this from a login/job environment where SCRATCH is available." >&2
  return 1 2>/dev/null || exit 1
fi

export LAYERMCP_GPT_OSS_CHECKPOINT="$SCRATCH/LayerMCP/checkpoints/gpt-oss-20b/original"
export LAYERMCP_PHI4_CHECKPOINT="$SCRATCH/LayerMCP/checkpoints/phi-4"
export LAYERMCP_LLAMA31_8B_CHECKPOINT="$SCRATCH/LayerMCP/checkpoints/llama-3.1-8b-instruct"
export LAYERMCP_QWEN36_CHECKPOINT="$SCRATCH/LayerMCP/checkpoints/qwen-3.6"
export LAYERMCP_GEMMA4_CHECKPOINT="$SCRATCH/LayerMCP/checkpoints/gemma-4"
