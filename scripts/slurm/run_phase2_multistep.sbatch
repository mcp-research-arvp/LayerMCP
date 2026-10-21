#!/bin/bash
# Submit from the repository root after sourcing .env. Required: DATASET_GROUP,
# RUN_KIND=short_test|full. Set LAYER_INDEX, ATTENTION_TARGET, METHOD, STRENGTH,
# and SEED to change the intervention. Short tests use WORKFLOW_LIMIT (default 2).
#SBATCH --job-name=gpt-oss-phase2-multistep
#SBATCH --account=def-yousefne_gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

: "${SLURM_JOB_ID:?Submit this script with sbatch}"
: "${LAYERMCP_REPO_ROOT:?Source the repository .env before submission}"
: "${LAYERMCP_GPT_OSS_CHECKPOINT:?Source the repository .env before submission}"
: "${TIKTOKEN_ENCODINGS_BASE:?Source the repository .env before submission}"
: "${DATASET_GROUP:?Set DATASET_GROUP to a supported multi-step group}"
: "${RUN_KIND:?Set RUN_KIND to short_test or full}"
[[ "$RUN_KIND" == short_test || "$RUN_KIND" == full ]] || {
  echo "RUN_KIND must be short_test or full" >&2
  exit 2
}

cd "$LAYERMCP_REPO_ROOT"
module load StdEnv/2023 python/3.11
source "$LAYERMCP_REPO_ROOT/.venv/bin/activate"

LAYER_INDEX="${LAYER_INDEX:-0}"
ATTENTION_TARGET="${ATTENTION_TARGET:-qkv}"
METHOD="${METHOD:-noise}"
STRENGTH="${STRENGTH:-0.01}"
SEED="${SEED:-1234}"
LIMIT_ARGS=()
if [[ "$RUN_KIND" == short_test ]]; then
  LIMIT_ARGS=(--workflow-limit "${WORKFLOW_LIMIT:-2}")
elif [[ -n "${WORKFLOW_LIMIT:-}" ]]; then
  echo "Unset WORKFLOW_LIMIT for a full run" >&2
  exit 2
fi

RUN_DIR="results/runs/phase2/${SLURM_JOB_ID}_gpt-oss_multistep_${RUN_KIND}_${DATASET_GROUP}_layer${LAYER_INDEX}_${ATTENTION_TARGET}_${METHOD}_${STRENGTH}_seed${SEED}"
python -m research.phase2.gpt_oss_multistep_intervention_eval \
  --dataset-group "$DATASET_GROUP" \
  --checkpoint "$LAYERMCP_GPT_OSS_CHECKPOINT" \
  --output-dir "$RUN_DIR" \
  --layer "$LAYER_INDEX" \
  --target "$ATTENTION_TARGET" \
  --method "$METHOD" \
  --strength "$STRENGTH" \
  --seed "$SEED" \
  "${LIMIT_ARGS[@]}"
