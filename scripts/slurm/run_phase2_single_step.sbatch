#!/bin/bash
# Paired GPT-OSS control/intervention evaluation of Phase 1 benchmark files.
# Submit through scripts/slurm/submit_phase2.sh so the required paths and
# intervention settings are validated and exported.
#SBATCH --job-name=gpt-oss-phase2-single
#SBATCH --account=def-yousefne_gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

: "${SLURM_JOB_ID:?Submit this script with sbatch}"
: "${LAYERMCP_REPO_ROOT:?Set LAYERMCP_REPO_ROOT}"
: "${LAYERMCP_GPT_OSS_CHECKPOINT:?Set LAYERMCP_GPT_OSS_CHECKPOINT}"
: "${TIKTOKEN_ENCODINGS_BASE:?Set TIKTOKEN_ENCODINGS_BASE}"

cd "$LAYERMCP_REPO_ROOT"
module load StdEnv/2023 python/3.11
source "$LAYERMCP_REPO_ROOT/scripts/slurm/phase2_environment.sh"
validate_phase2_environment
source "$LAYERMCP_VENV/bin/activate"

LAYER_INDEX="${LAYER_INDEX:-0}"
ATTENTION_TARGET="${ATTENTION_TARGET:-qkv}"
METHOD="${METHOD:-noise}"
STRENGTH="${STRENGTH:-0.01}"
SEED="${SEED:-1234}"

source "$LAYERMCP_REPO_ROOT/scripts/slurm/submit_common.sh"
BENCHMARKS=()
if [[ -n "${DATASET_SELECTION:-}" ]]; then
  IFS=':' read -r -a BENCHMARKS <<< "$DATASET_SELECTION"
  for benchmark in "${BENCHMARKS[@]}"; do
    allowed=0
    for id in "${SINGLE_STEP_PRIMARY_DATASET_IDS[@]}"; do
      [[ "$benchmark" == "${SINGLE_STEP_DATASETS[$id]}" ]] && allowed=1
    done
    [[ "$allowed" == 1 ]] || { echo "Unsupported Phase 2 single-step benchmark: $benchmark" >&2; exit 2; }
  done
else
  for id in "${SINGLE_STEP_PRIMARY_DATASET_IDS[@]}"; do
    BENCHMARKS+=("${SINGLE_STEP_DATASETS[$id]}")
  done
fi
[[ "${#BENCHMARKS[@]}" -gt 0 ]] || { echo "No single-step benchmarks selected" >&2; exit 2; }

RUN_DIR="results/runs/phase2/${SLURM_JOB_ID}_gpt-oss_single_step_full_layer${LAYER_INDEX}_${ATTENTION_TARGET}_${METHOD}_${STRENGTH}_seed${SEED}"
echo "Evaluating ${#BENCHMARKS[@]} direct single-step benchmark files in paired conditions"
python -m research.phase2.gpt_oss_intervention_eval \
  --benchmarks "${BENCHMARKS[@]}" \
  --checkpoint "$LAYERMCP_GPT_OSS_CHECKPOINT" \
  --output-dir "$RUN_DIR" \
  --layers "$LAYER_INDEX" \
  --target "$ATTENTION_TARGET" \
  --method "$METHOD" \
  --strength "$STRENGTH" \
  --seeds "$SEED"
