#!/bin/bash
# Submit full paired GPT-OSS attention-intervention evaluations.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd -P)"
SINGLE_LAUNCHER="$SCRIPT_DIR/run_phase2_single_step.sbatch"
MULTI_LAUNCHER="$SCRIPT_DIR/run_phase2_multistep.sbatch"

usage() {
  cat <<'EOF'
Usage: scripts/slurm/submit_phase2.sh [both|single|multi] [OPTIONS]

Modes (default: both):
  both       One full saved single-step job plus all active multi-step groups.
  single     One full saved single-step job only.
  multi      One full job per active multi-step group only.

Options:
  --primary-only       Exclude the optional math_controlled multi-step group.
  --layer INDEX        Attention layer (default: 0).
  --target TARGET      Attention target (default: qkv).
  --method METHOD      noise or replace (default: noise).
  --strength NUMBER    Finite non-negative strength (default: 0.01).
  --seed INTEGER       Intervention seed (default: 1234).
  --env-file PATH      Environment file (default: repository .env).
  --dry-run            Show all sbatch commands without submitting jobs.
  -h, --help           Show this help.

The script loads .env, validates its paths, and exports the selected settings.
Run it from any directory. Each multi-step dataset group gets its own job.
EOF
}

mode=both
if [[ $# -gt 0 && "$1" != -* ]]; then
  mode="$1"
  shift
fi
case "$mode" in
  both|single|multi) ;;
  *) echo "Invalid mode: $mode (expected both, single, or multi)" >&2; exit 2 ;;
esac

env_file="$REPO_ROOT/.env"
primary_only=0
dry_run=0
layer_override=""
target_override=""
method_override=""
strength_override=""
seed_override=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --primary-only) primary_only=1 ;;
    --dry-run) dry_run=1 ;;
    --layer|--target|--method|--strength|--seed|--env-file)
      if [[ $# -lt 2 || -z "$2" ]]; then
        echo "Missing value for $1" >&2
        exit 2
      fi
      case "$1" in
        --layer) layer_override="$2" ;;
        --target) target_override="$2" ;;
        --method) method_override="$2" ;;
        --strength) strength_override="$2" ;;
        --seed) seed_override="$2" ;;
        --env-file) env_file="$2" ;;
      esac
      shift
      ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done
if [[ "$mode" == single && "$primary_only" == 1 ]]; then
  echo "--primary-only applies only to multi-step groups" >&2
  exit 2
fi
if [[ ! -f "$env_file" ]]; then
  echo "Environment file does not exist: $env_file" >&2
  exit 2
fi

source "$env_file"

: "${LAYERMCP_REPO_ROOT:?Set LAYERMCP_REPO_ROOT in the environment file}"
: "${LAYERMCP_GPT_OSS_CHECKPOINT:?Set LAYERMCP_GPT_OSS_CHECKPOINT in the environment file}"
: "${TIKTOKEN_ENCODINGS_BASE:?Set TIKTOKEN_ENCODINGS_BASE in the environment file}"
if [[ ! -d "$LAYERMCP_REPO_ROOT" || "$(cd "$LAYERMCP_REPO_ROOT" && pwd -P)" != "$REPO_ROOT" ]]; then
  echo "LAYERMCP_REPO_ROOT must identify this checkout: $REPO_ROOT" >&2
  exit 2
fi
for required_file in \
  "$LAYERMCP_GPT_OSS_CHECKPOINT/config.json" \
  "$LAYERMCP_GPT_OSS_CHECKPOINT/model.safetensors" \
  "$TIKTOKEN_ENCODINGS_BASE/o200k_base.tiktoken"; do
  if [[ ! -s "$required_file" ]]; then
    echo "Required GPT-OSS input is missing or empty: $required_file" >&2
    exit 2
  fi
done
if [[ "$mode" != multi ]]; then
  : "${LAYERMCP_GPT_OSS_SOURCE_RUN:?Set LAYERMCP_GPT_OSS_SOURCE_RUN in the environment file}"
  for required_file in RUN_COMPLETE artifact_index.jsonl run_metadata.json; do
    if [[ ! -f "$LAYERMCP_GPT_OSS_SOURCE_RUN/$required_file" ]]; then
      echo "Saved single-step source run is missing $required_file" >&2
      exit 2
    fi
  done
  export LAYERMCP_GPT_OSS_SOURCE_RUN
fi
export LAYERMCP_REPO_ROOT LAYERMCP_GPT_OSS_CHECKPOINT TIKTOKEN_ENCODINGS_BASE

LAYER_INDEX="${layer_override:-${LAYER_INDEX:-0}}"
ATTENTION_TARGET="${target_override:-${ATTENTION_TARGET:-qkv}}"
METHOD="${method_override:-${METHOD:-noise}}"
STRENGTH="${strength_override:-${STRENGTH:-0.01}}"
SEED="${seed_override:-${SEED:-1234}}"
[[ "$LAYER_INDEX" =~ ^[0-9]+$ ]] || { echo "--layer must be a non-negative integer" >&2; exit 2; }
[[ "$SEED" =~ ^-?[0-9]+$ ]] || { echo "--seed must be an integer" >&2; exit 2; }
case "$ATTENTION_TARGET" in
  attention_all|q_proj|k_proj|v_proj|o_proj|qkv|out) ;;
  *) echo "Unsupported attention target: $ATTENTION_TARGET" >&2; exit 2 ;;
esac
case "$METHOD" in
  noise|replace) ;;
  *) echo "Unsupported intervention method: $METHOD" >&2; exit 2 ;;
esac
[[ "$STRENGTH" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][+-]?[0-9]+)?$ ]] || {
  echo "--strength must be a finite non-negative number" >&2
  exit 2
}
export LAYER_INDEX ATTENTION_TARGET METHOD STRENGTH SEED
export RUN_KIND=full
unset WORKFLOW_LIMIT

if (( ! dry_run )) && ! command -v sbatch >/dev/null 2>&1; then
  echo "sbatch is unavailable; run this on a Slurm login node" >&2
  exit 2
fi

common_exports="ALL,LAYERMCP_REPO_ROOT,LAYERMCP_GPT_OSS_CHECKPOINT,TIKTOKEN_ENCODINGS_BASE,LAYER_INDEX=$LAYER_INDEX,ATTENTION_TARGET=$ATTENTION_TARGET,METHOD=$METHOD,STRENGTH=$STRENGTH,SEED=$SEED"
submitted=0
submit_job() {
  local label="$1"
  local export_spec="$2"
  local launcher="$3"
  local -a command=(sbatch --parsable --chdir="$REPO_ROOT" --job-name="$label" --export="$export_spec" "$launcher")
  if (( dry_run )); then
    printf '%q ' "${command[@]}"
    printf '\n'
  else
    local job_id
    job_id="$("${command[@]}")"
    if [[ ! "$job_id" =~ ^[0-9]+(;[A-Za-z0-9_.-]+)?$ ]]; then
      echo "Unexpected sbatch response for $label: $job_id" >&2
      exit 1
    fi
    printf '%s: job %s\n' "$label" "$job_id"
  fi
  submitted=$((submitted + 1))
}

if [[ "$mode" != multi ]]; then
  submit_job \
    "p2-single-l${LAYER_INDEX}-${ATTENTION_TARGET}" \
    "$common_exports,LAYERMCP_GPT_OSS_SOURCE_RUN" \
    "$SINGLE_LAUNCHER"
fi

if [[ "$mode" != single ]]; then
  groups=(
    coding_sweagent
    coding_nebius_replay
    enterprise_tau2
    finance_convfinqa
    finance_finqa
    finance_finretrieval_replay
    math_public_mathqa
  )
  if (( ! primary_only )); then
    groups+=(math_controlled)
  fi
  for group in "${groups[@]}"; do
    submit_job \
      "p2-multi-${group}" \
      "$common_exports,RUN_KIND=full,DATASET_GROUP=$group" \
      "$MULTI_LAUNCHER"
  done
fi

if (( dry_run )); then
  printf 'Would submit %d job(s): mode=%s, layer=%s, target=%s, method=%s, strength=%s, seed=%s\n' \
    "$submitted" "$mode" "$LAYER_INDEX" "$ATTENTION_TARGET" "$METHOD" "$STRENGTH" "$SEED"
else
  printf 'Submitted %d job(s).\n' "$submitted"
fi
