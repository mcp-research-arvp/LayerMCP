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
  both       One full direct-benchmark single-step job plus selected multi-step groups.
  single     One full direct-benchmark single-step job only.
  multi      One full job per selected multi-step group only.

Options:
  --primary-only       Exclude the optional math_controlled multi-step group.
  --domains IDS        Legacy shortcut: select domains for every workflow type in the chosen mode.
  --single-domains IDS Select domains only for direct single-step datasets.
  --multi-domains IDS  Select domains only for multi-step dataset groups.
  --single-datasets IDS  Select comma-separated Phase 1 single-step dataset IDs.
  --multi-datasets IDS Select comma-separated multi-step dataset-group IDs.
  --list-datasets TYPE Print supported dataset IDs for single or multi, then exit.
  --layer INDEX        Attention layer (default: 0).
  --target TARGET      Attention target (default: qkv).
  --method METHOD      noise or replace (default: noise).
  --strength NUMBER    Finite non-negative strength (default: 0.01).
  --strength-mode MODE absolute or tensor_rms (default: absolute; tensor_rms uses noise only).
  --seed INTEGER       One intervention seed (default: 1234).
  --seeds IDS          Comma-separated intervention seeds for one screen chunk.
  --example-limit N    Evaluate the first N rows of the selected direct dataset(s).
  --plan-manifest PATH Frozen screen-plan JSON to link and validate this chunk.
  --screen-plan-output PATH  Create this frozen plan inside the single-step job.
  --combined-output PATH     Combine and integrity-check that single completed chunk inside the job.
  --env-file PATH      Environment file (default: repository .env).
  --dry-run            Show all sbatch commands without submitting jobs.
  -h, --help           Show this help.

The script loads .env, validates its paths, and exports the selected settings.
LAYERMCP_VENV defaults to $LAYERMCP_REPO_ROOT/.venv; external paths are supported.
Set optional LAYERMCP_SLURM_* and LAYERMCP_MODULES values in the environment
file for site-specific account/resources/module setup. No site values are hard-coded.
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
seeds_override=""
strength_mode_override=""
plan_manifest=""
screen_plan_output=""
combined_output=""
example_limit_override=""
selected_domains=""
single_domains=""
multi_domains=""
single_datasets=""
multi_datasets=""
list_datasets=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --primary-only) primary_only=1 ;;
    --dry-run) dry_run=1 ;;
    --layer|--target|--method|--strength|--strength-mode|--seed|--seeds|--example-limit|--plan-manifest|--screen-plan-output|--combined-output|--env-file|--domains|--single-domains|--multi-domains|--single-datasets|--multi-datasets|--list-datasets)
      if [[ $# -lt 2 || -z "$2" ]]; then
        echo "Missing value for $1" >&2
        exit 2
      fi
      case "$1" in
        --layer) layer_override="$2" ;;
        --target) target_override="$2" ;;
        --method) method_override="$2" ;;
        --strength) strength_override="$2" ;;
        --strength-mode) strength_mode_override="$2" ;;
        --seed) seed_override="$2" ;;
        --seeds) seeds_override="$2" ;;
        --example-limit) example_limit_override="$2" ;;
        --plan-manifest) plan_manifest="$2" ;;
        --screen-plan-output) screen_plan_output="$2" ;;
        --combined-output) combined_output="$2" ;;
        --env-file) env_file="$2" ;;
        --domains) selected_domains="$2" ;;
        --single-domains) single_domains="$2" ;;
        --multi-domains) multi_domains="$2" ;;
        --single-datasets) single_datasets="$2" ;;
        --multi-datasets) multi_datasets="$2" ;;
        --list-datasets) list_datasets="$2" ;;
      esac
      shift
      ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done
if [[ -n "$list_datasets" ]]; then
  source "$SCRIPT_DIR/submit_common.sh"
  case "$list_datasets" in
    single)
      printf 'single-step dataset ID\tdomain\tpath\n'
      for dataset_id in "${SINGLE_STEP_DATASET_IDS[@]}"; do
        printf '%s\t%s\t%s\n' "$dataset_id" "${SINGLE_STEP_DATASET_DOMAINS[$dataset_id]}" "${SINGLE_STEP_DATASETS[$dataset_id]}"
      done
      ;;
    multi)
      printf 'multi-step dataset ID\tdomain\n'
      for dataset_id in "${MULTI_STEP_DATASET_IDS[@]}"; do
        printf '%s\t%s\n' "$dataset_id" "${MULTI_STEP_DATASET_DOMAINS[$dataset_id]}"
      done
      ;;
    *)
      echo "--list-datasets must be 'single' or 'multi'" >&2
      exit 2
      ;;
  esac
  exit 0
fi
if [[ "$mode" == single && "$primary_only" == 1 ]]; then
  echo "--primary-only applies only to multi-step groups" >&2
  exit 2
fi
if [[ -n "$selected_domains" && ( -n "$single_domains" || -n "$multi_domains" ) ]]; then
  echo "Use --domains or workflow-specific domain selectors, not both" >&2
  exit 2
fi
if [[ -n "$selected_domains" ]]; then
  case "$mode" in
    single) single_domains="$selected_domains" ;;
    multi) multi_domains="$selected_domains" ;;
    both) single_domains="$selected_domains"; multi_domains="$selected_domains" ;;
  esac
fi
if [[ "$mode" == multi && ( -n "$single_domains" || -n "$single_datasets" ) ]]; then
  echo "--single-domains and --single-datasets require single-step submission" >&2
  exit 2
fi
if [[ "$mode" == single && ( -n "$multi_domains" || -n "$multi_datasets" ) ]]; then
  echo "--multi-domains and --multi-datasets require multi-step submission" >&2
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
source "$SCRIPT_DIR/phase2_environment.sh"
validate_phase2_environment
for required_file in \
  "$LAYERMCP_GPT_OSS_CHECKPOINT/config.json" \
  "$LAYERMCP_GPT_OSS_CHECKPOINT/model.safetensors" \
  "$TIKTOKEN_ENCODINGS_BASE/o200k_base.tiktoken"; do
  if [[ ! -s "$required_file" ]]; then
    echo "Required GPT-OSS input is missing or empty: $required_file" >&2
    exit 2
  fi
done
export LAYERMCP_REPO_ROOT LAYERMCP_GPT_OSS_CHECKPOINT TIKTOKEN_ENCODINGS_BASE
export LAYERMCP_MODULES LAYERMCP_SLURM_ACCOUNT LAYERMCP_SLURM_PARTITION LAYERMCP_SLURM_GRES
export LAYERMCP_SLURM_CPUS_PER_TASK LAYERMCP_SLURM_MEM LAYERMCP_SLURM_TIME
source "$SCRIPT_DIR/submit_common.sh"
if [[ "$mode" != multi ]]; then
  DATASET_SELECTION="$(resolve_single_dataset_selection "$single_domains" "$single_datasets" primary)"
  export DATASET_SELECTION
fi
if [[ "$mode" != single ]]; then
  MULTI_DATASET_SELECTION="$(resolve_multi_dataset_selection "$multi_domains" "$multi_datasets")"
fi

LAYER_INDEX="${layer_override:-${LAYER_INDEX:-0}}"
ATTENTION_TARGET="${target_override:-${ATTENTION_TARGET:-qkv}}"
METHOD="${method_override:-${METHOD:-noise}}"
STRENGTH="${strength_override:-${STRENGTH:-0.01}}"
SEED="${seed_override:-${SEED:-1234}}"
SEEDS="${seeds_override:-${SEEDS:-$SEED}}"
STRENGTH_MODE="${strength_mode_override:-${STRENGTH_MODE:-absolute}}"
EXAMPLE_LIMIT="${example_limit_override:-${EXAMPLE_LIMIT:-}}"
[[ "$LAYER_INDEX" =~ ^[0-9]+$ ]] || { echo "--layer must be a non-negative integer" >&2; exit 2; }
[[ "$SEED" =~ ^-?[0-9]+$ ]] || { echo "--seed must be an integer" >&2; exit 2; }
[[ "$SEEDS" =~ ^-?[0-9]+(,-?[0-9]+)*$ ]] || { echo "--seeds must be a comma-separated list of integers" >&2; exit 2; }
[[ -z "$EXAMPLE_LIMIT" || "$EXAMPLE_LIMIT" =~ ^[1-9][0-9]*$ ]] || { echo "--example-limit must be a positive integer" >&2; exit 2; }
IFS=',' read -r -a seed_values <<< "$SEEDS"
(( ${#seed_values[@]} > 0 )) || { echo "--seeds must not be empty" >&2; exit 2; }
declare -A seen_seeds=()
for seed_value in "${seed_values[@]}"; do
  [[ -z "${seen_seeds[$seed_value]:-}" ]] || { echo "--seeds contains duplicate seed: $seed_value" >&2; exit 2; }
  seen_seeds[$seed_value]=1
done
case "$ATTENTION_TARGET" in
  attention_all|q_proj|k_proj|v_proj|o_proj|qkv|out) ;;
  *) echo "Unsupported attention target: $ATTENTION_TARGET" >&2; exit 2 ;;
esac
case "$METHOD" in
  noise|replace) ;;
  *) echo "Unsupported intervention method: $METHOD" >&2; exit 2 ;;
esac
case "$STRENGTH_MODE" in
  absolute|tensor_rms) ;;
  *) echo "Unsupported strength mode: $STRENGTH_MODE" >&2; exit 2 ;;
esac
if [[ "$STRENGTH_MODE" == tensor_rms && "$METHOD" != noise ]]; then
  echo "tensor_rms strength mode is defined only for noise interventions" >&2
  exit 2
fi
if [[ -n "$plan_manifest" && ! -f "$plan_manifest" ]]; then
  echo "Plan manifest does not exist: $plan_manifest" >&2
  exit 2
fi
if [[ "$mode" == both && -n "$plan_manifest" ]]; then
  echo "--plan-manifest is workflow-specific; use single or multi, not both" >&2
  exit 2
fi
if [[ "$mode" != single && ( -n "$EXAMPLE_LIMIT" || -n "$screen_plan_output" || -n "$combined_output" ) ]]; then
  echo "--example-limit, --screen-plan-output, and --combined-output currently apply only to direct single-step Phase 2 jobs" >&2
  exit 2
fi
if [[ -n "$screen_plan_output" || -n "$combined_output" ]]; then
  if [[ "$mode" != single || -z "$screen_plan_output" || -z "$combined_output" ]]; then
    echo "--screen-plan-output and --combined-output must be supplied together for a direct single-step job" >&2
    exit 2
  fi
  if [[ -n "$plan_manifest" ]]; then
    echo "Use either --plan-manifest or --screen-plan-output, not both" >&2
    exit 2
  fi
  [[ ! -e "$screen_plan_output" ]] || { echo "Screen plan output already exists: $screen_plan_output" >&2; exit 2; }
  [[ ! -e "$combined_output" ]] || { echo "Combined output already exists: $combined_output" >&2; exit 2; }
fi
[[ "$STRENGTH" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][+-]?[0-9]+)?$ ]] || {
  echo "--strength must be a finite non-negative number" >&2
  exit 2
}
export LAYER_INDEX ATTENTION_TARGET METHOD STRENGTH STRENGTH_MODE SEED SEEDS EXAMPLE_LIMIT
if [[ -n "$plan_manifest" ]]; then
  PLAN_MANIFEST="$(cd "$(dirname "$plan_manifest")" && pwd -P)/$(basename "$plan_manifest")"
  export PLAN_MANIFEST
fi
if [[ -n "$screen_plan_output" ]]; then
  case "$screen_plan_output" in
    /*) SCREEN_PLAN_OUTPUT="$screen_plan_output" ;;
    *) SCREEN_PLAN_OUTPUT="$PWD/$screen_plan_output" ;;
  esac
  case "$combined_output" in
    /*) COMBINED_OUTPUT="$combined_output" ;;
    *) COMBINED_OUTPUT="$PWD/$combined_output" ;;
  esac
  export SCREEN_PLAN_OUTPUT COMBINED_OUTPUT
fi
export RUN_KIND=full
unset WORKFLOW_LIMIT

if (( ! dry_run )) && ! command -v sbatch >/dev/null 2>&1; then
  echo "sbatch is unavailable; run this on a Slurm login node" >&2
  exit 2
fi

common_exports="ALL,LAYERMCP_VENV,LAYERMCP_REPO_ROOT,LAYERMCP_GPT_OSS_CHECKPOINT,TIKTOKEN_ENCODINGS_BASE,LAYER_INDEX=$LAYER_INDEX,ATTENTION_TARGET=$ATTENTION_TARGET,METHOD=$METHOD,STRENGTH=$STRENGTH,STRENGTH_MODE=$STRENGTH_MODE,SEED=$SEED,SEEDS=$SEEDS"
[[ -z "$EXAMPLE_LIMIT" ]] || common_exports+=",EXAMPLE_LIMIT=$EXAMPLE_LIMIT"
if [[ -n "${PLAN_MANIFEST:-}" ]]; then common_exports+=",PLAN_MANIFEST=$PLAN_MANIFEST"; fi
if [[ -n "${SCREEN_PLAN_OUTPUT:-}" ]]; then common_exports+=",SCREEN_PLAN_OUTPUT=$SCREEN_PLAN_OUTPUT,COMBINED_OUTPUT=$COMBINED_OUTPUT"; fi
scheduler_args=()
[[ -z "${LAYERMCP_SLURM_ACCOUNT:-}" ]] || scheduler_args+=("--account=$LAYERMCP_SLURM_ACCOUNT")
[[ -z "${LAYERMCP_SLURM_PARTITION:-}" ]] || scheduler_args+=("--partition=$LAYERMCP_SLURM_PARTITION")
[[ -z "${LAYERMCP_SLURM_GRES:-}" ]] || scheduler_args+=("--gres=$LAYERMCP_SLURM_GRES")
[[ -z "${LAYERMCP_SLURM_CPUS_PER_TASK:-}" ]] || scheduler_args+=("--cpus-per-task=$LAYERMCP_SLURM_CPUS_PER_TASK")
[[ -z "${LAYERMCP_SLURM_MEM:-}" ]] || scheduler_args+=("--mem=$LAYERMCP_SLURM_MEM")
[[ -z "${LAYERMCP_SLURM_TIME:-}" ]] || scheduler_args+=("--time=$LAYERMCP_SLURM_TIME")
submitted=0
submit_job() {
  local label="$1"
  local export_spec="$2"
  local launcher="$3"
  local -a command=(sbatch --parsable --chdir="$REPO_ROOT" "${scheduler_args[@]}" --job-name="$label" --export="$export_spec" "$launcher")
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
    "$common_exports,DATASET_SELECTION=${DATASET_SELECTION}" \
    "$SINGLE_LAUNCHER"
fi

if [[ "$mode" != single ]]; then
  if [[ "$MULTI_DATASET_SELECTION" == all ]]; then
    groups=("${PRIMARY_MULTI_STEP_GROUPS[@]}")
    if (( ! primary_only )); then
      groups+=(math_controlled)
    fi
  else
    IFS=':' read -r -a groups <<< "$MULTI_DATASET_SELECTION"
    if (( primary_only )); then
      for group in "${groups[@]}"; do
        if [[ "$group" == math_controlled ]]; then
          echo "--primary-only cannot be combined with explicit multi-step dataset math_controlled" >&2
          exit 2
        fi
      done
    fi
  fi
  for group in "${groups[@]}"; do
    submit_job \
      "p2-multi-${group}" \
      "$common_exports,RUN_KIND=full,DATASET_GROUP=$group" \
      "$MULTI_LAUNCHER"
  done
fi

if (( dry_run )); then
  printf 'Would submit %d job(s): mode=%s, layer=%s, target=%s, method=%s, strength-mode=%s, strength=%s, seeds=%s, examples=%s\n' \
    "$submitted" "$mode" "$LAYER_INDEX" "$ATTENTION_TARGET" "$METHOD" "$STRENGTH_MODE" "$STRENGTH" "$SEEDS" "${EXAMPLE_LIMIT:-all}"
else
  printf 'Submitted %d job(s).\n' "$submitted"
fi
