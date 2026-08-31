#!/bin/bash
# Shared functions for LayerMCP Slurm submission wrappers. Source this file;
# run submit_all.sh, submit_single_step.sh, or submit_multi_step.sh instead.

set -euo pipefail

SUBMISSION_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
LAYERMCP_REPO_ROOT="$(cd "$SUBMISSION_SCRIPT_DIR/../.." && pwd -P)"
SINGLE_STEP_LAUNCHER="$LAYERMCP_REPO_ROOT/scripts/slurm/run_single_step.sbatch"
MULTI_STEP_LAUNCHER="$LAYERMCP_REPO_ROOT/scripts/slurm/run_multi_step.sbatch"

# These are evaluation conditions, rather than merely model names: Qwen and
# Gemma support both direct and native-reasoning runs, while Phi, Llama, and
# GPT-OSS each have one supported condition.
ALL_CONDITIONS=(
  "phi-4-local:direct"
  "llama-3.1-8b-local:direct"
  "qwen-3.6-local:direct"
  "qwen-3.6-local:reasoning"
  "gemma-4-local:direct"
  "gemma-4-local:reasoning"
  "gpt-oss-local:reasoning"
)
PRIMARY_MULTI_STEP_GROUPS=(
  coding_sweagent
  coding_nebius_replay
  enterprise_tau2
  finance_convfinqa
  finance_finqa
  finance_finretrieval_replay
  math_public_mathqa
)
SINGLE_STEP_DATASET_IDS=(
  coding_smoke
  finance_smoke
  math_controlled
  math_public
  math_public_math_dataset
  enterprise_controlled
  enterprise_tau2_single_step
  coding_controlled
  coding_upstream_inspired
  coding_codesearchnet_public_derived
  coding_conala_public_derived
  finance_controlled
  finance_upstream_inspired
  finance_tatqa_public_derived
  finance_finqa_test_single
)
SINGLE_STEP_SMOKE_DATASET_IDS=(
  coding_smoke
  finance_smoke
)
SINGLE_STEP_PRIMARY_DATASET_IDS=(
  math_controlled
  math_public
  math_public_math_dataset
  enterprise_controlled
  enterprise_tau2_single_step
  coding_controlled
  coding_upstream_inspired
  coding_codesearchnet_public_derived
  coding_conala_public_derived
  finance_controlled
  finance_upstream_inspired
  finance_tatqa_public_derived
  finance_finqa_test_single
)
declare -A SINGLE_STEP_DATASETS=(
  [coding_smoke]="benchmark/coding/coding_smoke.json"
  [finance_smoke]="benchmark/finance/finance_smoke.json"
  [math_controlled]="benchmark/math/math_controlled.json"
  [math_public]="benchmark/math/math_public.json"
  [math_public_math_dataset]="benchmark/math/math_public_math_dataset.json"
  [enterprise_controlled]="benchmark/enterprise/enterprise_controlled.json"
  [enterprise_tau2_single_step]="benchmark/enterprise/enterprise_tau2_single_step.json"
  [coding_controlled]="benchmark/coding/coding_controlled.json"
  [coding_upstream_inspired]="benchmark/coding/coding_upstream_inspired.json"
  [coding_codesearchnet_public_derived]="benchmark/coding/coding_codesearchnet_public_derived.json"
  [coding_conala_public_derived]="benchmark/coding/coding_conala_public_derived.json"
  [finance_controlled]="benchmark/finance/finance_controlled.json"
  [finance_upstream_inspired]="benchmark/finance/finance_upstream_inspired.json"
  [finance_tatqa_public_derived]="benchmark/finance/finance_tatqa_public_derived.json"
  [finance_finqa_test_single]="benchmark/finance/finance_finqa_test_single.json"
)
declare -A SINGLE_STEP_DATASET_DOMAINS=(
  [coding_smoke]=coding
  [finance_smoke]=finance
  [math_controlled]=math
  [math_public]=math
  [math_public_math_dataset]=math
  [enterprise_controlled]=enterprise
  [enterprise_tau2_single_step]=enterprise
  [coding_controlled]=coding
  [coding_upstream_inspired]=coding
  [coding_codesearchnet_public_derived]=coding
  [coding_conala_public_derived]=coding
  [finance_controlled]=finance
  [finance_upstream_inspired]=finance
  [finance_tatqa_public_derived]=finance
  [finance_finqa_test_single]=finance
)
declare -A MULTI_STEP_DATASET_DOMAINS=(
  [coding_sweagent]=coding
  [coding_nebius_replay]=coding
  [enterprise_tau2]=enterprise
  [finance_convfinqa]=finance
  [finance_finqa]=finance
  [finance_finretrieval_replay]=finance
  [math_public_mathqa]=math
  [math_controlled]=math
)
MULTI_STEP_DATASET_IDS=("${PRIMARY_MULTI_STEP_GROUPS[@]}" math_controlled)

submission_usage_common() {
  cat <<'EOF'
Options shared by the submission wrappers:
  --model MODEL       Submit only MODEL. Defaults to all supported models.
  --dry-run           Print commands without calling sbatch.
  -h, --help          Show usage.

Supported models:
  phi-4-local, llama-3.1-8b-local, qwen-3.6-local, gemma-4-local,
  gpt-oss-local

"All models" means all seven valid model/reasoning conditions. GPT-OSS uses
reasoning effort low automatically; unsupported conditions are never submitted.
EOF
}

append_csv_value() {
  local current="$1"
  local value="$2"
  if [[ -n "$current" ]]; then
    printf '%s,%s' "$current" "$value"
  else
    printf '%s' "$value"
  fi
}

validate_selector_values() {
  local values="$1"
  local label="$2"
  local allowed_name="$3"
  local value
  local -n allowed="$allowed_name"
  local -a parsed_values=()
  [[ -z "$values" ]] && return
  IFS=',' read -r -a parsed_values <<< "$values"
  for value in "${parsed_values[@]}"; do
    if [[ -z "$value" || -z "${allowed[$value]:-}" ]]; then
      echo "Unsupported $label: $value" >&2
      exit 2
    fi
  done
}

resolve_single_dataset_selection() {
  local domains="$1"
  local datasets="$2"
  local run_kind="$3"
  local id path selection=""
  local -a eligible_dataset_ids=()
  local -a parsed_domains=()
  local -a parsed_datasets=()
  local -A allowed_domains=([math]=1 [enterprise]=1 [coding]=1 [finance]=1)
  local -A allowed_datasets=()
  local -A requested_domains=()
  local -A requested_datasets=()

  case "$run_kind" in
    smoke) eligible_dataset_ids=("${SINGLE_STEP_SMOKE_DATASET_IDS[@]}") ;;
    primary) eligible_dataset_ids=("${SINGLE_STEP_PRIMARY_DATASET_IDS[@]}") ;;
    *)
      echo "Unsupported single-step run kind: $run_kind (expected smoke or primary)" >&2
      exit 2
      ;;
  esac
  for id in "${eligible_dataset_ids[@]}"; do
    allowed_datasets[$id]=1
  done
  validate_selector_values "$domains" domain allowed_domains
  validate_selector_values "$datasets" "single-step dataset" allowed_datasets
  [[ -n "$domains" ]] && IFS=',' read -r -a parsed_domains <<< "$domains" || parsed_domains=()
  [[ -n "$datasets" ]] && IFS=',' read -r -a parsed_datasets <<< "$datasets" || parsed_datasets=()
  for id in "${parsed_domains[@]}"; do requested_domains[$id]=1; done
  for id in "${parsed_datasets[@]}"; do requested_datasets[$id]=1; done

  for id in "${eligible_dataset_ids[@]}"; do
    if [[ -n "${requested_domains[${SINGLE_STEP_DATASET_DOMAINS[$id]}]:-}" || -n "${requested_datasets[$id]:-}" ]]; then
      path="${SINGLE_STEP_DATASETS[$id]}"
      selection+="${selection:+:}$path"
    fi
  done
  printf '%s' "$selection"
}

resolve_multi_dataset_selection() {
  local domains="$1"
  local datasets="$2"
  local id selection=""
  local -a parsed_domains=()
  local -a parsed_datasets=()
  local -A allowed_domains=([math]=1 [enterprise]=1 [coding]=1 [finance]=1)
  local -A allowed_datasets=()
  local -A requested_domains=()
  local -A requested_datasets=()

  for id in "${MULTI_STEP_DATASET_IDS[@]}"; do
    allowed_datasets[$id]=1
  done
  validate_selector_values "$domains" domain allowed_domains
  validate_selector_values "$datasets" "multi-step dataset group" allowed_datasets
  if [[ -z "$domains" && -z "$datasets" ]]; then
    printf 'all'
    return
  fi
  [[ -n "$domains" ]] && IFS=',' read -r -a parsed_domains <<< "$domains" || parsed_domains=()
  [[ -n "$datasets" ]] && IFS=',' read -r -a parsed_datasets <<< "$datasets" || parsed_datasets=()
  for id in "${parsed_domains[@]}"; do requested_domains[$id]=1; done
  for id in "${parsed_datasets[@]}"; do requested_datasets[$id]=1; done

  for id in "${MULTI_STEP_DATASET_IDS[@]}"; do
    if [[ -n "${requested_datasets[$id]:-}" || ( -n "${requested_domains[${MULTI_STEP_DATASET_DOMAINS[$id]}]:-}" && "$id" != math_controlled ) ]]; then
      selection+="${selection:+:}$id"
    fi
  done
  printf '%s' "$selection"
}

require_submission_repository() {
  local required
  for required in \
    "$SINGLE_STEP_LAUNCHER" \
    "$MULTI_STEP_LAUNCHER" \
    "$LAYERMCP_REPO_ROOT/evaluation/evaluate.py" \
    "$LAYERMCP_REPO_ROOT/pyproject.toml"; do
    if [[ ! -f "$required" ]]; then
      echo "LayerMCP submission helper could not find required file: $required" >&2
      exit 2
    fi
  done
}

condition_matches_model() {
  local condition="$1"
  local selected_model="$2"
  [[ "$selected_model" == all || "${condition%%:*}" == "$selected_model" ]]
}

validate_model() {
  local model="$1"
  case "$model" in
    all|phi-4-local|llama-3.1-8b-local|qwen-3.6-local|gemma-4-local|gpt-oss-local)
      ;;
    *)
      echo "Unsupported model: $model" >&2
      submission_usage_common >&2
      exit 2
      ;;
  esac
}

validate_single_run_kind() {
  case "$1" in
    smoke|primary) ;;
    *)
      echo "Unsupported single-step run kind: $1 (expected smoke or primary)" >&2
      exit 2
      ;;
  esac
}

validate_multi_run_kind() {
  case "$1" in
    short_test|full) ;;
    *)
      echo "Unsupported multi-step run kind: $1 (expected short_test or full)" >&2
      exit 2
      ;;
  esac
}

validate_multi_group() {
  case "$1" in
    all|coding_sweagent|coding_nebius_replay|enterprise_tau2|finance_convfinqa|finance_finqa|finance_finretrieval_replay|math_public_mathqa|math_controlled)
      ;;
    *)
      echo "Unsupported multi-step dataset group: $1" >&2
      exit 2
      ;;
  esac
}

submit_sbatch() {
  local dry_run="$1"
  shift
  if [[ "$dry_run" == 1 ]]; then
    printf 'sbatch'
    printf ' %q' "$@"
    printf '\n'
  else
    sbatch "$@"
  fi
}

submit_single_condition() {
  local model="$1"
  local reasoning_mode="$2"
  local run_kind="$3"
  local dry_run="$4"
  local dataset_selection="$5"
  local condition_name="${model}-${reasoning_mode}"
  if [[ "$model" == gpt-oss-local ]]; then
    condition_name+="-low"
  fi
  local job_name="${condition_name}-single-${run_kind}"
  local exports="ALL,LAYERMCP_REPO_ROOT=$LAYERMCP_REPO_ROOT,MODEL=$model,REASONING_MODE=$reasoning_mode,RUN_KIND=$run_kind"

  if [[ "$model" == gpt-oss-local ]]; then
    exports+=",REASONING_EFFORT=low"
  else
    exports+=",REASONING_EFFORT="
  fi
  exports+=",DATASET_SELECTION=$dataset_selection"
  submit_sbatch "$dry_run" --job-name="$job_name" --export="$exports" "$SINGLE_STEP_LAUNCHER"
}

submit_multi_condition() {
  local model="$1"
  local reasoning_mode="$2"
  local run_kind="$3"
  local dataset_group="$4"
  local dry_run="$5"
  local condition_name="${model}-${reasoning_mode}"
  if [[ "$model" == gpt-oss-local ]]; then
    condition_name+="-low"
  fi
  local job_name="${condition_name}-multi-${run_kind}-${dataset_group}"
  local exports="ALL,LAYERMCP_REPO_ROOT=$LAYERMCP_REPO_ROOT,MODEL=$model,REASONING_MODE=$reasoning_mode,RUN_KIND=$run_kind,DATASET_GROUP=$dataset_group"

  if [[ "$model" == gpt-oss-local ]]; then
    exports+=",REASONING_EFFORT=low"
  else
    exports+=",REASONING_EFFORT="
  fi
  submit_sbatch "$dry_run" --job-name="$job_name" --export="$exports" "$MULTI_STEP_LAUNCHER"
}

submit_single_steps() {
  local selected_model="$1"
  local run_kind="$2"
  local dataset_selection="$3"
  local dry_run="$4"
  local condition model reasoning_mode submitted=0

  require_submission_repository
  validate_model "$selected_model"
  validate_single_run_kind "$run_kind"
  for condition in "${ALL_CONDITIONS[@]}"; do
    if ! condition_matches_model "$condition" "$selected_model"; then
      continue
    fi
    model="${condition%%:*}"
    reasoning_mode="${condition##*:}"
    submit_single_condition "$model" "$reasoning_mode" "$run_kind" "$dry_run" "$dataset_selection"
    ((submitted += 1))
  done
  echo "Submitted $submitted single-step job(s)."
}

submit_multi_steps() {
  local selected_model="$1"
  local run_kind="$2"
  local requested_groups="$3"
  local dry_run="$4"
  local condition model reasoning_mode group submitted=0
  local groups=()
  local requested_group_list=()

  require_submission_repository
  validate_model "$selected_model"
  validate_multi_run_kind "$run_kind"
  IFS=':' read -r -a requested_group_list <<< "$requested_groups"
  for group in "${requested_group_list[@]}"; do
    validate_multi_group "$group"
  done
  if [[ "$requested_groups" == all && "$run_kind" == full ]]; then
    # The canonical launcher deliberately disallows one giant full run. Submit
    # one complete, independently validated job for each primary group instead.
    groups=("${PRIMARY_MULTI_STEP_GROUPS[@]}")
  else
    groups=("${requested_group_list[@]}")
  fi

  for condition in "${ALL_CONDITIONS[@]}"; do
    if ! condition_matches_model "$condition" "$selected_model"; then
      continue
    fi
    model="${condition%%:*}"
    reasoning_mode="${condition##*:}"
    for group in "${groups[@]}"; do
      submit_multi_condition "$model" "$reasoning_mode" "$run_kind" "$group" "$dry_run"
      ((submitted += 1))
    done
  done
  echo "Submitted $submitted multi-step job(s)."
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "Source submit_common.sh from a submission wrapper; do not run it directly." >&2
  exit 2
fi
