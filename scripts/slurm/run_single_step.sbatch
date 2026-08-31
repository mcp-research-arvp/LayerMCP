#!/bin/bash
# Submit with a job name that identifies all run dimensions, for example:
# sbatch --job-name=phi-4-local-direct-smoke \
#   --export=ALL,MODEL=phi-4-local,REASONING_MODE=direct,RUN_KIND=smoke scripts/slurm/run_single_step.sbatch
# sbatch --job-name=qwen-3.6-local-reasoning-primary \
#   --export=ALL,MODEL=qwen-3.6-local,REASONING_MODE=reasoning,RUN_KIND=primary scripts/slurm/run_single_step.sbatch
# sbatch --job-name=gpt-oss-local-reasoning-low-single-primary \
#   --export=ALL,MODEL=gpt-oss-local,REASONING_MODE=reasoning,REASONING_EFFORT=low,RUN_KIND=primary scripts/slurm/run_single_step.sbatch
#SBATCH --job-name=lmcp-std-pr30
#SBATCH --account=def-yousefne_gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
LAUNCHER_PATH="$(realpath "${BASH_SOURCE[0]}")"
if [[ -n "${LAYERMCP_REPO_ROOT:-}" ]]; then
  REPO_ROOT_CANDIDATE="$LAYERMCP_REPO_ROOT"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  REPO_ROOT_CANDIDATE="$SLURM_SUBMIT_DIR"
else
  REPO_ROOT_CANDIDATE="$PWD"
fi
if ! REPO_ROOT="$(cd "$REPO_ROOT_CANDIDATE" 2>/dev/null && pwd -P)"; then
  echo "LayerMCP repository root does not exist: $REPO_ROOT_CANDIDATE" >&2
  echo "Submit sbatch from the LayerMCP repository root or set LAYERMCP_REPO_ROOT explicitly." >&2
  exit 2
fi
for required_repo_file in \
  pyproject.toml \
  evaluation/evaluate.py \
  scripts/slurm/run_single_step.sbatch \
  scripts/slurm/run_multi_step.sbatch; do
  if [[ ! -f "$REPO_ROOT/$required_repo_file" ]]; then
    echo "Selected directory is not the LayerMCP repository root: $REPO_ROOT" >&2
    echo "Missing required project file: $required_repo_file" >&2
    echo "Submit sbatch from the LayerMCP repository root or set LAYERMCP_REPO_ROOT explicitly." >&2
    exit 2
  fi
done
cd "$REPO_ROOT"

if [[ "${LAYERMCP_VALIDATE_REPO_ROOT_ONLY:-0}" == 1 ]]; then
  printf '%s\n' "$REPO_ROOT"
  exit 0
fi

ACTUAL_COMMIT="$(git rev-parse HEAD)"

if grep -Eq "MAX_TOOL_CORRECTIONS|repair_tool_call" evaluation/evaluate.py; then
  echo "Refusing to run: retry/repair behavior from PR #19 is present." >&2
  exit 1
fi

: "${MODEL:?Set MODEL to phi-4-local, llama-3.1-8b-local, qwen-3.6-local, gemma-4-local, or gpt-oss-local}"
: "${REASONING_MODE:?Set REASONING_MODE to direct or reasoning}"
: "${RUN_KIND:?Set RUN_KIND to smoke or primary}"
: "${SLURM_JOB_ID:?This launcher must run under Slurm}"
: "${SLURM_JOB_NAME:?Slurm must provide a job name}"
[[ "$REASONING_MODE" == direct || "$REASONING_MODE" == reasoning ]] || {
  echo "Unsupported REASONING_MODE: $REASONING_MODE (expected direct or reasoning)" >&2
  exit 2
}
case "$MODEL:$REASONING_MODE" in
  phi-4-local:reasoning|llama-3.1-8b-local:reasoning)
    echo "$MODEL does not implement reasoning mode; use REASONING_MODE=direct" >&2
    exit 2
    ;;
esac
if [[ "$MODEL" == gpt-oss-local ]]; then
  [[ "$REASONING_MODE" == reasoning ]] || {
    echo "gpt-oss-local requires REASONING_MODE=reasoning" >&2
    exit 2
  }
  [[ -n "${REASONING_EFFORT:-}" ]] || {
    echo "Set REASONING_EFFORT=low for gpt-oss-local" >&2
    exit 2
  }
  [[ "$REASONING_EFFORT" == low ]] || {
    echo "Unsupported GPT-OSS REASONING_EFFORT: $REASONING_EFFORT (expected low)" >&2
    exit 2
  }
elif [[ -n "${REASONING_EFFORT:-}" ]]; then
  echo "REASONING_EFFORT is GPT-OSS-only; omit it for $MODEL" >&2
  exit 2
fi
REASONING_EFFORT_VALUE="${REASONING_EFFORT:-}"
if [[ "${LAYERMCP_VALIDATE_REASONING_ONLY:-0}" == 1 ]]; then
  printf '%s\t%s\t%s\n' "$MODEL" "$REASONING_MODE" "$REASONING_EFFORT_VALUE"
  exit 0
fi

module load python/3.11
source "$HOME/venvs/layermcp/bin/activate"

export HF_HOME="$SCRATCH/hf_cache"
export HF_HUB_CACHE="$SCRATCH/hf_cache/hub"
unset TRANSFORMERS_CACHE || true

REGISTRY_METADATA_TMP="$(mktemp)"
trap 'rm -f "$REGISTRY_METADATA_TMP"' EXIT
python -m analysis.run_artifacts \
  --capture-live-registry \
  --server mcp_server/server.py \
  --registry-metadata-out "$REGISTRY_METADATA_TMP"
mapfile -t REGISTRY_VALUES < <(python - "$REGISTRY_METADATA_TMP" <<'PY'
import json
from pathlib import Path
import sys

metadata = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
for field in (
    "tool_pool",
    "tool_count",
    "tool_registry_fingerprint",
    "tool_registry_fingerprint_version",
):
    value = metadata.get(field)
    if value is None or value == "":
        raise SystemExit(f"Missing captured registry metadata: {field}")
    print(value)
PY
)
if [[ "${#REGISTRY_VALUES[@]}" -ne 4 ]]; then
  echo "Live registry metadata capture returned an unexpected shape." >&2
  exit 2
fi
TOOL_POOL="${REGISTRY_VALUES[0]}"
TOOL_COUNT="${REGISTRY_VALUES[1]}"
EXPECTED_REGISTRY_FINGERPRINT="${REGISTRY_VALUES[2]}"
REGISTRY_FINGERPRINT_VERSION="${REGISTRY_VALUES[3]}"
[[ "$TOOL_COUNT" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid live tool count" >&2; exit 2; }
[[ "$EXPECTED_REGISTRY_FINGERPRINT" =~ ^sha256:[0-9a-f]{64}$ ]] || {
  echo "Invalid live registry fingerprint" >&2
  exit 2
}
rm -f "$REGISTRY_METADATA_TMP"
trap - EXIT

safe_component() {
  local label="$1"
  local value="$2"
  python -m analysis.run_artifacts --safe-component "$label" "$value"
}

REQUIRED_CHECKPOINT_FILES=(config.json)
case "$MODEL" in
  phi-4-local)
    : "${LAYERMCP_PHI4_CHECKPOINT:=$REPO_ROOT/checkpoints/phi-4}"
    export LAYERMCP_PHI4_CHECKPOINT
    CHECKPOINT="$LAYERMCP_PHI4_CHECKPOINT"
    PROMPT_TEMPLATE_ID="tool_name_only_v1"
    EXPECTED_MODEL_NAME="microsoft/phi-4"
    REASONING_METHOD="none"
    GENERATION_LIMIT=128
    ;;
  llama-3.1-8b-local)
    : "${LAYERMCP_LLAMA31_8B_CHECKPOINT:=$REPO_ROOT/checkpoints/llama-3.1-8b-instruct}"
    export LAYERMCP_LLAMA31_8B_CHECKPOINT
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    CHECKPOINT="$LAYERMCP_LLAMA31_8B_CHECKPOINT"
    PROMPT_TEMPLATE_ID="structured_tool_call_v1"
    EXPECTED_MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
    REASONING_METHOD="none"
    GENERATION_LIMIT=128
    ;;
  qwen-3.6-local)
    : "${LAYERMCP_QWEN36_CHECKPOINT:=$REPO_ROOT/checkpoints/qwen-3.6}"
    export LAYERMCP_QWEN36_CHECKPOINT
    CHECKPOINT="$LAYERMCP_QWEN36_CHECKPOINT"
    PROMPT_TEMPLATE_ID="tool_name_only_v1"
    EXPECTED_MODEL_NAME="Qwen/Qwen3.6"
    GENERATION_LIMIT=4096
    [[ "$REASONING_MODE" == reasoning ]] && REASONING_METHOD="native_enabled" || REASONING_METHOD="native_disabled"
    ;;
  gemma-4-local)
    : "${LAYERMCP_GEMMA4_CHECKPOINT:=$REPO_ROOT/checkpoints/gemma-4}"
    export LAYERMCP_GEMMA4_CHECKPOINT
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    CHECKPOINT="$LAYERMCP_GEMMA4_CHECKPOINT"
    PROMPT_TEMPLATE_ID="tool_name_only_v1"
    EXPECTED_MODEL_NAME="google/gemma-4-26b-a4b-it"
    GENERATION_LIMIT=4096
    [[ "$REASONING_MODE" == reasoning ]] && REASONING_METHOD="native_enabled" || REASONING_METHOD="native_disabled"
    ;;
  gpt-oss-local)
    : "${LAYERMCP_GPT_OSS_CHECKPOINT:=$REPO_ROOT/checkpoints/gpt-oss-20b/original}"
    export LAYERMCP_GPT_OSS_CHECKPOINT
    CHECKPOINT="$LAYERMCP_GPT_OSS_CHECKPOINT"
    PROMPT_TEMPLATE_ID="harmony_structured_context_sql_v2"
    EXPECTED_MODEL_NAME="openai/gpt-oss-20b"
    REASONING_METHOD="harmony"
    GENERATION_LIMIT=4096
    REQUIRED_CHECKPOINT_FILES=(config.json model.safetensors)
    ;;
  *)
    echo "Unsupported MODEL: $MODEL" >&2
    exit 2
    ;;
esac

for required_checkpoint_file in "${REQUIRED_CHECKPOINT_FILES[@]}"; do
  [[ -s "$CHECKPOINT/$required_checkpoint_file" ]] || {
    echo "Missing required checkpoint file: $CHECKPOINT/$required_checkpoint_file" >&2
    exit 2
  }
done

SMOKE_BENCHMARKS=(
  "benchmark/coding/coding_smoke.json"
  "benchmark/finance/finance_smoke.json"
)

PRIMARY_BENCHMARKS=(
  "benchmark/math/math_controlled.json"
  "benchmark/math/math_public.json"
  "benchmark/math/math_public_math_dataset.json"
  "benchmark/enterprise/enterprise_controlled.json"
  "benchmark/enterprise/enterprise_tau2_single_step.json"
  "benchmark/coding/coding_controlled.json"
  "benchmark/coding/coding_upstream_inspired.json"
  "benchmark/coding/coding_codesearchnet_public_derived.json"
  "benchmark/coding/coding_conala_public_derived.json"
  "benchmark/finance/finance_controlled.json"
  "benchmark/finance/finance_upstream_inspired.json"
  "benchmark/finance/finance_tatqa_public_derived.json"
  "benchmark/finance/finance_finqa_test_single.json"
)

case "$RUN_KIND" in
  smoke|primary) ;;
  *)
    echo "Unsupported RUN_KIND: $RUN_KIND (expected smoke or primary)" >&2
    exit 2
    ;;
esac

if [[ -n "${DATASET_SELECTION:-}" ]]; then
  IFS=':' read -r -a BENCHMARKS <<< "$DATASET_SELECTION"
  [[ "${#BENCHMARKS[@]}" -gt 0 ]] || {
    echo "DATASET_SELECTION must contain at least one colon-separated benchmark path" >&2
    exit 2
  }
  for selected_benchmark in "${BENCHMARKS[@]}"; do
    if [[ -z "$selected_benchmark" ]]; then
      echo "DATASET_SELECTION must not contain empty benchmark paths" >&2
      exit 2
    fi
    is_allowed_benchmark=0
    for allowed_benchmark in "${SMOKE_BENCHMARKS[@]}" "${PRIMARY_BENCHMARKS[@]}"; do
      if [[ "$selected_benchmark" == "$allowed_benchmark" ]]; then
        is_allowed_benchmark=1
        break
      fi
    done
    if [[ "$is_allowed_benchmark" != 1 ]]; then
      echo "DATASET_SELECTION contains an unsupported single-step benchmark: $selected_benchmark" >&2
      exit 2
    fi
  done
else
  case "$RUN_KIND" in
    smoke) BENCHMARKS=("${SMOKE_BENCHMARKS[@]}") ;;
    primary) BENCHMARKS=("${PRIMARY_BENCHMARKS[@]}") ;;
  esac
fi

PROMPT_MODE="$REASONING_MODE"
MODEL_COMPONENT="$(safe_component model "$MODEL")"
REASONING_COMPONENT="$(safe_component reasoning_mode "$REASONING_MODE")"
if [[ -n "$REASONING_EFFORT_VALUE" ]]; then
  REASONING_EFFORT_COMPONENT="$(safe_component reasoning_effort "$REASONING_EFFORT_VALUE")"
else
  REASONING_EFFORT_COMPONENT=""
fi
RUN_KIND_COMPONENT="$(safe_component run_kind "$RUN_KIND")"
JOB_ID_COMPONENT="$(safe_component slurm_job_id "$SLURM_JOB_ID")"
JOB_NAME_COMPONENT="$(safe_component slurm_job_name "$SLURM_JOB_NAME")"
for required_name_part in "$MODEL_COMPONENT" "$REASONING_COMPONENT" "$RUN_KIND_COMPONENT"; do
  if [[ "$JOB_NAME_COMPONENT" != *"$required_name_part"* ]]; then
    echo "Slurm job name must identify model, reasoning mode, and run kind: " \
      "$SLURM_JOB_NAME is missing $required_name_part" >&2
    exit 2
  fi
done
if [[ -n "$REASONING_EFFORT_COMPONENT" && "$JOB_NAME_COMPONENT" != *"$REASONING_EFFORT_COMPONENT"* ]]; then
  echo "Slurm job name must identify reasoning effort: $SLURM_JOB_NAME is missing $REASONING_EFFORT_COMPONENT" >&2
  exit 2
fi
if [[ "$MODEL" == gpt-oss-local ]]; then
  EXPECTED_GPT_OSS_JOB_NAME="gpt-oss-local-reasoning-low-single-${RUN_KIND_COMPONENT}"
  [[ "$JOB_NAME_COMPONENT" == "$EXPECTED_GPT_OSS_JOB_NAME" ]] || {
    echo "GPT-OSS job name must be: $EXPECTED_GPT_OSS_JOB_NAME" >&2
    exit 2
  }
fi

RUNS_ROOT="results/runs/single_step"
RUN_DATE="$(date -u +%Y%m%dT%H%M%SZ)"
CONDITION_COMPONENT="$REASONING_COMPONENT"
if [[ -n "$REASONING_EFFORT_COMPONENT" ]]; then
  CONDITION_COMPONENT="${CONDITION_COMPONENT}_${REASONING_EFFORT_COMPONENT}"
fi
RUN_DIR="$RUNS_ROOT/${RUN_DATE}_${JOB_ID_COMPONENT}_${MODEL_COMPONENT}_${CONDITION_COMPONENT}_${RUN_KIND_COMPONENT}"
mkdir -p "$RUNS_ROOT"
mkdir "$RUN_DIR"

SLURM_STDOUT_PATH="$SUBMIT_DIR/slurm-${JOB_NAME_COMPONENT}-${JOB_ID_COMPONENT}.out"
SLURM_STDERR_PATH="$SUBMIT_DIR/slurm-${JOB_NAME_COMPONENT}-${JOB_ID_COMPONENT}.err"

: "${RUN_DIR:?RUN_DIR must be set before preserving the launcher}"
cp "$LAUNCHER_PATH" "$RUN_DIR/launcher.sbatch"
sha256sum "$LAUNCHER_PATH" > "$RUN_DIR/launcher.sha256"

for benchmark in "${BENCHMARKS[@]}"; do
  test -f "$benchmark"
  realpath "$benchmark"
done > "$RUN_DIR/resolved_datasets.txt"

python - "$RUN_DIR/run_metadata.json" "$RUN_DIR/resolved_datasets.txt" <<PY
import json
from pathlib import Path
import sys

metadata_path = Path(sys.argv[1])
datasets = Path(sys.argv[2]).read_text(encoding="utf-8").splitlines()
metadata = {
    "git_branch": "$(git branch --show-current)",
    "git_commit": "$ACTUAL_COMMIT",
    "slurm_job_id": "$SLURM_JOB_ID",
    "slurm_job_name": "$SLURM_JOB_NAME",
    "slurm_stdout_path": "$SLURM_STDOUT_PATH",
    "slurm_stderr_path": "$SLURM_STDERR_PATH",
    "model": "$MODEL",
    "expected_model_name": "$EXPECTED_MODEL_NAME",
    "checkpoint": "$CHECKPOINT",
    "prompt_condition": "$PROMPT_MODE",
    "reasoning_mode": "$REASONING_MODE",
    "reasoning_method": "$REASONING_METHOD",
    "effective_generation_limit": $GENERATION_LIMIT,
    "effective_generation_limit_unit": "tokens",
    "run_kind": "$RUN_KIND",
    "prompt_template_id": "$PROMPT_TEMPLATE_ID",
    "benchmark_mode": "grounded_tool_execution",
    "benchmark_paths": datasets,
    "tool_pool": "$TOOL_POOL",
    "tool_count": $TOOL_COUNT,
    "tool_registry_fingerprint": "$EXPECTED_REGISTRY_FINGERPRINT",
    "tool_registry_fingerprint_version": "$REGISTRY_FINGERPRINT_VERSION",
    "registry_metadata_source": "live_mcp_registry_at_job_startup",
    "evaluation_protocol": "single_step_tool_routing_v1",
    "final_outcome_matchers": [
        "recursive_json_subset_v1",
        "finance_query_table_rows_v1",
    ],
    "generation": {
        "temperature": 0.0,
        "max_tokens": $GENERATION_LIMIT,
        "effective_generation_limit": $GENERATION_LIMIT,
        "effective_generation_limit_unit": "tokens",
        "seed": "not_applicable_deterministic_greedy",
    },
    "model_attempts_per_sample": 1,
    "tool_call_retries": 0,
}
reasoning_effort = "$REASONING_EFFORT_VALUE"
if reasoning_effort:
    metadata["reasoning_effort"] = reasoning_effort
with metadata_path.open("x", encoding="utf-8") as handle:
    json.dump(metadata, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY

python -m compileall benchmark evaluation mcp_server tests

for benchmark in "${BENCHMARKS[@]}"; do
  test -f "$benchmark"
  domain="$(safe_component domain "$(printf '%s' "$benchmark" | cut -d/ -f2)")"
  name="$(safe_component dataset "$(basename "$benchmark" .json)")"
  dataset_dir="$RUN_DIR/domains/$domain/$name"
  mkdir -p "$(dirname "$dataset_dir")"
  mkdir "$dataset_dir"
  log_path="$dataset_dir/evaluation.log"

  echo "------------------------------------------------------------"
  echo "Model: $MODEL"
  echo "Benchmark: $benchmark"
  echo "Log: $log_path"
  echo "------------------------------------------------------------"

  REASONING_EFFORT_ARGS=()
  if [[ -n "$REASONING_EFFORT_VALUE" ]]; then
    REASONING_EFFORT_ARGS=(--reasoning-effort "$REASONING_EFFORT_VALUE")
  fi
  python -m evaluation.evaluate \
    --dataset "$benchmark" \
    --router "$MODEL" \
    --reasoning-mode "$REASONING_MODE" \
    "${REASONING_EFFORT_ARGS[@]}" \
    --call-predicted-tools \
    --output-dir "$dataset_dir" \
    2>&1 | tee "$log_path"

  python -m analysis.run_artifacts \
    --dataset-dir "$dataset_dir" \
    --index "$RUN_DIR/artifact_index.jsonl" \
    --benchmark "$benchmark" \
    --model "$EXPECTED_MODEL_NAME" \
    --prompt-template "$PROMPT_TEMPLATE_ID" \
    --registry-fingerprint "$EXPECTED_REGISTRY_FINGERPRINT" \
    --registry-fingerprint-version "$REGISTRY_FINGERPRINT_VERSION" \
    --tool-count "$TOOL_COUNT" \
    --tool-pool "$TOOL_POOL"
done

python -m analysis.run_artifacts \
  --validate-run \
  --index "$RUN_DIR/artifact_index.jsonl" \
  --resolved-datasets "$RUN_DIR/resolved_datasets.txt" \
  --run-metadata "$RUN_DIR/run_metadata.json"

test ! -e "$RUN_DIR/RUN_COMPLETE"
( set -o noclobber; printf 'git_commit=%s\n' "$ACTUAL_COMMIT" > "$RUN_DIR/RUN_COMPLETE" )
echo "Completed $MODEL $REASONING_MODE $RUN_KIND evaluation."
echo "Results saved under: $RUN_DIR"
