#!/bin/bash
# Example:
# MODEL=phi-4-local REASONING_MODE=direct DATASET_GROUP=math_controlled RUN_KIND=short_test
# sbatch --job-name="${MODEL}-${REASONING_MODE}-${RUN_KIND}-${DATASET_GROUP}" \
#   --export=ALL,MODEL,REASONING_MODE,DATASET_GROUP,RUN_KIND \
#   scripts/slurm/run_multi_step.sbatch
#SBATCH --job-name=lmcp-multistep
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

DATASET_GROUP_ORDER=(
  coding_sweagent
  coding_nebius_replay
  enterprise_tau2
  finance_convfinqa
  finance_finqa
  finance_finretrieval_replay
  math_controlled
)
ACCEPTED_DATASET_GROUPS="${DATASET_GROUP_ORDER[*]} all"
declare -A DATASETS=(
  [coding_sweagent]="benchmark/coding/coding_sweagent_multistep.json"
  [coding_nebius_replay]="benchmark/coding/coding_nebius_sweagent_replay_multistep.json"
  [enterprise_tau2]="benchmark/enterprise/enterprise_public_workflows.json"
  [finance_convfinqa]="benchmark/finance/finance_convfinqa_multistep.json"
  [finance_finqa]="benchmark/finance/finance_finqa_test_multistep.json"
  [finance_finretrieval_replay]="benchmark/finance/finance_finretrieval_replay_multistep.json"
  [math_controlled]="benchmark/math/math_multistep_controlled.json"
)
declare -A DATASET_DOMAINS=(
  [coding_sweagent]="coding"
  [coding_nebius_replay]="coding"
  [enterprise_tau2]="enterprise"
  [finance_convfinqa]="finance"
  [finance_finqa]="finance"
  [finance_finretrieval_replay]="finance"
  [math_controlled]="math"
)
declare -A DATASET_BENCHMARK_MODES=(
  [coding_sweagent]="grounded_tool_execution"
  [coding_nebius_replay]="offline_trace_replay"
  [enterprise_tau2]="grounded_tool_execution"
  [finance_convfinqa]="grounded_tool_execution"
  [finance_finqa]="grounded_tool_execution"
  [finance_finretrieval_replay]="offline_trace_replay"
  [math_controlled]="grounded_tool_execution"
)
declare -A EXPECTED_WORKFLOW_COUNTS=(
  [coding_sweagent]=5
  [coding_nebius_replay]=33
  [enterprise_tau2]=69
  [finance_convfinqa]=10
  [finance_finqa]=490
  [finance_finretrieval_replay]=485
  [math_controlled]=50
)
declare -A EXPECTED_STEP_COUNTS=(
  [coding_sweagent]=11
  [coding_nebius_replay]=139
  [enterprise_tau2]=350
  [finance_convfinqa]=35
  [finance_finqa]=1111
  [finance_finretrieval_replay]=1490
  [math_controlled]=105
)

if [[ -z "${DATASET_GROUP:-}" ]]; then
  echo "DATASET_GROUP is required. Accepted values: $ACCEPTED_DATASET_GROUPS" >&2
  exit 2
fi
if [[ -z "${RUN_KIND:-}" ]]; then
  echo "RUN_KIND is required (expected short_test or full)" >&2
  exit 2
fi
: "${REASONING_MODE:?Set REASONING_MODE to direct or reasoning}"
[[ "$REASONING_MODE" == direct || "$REASONING_MODE" == reasoning ]] || {
  echo "Unsupported REASONING_MODE: $REASONING_MODE (expected direct or reasoning)" >&2
  exit 2
}
[[ "$RUN_KIND" == short_test || "$RUN_KIND" == full ]] || {
  echo "Unsupported RUN_KIND: $RUN_KIND (expected short_test or full)" >&2
  exit 2
}

if [[ "$DATASET_GROUP" == all ]]; then
  [[ "$RUN_KIND" == short_test ]] || {
    echo "DATASET_GROUP=all is short_test-only" >&2
    exit 2
  }
  SELECTED_DATASET_GROUPS=("${DATASET_GROUP_ORDER[@]}")
elif [[ -n "${DATASETS[$DATASET_GROUP]:-}" ]]; then
  SELECTED_DATASET_GROUPS=("$DATASET_GROUP")
else
  echo "Unsupported DATASET_GROUP: $DATASET_GROUP. Accepted values: $ACCEPTED_DATASET_GROUPS" >&2
  exit 2
fi

for dataset_group_name in "${SELECTED_DATASET_GROUPS[@]}"; do
  for metadata_value in \
    "${DATASETS[$dataset_group_name]:-}" \
    "${DATASET_DOMAINS[$dataset_group_name]:-}" \
    "${DATASET_BENCHMARK_MODES[$dataset_group_name]:-}" \
    "${EXPECTED_WORKFLOW_COUNTS[$dataset_group_name]:-}" \
    "${EXPECTED_STEP_COUNTS[$dataset_group_name]:-}"; do
    if [[ -z "$metadata_value" ]]; then
      echo "Missing launcher metadata for dataset group: $dataset_group_name" >&2
      exit 2
    fi
  done
  if [[ ! -f "${DATASETS[$dataset_group_name]}" ]]; then
    echo "Dataset for group $dataset_group_name does not exist: ${DATASETS[$dataset_group_name]}" >&2
    exit 2
  fi
done

if [[ "${LAYERMCP_VALIDATE_DATASET_GROUPS_ONLY:-0}" == 1 ]]; then
  for dataset_group_name in "${SELECTED_DATASET_GROUPS[@]}"; do
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$dataset_group_name" \
      "${DATASETS[$dataset_group_name]}" \
      "${DATASET_DOMAINS[$dataset_group_name]}" \
      "${DATASET_BENCHMARK_MODES[$dataset_group_name]}" \
      "${EXPECTED_WORKFLOW_COUNTS[$dataset_group_name]}" \
      "${EXPECTED_STEP_COUNTS[$dataset_group_name]}"
  done
  exit 0
fi

: "${MODEL:?Set MODEL to phi-4-local, llama-3.1-8b-local, qwen-3.6-local, gemma-4-local, or gpt-oss-local}"
: "${SLURM_JOB_ID:?This launcher must run under Slurm}"
: "${SLURM_JOB_NAME:?Slurm must provide a job name}"

module load python/3.11
source "$HOME/venvs/layermcp/bin/activate"
export HF_HOME="$SCRATCH/hf_cache"
export HF_HUB_CACHE="$SCRATCH/hf_cache/hub"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset TRANSFORMERS_CACHE || true

for dataset_group_name in "${SELECTED_DATASET_GROUPS[@]}"; do
  python - \
    "${DATASETS[$dataset_group_name]}" \
    "${DATASET_BENCHMARK_MODES[$dataset_group_name]}" \
    "${EXPECTED_WORKFLOW_COUNTS[$dataset_group_name]}" \
    "${EXPECTED_STEP_COUNTS[$dataset_group_name]}" <<'PY'
import json
from pathlib import Path
import sys

dataset_path = Path(sys.argv[1])
expected_mode = sys.argv[2]
expected_workflows = int(sys.argv[3])
expected_steps = int(sys.argv[4])
rows = json.loads(dataset_path.read_text(encoding="utf-8"))
workflows = len(rows)
steps = sum(len(row.get("expected_steps", ())) for row in rows)
modes = {row.get("benchmark_mode") or "grounded_tool_execution" for row in rows}
if workflows != expected_workflows or steps != expected_steps:
    raise SystemExit(
        f"Dataset count mismatch for {dataset_path}: "
        f"expected {expected_workflows}/{expected_steps}, got {workflows}/{steps}"
    )
if modes != {expected_mode}:
    raise SystemExit(
        f"Benchmark mode mismatch for {dataset_path}: "
        f"expected {expected_mode!r}, got {sorted(modes)!r}"
    )
PY
done

ACTUAL_COMMIT="$(git rev-parse HEAD)"
EVALUATION_PROTOCOL="guided_predicted_rollout_v1"
WORKFLOW_EXECUTION_MODE="predicted_sequence"
if grep -Eq "MAX_TOOL_CORRECTIONS|repair_tool_call" evaluation/evaluate.py; then
  echo "Refusing to run: this protocol requires one attempt and zero retries." >&2
  exit 1
fi
grep -q "MULTISTEP_EVALUATION_PROTOCOL = \"$EVALUATION_PROTOCOL\"" evaluation/evaluate.py || {
  echo "Evaluator does not implement $EVALUATION_PROTOCOL" >&2
  exit 2
}

REQUIRED_CHECKPOINT_FILES=(config.json)
case "$MODEL" in
  phi-4-local)
    [[ "$REASONING_MODE" == direct ]] || {
      echo "phi-4-local does not implement reasoning mode; use REASONING_MODE=direct" >&2
      exit 2
    }
    : "${LAYERMCP_PHI4_CHECKPOINT:=$REPO_ROOT/checkpoints/phi-4}"
    export LAYERMCP_PHI4_CHECKPOINT
    CHECKPOINT="$LAYERMCP_PHI4_CHECKPOINT"; EXPECTED_MODEL_NAME="microsoft/phi-4"
    PROMPT_TEMPLATE_ID="tool_name_only_v1"; REASONING_METHOD="none"; GENERATION_LIMIT=128
    ;;
  llama-3.1-8b-local)
    [[ "$REASONING_MODE" == direct ]] || {
      echo "llama-3.1-8b-local does not implement reasoning mode; use REASONING_MODE=direct" >&2
      exit 2
    }
    : "${LAYERMCP_LLAMA31_8B_CHECKPOINT:=$REPO_ROOT/checkpoints/llama-3.1-8b-instruct}"
    export LAYERMCP_LLAMA31_8B_CHECKPOINT
    CHECKPOINT="$LAYERMCP_LLAMA31_8B_CHECKPOINT"; EXPECTED_MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
    PROMPT_TEMPLATE_ID="structured_tool_call_v1"; REASONING_METHOD="none"; GENERATION_LIMIT=128
    ;;
  qwen-3.6-local)
    : "${LAYERMCP_QWEN36_CHECKPOINT:=$REPO_ROOT/checkpoints/qwen-3.6}"
    export LAYERMCP_QWEN36_CHECKPOINT
    CHECKPOINT="$LAYERMCP_QWEN36_CHECKPOINT"; EXPECTED_MODEL_NAME="Qwen/Qwen3.6"
    PROMPT_TEMPLATE_ID="tool_name_only_v1"; GENERATION_LIMIT=4096
    [[ "$REASONING_MODE" == reasoning ]] && REASONING_METHOD="native_enabled" || REASONING_METHOD="native_disabled"
    ;;
  gemma-4-local)
    : "${LAYERMCP_GEMMA4_CHECKPOINT:=$REPO_ROOT/checkpoints/gemma-4}"
    export LAYERMCP_GEMMA4_CHECKPOINT
    CHECKPOINT="$LAYERMCP_GEMMA4_CHECKPOINT"; EXPECTED_MODEL_NAME="google/gemma-4-26b-a4b-it"
    PROMPT_TEMPLATE_ID="tool_name_only_v1"; GENERATION_LIMIT=4096
    [[ "$REASONING_MODE" == reasoning ]] && REASONING_METHOD="native_enabled" || REASONING_METHOD="native_disabled"
    ;;
  gpt-oss-local)
    [[ "$REASONING_MODE" == direct ]] || {
      echo "gpt-oss-local does not implement reasoning mode; use REASONING_MODE=direct" >&2
      exit 2
    }
    : "${LAYERMCP_GPT_OSS_CHECKPOINT:=$REPO_ROOT/checkpoints/gpt-oss-20b/original}"
    export LAYERMCP_GPT_OSS_CHECKPOINT
    CHECKPOINT="$LAYERMCP_GPT_OSS_CHECKPOINT"; EXPECTED_MODEL_NAME="openai/gpt-oss-20b"
    PROMPT_TEMPLATE_ID="harmony_structured_context_sql_v2"; REASONING_METHOD="none"; GENERATION_LIMIT=128
    REQUIRED_CHECKPOINT_FILES=(config.json model.safetensors)
    ;;
  *) echo "Unsupported MODEL: $MODEL" >&2; exit 2 ;;
esac
for required_checkpoint_file in "${REQUIRED_CHECKPOINT_FILES[@]}"; do
  [[ -s "$CHECKPOINT/$required_checkpoint_file" ]] || {
    echo "Missing required checkpoint file: $CHECKPOINT/$required_checkpoint_file" >&2
    exit 2
  }
done

safe_component() { python -m analysis.run_artifacts --safe-component "$1" "$2"; }
MODEL_COMPONENT="$(safe_component model "$MODEL")"
REASONING_COMPONENT="$(safe_component reasoning_mode "$REASONING_MODE")"
GROUP_COMPONENT="$(safe_component dataset_group "$DATASET_GROUP")"
RUN_KIND_COMPONENT="$(safe_component run_kind "$RUN_KIND")"
JOB_ID_COMPONENT="$(safe_component slurm_job_id "$SLURM_JOB_ID")"
JOB_NAME_COMPONENT="$(safe_component slurm_job_name "$SLURM_JOB_NAME")"
for required in "$MODEL_COMPONENT" "$REASONING_COMPONENT" "$RUN_KIND_COMPONENT" "$GROUP_COMPONENT"; do
  [[ "$JOB_NAME_COMPONENT" == *"$required"* ]] || {
    echo "Slurm job name must include model, reasoning mode, run kind, and dataset group: missing $required" >&2
    exit 2
  }
done

RUN_DATE="$(date -u +%Y%m%dT%H%M%SZ)"
RUNS_ROOT="results/runs/multi_step"
RUN_DIR="$RUNS_ROOT/${RUN_DATE}_${JOB_ID_COMPONENT}_${MODEL_COMPONENT}_${REASONING_COMPONENT}_${RUN_KIND_COMPONENT}_${GROUP_COMPONENT}"
mkdir -p "$RUNS_ROOT"; mkdir "$RUN_DIR"
cp "$LAUNCHER_PATH" "$RUN_DIR/launcher.sbatch"
sha256sum "$LAUNCHER_PATH" > "$RUN_DIR/launcher.sha256"

REGISTRY_SNAPSHOT="$RUN_DIR/live_registry_snapshot.json"
python -m analysis.run_artifacts --capture-live-registry --include-catalog \
  --server mcp_server/server.py --registry-metadata-out "$REGISTRY_SNAPSHOT"
mapfile -t REGISTRY_VALUES < <(python - "$REGISTRY_SNAPSHOT" <<'PY'
import json,re,sys
from pathlib import Path
x=json.loads(Path(sys.argv[1]).read_text())
for f in ("tool_pool","tool_count","tool_registry_fingerprint","tool_registry_fingerprint_version"):
    if x.get(f) in (None, ""): raise SystemExit(f"Missing live registry field: {f}")
if not re.fullmatch(r"sha256:[0-9a-f]{64}",x["tool_registry_fingerprint"]): raise SystemExit("Malformed fingerprint")
for f in ("tool_names","tool_schemas","tool_descriptions"):
    if not x.get(f): raise SystemExit(f"Missing live registry catalog: {f}")
print(x["tool_pool"]);print(x["tool_count"]);print(x["tool_registry_fingerprint"]);print(x["tool_registry_fingerprint_version"])
PY
)
TOOL_POOL="${REGISTRY_VALUES[0]}"; TOOL_COUNT="${REGISTRY_VALUES[1]}"
EXPECTED_REGISTRY_FINGERPRINT="${REGISTRY_VALUES[2]}"; REGISTRY_FINGERPRINT_VERSION="${REGISTRY_VALUES[3]}"

SOURCE_LIST="$RUN_DIR/resolved_datasets.txt"
for group in "${SELECTED_DATASET_GROUPS[@]}"; do realpath "${DATASETS[$group]}"; done > "$SOURCE_LIST"
declare -A EVALUATED PROVENANCE
for group in "${SELECTED_DATASET_GROUPS[@]}"; do
  source_benchmark="${DATASETS[$group]}"; EVALUATED[$group]="$source_benchmark"; PROVENANCE[$group]=""
  if [[ "$RUN_KIND" == short_test ]]; then
    subset_dir="$RUN_DIR/short_test_subsets/$group"; mkdir -p "$subset_dir"
    EVALUATED[$group]="$subset_dir/subset.json"; PROVENANCE[$group]="$subset_dir/provenance.json"
    python -m analysis.multi_step_run --make-short-test \
      --source-benchmark "$source_benchmark" --subset "${EVALUATED[$group]}" \
      --provenance "${PROVENANCE[$group]}" --model "$MODEL" \
      --checkpoint "$CHECKPOINT" --registry-snapshot "$REGISTRY_SNAPSHOT" \
      --reasoning-mode "$REASONING_MODE"
  fi
done

SLURM_STDOUT_PATH="$SUBMIT_DIR/slurm-${JOB_NAME_COMPONENT}-${JOB_ID_COMPONENT}.out"
SLURM_STDERR_PATH="$SUBMIT_DIR/slurm-${JOB_NAME_COMPONENT}-${JOB_ID_COMPONENT}.err"
python - "$RUN_DIR/run_metadata.json" "$SOURCE_LIST" "$RUN_DIR" <<PY
import collections,json,sys
from pathlib import Path
paths=Path(sys.argv[2]).read_text().splitlines(); run=Path(sys.argv[3])
counts={}; modes={}; declared={}; selections={}
for path in paths:
    rows=json.loads(Path(path).read_text()); counts[path]={"workflows":len(rows),"routed_steps":sum(len(r["expected_steps"]) for r in rows)}
    modes[path]=dict(sorted(collections.Counter(r.get("benchmark_mode") or "grounded_tool_execution" for r in rows).items()))
    declared[path]=dict(sorted(collections.Counter(r.get("workflow_execution_mode") or "isolated_step" for r in rows).items()))
for p in sorted(run.glob("short_test_subsets/*/provenance.json")):
    selections[p.parent.name]=json.loads(p.read_text())
data={"git_commit":"$ACTUAL_COMMIT","model":"$MODEL","expected_model_name":"$EXPECTED_MODEL_NAME","checkpoint":"$CHECKPOINT","prompt_condition":"$REASONING_MODE","prompt_template_id":"$PROMPT_TEMPLATE_ID","reasoning_mode":"$REASONING_MODE","reasoning_method":"$REASONING_METHOD","effective_generation_limit":$GENERATION_LIMIT,"effective_generation_limit_unit":"tokens","run_kind":"$RUN_KIND","headline_eligible":False if "$RUN_KIND"=="short_test" else True,"dataset_group":"$DATASET_GROUP","benchmark_paths":paths,"source_counts":counts,"short_test_selection":selections,"benchmark_mode_distributions":modes,"declared_workflow_execution_mode_distributions":declared,"workflow_execution_mode":"$WORKFLOW_EXECUTION_MODE","evaluation_protocol":"$EVALUATION_PROTOCOL","model_attempts_per_step":1,"tool_call_retries":0,"generation":{"temperature":0.0,"max_tokens":$GENERATION_LIMIT,"seed":"not_applicable_deterministic_greedy"},"tool_pool":"$TOOL_POOL","tool_count":int("$TOOL_COUNT"),"tool_registry_fingerprint":"$EXPECTED_REGISTRY_FINGERPRINT","tool_registry_fingerprint_version":"$REGISTRY_FINGERPRINT_VERSION","registry_metadata_source":"live_mcp_registry_at_job_startup","slurm_job_id":"$SLURM_JOB_ID","slurm_job_name":"$SLURM_JOB_NAME","slurm_stdout_path":"$SLURM_STDOUT_PATH","slurm_stderr_path":"$SLURM_STDERR_PATH"}
with Path(sys.argv[1]).open("x") as f:json.dump(data,f,indent=2,sort_keys=True);f.write("\n")
PY

for group in "${SELECTED_DATASET_GROUPS[@]}"; do
  source_benchmark="${DATASETS[$group]}"; evaluated_benchmark="${EVALUATED[$group]}"
  provenance_args=(); [[ -z "${PROVENANCE[$group]}" ]] || provenance_args=(--short-test-provenance "${PROVENANCE[$group]}")
  domain="$(safe_component domain "${DATASET_DOMAINS[$group]}")"
  dataset="$(safe_component dataset "$(basename "$source_benchmark" .json)")"
  dataset_dir="$RUN_DIR/domains/$domain/$dataset"; mkdir -p "$(dirname "$dataset_dir")"; mkdir "$dataset_dir"
  python -m evaluation.evaluate --dataset "$evaluated_benchmark" --router "$MODEL" \
    --reasoning-mode "$REASONING_MODE" --call-predicted-tools --output-dir "$dataset_dir" \
    2>&1 | tee "$dataset_dir/evaluation.log"
  python -m analysis.multi_step_run --dataset-dir "$dataset_dir" --index "$RUN_DIR/artifact_index.jsonl" \
    --source-benchmark "$source_benchmark" --evaluated-benchmark "$evaluated_benchmark" \
    --model "$EXPECTED_MODEL_NAME" --prompt-template "$PROMPT_TEMPLATE_ID" \
    --reasoning-mode "$REASONING_MODE" --reasoning-method "$REASONING_METHOD" \
    --generation-limit "$GENERATION_LIMIT" --registry-fingerprint "$EXPECTED_REGISTRY_FINGERPRINT" \
    --registry-fingerprint-version "$REGISTRY_FINGERPRINT_VERSION" --tool-count "$TOOL_COUNT" \
    --tool-pool "$TOOL_POOL" "${provenance_args[@]}"
done

python -m analysis.multi_step_run --validate-index --index "$RUN_DIR/artifact_index.jsonl" \
  --source-benchmark "$SOURCE_LIST" --run-metadata "$RUN_DIR/run_metadata.json"
test "$(wc -l < "$RUN_DIR/artifact_index.jsonl")" -eq "${#SELECTED_DATASET_GROUPS[@]}"
test ! -e "$RUN_DIR/RUN_COMPLETE"
( set -o noclobber; printf 'git_commit=%s\n' "$ACTUAL_COMMIT" > "$RUN_DIR/RUN_COMPLETE" )
echo "Completed $MODEL $REASONING_MODE $RUN_KIND $DATASET_GROUP under $RUN_DIR"
