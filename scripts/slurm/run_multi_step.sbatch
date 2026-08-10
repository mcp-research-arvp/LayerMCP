#!/bin/bash
# Full examples (exactly one dataset group):
# sbatch --job-name=phi-4-local-standard_prompt-full-finqa --export=ALL,MODEL=phi-4-local,DATASET_GROUP=finqa,RUN_KIND=full scripts/slurm/run_multi_step.sbatch
# sbatch --job-name=llama-3.1-8b-local-standard_prompt-full-enterprise_tau2 --export=ALL,MODEL=llama-3.1-8b-local,DATASET_GROUP=enterprise_tau2,RUN_KIND=full scripts/slurm/run_multi_step.sbatch
# Worst-case preflight across all seven groups (override walltime at submission):
# sbatch --time=02:00:00 --job-name=phi-4-local-standard_prompt-preflight-all --export=ALL,MODEL=phi-4-local,DATASET_GROUP=all,RUN_KIND=preflight scripts/slurm/run_multi_step.sbatch
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
cd "$SCRATCH/layermcp/LayerMCP"
module load python/3.11
source "$HOME/venvs/layermcp/bin/activate"
export HF_HOME="$SCRATCH/hf_cache"
export HF_HUB_CACHE="$SCRATCH/hf_cache/hub"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset TRANSFORMERS_CACHE || true

: "${MODEL:?Set MODEL to phi-4-local or llama-3.1-8b-local}"
: "${DATASET_GROUP:?Set DATASET_GROUP to a supported group or all}"
: "${RUN_KIND:?Set RUN_KIND to preflight or full}"
: "${SLURM_JOB_ID:?This launcher must run under Slurm}"
: "${SLURM_JOB_NAME:?Slurm must provide a job name}"

ACTUAL_COMMIT="$(git rev-parse HEAD)"
PROMPT_MODE="standard_prompt"
if grep -Eq "MAX_TOOL_CORRECTIONS|repair_tool_call" evaluation/evaluate.py; then
  echo "Refusing to run: retry/repair behavior is present; this protocol requires zero retries." >&2
  exit 1
fi

safe_component() { python -m analysis.run_artifacts --safe-component "$1" "$2"; }
MODEL_COMPONENT="$(safe_component model "$MODEL")"
GROUP_COMPONENT="$(safe_component dataset_group "$DATASET_GROUP")"
RUN_KIND_COMPONENT="$(safe_component run_kind "$RUN_KIND")"
PROMPT_COMPONENT="$(safe_component prompt_condition "$PROMPT_MODE")"
JOB_ID_COMPONENT="$(safe_component slurm_job_id "$SLURM_JOB_ID")"
JOB_NAME_COMPONENT="$(safe_component slurm_job_name "$SLURM_JOB_NAME")"
for required in "$MODEL_COMPONENT" "$PROMPT_COMPONENT" "$RUN_KIND_COMPONENT" "$GROUP_COMPONENT"; do
  [[ "$JOB_NAME_COMPONENT" == *"$required"* ]] || {
    echo "Slurm job name must include model, prompt condition, run kind, and dataset group: missing $required" >&2
    exit 2
  }
done

case "$MODEL" in
  phi-4-local)
    export LAYERMCP_PHI4_CHECKPOINT="$SCRATCH/layermcp/LayerMCP/checkpoints/phi-4"
    CHECKPOINT="$LAYERMCP_PHI4_CHECKPOINT"
    EXPECTED_MODEL_NAME="microsoft/phi-4"
    PROMPT_TEMPLATE_ID="tool_name_only_v1"
    ;;
  llama-3.1-8b-local)
    export LAYERMCP_LLAMA31_8B_CHECKPOINT="$SCRATCH/layermcp/LayerMCP/checkpoints/llama-3.1-8b-instruct"
    CHECKPOINT="$LAYERMCP_LLAMA31_8B_CHECKPOINT"
    EXPECTED_MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
    PROMPT_TEMPLATE_ID="structured_tool_call_v1"
    ;;
  *) echo "Unsupported MODEL: $MODEL" >&2; exit 2 ;;
esac
test -f "$CHECKPOINT/config.json"

declare -A DATASETS=(
  [coding_sweagent]="benchmark/coding/coding_sweagent_multistep.json"
  [coding_nebius_replay]="benchmark/coding/coding_nebius_sweagent_replay_multistep.json"
  [enterprise_tau2]="benchmark/enterprise/enterprise_public_workflows.json"
  [convfinqa]="benchmark/finance/finance_convfinqa_multistep.json"
  [finqa]="benchmark/finance/finance_finqa_test_multistep.json"
  [finretrieval_replay]="benchmark/finance/finance_finretrieval_replay_multistep.json"
  [mathematics]="benchmark/math/math_multistep_controlled.json"
)
ORDER=(coding_sweagent coding_nebius_replay enterprise_tau2 convfinqa finqa finretrieval_replay mathematics)
if [[ "$DATASET_GROUP" == all ]]; then
  [[ "$RUN_KIND" == preflight ]] || { echo "DATASET_GROUP=all is preflight-only" >&2; exit 2; }
  GROUPS=("${ORDER[@]}")
else
  [[ -n "${DATASETS[$DATASET_GROUP]:-}" ]] || { echo "Unsupported DATASET_GROUP" >&2; exit 2; }
  GROUPS=("$DATASET_GROUP")
fi
[[ "$RUN_KIND" == preflight || "$RUN_KIND" == full ]] || { echo "Unsupported RUN_KIND" >&2; exit 2; }

RUN_DATE="$(date -u +%Y%m%dT%H%M%SZ)"
RUNS_ROOT="results/runs/multi_step"
RUN_DIR="$RUNS_ROOT/${RUN_DATE}_${JOB_ID_COMPONENT}_${MODEL_COMPONENT}_${PROMPT_COMPONENT}_${RUN_KIND_COMPONENT}_${GROUP_COMPONENT}"
mkdir -p "$RUNS_ROOT"
mkdir "$RUN_DIR"
: "${RUN_DIR:?RUN_DIR must be set before preserving the launcher}"
LAUNCHER_PATH="$(realpath "${BASH_SOURCE[0]}")"
cp "$LAUNCHER_PATH" "$RUN_DIR/launcher.sbatch"
sha256sum "$LAUNCHER_PATH" > "$RUN_DIR/launcher.sha256"

REGISTRY_SNAPSHOT="$RUN_DIR/live_registry_snapshot.json"
python -m analysis.run_artifacts --capture-live-registry --include-catalog \
  --server mcp_server/server.py --registry-metadata-out "$REGISTRY_SNAPSHOT"
mapfile -t REGISTRY_VALUES < <(python - "$REGISTRY_SNAPSHOT" <<'PY'
import json, re, sys
from pathlib import Path
x=json.loads(Path(sys.argv[1]).read_text())
for field in ("tool_pool","tool_count","tool_registry_fingerprint","tool_registry_fingerprint_version"):
    if x.get(field) in (None, ""): raise SystemExit(f"Missing live registry field: {field}")
if not re.fullmatch(r"sha256:[0-9a-f]{64}", x["tool_registry_fingerprint"]): raise SystemExit("Malformed fingerprint")
for field in ("tool_names","tool_schemas","tool_descriptions"):
    if not x.get(field): raise SystemExit(f"Missing live registry catalog: {field}")
print(x["tool_pool"]); print(x["tool_count"]); print(x["tool_registry_fingerprint"]); print(x["tool_registry_fingerprint_version"])
PY
)
TOOL_POOL="${REGISTRY_VALUES[0]}"; TOOL_COUNT="${REGISTRY_VALUES[1]}"
EXPECTED_REGISTRY_FINGERPRINT="${REGISTRY_VALUES[2]}"; REGISTRY_FINGERPRINT_VERSION="${REGISTRY_VALUES[3]}"

SOURCE_LIST="$RUN_DIR/resolved_datasets.txt"
for group in "${GROUPS[@]}"; do realpath "${DATASETS[$group]}"; done > "$SOURCE_LIST"
SLURM_STDOUT_PATH="$SUBMIT_DIR/slurm-${JOB_NAME_COMPONENT}-${JOB_ID_COMPONENT}.out"
SLURM_STDERR_PATH="$SUBMIT_DIR/slurm-${JOB_NAME_COMPONENT}-${JOB_ID_COMPONENT}.err"

python - "$RUN_DIR/run_metadata.json" "$SOURCE_LIST" <<PY
import collections,json,sys
from pathlib import Path
paths=Path(sys.argv[2]).read_text().splitlines()
counts={}; benchmark_modes={}; execution_modes={}
for path in paths:
    rows=json.loads(Path(path).read_text())
    counts[path]={"workflows":len(rows),"routed_steps":sum(len(row["expected_steps"]) for row in rows)}
    benchmark_modes[path]=dict(sorted(collections.Counter(row.get("benchmark_mode") or "grounded_tool_execution" for row in rows).items()))
    execution_modes[path]=dict(sorted(collections.Counter(row.get("workflow_execution_mode") or "isolated_step" for row in rows).items()))
data={"git_commit":"$ACTUAL_COMMIT","model":"$MODEL","expected_model_name":"$EXPECTED_MODEL_NAME","checkpoint":"$CHECKPOINT","prompt_condition":"$PROMPT_MODE","prompt_template_id":"$PROMPT_TEMPLATE_ID","run_kind":"$RUN_KIND","dataset_group":"$DATASET_GROUP","benchmark_paths":paths,"source_counts":counts,"benchmark_mode_distributions":benchmark_modes,"workflow_execution_mode_distributions":execution_modes,"evaluation_protocol":"teacher_forced_step_routing_v1","model_attempts_per_step":1,"tool_call_retries":0,"generation":{"temperature":0.0,"max_tokens":128,"seed":"not_applicable_deterministic_greedy"},"tool_pool":"$TOOL_POOL","tool_count":int("$TOOL_COUNT"),"tool_registry_fingerprint":"$EXPECTED_REGISTRY_FINGERPRINT","tool_registry_fingerprint_version":"$REGISTRY_FINGERPRINT_VERSION","registry_metadata_source":"live_mcp_registry_at_job_startup","slurm_job_id":"$SLURM_JOB_ID","slurm_job_name":"$SLURM_JOB_NAME","slurm_stdout_path":"$SLURM_STDOUT_PATH","slurm_stderr_path":"$SLURM_STDERR_PATH","preflight_headline_eligible":False if "$RUN_KIND"=="preflight" else None}
with Path(sys.argv[1]).open('x') as f: json.dump(data,f,indent=2,sort_keys=True); f.write('\n')
PY

for group in "${GROUPS[@]}"; do
  source_benchmark="${DATASETS[$group]}"
  evaluated_benchmark="$source_benchmark"
  provenance_args=()
  if [[ "$RUN_KIND" == preflight ]]; then
    subset_dir="$RUN_DIR/preflight_subsets/$group"; mkdir -p "$subset_dir"
    evaluated_benchmark="$subset_dir/subset.json"
    provenance="$subset_dir/provenance.json"
    python -m analysis.multi_step_run --make-preflight \
      --source-benchmark "$source_benchmark" --subset "$evaluated_benchmark" \
      --provenance "$provenance" --model "$MODEL" --checkpoint "$CHECKPOINT" \
      --registry-snapshot "$REGISTRY_SNAPSHOT"
    provenance_args=(--preflight-provenance "$provenance")
  fi
  domain="$(safe_component domain "$(cut -d/ -f2 <<<"$source_benchmark")")"
  dataset="$(safe_component dataset "$(basename "$source_benchmark" .json)")"
  dataset_dir="$RUN_DIR/domains/$domain/$dataset"; mkdir -p "$(dirname "$dataset_dir")"; mkdir "$dataset_dir"
  python -m evaluation.evaluate --dataset "$evaluated_benchmark" --router "$MODEL" --call-predicted-tools --output-dir "$dataset_dir" 2>&1 | tee "$dataset_dir/evaluation.log"
  python -m analysis.multi_step_run --dataset-dir "$dataset_dir" --index "$RUN_DIR/artifact_index.jsonl" --source-benchmark "$source_benchmark" --evaluated-benchmark "$evaluated_benchmark" --model "$EXPECTED_MODEL_NAME" --prompt-template "$PROMPT_TEMPLATE_ID" --registry-fingerprint "$EXPECTED_REGISTRY_FINGERPRINT" --registry-fingerprint-version "$REGISTRY_FINGERPRINT_VERSION" --tool-count "$TOOL_COUNT" --tool-pool "$TOOL_POOL" "${provenance_args[@]}"
done

python -m analysis.multi_step_run --validate-index --index "$RUN_DIR/artifact_index.jsonl" --source-benchmark "$SOURCE_LIST"
test "$(wc -l < "$RUN_DIR/artifact_index.jsonl")" -eq "${#GROUPS[@]}"
test ! -e "$RUN_DIR/RUN_COMPLETE"
( set -o noclobber; printf 'git_commit=%s\n' "$ACTUAL_COMMIT" > "$RUN_DIR/RUN_COMPLETE" )
echo "Completed $MODEL $RUN_KIND $DATASET_GROUP under $RUN_DIR"
