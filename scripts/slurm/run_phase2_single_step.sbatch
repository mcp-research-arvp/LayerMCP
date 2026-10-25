#!/bin/bash
# Paired GPT-OSS control/intervention evaluation of Phase 1 benchmark files.
# Submit through scripts/slurm/submit_phase2.sh so the required paths and
# intervention settings are validated and exported.
#SBATCH --job-name=gpt-oss-phase2-single
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

: "${SLURM_JOB_ID:?Submit this script with sbatch}"
: "${LAYERMCP_REPO_ROOT:?Set LAYERMCP_REPO_ROOT}"
: "${LAYERMCP_GPT_OSS_CHECKPOINT:?Set LAYERMCP_GPT_OSS_CHECKPOINT}"
: "${TIKTOKEN_ENCODINGS_BASE:?Set TIKTOKEN_ENCODINGS_BASE}"

cd "$LAYERMCP_REPO_ROOT"
if [[ -n "${LAYERMCP_MODULES:-}" ]]; then
  # Intentional word splitting: module names are configured as a whitespace-separated list.
  module load $LAYERMCP_MODULES
fi
source "$LAYERMCP_REPO_ROOT/scripts/slurm/phase2_environment.sh"
validate_phase2_environment
source "$LAYERMCP_VENV/bin/activate"
validate_phase2_harmony_runtime

LAYER_INDEX="${LAYER_INDEX:-0}"
ATTENTION_TARGET="${ATTENTION_TARGET:-qkv}"
METHOD="${METHOD:-noise}"
STRENGTH="${STRENGTH:-0.01}"
STRENGTH_MODE="${STRENGTH_MODE:-absolute}"
SEED="${SEED:-1234}"
SEEDS="${SEEDS:-$SEED}"
PLAN_MANIFEST="${PLAN_MANIFEST:-}"
EXAMPLE_LIMIT="${EXAMPLE_LIMIT:-}"
SCREEN_PLAN_OUTPUT="${SCREEN_PLAN_OUTPUT:-}"
COMBINED_OUTPUT="${COMBINED_OUTPUT:-}"

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

RUN_SCOPE="full"
[[ -z "$EXAMPLE_LIMIT" ]] || RUN_SCOPE="limit${EXAMPLE_LIMIT}"
RUN_DIR="results/runs/phase2/${SLURM_JOB_ID}_gpt-oss_single_step_${RUN_SCOPE}_layer${LAYER_INDEX}_${ATTENTION_TARGET}_${METHOD}_${STRENGTH_MODE}_${STRENGTH}_seeds${SEEDS//,/-}"
echo "Evaluating ${#BENCHMARKS[@]} direct single-step benchmark files in paired conditions"
base_command=(python -m research.phase2.gpt_oss_intervention_eval \
  --benchmarks "${BENCHMARKS[@]}" \
  --checkpoint "$LAYERMCP_GPT_OSS_CHECKPOINT" \
  --layers "$LAYER_INDEX" \
  --target "$ATTENTION_TARGET" \
  --method "$METHOD" \
  --strength "$STRENGTH" \
  --strength-mode "$STRENGTH_MODE" \
  --seeds "$SEEDS")
if [[ -n "$EXAMPLE_LIMIT" ]]; then
  base_command+=(--example-limit "$EXAMPLE_LIMIT")
fi
if [[ -n "$SCREEN_PLAN_OUTPUT" || -n "$COMBINED_OUTPUT" ]]; then
  [[ -n "$SCREEN_PLAN_OUTPUT" && -n "$COMBINED_OUTPUT" && -z "$PLAN_MANIFEST" ]] || {
    echo "Screen validation requires SCREEN_PLAN_OUTPUT and COMBINED_OUTPUT, without PLAN_MANIFEST" >&2
    exit 2
  }
  "${base_command[@]}" --plan-only --plan-output "$SCREEN_PLAN_OUTPUT"
  PLAN_MANIFEST="$SCREEN_PLAN_OUTPUT"
fi
command=("${base_command[@]}" --output-dir "$RUN_DIR")
if [[ -n "$PLAN_MANIFEST" ]]; then
  command+=(--plan-manifest "$PLAN_MANIFEST")
fi
"${command[@]}"
if [[ -n "$COMBINED_OUTPUT" ]]; then
  python -m analysis.phase2_paired_results \
    --run-dir "$RUN_DIR" \
    --expected-plan "$PLAN_MANIFEST" \
    --output "$COMBINED_OUTPUT"
  test -f "$RUN_DIR/RUN_COMPLETE"
  test ! -e "$RUN_DIR/RUN_IN_PROGRESS"
  test ! -e "$RUN_DIR/RUN_FAILED.json"
  python - "$PLAN_MANIFEST" "$RUN_DIR" "$COMBINED_OUTPUT" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

plan_path, run_dir, report_path = map(Path, sys.argv[1:])
plan_bytes = plan_path.read_bytes()
plan = json.loads(plan_bytes)
config = json.loads((run_dir / "run_config.json").read_text())
manifest_bytes = (run_dir / "run_manifest.json").read_bytes()
manifest = json.loads(manifest_bytes)
status = json.loads((run_dir / "run_status.json").read_text())
records = [json.loads(line) for line in (run_dir / "paired_records.jsonl").read_text().splitlines()]
report = json.loads(report_path.read_text())

screen = plan["screen_plan"]
assert screen["selected_layers"] == config["screen_plan"]["selected_layers"]
assert screen["sample_ids"] == config["screen_plan"]["sample_ids"]
assert screen["seeds"] == config["screen_plan"]["seeds"]
assert screen["total_pair_count"] == len(records)
assert status["status"] == "completed" and status.get("completed_at")
assert manifest["run_id"] == config["run_id"]
assert manifest["job_details"]["actual"]["slurm_job_id"]
assert manifest["runtime"].get("git_commit")
assert manifest["tool_registry"].get("metadata")
plan_hash = "sha256:" + hashlib.sha256(plan_bytes).hexdigest()
manifest_hash = "sha256:" + hashlib.sha256(manifest_bytes).hexdigest()
assert config["parent_plan"]["sha256"] == plan_hash
for record in records:
    assert record["control_verification"]["perturbation"]["status"] == "disabled"
    assert record["control_verification"]["perturbation"]["changed_elements"] == 0
    assert "final_outcome_correct" in record["control"]
    perturbation = record["intervention_verification"]["perturbation"]
    assert perturbation["status"] == "changed" and perturbation["changed_elements"] > 0
    assert record["intervention_verification"]["restored_exactly"] is True
    assert record["registry_exact_match"] is True
    assert record["run_manifest"]["sha256"] == manifest_hash
assert report["expected_plan"]["sha256"] == plan_hash
assert report["paired_observation_count"] == screen["total_pair_count"]
print("Phase 2 screen validation integrity checks passed.")
PY
fi
