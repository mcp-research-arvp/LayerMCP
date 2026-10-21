#!/bin/bash
# Paired GPT-OSS control/intervention evaluation of every saved single-step row.
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
: "${LAYERMCP_GPT_OSS_SOURCE_RUN:?Set LAYERMCP_GPT_OSS_SOURCE_RUN}"
: "${TIKTOKEN_ENCODINGS_BASE:?Set TIKTOKEN_ENCODINGS_BASE}"

cd "$LAYERMCP_REPO_ROOT"
module load StdEnv/2023 python/3.11
source "$LAYERMCP_REPO_ROOT/.venv/bin/activate"

LAYER_INDEX="${LAYER_INDEX:-0}"
ATTENTION_TARGET="${ATTENTION_TARGET:-qkv}"
METHOD="${METHOD:-noise}"
STRENGTH="${STRENGTH:-0.01}"
SEED="${SEED:-1234}"

# The source run's artifact index is authoritative. Count its saved rows so
# this launcher continues to cover the entire run if the baseline is rebuilt.
EXAMPLE_COUNT="$(python - "$LAYERMCP_GPT_OSS_SOURCE_RUN" <<'PY'
import json
from pathlib import Path
import sys

source = Path(sys.argv[1]).expanduser().resolve(strict=True)
if not (source / "RUN_COMPLETE").is_file():
    raise SystemExit(f"Saved single-step source run is incomplete: {source}")
index = source / "artifact_index.jsonl"
count = 0
for line in index.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    item = json.loads(line)
    raw_path = item.get("samples_path") or item.get("samples")
    if not isinstance(raw_path, str) or not raw_path:
        raise SystemExit(f"Artifact index entry has no samples path: {item}")
    path = Path(raw_path)
    if not path.is_absolute():
        path = source / path
    with path.open(encoding="utf-8") as rows:
        count += sum(bool(row.strip()) for row in rows)
if count <= 0:
    raise SystemExit(f"Saved single-step source run has no rows: {source}")
print(count)
PY
)"

RUN_DIR="results/runs/phase2/${SLURM_JOB_ID}_gpt-oss_single_step_full_layer${LAYER_INDEX}_${ATTENTION_TARGET}_${METHOD}_${STRENGTH}_seed${SEED}"
echo "Evaluating ${EXAMPLE_COUNT} saved single-step examples in paired conditions"
python -m research.phase2.gpt_oss_intervention_eval \
  --source-run-dir "$LAYERMCP_GPT_OSS_SOURCE_RUN" \
  --checkpoint "$LAYERMCP_GPT_OSS_CHECKPOINT" \
  --output-dir "$RUN_DIR" \
  --layers "$LAYER_INDEX" \
  --target "$ATTENTION_TARGET" \
  --method "$METHOD" \
  --strength "$STRENGTH" \
  --seeds "$SEED" \
  --example-limit "$EXAMPLE_COUNT"
