#!/bin/bash
#SBATCH --account=def-yousefne
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --job-name=layermcp_qwen_all
#SBATCH --output=slurm_logs/layermcp_qwen_all_%j.out
#SBATCH --error=slurm_logs/layermcp_qwen_all_%j.err

set -euo pipefail

cd "$HOME/my_projects/LayerMCP"
mkdir -p slurm_logs

source .venv/bin/activate
source env.narval.sh

export LAYERMCP_QWEN36_DEVICE_MAP="cuda:0,cuda:1"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

echo "Host: $(hostname)"
echo "Started: $(date)"
echo "Qwen checkpoint: ${LAYERMCP_QWEN36_CHECKPOINT}"
echo "Qwen device map: ${LAYERMCP_QWEN36_DEVICE_MAP}"
nvidia-smi

echo "Running math public derived benchmark"
python -m evaluation.evaluate \
  --benchmark benchmark/math/tool_routing_math_public_derived.json \
  --router qwen-3.6-local

echo "Running enterprise public adapted benchmark"
python -m evaluation.evaluate \
  --benchmark benchmark/enterprise/tool_routing_enterprise_public_adapted.json \
  --router qwen-3.6-local

echo "Running math controlled benchmark"
python -m evaluation.evaluate \
  --benchmark benchmark/math/tool_routing_math_controlled.json \
  --router qwen-3.6-local

echo "Running enterprise v2 controlled benchmark"
python -m evaluation.evaluate \
  --benchmark benchmark/enterprise/tool_routing_enterprise_v2_controlled.json \
  --router qwen-3.6-local

echo "Finished: $(date)"

