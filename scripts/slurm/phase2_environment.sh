# Shared Phase 2 virtual-environment preflight. Relative paths use the checkout.
validate_phase2_environment() {
  LAYERMCP_VENV="${LAYERMCP_VENV:-$LAYERMCP_REPO_ROOT/.venv}"
  [[ "$LAYERMCP_VENV" == /* ]] || LAYERMCP_VENV="$LAYERMCP_REPO_ROOT/$LAYERMCP_VENV"
  if [[ ! -r "$LAYERMCP_VENV/bin/activate" || ! -x "$LAYERMCP_VENV/bin/python" ]]; then
    echo "Invalid LAYERMCP_VENV: $LAYERMCP_VENV (requires readable bin/activate and executable bin/python)" >&2
    return 2
  fi
  "$LAYERMCP_VENV/bin/python" --version >/dev/null || return 2
  export LAYERMCP_VENV
}

validate_phase2_harmony_runtime() {
  : "${TIKTOKEN_ENCODINGS_BASE:?Set TIKTOKEN_ENCODINGS_BASE}"
  local tokenizer_asset="$TIKTOKEN_ENCODINGS_BASE/o200k_base.tiktoken"
  [[ -s "$tokenizer_asset" ]] || {
    echo "Missing local Harmony tokenizer asset: $tokenizer_asset" >&2
    return 2
  }
  export TIKTOKEN_ENCODINGS_BASE
  python -c 'from openai_harmony import HarmonyEncodingName, load_harmony_encoding; print(load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS).name)'
}

validate_phase2_cuda_runtime() {
  # Phase 2 GPT-OSS runs are H100 experiments.  Do not silently construct the
  # model on CPU when CUDA is unavailable inside an allocated Slurm job.
  python - <<'PY'
import os
import sys

import torch

visible = os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")
if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
    raise SystemExit(
        "Phase 2 GPT-OSS requires a CUDA-visible GPU after environment setup; "
        f"CUDA_VISIBLE_DEVICES={visible!r}, "
        f"torch.cuda.is_available()={torch.cuda.is_available()}, "
        f"device_count={torch.cuda.device_count()}"
    )
print(
    "Phase 2 CUDA preflight: "
    f"CUDA_VISIBLE_DEVICES={visible!r}; "
    f"device_count={torch.cuda.device_count()}; "
    f"device_0={torch.cuda.get_device_name(0)}"
)
PY
}
