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
