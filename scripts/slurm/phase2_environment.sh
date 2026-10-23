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
