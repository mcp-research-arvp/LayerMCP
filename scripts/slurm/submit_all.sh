#!/bin/bash
# Submit LayerMCP's supported model conditions to Slurm.
# Defaults: primary single-step benchmarks and all primary full multi-step jobs.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
source "$SCRIPT_DIR/submit_common.sh"

run_single=1
run_multi=1
selected_model=all
single_run_kind=primary
multi_run_kind=full
selected_domains=""
generic_datasets=""
single_datasets=""
multi_datasets=""
dry_run=0

usage() {
  cat <<'EOF'
Usage: scripts/slurm/submit_all.sh [OPTIONS]

Submit complete LayerMCP evaluation jobs. With no options, this submits:
  - 7 primary single-step jobs (one per supported model/reasoning condition)
  - 49 full multi-step jobs (7 conditions × 7 primary dataset groups)

Options:
  --single-only              Submit only single-step jobs.
  --multi-only               Submit only multi-step jobs.
  --model MODEL              Limit submissions to one model.
  --single-run-kind KIND     smoke or primary (default: primary).
  --multi-run-kind KIND      short_test or full (default: full).
  --domains IDS              Comma-separated domains: math, enterprise, coding, finance.
  --datasets IDS             Dataset IDs when submitting one run type.
  --single-datasets IDS      Comma-separated single-step dataset IDs.
  --multi-datasets IDS       Comma-separated multi-step dataset-group IDs.
  --multi-dataset-group ID   Deprecated alias for --multi-datasets; all selects primary groups.
  --dry-run                  Print sbatch commands without submitting.
  -h, --help                 Show this help.

For --multi-run-kind full with all selected primary groups, this wrapper submits
one job per primary group because the canonical launcher rejects a monolithic
full run. For short_test with all, it submits one launcher job per condition.
Domain and dataset selectors are combined as a union. See the README for IDs.
EOF
  submission_usage_common
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --single-only)
      run_multi=0
      ;;
    --multi-only)
      run_single=0
      ;;
    --model|--single-run-kind|--multi-run-kind|--domains|--datasets|--single-datasets|--multi-datasets|--multi-dataset-group)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for $1" >&2
        exit 2
      fi
      if [[ -z "$2" ]]; then
        echo "Empty value is not allowed for $1" >&2
        exit 2
      fi
      case "$1" in
        --model) selected_model="$2" ;;
        --single-run-kind) single_run_kind="$2" ;;
        --multi-run-kind) multi_run_kind="$2" ;;
        --domains) selected_domains="$(append_csv_value "$selected_domains" "$2")" ;;
        --datasets) generic_datasets="$(append_csv_value "$generic_datasets" "$2")" ;;
        --single-datasets) single_datasets="$(append_csv_value "$single_datasets" "$2")" ;;
        --multi-datasets) multi_datasets="$(append_csv_value "$multi_datasets" "$2")" ;;
        --multi-dataset-group) multi_datasets="$(append_csv_value "$multi_datasets" "$2")" ;;
      esac
      shift
      ;;
    --dry-run)
      dry_run=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

if (( ! run_single && ! run_multi )); then
  echo "Select at least one of --single-only or --multi-only." >&2
  exit 2
fi
if (( ! run_single )) && [[ -n "$single_datasets" ]]; then
  echo "--single-datasets requires single-step submission." >&2
  exit 2
fi
if (( ! run_multi )) && [[ -n "$multi_datasets" ]]; then
  echo "--multi-datasets requires multi-step submission." >&2
  exit 2
fi
if [[ -n "$generic_datasets" ]]; then
  if (( run_single && run_multi )); then
    echo "--datasets is ambiguous when submitting both run types; use --single-datasets and --multi-datasets." >&2
    exit 2
  elif (( run_single )); then
    single_datasets="$(append_csv_value "$single_datasets" "$generic_datasets")"
  else
    multi_datasets="$(append_csv_value "$multi_datasets" "$generic_datasets")"
  fi
fi

if (( run_single )); then
  validate_single_run_kind "$single_run_kind"
  single_selection="$(resolve_single_dataset_selection "$selected_domains" "$single_datasets" "$single_run_kind")"
  submit_single_steps "$selected_model" "$single_run_kind" "$single_selection" "$dry_run"
fi
if (( run_multi )); then
  multi_selection="$(resolve_multi_dataset_selection "$selected_domains" "$multi_datasets")"
  submit_multi_steps "$selected_model" "$multi_run_kind" "$multi_selection" "$dry_run"
fi
