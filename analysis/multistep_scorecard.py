"""Render separated outcome metrics from completed multi-step runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from evaluation.evaluate import OUTCOME_METRIC_NAMES


_LEGACY_METRIC = "workflow_final_answer_accuracy"
_DATASET_LABELS = {
    "math_public_mathqa_multistep": "MathQA public-derived",
    "math_multistep_controlled": "Math controlled diagnostic",
}


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _contains_key(value: Any, key: str) -> bool:
    if isinstance(value, dict):
        return key in value or any(_contains_key(item, key) for item in value.values())
    if isinstance(value, list):
        return any(_contains_key(item, key) for item in value)
    return False


_REQUIRED_SUMMARY_FIELDS = (
    "tool_selection_accuracy",
    "argument_accuracy",
    "step_outcome_accuracy",
    "step_outcome_scored",
    "step_outcome_status_counts",
    "step_outcome_matchers",
    "all_tools_correct_accuracy",
    "all_arguments_correct_accuracy",
    "all_steps_correct_accuracy",
    "all_steps_correct_scored",
    "final_step_outcome_accuracy",
    "final_step_outcome_gold",
    "final_step_outcome_scored",
    "final_step_outcome_correct",
    "final_step_outcome_mismatch",
    "final_step_outcome_extraction_error",
    "final_step_outcome_unavailable",
    "final_step_outcome_status_counts",
    "final_step_outcome_contracts",
    "final_step_outcome_matchers",
)
_FINAL_PROGRAM_FIELDS = tuple(
    f"final_program_execution_{suffix}"
    for suffix in (
        "accuracy",
        "gold",
        "scored",
        "correct",
        "mismatch",
        "extraction_error",
        "unavailable",
        "status_counts",
        "contracts",
        "matchers",
    )
)


def _reject_historical_scalar(value: dict[str, Any], path: Path) -> None:
    if _contains_key(value, _LEGACY_METRIC):
        raise ValueError(f"Historical scalar workflow-final metric in {path}")


def _validate_metric_names(value: dict[str, Any], path: Path) -> None:
    if value.get("outcome_metric_names") != list(OUTCOME_METRIC_NAMES):
        raise ValueError(f"Unexpected outcome metric names in {path}")


def _validate_summary(value: dict[str, Any], path: Path) -> None:
    _reject_historical_scalar(value, path)
    _validate_metric_names(value, path)
    missing = [field for field in _REQUIRED_SUMMARY_FIELDS if field not in value]
    if missing:
        raise ValueError(
            f"Corrected multi-step summary fields are missing in {path}: "
            + ", ".join(missing)
        )
    present_program_fields = [field for field in _FINAL_PROGRAM_FIELDS if field in value]
    if present_program_fields and len(present_program_fields) != len(
        _FINAL_PROGRAM_FIELDS
    ):
        raise ValueError(f"Incomplete final program metrics in {path}")
    generation_limit = value.get("effective_generation_limit")
    if (
        isinstance(generation_limit, bool)
        or not isinstance(generation_limit, int)
        or generation_limit <= 0
        or value.get("effective_generation_limit_unit") != "tokens"
    ):
        raise ValueError(f"Invalid generation-limit metadata in {path}")
    model = value.get("model_name")
    effort = value.get("reasoning_effort")
    if model == "openai/gpt-oss-20b":
        if (
            value.get("reasoning_mode"),
            value.get("reasoning_method"),
            effort,
        ) != ("reasoning", "harmony", "low"):
            raise ValueError(f"Invalid GPT-OSS Harmony LOW condition in {path}")
        if value.get("effective_generation_limit") != 4096:
            raise ValueError(f"Invalid GPT-OSS generation limit in {path}")
    elif effort is not None:
        raise ValueError(f"reasoning_effort is GPT-OSS-only in {path}")


def _condition_label(value: dict[str, Any]) -> str:
    if (
        value.get("reasoning_mode"),
        value.get("reasoning_method"),
        value.get("reasoning_effort"),
    ) == ("reasoning", "harmony", "low"):
        return "Reasoning — Harmony LOW"
    return str(value.get("reasoning_mode"))


def load_rows(run_directories: Sequence[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run_value in run_directories:
        run = run_value.resolve(strict=True)
        if not (run / "RUN_COMPLETE").is_file():
            raise ValueError(f"Multi-step run is incomplete: {run}")
        metadata_path = run / "run_metadata.json"
        metadata = _load_object(metadata_path)
        _reject_historical_scalar(metadata, metadata_path)
        _validate_metric_names(metadata, metadata_path)
        if metadata.get("headline_eligible") is False:
            continue
        for summary_path in sorted(run.glob("domains/*/*/summary.json")):
            summary = _load_object(summary_path)
            _validate_summary(summary, summary_path)
            rows.append(
                {
                    "run": run.name,
                    "dataset": _DATASET_LABELS.get(
                        summary_path.parent.name, summary_path.parent.name
                    ),
                    "model": summary.get("model_name"),
                    "condition": _condition_label(summary),
                    "reasoning_mode": summary.get("reasoning_mode"),
                    "reasoning_method": summary.get("reasoning_method"),
                    "reasoning_effort": summary.get("reasoning_effort"),
                    "effective_generation_limit": summary.get(
                        "effective_generation_limit"
                    ),
                    "effective_generation_limit_unit": summary.get(
                        "effective_generation_limit_unit"
                    ),
                    "tool_selection_accuracy": summary.get(
                        "tool_selection_accuracy"
                    ),
                    "argument_accuracy": summary.get("argument_accuracy"),
                    "step_outcome_accuracy": summary.get("step_outcome_accuracy"),
                    "all_steps_correct_accuracy": summary.get(
                        "all_steps_correct_accuracy"
                    ),
                    "final_step_outcome_accuracy": summary.get(
                        "final_step_outcome_accuracy"
                    ),
                    "final_program_execution_accuracy": summary.get(
                        "final_program_execution_accuracy"
                    ),
                }
            )
    return rows


def _accuracy(value: Any) -> str:
    return "—" if value is None else f"{float(value):.2%}"


def build_scorecard(run_directories: Sequence[Path]) -> str:
    rows = load_rows(run_directories)
    include_program = any(
        row["final_program_execution_accuracy"] is not None for row in rows
    )
    metric_columns = [
        ("Tool Selection", "tool_selection_accuracy"),
        ("Argument Accuracy", "argument_accuracy"),
        ("Step Outcome Accuracy", "step_outcome_accuracy"),
        ("All Steps Correct", "all_steps_correct_accuracy"),
        ("Final Step Outcome", "final_step_outcome_accuracy"),
    ]
    if include_program:
        metric_columns.append(
            ("Final Program Execution", "final_program_execution_accuracy")
        )
    headers = [
        "Run",
        "Dataset",
        "Model",
        "Condition",
        "Max generated tokens",
    ] + [
        heading for heading, _ in metric_columns
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    for row in rows:
        values = [
            row["run"],
            row["dataset"],
            row["model"],
            row["condition"],
            (
                f'{row["effective_generation_limit"]} '
                f'{row["effective_generation_limit_unit"]}'
            ),
        ]
        values.extend(_accuracy(row[field]) for _, field in metric_columns)
        lines.append(
            "| "
            + " | ".join(str(value).replace("|", "\\|") for value in values)
            + " |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directories", nargs="+", type=Path)
    args = parser.parse_args()
    print(build_scorecard(args.run_directories), end="")


if __name__ == "__main__":
    main()
