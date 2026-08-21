"""Render version-strict workflow-level metrics from completed multi-step runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from evaluation.evaluate import WORKFLOW_FINAL_SCORING_VERSION


_PREFIXES = (
    "workflow_final_answer",
    "workflow_final_program_execution",
    "workflow_final_tool_result",
)


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def load_rows(run_directories: Sequence[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run_value in run_directories:
        run = run_value.resolve(strict=True)
        if not (run / "RUN_COMPLETE").is_file():
            raise ValueError(f"Multi-step run is incomplete: {run}")
        metadata = _load_object(run / "run_metadata.json")
        version = metadata.get("workflow_final_scoring_version")
        if version != WORKFLOW_FINAL_SCORING_VERSION:
            raise ValueError(
                f"Incompatible workflow-final scoring in {run}: {version!r}"
            )
        for summary_path in sorted(run.glob("domains/*/*/summary.json")):
            summary = _load_object(summary_path)
            if summary.get("workflow_final_scoring_version") != version:
                raise ValueError(f"Summary scoring version mismatch: {summary_path}")
            for prefix in _PREFIXES:
                if summary.get(f"{prefix}_scoring_version") != version:
                    raise ValueError(f"Metric scoring version mismatch: {summary_path}")
            rows.append(
                {
                    "run": run.name,
                    "dataset": summary_path.parent.name,
                    "model": summary.get("model_name"),
                    "condition": summary.get("reasoning_mode"),
                    "source_run_identity": summary.get(
                        "source_run_identity", run.name
                    ),
                    "workflow_final_scoring_version": version,
                    **{
                        key: summary.get(key)
                        for prefix in _PREFIXES
                        for key in (
                            f"{prefix}_accuracy",
                            f"{prefix}_gold",
                            f"{prefix}_scored",
                            f"{prefix}_correct",
                            f"{prefix}_mismatch",
                            f"{prefix}_extraction_error",
                            f"{prefix}_unavailable",
                            f"{prefix}_status_counts",
                            f"{prefix}_contracts",
                            f"{prefix}_matchers",
                        )
                    },
                }
            )
    return rows


def build_scorecard(run_directories: Sequence[Path]) -> str:
    rows = load_rows(run_directories)
    lines = [
        f"Workflow-final scoring version: `{WORKFLOW_FINAL_SCORING_VERSION}`",
        "",
        "| run | dataset | model | condition | metric | accuracy | gold | scored | correct | mismatch | extraction error | unavailable | contract | matcher | statuses | source run |",
        "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|",
    ]
    for row in rows:
        for prefix in _PREFIXES:
            accuracy = row[f"{prefix}_accuracy"]
            rendered_accuracy = "unavailable" if accuracy is None else f"{accuracy:.6f}"
            lines.append(
                "| "
                + " | ".join(
                    str(value).replace("|", "\\|")
                    for value in (
                        row["run"], row["dataset"], row["model"], row["condition"],
                        prefix, rendered_accuracy, row[f"{prefix}_gold"],
                        row[f"{prefix}_scored"], row[f"{prefix}_correct"],
                        row[f"{prefix}_mismatch"],
                        row[f"{prefix}_extraction_error"],
                        row[f"{prefix}_unavailable"],
                        json.dumps(row[f"{prefix}_contracts"], sort_keys=True),
                        json.dumps(row[f"{prefix}_matchers"], sort_keys=True),
                        json.dumps(row[f"{prefix}_status_counts"], sort_keys=True),
                        row["source_run_identity"],
                    )
                )
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
