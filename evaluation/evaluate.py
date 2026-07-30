from __future__ import annotations

import argparse
import asyncio
from collections import Counter, defaultdict
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

BENCHMARK_PATH = (
    PROJECT_ROOT / "benchmark" / "archive" / "root" / "tool_routing.json"
)
SERVER_PATH = PROJECT_ROOT / "mcp_server" / "server.py"
RESULTS_DIR = PROJECT_ROOT / "results"

RETAIL_TOOL_NAMES = {
    "find_user_id_by_email",
    "find_user_id_by_name_zip",
    "get_user_details",
    "get_order_details",
    "get_product_details",
    "cancel_pending_order",
    "modify_pending_order_items",
    "modify_pending_order_address",
    "modify_user_address",
    "return_delivered_order_items",
    "exchange_delivered_order_items",
    "transfer_to_human_agents",
}

MULTISTEP_OVERALL_TASK_CHAR_LIMIT = 8_000
MULTISTEP_CURRENT_STEP_CHAR_LIMIT = 16_000
MULTISTEP_HISTORY_STEP_LIMIT = 2
MULTISTEP_HISTORY_ITEM_CHAR_LIMIT = 3_500
PROMPT_CONTEXT_CHAR_LIMIT = MULTISTEP_CURRENT_STEP_CHAR_LIMIT

SINGLE_STEP_EVALUATION_PROTOCOL = "single_step_tool_routing_v1"
MULTISTEP_EVALUATION_PROTOCOL = "guided_predicted_rollout_v1"
EVALUATION_PROTOCOL_DESCRIPTIONS = {
    SINGLE_STEP_EVALUATION_PROTOCOL: (
        "Each sample is evaluated as one independently routed tool call."
    ),
    MULTISTEP_EVALUATION_PROTOCOL: (
        "The benchmark supplies each step instruction, while every subsequent "
        "step receives only the model's predicted calls and executed results from "
        "earlier steps. Gold prior-step answers and gold state replay are not used. "
        "This evaluates error-propagating guided rollout, not autonomous planning."
    ),
}

DEFAULT_BENCHMARK_MODE = "grounded_tool_execution"
TOOL_REGISTRY_FINGERPRINT_VERSION = (
    "tool_registry_name_schema_description_v1"
)
ALLOWED_BENCHMARK_MODES = frozenset(
    {
        DEFAULT_BENCHMARK_MODE,
        "offline_trace_replay",
    }
)
DEFAULT_WORKFLOW_EXECUTION_MODE = "isolated_step"
PREDICTED_ROLLOUT_EXECUTION_MODE = "predicted_sequence"
REASONING_MODES = ("direct", "reasoning")
REFERENCE_PREFIX_REPLAY_MODE = "isolated_reference_prefix_replay"
ALLOWED_WORKFLOW_EXECUTION_MODES = frozenset(
    {DEFAULT_WORKFLOW_EXECUTION_MODE, REFERENCE_PREFIX_REPLAY_MODE}
)


@dataclass(frozen=True)
class BenchmarkStep:
    id: str
    query: str
    expected_tool: str
    expected_args: dict[str, Any]
    expected_answer: Any
    depends_on: tuple[str, ...]
    source_program: str | None
    prompt_context: str = ""


@dataclass(frozen=True)
class BenchmarkSample:
    id: str
    domain: str
    task_type: str
    difficulty: str
    source: str
    query: str
    expected_tool: str
    expected_args: dict[str, Any]
    expected_answer: Any
    perturbation_type: str
    notes: str
    benchmark_family: str = "unspecified"
    expected_final_answer: Any = None
    prompt_context: str = ""
    expected_steps: tuple[BenchmarkStep, ...] = ()
    benchmark_mode: str = DEFAULT_BENCHMARK_MODE
    workflow_execution_mode: str = DEFAULT_WORKFLOW_EXECUTION_MODE


@dataclass(frozen=True)
class EvaluationArtifactPaths:
    directory: Path
    samples: Path
    summary: Path


def _evaluation_artifact_paths(
    output_dir: Path | None,
    timestamp: str,
) -> EvaluationArtifactPaths:
    if output_dir is None:
        directory = RESULTS_DIR
        samples_name = f"{timestamp}_samples.jsonl"
        summary_name = f"{timestamp}_summary.json"
    else:
        directory = Path(output_dir)
        samples_name = "samples.jsonl"
        summary_name = "summary.json"

    directory.mkdir(parents=True, exist_ok=True)
    paths = EvaluationArtifactPaths(
        directory=directory,
        samples=directory / samples_name,
        summary=directory / summary_name,
    )
    for path in (paths.samples, paths.summary):
        if path.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing evaluation artifact: {path}"
            )
    return paths


def _open_samples_exclusive(path: Path):
    try:
        return path.open("x", encoding="utf-8")
    except FileExistsError as exc:
        raise FileExistsError(
            f"Refusing to overwrite existing evaluation artifact: {path}"
        ) from exc


def _write_summary_exclusive(path: Path, summary: dict[str, Any]) -> None:
    temporary_path = path.with_name(f".{path.name}.incomplete-{os.getpid()}")
    try:
        with temporary_path.open("x", encoding="utf-8") as summary_handle:
            json.dump(summary, summary_handle, ensure_ascii=True, indent=2)
            summary_handle.write("\n")
            summary_handle.flush()
            os.fsync(summary_handle.fileno())
        os.link(temporary_path, path)
    except FileExistsError as exc:
        raise FileExistsError(
            f"Refusing to overwrite existing evaluation artifact: {path}"
        ) from exc
    finally:
        temporary_path.unlink(missing_ok=True)


def _normalize_prompt_context(value: Any, location: str) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError(f"{location} prompt_context must be a string.")
    if len(value) > PROMPT_CONTEXT_CHAR_LIMIT:
        raise ValueError(
            f"{location} prompt_context exceeds {PROMPT_CONTEXT_CHAR_LIMIT} "
            "characters."
        )
    return value


def _normalize_benchmark_mode(value: Any, location: str) -> str:
    if value is None:
        return DEFAULT_BENCHMARK_MODE
    if not isinstance(value, str):
        raise ValueError(f"{location} benchmark_mode must be a string.")
    if value not in ALLOWED_BENCHMARK_MODES:
        allowed = ", ".join(sorted(ALLOWED_BENCHMARK_MODES))
        raise ValueError(
            f"{location} benchmark_mode must be one of: {allowed}."
        )
    return value


def _normalize_workflow_execution_mode(value: Any, location: str) -> str:
    if value is None:
        return DEFAULT_WORKFLOW_EXECUTION_MODE
    if not isinstance(value, str):
        raise ValueError(f"{location} workflow_execution_mode must be a string.")
    if value not in ALLOWED_WORKFLOW_EXECUTION_MODES:
        allowed = ", ".join(sorted(ALLOWED_WORKFLOW_EXECUTION_MODES))
        raise ValueError(
            f"{location} workflow_execution_mode must be one of: {allowed}."
        )
    return value


def _normalize_step(
    step: dict[str, Any],
    *,
    sample_index: int,
    step_index: int,
) -> BenchmarkStep:
    if not isinstance(step, dict):
        raise ValueError(
            f"Sample {sample_index} expected_steps[{step_index}] must be an object."
        )
    expected_args = step.get("expected_args", {})
    if not isinstance(expected_args, dict):
        raise ValueError(
            f"Sample {sample_index} expected_steps[{step_index}].expected_args "
            "must be an object."
        )
    depends_on = step.get("depends_on", [])
    if not isinstance(depends_on, list) or not all(
        isinstance(item, str) for item in depends_on
    ):
        raise ValueError(
            f"Sample {sample_index} expected_steps[{step_index}].depends_on "
            "must be a list of strings."
        )
    return BenchmarkStep(
        id=str(step.get("id") or f"step_{step_index + 1:03d}"),
        query=str(step["query"]),
        expected_tool=str(step["expected_tool"]),
        expected_args=expected_args,
        expected_answer=step.get("expected_answer"),
        depends_on=tuple(depends_on),
        source_program=(
            str(step["source_program"]) if step.get("source_program") is not None else None
        ),
        prompt_context=_normalize_prompt_context(
            step.get("prompt_context"),
            f"Sample {sample_index} expected_steps[{step_index}]",
        ),
    )


@dataclass(frozen=True)
class SampleScore:
    tool_selection_correct: bool
    argument_match_correct: bool
    execution_success: bool
    failure_category: str


@dataclass(frozen=True)
class OutcomeMatch:
    matched: bool
    diagnostic: str | None


@dataclass(frozen=True)
class ToolResultExtraction:
    value: Any
    diagnostic: str | None


@dataclass(frozen=True)
class FinalOutcomeScore:
    correct: bool | None
    status: str
    matcher: str | None
    diagnostic: str | None


FINAL_OUTCOME_MATCHER = "recursive_json_subset_v1"
WORKFLOW_FINAL_ANSWER_MATCHER = "formatted_final_scalar_v1"
FINANCE_QUERY_TABLE_RESULT_MATCHER = "finance_query_table_rows_v1"
NUMERIC_REL_TOL = 1e-9
NUMERIC_ABS_TOL = 1e-6
_SYMBOLIC_MATH_FIELDS = {
    "simplified",
    "factored",
    "expanded",
    "derivative",
    "solutions",
}


def _normalize_sample(sample: dict[str, Any], index: int) -> BenchmarkSample:
    raw_steps = sample.get("expected_steps", [])
    if not isinstance(raw_steps, list):
        raise ValueError(f"Sample {index} expected_steps must be a list.")
    expected_steps = tuple(
        _normalize_step(step, sample_index=index, step_index=step_index)
        for step_index, step in enumerate(raw_steps)
    )
    completed_step_ids: set[str] = set()
    for step in expected_steps:
        if step.id in completed_step_ids:
            raise ValueError(f"Sample {index} repeats expected step ID {step.id!r}.")
        unknown_dependencies = set(step.depends_on) - completed_step_ids
        if unknown_dependencies:
            raise ValueError(
                f"Sample {index} step {step.id!r} depends on unknown or future "
                f"steps: {sorted(unknown_dependencies)!r}."
            )
        completed_step_ids.add(step.id)
    if sample.get("task_type") == "multi_step_tool_routing" and len(expected_steps) < 2:
        raise ValueError(
            f"Sample {index} multi_step_tool_routing requires at least two steps."
        )

    expected_args = sample.get("expected_args")
    if expected_args is None:
        expected_args = sample.get("tool_args", {})
    if expected_args is None:
        expected_args = {}
    if not isinstance(expected_args, dict):
        raise ValueError(f"Sample {index} expected_args/tool_args must be an object.")

    sample_id = sample.get("id") or f"sample_{index + 1:04d}"

    expected_tool = sample.get("expected_tool")
    if expected_tool is None and expected_steps:
        expected_tool = expected_steps[0].expected_tool
    if expected_tool is None:
        raise ValueError(f"Sample {index} expected_tool is required.")

    return BenchmarkSample(
        id=str(sample_id),
        domain=str(sample.get("domain", "unknown")),
        task_type=str(sample.get("task_type", "tool_routing")),
        difficulty=str(sample.get("difficulty", "unspecified")),
        source=str(sample.get("source", "unspecified")),
        query=str(sample["query"]),
        expected_tool=str(expected_tool),
        expected_args=expected_args,
        expected_answer=sample.get("expected_answer"),
        perturbation_type=str(sample.get("perturbation_type", "none")),
        notes=str(sample.get("notes", "")),
        benchmark_family=str(
            sample.get("source_dataset") or sample.get("source") or "unspecified"
        ).strip().casefold(),
        benchmark_mode=_normalize_benchmark_mode(
            sample.get("benchmark_mode"),
            f"Sample {index}",
        ),
        workflow_execution_mode=_normalize_workflow_execution_mode(
            sample.get("workflow_execution_mode"),
            f"Sample {index}",
        ),
        expected_final_answer=sample.get("expected_final_answer"),
        prompt_context=_normalize_prompt_context(
            sample.get("prompt_context"),
            f"Sample {index}",
        ),
        expected_steps=expected_steps,
    )


def load_benchmark(path: Path) -> list[BenchmarkSample]:
    if not path.exists():
        raise FileNotFoundError(
            f"Benchmark dataset not found: {path}. "
            "Pass an existing benchmark JSON path with --dataset."
        )

    with path.open("r", encoding="utf-8") as handle:
        dataset = json.load(handle)

    if not isinstance(dataset, list):
        raise ValueError("Benchmark dataset must be a JSON list.")

    return [_normalize_sample(sample, index) for index, sample in enumerate(dataset)]


def _normalize_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _exact_argument_match(selected_args: dict[str, Any], expected_args: dict[str, Any]) -> bool:
    return _normalize_json(selected_args) == _normalize_json(expected_args)


def _symbolically_equivalent(actual: Any, expected: Any) -> bool:
    if not isinstance(actual, (str, int, float)) or isinstance(actual, bool):
        return False
    if not isinstance(expected, (str, int, float)) or isinstance(expected, bool):
        return False

    try:
        import sympy

        actual_expression = sympy.sympify(str(actual).replace("^", "**"))
        expected_expression = sympy.sympify(str(expected).replace("^", "**"))
        return bool(sympy.simplify(actual_expression - expected_expression) == 0)
    except Exception:
        return False


def _match_expected_answer(
    actual: Any,
    expected: Any,
    *,
    domain: str,
    path: str = "$",
    symbolic_math_value: bool = False,
) -> OutcomeMatch:
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return OutcomeMatch(
                False,
                f"{path}: expected an object, got {type(actual).__name__}",
            )
        for key, expected_value in expected.items():
            child_path = f"{path}.{key}"
            if key not in actual:
                return OutcomeMatch(False, f"{child_path}: expected key is missing")
            match = _match_expected_answer(
                actual[key],
                expected_value,
                domain=domain,
                path=child_path,
                symbolic_math_value=(
                    domain == "mathematics" and key in _SYMBOLIC_MATH_FIELDS
                ),
            )
            if not match.matched:
                return match
        return OutcomeMatch(True, None)

    if isinstance(expected, list):
        if not isinstance(actual, list):
            return OutcomeMatch(
                False,
                f"{path}: expected an array, got {type(actual).__name__}",
            )
        if len(actual) != len(expected):
            return OutcomeMatch(
                False,
                f"{path}: expected {len(expected)} items, got {len(actual)}",
            )
        for index, (actual_item, expected_item) in enumerate(zip(actual, expected)):
            match = _match_expected_answer(
                actual_item,
                expected_item,
                domain=domain,
                path=f"{path}[{index}]",
                symbolic_math_value=symbolic_math_value,
            )
            if not match.matched:
                return match
        return OutcomeMatch(True, None)

    if isinstance(expected, bool):
        if isinstance(actual, bool) and actual == expected:
            return OutcomeMatch(True, None)
        return OutcomeMatch(False, f"{path}: expected {expected!r}, got {actual!r}")

    if symbolic_math_value and _symbolically_equivalent(actual, expected):
        return OutcomeMatch(True, None)

    if isinstance(expected, int) and not isinstance(expected, bool):
        if isinstance(actual, int) and not isinstance(actual, bool):
            if actual == expected:
                return OutcomeMatch(True, None)
            return OutcomeMatch(False, f"{path}: expected {expected!r}, got {actual!r}")
        if isinstance(actual, float) and math.isclose(
            actual,
            expected,
            rel_tol=NUMERIC_REL_TOL,
            abs_tol=NUMERIC_ABS_TOL,
        ):
            return OutcomeMatch(True, None)
        return OutcomeMatch(False, f"{path}: expected {expected!r}, got {actual!r}")

    if isinstance(expected, float):
        if (
            isinstance(actual, (int, float))
            and not isinstance(actual, bool)
            and math.isclose(
                actual,
                expected,
                rel_tol=NUMERIC_REL_TOL,
                abs_tol=NUMERIC_ABS_TOL,
            )
        ):
            return OutcomeMatch(True, None)
        return OutcomeMatch(False, f"{path}: expected {expected!r}, got {actual!r}")

    if type(actual) is type(expected) and actual == expected:
        return OutcomeMatch(True, None)
    return OutcomeMatch(False, f"{path}: expected {expected!r}, got {actual!r}")


def _match_finance_query_table_result(actual: Any, expected: Any) -> OutcomeMatch:
    """Match a table-query result by its bounded data payload, not SQL aliases."""
    if not isinstance(expected, dict):
        return _match_expected_answer(actual, expected, domain="finance")

    required_keys = ("dataset_id", "columns", "rows", "row_count", "truncated")
    if not all(key in expected for key in required_keys):
        return _match_expected_answer(actual, expected, domain="finance")

    if not isinstance(actual, dict):
        return OutcomeMatch(False, f"$: expected object, got {type(actual).__name__}")

    expected_columns = expected["columns"]
    actual_columns = actual.get("columns")
    if not isinstance(expected_columns, list):
        return _match_expected_answer(actual, expected, domain="finance")
    if not isinstance(actual_columns, list):
        return OutcomeMatch(
            False,
            "$.columns: expected a list with "
            f"{len(expected_columns)} columns, got {actual_columns!r}",
        )
    if len(actual_columns) != len(expected_columns):
        return OutcomeMatch(
            False,
            "$.columns: expected "
            f"{len(expected_columns)} columns, got {len(actual_columns)}",
        )

    expected_payload = {
        key: expected[key] for key in required_keys if key != "columns"
    }
    return _match_expected_answer(actual, expected_payload, domain="finance")


def _score_final_outcome(
    *,
    expected_answer: Any,
    tool_result_value: Any,
    result_extraction_diagnostic: str | None,
    domain: str,
    call_predicted_tools: bool,
    no_tool_call: bool,
    execution_success: bool,
    expected_tool: str | None = None,
    called_tool: str | None = None,
) -> FinalOutcomeScore:
    if expected_answer is None:
        return FinalOutcomeScore(
            None,
            "missing_expected_answer",
            None,
            "Benchmark row does not provide expected_answer.",
        )
    if not call_predicted_tools:
        return FinalOutcomeScore(
            None,
            "execution_disabled",
            None,
            "Predicted-tool execution is disabled.",
        )
    if no_tool_call:
        return FinalOutcomeScore(
            False,
            "no_tool_call",
            FINAL_OUTCOME_MATCHER,
            "No registered predicted tool was available to execute.",
        )
    if not execution_success:
        return FinalOutcomeScore(
            False,
            "execution_error",
            FINAL_OUTCOME_MATCHER,
            "The predicted tool did not execute successfully.",
        )
    if result_extraction_diagnostic is not None:
        return FinalOutcomeScore(
            False,
            "result_extraction_error",
            FINAL_OUTCOME_MATCHER,
            result_extraction_diagnostic,
        )

    if expected_tool == called_tool == "finance_query_table":
        match = _match_finance_query_table_result(
            tool_result_value,
            expected_answer,
        )
        matcher = FINANCE_QUERY_TABLE_RESULT_MATCHER
    else:
        match = _match_expected_answer(
            tool_result_value,
            expected_answer,
            domain=domain,
        )
        matcher = FINAL_OUTCOME_MATCHER
    return FinalOutcomeScore(
        match.matched,
        "correct" if match.matched else "mismatch",
        matcher,
        match.diagnostic,
    )


def _final_outcome_record_fields(score: FinalOutcomeScore) -> dict[str, Any]:
    return {
        "final_outcome_correct": score.correct,
        "final_outcome_status": score.status,
        "final_outcome_matcher": score.matcher,
        "final_outcome_diagnostic": score.diagnostic,
    }


def _extract_workflow_final_scalar(tool_result_value: Any) -> tuple[Any, str | None]:
    if isinstance(tool_result_value, dict):
        if "result" in tool_result_value:
            return tool_result_value["result"], None
        rows = tool_result_value.get("rows")
        if (
            isinstance(rows, list)
            and len(rows) == 1
            and isinstance(rows[0], list)
            and len(rows[0]) == 1
        ):
            return rows[0][0], None
        return None, "Final tool result does not expose one scalar result."
    if isinstance(tool_result_value, (str, int, float, bool)):
        return tool_result_value, None
    return None, "Final tool result is not a supported scalar value."


def _display_decimal_places(value: str) -> int:
    normalized = value.strip().rstrip("%").replace(",", "")
    if "." not in normalized:
        return 0
    return len(normalized.rsplit(".", 1)[1])


def _parse_display_number(value: str) -> float | None:
    normalized = value.strip().rstrip("%").replace(",", "").replace("$", "")
    try:
        return float(normalized)
    except ValueError:
        return None


def _score_workflow_final_answer(
    *,
    expected_final_answer: Any,
    final_tool_result_value: Any,
    call_predicted_tools: bool,
    benchmark_family: str,
) -> FinalOutcomeScore:
    if expected_final_answer is None or expected_final_answer == "":
        return FinalOutcomeScore(
            None,
            "missing_expected_final_answer",
            None,
            "Benchmark workflow does not provide a non-empty expected_final_answer.",
        )
    if not call_predicted_tools:
        return FinalOutcomeScore(
            None,
            "execution_disabled",
            None,
            "Predicted-tool execution is disabled.",
        )

    actual, extraction_error = _extract_workflow_final_scalar(
        final_tool_result_value
    )
    if extraction_error is not None:
        return FinalOutcomeScore(
            False,
            "result_extraction_error",
            WORKFLOW_FINAL_ANSWER_MATCHER,
            extraction_error,
        )

    if isinstance(expected_final_answer, str):
        expected_number = _parse_display_number(expected_final_answer)
        if expected_number is not None and isinstance(actual, (int, float)):
            actual_number = float(actual)
            if expected_final_answer.strip().endswith("%") and benchmark_family in {
                "finqa",
                "convfinqa",
            }:
                actual_number *= 100.0
            decimal_places = _display_decimal_places(expected_final_answer)
            matched = math.isclose(
                round(actual_number, decimal_places),
                expected_number,
                rel_tol=0.0,
                abs_tol=10 ** (-(decimal_places + 6)),
            )
            return FinalOutcomeScore(
                matched,
                "correct" if matched else "mismatch",
                WORKFLOW_FINAL_ANSWER_MATCHER,
                None
                if matched
                else (
                    f"Expected displayed final answer {expected_final_answer!r}, "
                    f"got scalar {actual!r}."
                ),
            )
        matched = str(actual).strip().casefold() == expected_final_answer.strip().casefold()
    else:
        matched = _match_expected_answer(
            actual,
            expected_final_answer,
            domain="finance",
        ).matched

    return FinalOutcomeScore(
        matched,
        "correct" if matched else "mismatch",
        WORKFLOW_FINAL_ANSWER_MATCHER,
        None
        if matched
        else f"Expected final answer {expected_final_answer!r}, got {actual!r}.",
    )


def _score_sample(
    *,
    expected_tool: str,
    selected_tool: str | None,
    expected_args: dict[str, Any],
    selected_args: dict[str, Any],
    execution_success: bool,
    execution_attempted: bool,
) -> SampleScore:
    no_tool_call = selected_tool is None or selected_tool == "hallucinated_tool"
    if no_tool_call:
        return SampleScore(
            tool_selection_correct=False,
            argument_match_correct=False,
            execution_success=False,
            failure_category="no_tool_call",
        )

    tool_selection_correct = selected_tool == expected_tool
    argument_match_correct = tool_selection_correct and _exact_argument_match(
        selected_args,
        expected_args,
    )

    if not tool_selection_correct:
        failure_category = "wrong_tool"
    elif not argument_match_correct:
        failure_category = "wrong_args"
    elif execution_attempted and not execution_success:
        failure_category = "execution_error"
    else:
        failure_category = "correct"

    return SampleScore(
        tool_selection_correct=tool_selection_correct,
        argument_match_correct=argument_match_correct,
        execution_success=execution_success,
        failure_category=failure_category,
    )


def _build_aggregate_metrics(
    records: list[dict[str, Any]],
    *,
    include_mode_breakdowns: bool = True,
) -> dict[str, Any]:
    total = len(records)
    expected_tools = sorted({record["expected_tool"] for record in records})

    per_tool_totals: Counter[str] = Counter(record["expected_tool"] for record in records)
    per_tool_correct: Counter[str] = Counter(
        record["expected_tool"]
        for record in records
        if record["tool_selection_correct"]
    )
    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for record in records:
        selected = record["selected_tool"] or "no_tool_call"
        confusion[record["expected_tool"]][selected] += 1

    final_outcome_gold_samples = sum(
        1 for record in records if record.get("expected_answer") is not None
    )
    final_outcome_scored_samples = sum(
        1 for record in records if record.get("final_outcome_correct") is not None
    )
    final_outcome_correct_samples = sum(
        1 for record in records if record.get("final_outcome_correct") is True
    )

    metrics = {
        "total_samples": total,
        "benchmark_mode_counts": _benchmark_mode_counts(records),
        "tool_selection_accuracy": (
            sum(1 for record in records if record["tool_selection_correct"]) / total
            if total
            else 0.0
        ),
        "exact_argument_match_accuracy": (
            sum(1 for record in records if record["argument_match_correct"]) / total
            if total
            else 0.0
        ),
        "execution_success_rate": (
            sum(1 for record in records if record["execution_success"]) / total
            if total
            else 0.0
        ),
        "no_tool_call_rate": (
            sum(1 for record in records if record["failure_category"] == "no_tool_call") / total
            if total
            else 0.0
        ),
        "final_outcome_accuracy": (
            final_outcome_correct_samples / final_outcome_scored_samples
            if final_outcome_scored_samples
            else None
        ),
        "final_outcome_correct_samples": final_outcome_correct_samples,
        "final_outcome_scored_samples": final_outcome_scored_samples,
        "final_outcome_gold_samples": final_outcome_gold_samples,
        "final_outcome_gold_coverage": (
            final_outcome_gold_samples / total if total else 0.0
        ),
        "per_tool_accuracy": {
            tool: (
                per_tool_correct[tool] / per_tool_totals[tool]
                if per_tool_totals[tool]
                else 0.0
            )
            for tool in expected_tools
        },
        "confusion_matrix": {
            expected: dict(sorted(selected_counts.items()))
            for expected, selected_counts in sorted(confusion.items())
        },
    }
    if include_mode_breakdowns:
        records_by_mode: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            records_by_mode[
                record.get("benchmark_mode", DEFAULT_BENCHMARK_MODE)
            ].append(record)
        metrics["metrics_by_benchmark_mode"] = {
            mode: _build_aggregate_metrics(
                mode_records,
                include_mode_breakdowns=False,
            )
            for mode, mode_records in sorted(records_by_mode.items())
        }
    return metrics


def _summarize_tool_result(result: Any) -> str:
    structured = getattr(result, "structuredContent", None)
    if structured is not None:
        return json.dumps(structured, ensure_ascii=True)

    content = getattr(result, "content", None) or []
    if not content:
        return "<no content>"

    first_item = content[0]
    text = getattr(first_item, "text", None)
    if text:
        return text

    return repr(first_item)


def _extract_structured_tool_result(result: Any) -> ToolResultExtraction:
    structured = getattr(result, "structuredContent", None)
    if structured is not None:
        return ToolResultExtraction(structured, None)

    content = getattr(result, "content", None) or []
    for item in content:
        text = getattr(item, "text", None)
        if not isinstance(text, str):
            continue
        try:
            return ToolResultExtraction(json.loads(text), None)
        except json.JSONDecodeError:
            return ToolResultExtraction(text, None)

    return ToolResultExtraction(
        None,
        "MCP tool result contained neither structuredContent nor text content.",
    )


@asynccontextmanager
async def _run_server_session(server_path: Path):
    from mcp import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    server = StdioServerParameters(
        command=sys.executable,
        args=[str(server_path)],
        cwd=str(PROJECT_ROOT),
    )

    async with stdio_client(server) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            yield session


async def _call_tool_with_sample_isolation(
    session: ClientSession,
    server_path: Path,
    tool_name: str,
    tool_args: dict[str, Any],
) -> Any:
    if tool_name not in RETAIL_TOOL_NAMES:
        return await session.call_tool(tool_name, tool_args)

    async with _run_server_session(server_path) as isolated_session:
        return await isolated_session.call_tool(tool_name, tool_args)


async def _call_tool_with_workflow_isolation(
    session: ClientSession,
    server_path: Path,
    tool_name: str,
    tool_args: dict[str, Any],
    prior_predictions: list[dict[str, Any]],
) -> Any:
    """Execute against predicted, never gold, workflow state.

    Retail tools mutate server state. Recreate that state in an isolated server
    by replaying earlier successful predicted calls before the current call.
    Other tool families are non-mutating and can use the evaluation session.
    """
    if tool_name not in RETAIL_TOOL_NAMES:
        return await _call_tool_with_sample_isolation(
            session,
            server_path,
            tool_name,
            tool_args,
        )

    async with _run_server_session(server_path) as isolated_session:
        for prior_step in prior_predictions:
            prior_tool = prior_step.get("called_tool")
            if (
                prior_tool not in RETAIL_TOOL_NAMES
                or not prior_step.get("execution_success", False)
            ):
                continue
            setup_result = await isolated_session.call_tool(
                prior_tool,
                prior_step.get("selected_args", {}),
            )
            if bool(getattr(setup_result, "isError", False)):
                raise RuntimeError(
                    "Predicted workflow state replay failed at "
                    f"{prior_step.get('step_id')}: "
                    f"{_summarize_tool_result(setup_result)}"
                )
        return await isolated_session.call_tool(tool_name, tool_args)


def _tool_schema(tool: Any) -> dict[str, Any]:
    schema = getattr(tool, "inputSchema", None)
    if schema is None:
        schema = getattr(tool, "parameters", None)
    return schema if isinstance(schema, dict) else {}


def _reasoning_metadata(router: Any, reasoning_mode: str) -> dict[str, str]:
    if reasoning_mode not in REASONING_MODES:
        raise ValueError(
            f"Unsupported reasoning mode {reasoning_mode!r}; "
            f"expected one of: {', '.join(REASONING_MODES)}."
        )
    supports_reasoning = bool(getattr(router, "SUPPORTS_REASONING_MODE", False))
    if reasoning_mode == "reasoning" and not supports_reasoning:
        raise ValueError(
            f"Router {getattr(router, 'ROUTER_ID', 'unknown')!r} does not "
            "implement reasoning mode yet."
        )
    method = "none"
    if supports_reasoning and hasattr(router, "reasoning_method"):
        method = str(router.reasoning_method(reasoning_mode))
    return {
        "reasoning_mode": reasoning_mode,
        "reasoning_method": method,
    }


def _generation_metadata(router: Any) -> dict[str, Any]:
    limit = getattr(router, "MAX_GENERATED_TOKENS", None)
    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise ValueError(
            f"Router {getattr(router, 'ROUTER_ID', 'unknown')!r} must expose a "
            "positive MAX_GENERATED_TOKENS value."
        )
    return {
        "effective_generation_limit": limit,
        "effective_generation_limit_unit": "tokens",
    }


def _validate_expected_tools(
    dataset: list[BenchmarkSample],
    live_tool_set: set[str],
) -> None:
    expected_tools = {sample.expected_tool for sample in dataset}
    expected_tools.update(
        step.expected_tool
        for sample in dataset
        for step in sample.expected_steps
    )
    missing = sorted(expected_tools - live_tool_set)
    if missing:
        raise ValueError(
            "Benchmark expected_tool values are not registered by the MCP server: "
            + ", ".join(missing)
        )


def _route_query(
    router: Any,
    query: str,
    live_tools: list[str],
    tool_schemas: dict[str, dict[str, Any]],
    tool_descriptions: dict[str, str],
    reasoning_mode: str = "direct",
) -> tuple[str | None, dict[str, Any], str, str, str | None, str | None]:
    if hasattr(router, "choose_tool_call"):
        if getattr(router, "SUPPORTS_STRUCTURED_TOOL_DESCRIPTIONS", False):
            if getattr(router, "SUPPORTS_REASONING_MODE", False):
                prediction = router.choose_tool_call(
                    query,
                    live_tools,
                    tool_schemas,
                    tool_descriptions,
                    reasoning_mode=reasoning_mode,
                )
            else:
                prediction = router.choose_tool_call(
                    query,
                    live_tools,
                    tool_schemas,
                    tool_descriptions,
                )
        else:
            prediction = router.choose_tool_call(
                query,
                live_tools,
                tool_schemas,
            )
        return (
            prediction.selected_tool,
            prediction.selected_args,
            prediction.raw_output,
            getattr(prediction, "parse_status", "ok"),
            getattr(prediction, "attempted_tool", None),
            getattr(prediction, "diagnostic", None),
        )

    if getattr(router, "SUPPORTS_TOOL_DESCRIPTIONS", False):
        selected_tool = router.choose_tool(
            query,
            live_tools,
            tool_descriptions,
        )
    else:
        selected_tool = router.choose_tool(query, live_tools)
    return selected_tool, {}, selected_tool, "legacy_router", selected_tool, None


def _query_with_context(query: str, prompt_context: str) -> str:
    if not prompt_context.strip():
        return query
    return f"{query}\n\nGrounding context:\n{prompt_context}"


def _route_sample(
    router: Any,
    sample: BenchmarkSample,
    live_tools: list[str],
    tool_schemas: dict[str, dict[str, Any]],
    tool_descriptions: dict[str, str],
    reasoning_mode: str = "direct",
) -> tuple[str | None, dict[str, Any], str, str, str | None, str | None]:
    return _route_query(
        router,
        _query_with_context(sample.query, sample.prompt_context),
        live_tools,
        tool_schemas,
        tool_descriptions,
        reasoning_mode,
    )


def _is_no_tool_call(
    selected_tool: str | None,
    hallucinated_tool: str,
    live_tool_set: set[str],
) -> bool:
    return selected_tool == hallucinated_tool or selected_tool not in live_tool_set


def _tool_pool_metadata(
    live_tools: list[str],
    tool_schemas: dict[str, dict[str, Any]],
    tool_descriptions: dict[str, str],
) -> dict[str, Any]:
    tool_names = sorted(live_tools)
    registry_payload = [
        {
            "name": name,
            "schema": tool_schemas.get(name, {}),
            "description": tool_descriptions.get(name, ""),
        }
        for name in tool_names
    ]
    registry_digest = hashlib.sha256(
        _normalize_json(registry_payload).encode("utf-8")
    ).hexdigest()
    return {
        "tool_pool": "full_mcp_registry",
        "tool_count": len(tool_names),
        "tool_names": tool_names,
        "tool_registry_fingerprint": f"sha256:{registry_digest}",
        "tool_registry_fingerprint_version": TOOL_REGISTRY_FINGERPRINT_VERSION,
    }


def _benchmark_mode_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(
        record.get("benchmark_mode", DEFAULT_BENCHMARK_MODE)
        for record in records
    )
    return dict(sorted(counts.items()))


def _multistep_step_metrics(
    step_records: list[dict[str, Any]],
) -> dict[str, Any]:
    total_steps = len(step_records)
    semantic_output_scores = [
        step["final_outcome_correct"]
        for step in step_records
        if step["final_outcome_correct"] is not None
    ]
    tool_accuracy = (
        sum(step["tool_selection_correct"] for step in step_records) / total_steps
        if total_steps
        else 0.0
    )
    argument_accuracy = (
        sum(step["argument_match_correct"] for step in step_records) / total_steps
        if total_steps
        else 0.0
    )
    result_accuracy = (
        sum(score is True for score in semantic_output_scores)
        / len(semantic_output_scores)
        if semantic_output_scores
        else None
    )
    return {
        "total_steps": total_steps,
        "step_correct_tool_accuracy": tool_accuracy,
        "step_correct_arguments_accuracy": argument_accuracy,
        "step_correct_result_accuracy": result_accuracy,
        # Backwards-compatible metric names retained for historical consumers.
        "step_tool_selection_accuracy": tool_accuracy,
        "step_exact_argument_match_accuracy": argument_accuracy,
        "step_semantic_output_accuracy": result_accuracy,
        "step_semantic_output_scored": len(semantic_output_scores),
    }


def _multistep_workflow_metrics(
    workflow_records: list[dict[str, Any]],
) -> dict[str, Any]:
    total_workflows = len(workflow_records)
    semantic_output_scores = [
        workflow["sequence_semantic_output_correct"]
        for workflow in workflow_records
        if workflow["sequence_semantic_output_correct"] is not None
    ]
    final_answer_scores = [
        workflow["workflow_final_answer_correct"]
        for workflow in workflow_records
        if workflow.get("workflow_final_answer_correct") is not None
    ]
    all_tools_accuracy = (
        sum(
            workflow["sequence_tool_selection_correct"]
            for workflow in workflow_records
        )
        / total_workflows
        if total_workflows
        else 0.0
    )
    all_arguments_accuracy = (
        sum(
            workflow["sequence_argument_match_correct"]
            for workflow in workflow_records
        )
        / total_workflows
        if total_workflows
        else 0.0
    )
    all_step_results_accuracy = (
        sum(score is True for score in semantic_output_scores)
        / len(semantic_output_scores)
        if semantic_output_scores
        else None
    )
    final_answer_accuracy = (
        sum(score is True for score in final_answer_scores)
        / len(final_answer_scores)
        if final_answer_scores
        else None
    )
    return {
        "total_workflows": total_workflows,
        "workflow_all_tools_correct_accuracy": all_tools_accuracy,
        "workflow_all_arguments_correct_accuracy": all_arguments_accuracy,
        "workflow_all_step_results_correct_accuracy": all_step_results_accuracy,
        "workflow_final_answer_accuracy": final_answer_accuracy,
        "workflow_final_answer_scored": len(final_answer_scores),
        "workflow_final_answer_gold": sum(
            workflow.get("expected_final_answer") not in {None, ""}
            for workflow in workflow_records
        ),
        # Backwards-compatible metric names retained for historical consumers.
        "workflow_tool_sequence_accuracy": all_tools_accuracy,
        "workflow_exact_sequence_accuracy": all_arguments_accuracy,
        "workflow_semantic_output_sequence_accuracy": all_step_results_accuracy,
        "workflow_semantic_output_sequence_scored": len(
            semantic_output_scores
        ),
        "workflow_expected_final_answer_gold": sum(
            workflow.get("expected_final_answer") not in {None, ""}
            for workflow in workflow_records
        ),
    }


def _build_multistep_metrics(
    workflow_records: list[dict[str, Any]],
    step_records: list[dict[str, Any]],
) -> dict[str, Any]:
    workflows_by_mode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for workflow in workflow_records:
        workflows_by_mode[
            workflow.get("benchmark_mode", DEFAULT_BENCHMARK_MODE)
        ].append(workflow)

    steps_by_mode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for step in step_records:
        steps_by_mode[
            step.get("benchmark_mode", DEFAULT_BENCHMARK_MODE)
        ].append(step)

    return {
        "benchmark_mode_counts": _benchmark_mode_counts(workflow_records),
        "step_benchmark_mode_counts": _benchmark_mode_counts(step_records),
        **_multistep_workflow_metrics(workflow_records),
        **_multistep_step_metrics(step_records),
        "workflow_metrics_by_benchmark_mode": {
            mode: _multistep_workflow_metrics(mode_records)
            for mode, mode_records in sorted(workflows_by_mode.items())
        },
        "step_metrics_by_benchmark_mode": {
            mode: _multistep_step_metrics(mode_records)
            for mode, mode_records in sorted(steps_by_mode.items())
        },
    }


def _predicted_history_item(step_record: dict[str, Any]) -> dict[str, Any]:
    """Build rollout context containing predictions and execution only."""
    return {
        "step_id": step_record["step_id"],
        "query": step_record["query"],
        "selected_tool": step_record["selected_tool"],
        "selected_args": step_record["selected_args"],
        "execution_success": step_record["execution_success"],
        "tool_result_value": step_record["tool_result_value"],
        "tool_error": step_record["tool_error"],
    }


def _multistep_query(
    sample: BenchmarkSample,
    step: BenchmarkStep,
    history: list[dict[str, Any]],
) -> str:
    parts = [
        "Overall task: "
        + _bounded_prompt_text(
            sample.query,
            MULTISTEP_OVERALL_TASK_CHAR_LIMIT,
        ),
        "Current step: "
        + _bounded_prompt_text(
            step.query,
            MULTISTEP_CURRENT_STEP_CHAR_LIMIT,
        ),
    ]
    if sample.prompt_context.strip():
        parts.append(
            "Overall grounding context: "
            + _bounded_prompt_text(
                sample.prompt_context,
                MULTISTEP_CURRENT_STEP_CHAR_LIMIT,
            )
        )
    # Some dependent-step contexts contain expressions resolved with gold prior
    # answers. Supplying those would silently restore teacher forcing. Other
    # contexts (for example repository coordinates) remain necessary grounding.
    if step.prompt_context.strip() and not _is_gold_resolved_step_context(step):
        parts.append(
            "Current-step grounding context: "
            + _bounded_prompt_text(
                step.prompt_context,
                MULTISTEP_CURRENT_STEP_CHAR_LIMIT,
            )
        )
    if history:
        dependency_ids = set(step.depends_on)
        recent_non_dependencies = [
            item
            for item in history
            if item.get("step_id") not in dependency_ids
        ][-MULTISTEP_HISTORY_STEP_LIMIT:]
        visible_step_ids = dependency_ids | {
            str(item.get("step_id")) for item in recent_non_dependencies
        }
        visible_history = [
            item
            for item in history
            if item.get("step_id") in visible_step_ids
        ]
        omitted_count = len(history) - len(visible_history)
        history_heading = "Prior predicted calls and executed results"
        if omitted_count:
            history_heading += (
                f" (showing every declared dependency and up to the latest "
                f"{MULTISTEP_HISTORY_STEP_LIMIT} other step(s); "
                f"{omitted_count} earlier step(s) omitted)"
            )
        parts.append(
            history_heading
            + ":\n"
            + "\n".join(
                _bounded_prompt_text(
                    json.dumps(item, ensure_ascii=True, sort_keys=True),
                    MULTISTEP_HISTORY_ITEM_CHAR_LIMIT,
                )
                for item in visible_history
            )
        )
    parts.append("Choose and call the one tool needed for the current step.")
    return "\n\n".join(parts)


def _is_gold_resolved_step_context(step: BenchmarkStep) -> bool:
    if not step.depends_on or not step.prompt_context.strip():
        return False
    try:
        context = json.loads(step.prompt_context)
    except json.JSONDecodeError:
        return False
    if not isinstance(context, dict):
        return False
    return context.get("kind") in {
        "finance_calculator_call_grounding_v1",
        "math_controlled_current_step_v1",
    }


def _bounded_prompt_text(value: str, maximum_chars: int) -> str:
    """Return a deterministic head/tail prompt excerpt with a full-text hash."""
    if len(value) <= maximum_chars:
        return value
    if maximum_chars < 200:
        raise ValueError("Prompt text bounds must allow a hash marker.")
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    marker = (
        "\n...[bounded prompt excerpt; "
        f"original_chars={len(value)}; sha256={digest}]...\n"
    )
    remaining = maximum_chars - len(marker)
    head_chars = (remaining * 3) // 4
    tail_chars = remaining - head_chars
    return value[:head_chars] + marker + value[-tail_chars:]


async def _evaluate_multistep_with_server(
    dataset: list[BenchmarkSample],
    benchmark_path: Path,
    server_path: Path,
    call_predicted_tools: bool,
    router_name: str,
    reasoning_mode: str = "direct",
    output_dir: Path | None = None,
) -> None:
    from models.routers.registry import load_router

    if not dataset or not all(sample.expected_steps for sample in dataset):
        raise ValueError(
            "Multi-step evaluation requires expected_steps on every dataset sample."
        )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    artifacts = _evaluation_artifact_paths(output_dir, timestamp)
    samples_path = artifacts.samples
    summary_path = artifacts.summary

    workflow_records: list[dict[str, Any]] = []
    step_records: list[dict[str, Any]] = []
    latencies: list[float] = []
    executed_tool_calls = 0
    errors_count = 0

    async with _run_server_session(server_path) as session:
        listed_tools = await session.list_tools()
        live_tools = [tool.name for tool in listed_tools.tools]
        live_tool_set = set(live_tools)
        tool_schemas = {
            tool.name: _tool_schema(tool) for tool in listed_tools.tools
        }
        tool_descriptions = {
            tool.name: str(getattr(tool, "description", "") or "")
            for tool in listed_tools.tools
        }
        _validate_expected_tools(dataset, live_tool_set)

        router = load_router(router_name)
        reasoning_metadata = _reasoning_metadata(router, reasoning_mode)
        generation_metadata = _generation_metadata(router)
        hallucinated_tool = router.HALLUCINATED_TOOL
        model_name = router.MODEL_NAME
        prompt_template = router.PROMPT_TEMPLATE
        tool_pool_metadata = _tool_pool_metadata(
            live_tools,
            tool_schemas,
            tool_descriptions,
        )

        print(f"Discovered MCP tools: {', '.join(live_tools)}")
        print(
            f"Evaluation protocol: {MULTISTEP_EVALUATION_PROTOCOL} -- "
            f"{EVALUATION_PROTOCOL_DESCRIPTIONS[MULTISTEP_EVALUATION_PROTOCOL]}"
        )

        with _open_samples_exclusive(samples_path) as sample_handle:
            for sample in tqdm(dataset):
                history: list[dict[str, Any]] = []
                workflow_steps: list[dict[str, Any]] = []

                for step_index, step in enumerate(sample.expected_steps):
                    query = _multistep_query(sample, step, history)
                    start = time.perf_counter()
                    (
                        selected_tool,
                        selected_args,
                        raw_model_output,
                        parse_status,
                        attempted_tool,
                        parse_diagnostic,
                    ) = _route_query(
                        router,
                        query,
                        live_tools,
                        tool_schemas,
                        tool_descriptions,
                        reasoning_mode,
                    )
                    latency = time.perf_counter() - start
                    latencies.append(latency)

                    no_tool_call = _is_no_tool_call(
                        selected_tool,
                        hallucinated_tool,
                        live_tool_set,
                    )
                    called_tool = None
                    tool_result = None
                    tool_result_value = None
                    result_extraction_diagnostic = None
                    tool_error = None
                    execution_success = False
                    execution_attempted = False

                    if call_predicted_tools and not no_tool_call:
                        called_tool = selected_tool
                        execution_attempted = True
                        try:
                            call_result = await _call_tool_with_workflow_isolation(
                                session,
                                server_path,
                                selected_tool,
                                selected_args,
                                workflow_steps,
                            )
                            executed_tool_calls += 1
                            tool_result = _summarize_tool_result(call_result)
                            extracted_result = _extract_structured_tool_result(
                                call_result
                            )
                            tool_result_value = extracted_result.value
                            result_extraction_diagnostic = (
                                extracted_result.diagnostic
                            )
                            execution_success = not bool(
                                getattr(call_result, "isError", False)
                            )
                            if not execution_success:
                                errors_count += 1
                                tool_error = tool_result
                        except Exception as exc:  # pragma: no cover
                            errors_count += 1
                            tool_error = str(exc)

                    score = _score_sample(
                        expected_tool=step.expected_tool,
                        selected_tool=None if no_tool_call else selected_tool,
                        expected_args=step.expected_args,
                        selected_args=selected_args,
                        execution_success=execution_success,
                        execution_attempted=execution_attempted,
                    )
                    final_outcome = _score_final_outcome(
                        expected_answer=step.expected_answer,
                        tool_result_value=tool_result_value,
                        result_extraction_diagnostic=result_extraction_diagnostic,
                        domain=sample.domain,
                        call_predicted_tools=call_predicted_tools,
                        no_tool_call=no_tool_call,
                        execution_success=execution_success,
                        expected_tool=step.expected_tool,
                        called_tool=called_tool,
                    )
                    step_record = {
                        "sample_id": sample.id,
                        "benchmark_path": str(benchmark_path),
                        "step_id": step.id,
                        "step_index": step_index,
                        "domain": sample.domain,
                        "benchmark_mode": sample.benchmark_mode,
                        "workflow_execution_mode": PREDICTED_ROLLOUT_EXECUTION_MODE,
                        "declared_workflow_execution_mode": (
                            sample.workflow_execution_mode
                        ),
                        "evaluation_protocol": MULTISTEP_EVALUATION_PROTOCOL,
                        "evaluation_protocol_description": (
                            EVALUATION_PROTOCOL_DESCRIPTIONS[
                                MULTISTEP_EVALUATION_PROTOCOL
                            ]
                        ),
                        "query": step.query,
                        "prompt_context": step.prompt_context,
                        "routed_query": query,
                        "expected_tool": step.expected_tool,
                        "selected_tool": None if no_tool_call else selected_tool,
                        "expected_args": step.expected_args,
                        "expected_answer": step.expected_answer,
                        "selected_args": selected_args,
                        "tool_selection_correct": score.tool_selection_correct,
                        "argument_match_correct": score.argument_match_correct,
                        "execution_success": score.execution_success,
                        "failure_category": score.failure_category,
                        "depends_on": list(step.depends_on),
                        "source_program": step.source_program,
                        **tool_pool_metadata,
                        "raw_model_output": raw_model_output,
                        "parse_status": parse_status,
                        "attempted_tool": attempted_tool,
                        "parse_diagnostic": parse_diagnostic,
                        "latency_seconds": latency,
                        **reasoning_metadata,
                        **generation_metadata,
                        "called_tool": called_tool,
                        "tool_result": tool_result,
                        "tool_result_value": tool_result_value,
                        "tool_error": tool_error,
                        **_final_outcome_record_fields(final_outcome),
                    }
                    step_records.append(step_record)
                    workflow_steps.append(step_record)
                    history.append(_predicted_history_item(step_record))

                final_step_result_value = (
                    workflow_steps[-1]["tool_result_value"]
                    if workflow_steps
                    else None
                )
                workflow_final_answer = _score_workflow_final_answer(
                    expected_final_answer=sample.expected_final_answer,
                    final_tool_result_value=final_step_result_value,
                    call_predicted_tools=call_predicted_tools,
                    benchmark_family=sample.benchmark_family,
                )
                workflow_record = {
                    "sample_id": sample.id,
                    "benchmark_path": str(benchmark_path),
                    "domain": sample.domain,
                    "benchmark_mode": sample.benchmark_mode,
                    "benchmark_family": sample.benchmark_family,
                    "workflow_execution_mode": PREDICTED_ROLLOUT_EXECUTION_MODE,
                    "declared_workflow_execution_mode": (
                        sample.workflow_execution_mode
                    ),
                    "evaluation_protocol": MULTISTEP_EVALUATION_PROTOCOL,
                    "evaluation_protocol_description": (
                        EVALUATION_PROTOCOL_DESCRIPTIONS[
                            MULTISTEP_EVALUATION_PROTOCOL
                        ]
                    ),
                    "query": sample.query,
                    "prompt_context": sample.prompt_context,
                    "expected_final_answer": sample.expected_final_answer,
                    "final_step_result_value": final_step_result_value,
                    "workflow_final_answer_correct": workflow_final_answer.correct,
                    "workflow_final_answer_status": workflow_final_answer.status,
                    "workflow_final_answer_matcher": workflow_final_answer.matcher,
                    "workflow_final_answer_diagnostic": workflow_final_answer.diagnostic,
                    "task_type": sample.task_type,
                    "difficulty": sample.difficulty,
                    "source": sample.source,
                    **tool_pool_metadata,
                    "sequence_tool_selection_correct": all(
                        step["tool_selection_correct"] for step in workflow_steps
                    ),
                    "sequence_argument_match_correct": all(
                        step["argument_match_correct"] for step in workflow_steps
                    ),
                    "sequence_execution_success": (
                        call_predicted_tools
                        and all(step["execution_success"] for step in workflow_steps)
                    ),
                    "sequence_semantic_output_correct": (
                        all(
                            step["final_outcome_correct"] is True
                            for step in workflow_steps
                        )
                        if workflow_steps
                        and all(
                            step["final_outcome_correct"] is not None
                            for step in workflow_steps
                        )
                        else None
                    ),
                    "steps": workflow_steps,
                    "model_name": model_name,
                    "router_id": getattr(router, "ROUTER_ID", router_name),
                    "router_backend": getattr(
                        router, "ROUTER_BACKEND", "unknown"
                    ),
                    "prompt_template": prompt_template,
                    **reasoning_metadata,
                    **generation_metadata,
                }
                workflow_records.append(workflow_record)
                sample_handle.write(
                    json.dumps(workflow_record, ensure_ascii=True) + "\n"
                )

    multistep_metrics = _build_multistep_metrics(
        workflow_records,
        step_records,
    )
    total_steps = multistep_metrics["total_steps"]
    total_workflows = multistep_metrics["total_workflows"]
    summary = {
        "timestamp": timestamp,
        "benchmark_path": str(benchmark_path),
        "model_name": model_name,
        "router_id": getattr(router, "ROUTER_ID", router_name),
        "router_backend": getattr(router, "ROUTER_BACKEND", "unknown"),
        "architecture_source": getattr(router, "ARCHITECTURE_SOURCE", "unknown"),
        "weight_source": getattr(router, "WEIGHT_SOURCE", "unknown"),
        "prompt_template": prompt_template,
        **reasoning_metadata,
        **generation_metadata,
        "evaluation_protocol": MULTISTEP_EVALUATION_PROTOCOL,
        "evaluation_protocol_description": EVALUATION_PROTOCOL_DESCRIPTIONS[
            MULTISTEP_EVALUATION_PROTOCOL
        ],
        "workflow_execution_modes": [PREDICTED_ROLLOUT_EXECUTION_MODE],
        "declared_workflow_execution_modes": sorted(
            {sample.workflow_execution_mode for sample in dataset}
        ),
        **tool_pool_metadata,
        **multistep_metrics,
        "average_step_latency_seconds": (
            sum(latencies) / len(latencies) if latencies else 0.0
        ),
        "executed_tool_calls": executed_tool_calls,
        "errors_count": errors_count,
    }
    _write_summary_exclusive(summary_path, summary)

    print("\n===================")
    print(f"Evaluation protocol: {MULTISTEP_EVALUATION_PROTOCOL}")
    print("Metric scope: guided rollout with predicted-result error propagation")
    print(f"Workflows: {total_workflows}")
    print(f"Steps: {total_steps}")
    print(
        "Correct tool per provided step: "
        f"{summary['step_correct_tool_accuracy']:.2%}"
    )
    print(
        "Correct arguments per provided step: "
        f"{summary['step_correct_arguments_accuracy']:.2%}"
    )
    step_semantic_output_accuracy = summary["step_correct_result_accuracy"]
    print(
        "Correct result per provided step: "
        + (
            f"{step_semantic_output_accuracy:.2%}"
            if step_semantic_output_accuracy is not None
            else "not scored"
        )
    )
    workflow_semantic_output_accuracy = summary[
        "workflow_all_step_results_correct_accuracy"
    ]
    print(
        "All provided step results correct: "
        + (
            f"{workflow_semantic_output_accuracy:.2%}"
            if workflow_semantic_output_accuracy is not None
            else "not scored"
        )
    )
    workflow_final_answer_accuracy = summary["workflow_final_answer_accuracy"]
    print(
        "Final answer accuracy: "
        + (
            f"{workflow_final_answer_accuracy:.2%}"
            if workflow_final_answer_accuracy is not None
            else "not scored"
        )
    )
    for mode, workflow_metrics in summary[
        "workflow_metrics_by_benchmark_mode"
    ].items():
        step_metrics = summary["step_metrics_by_benchmark_mode"].get(mode, {})
        print(
            f"Benchmark mode {mode}: "
            f"workflows={workflow_metrics['total_workflows']}, "
            "all-step argument accuracy="
            f"{workflow_metrics['workflow_exact_sequence_accuracy']:.2%}, "
            f"steps={step_metrics.get('total_steps', 0)}, "
            "provided-step tool-selection accuracy="
            f"{step_metrics.get('step_tool_selection_accuracy', 0.0):.2%}"
        )
    print(f"Results: {samples_path}")
    print(f"Summary: {summary_path}")


async def _evaluate_with_server(
    dataset: list[BenchmarkSample],
    benchmark_path: Path,
    server_path: Path,
    call_predicted_tools: bool,
    router_name: str,
    reasoning_mode: str = "direct",
    output_dir: Path | None = None,
) -> None:
    if any(sample.expected_steps for sample in dataset):
        await _evaluate_multistep_with_server(
            dataset,
            benchmark_path,
            server_path,
            call_predicted_tools,
            router_name,
            reasoning_mode=reasoning_mode,
            output_dir=output_dir,
        )
        return

    from models.routers.registry import load_router

    latencies: list[float] = []
    executed_tool_calls = 0
    errors_count = 0
    records: list[dict[str, Any]] = []
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    artifacts = _evaluation_artifact_paths(output_dir, timestamp)
    samples_path = artifacts.samples
    summary_path = artifacts.summary

    async with _run_server_session(server_path) as session:
        listed_tools = await session.list_tools()
        live_tools = [tool.name for tool in listed_tools.tools]
        live_tool_set = set(live_tools)
        tool_schemas = {tool.name: _tool_schema(tool) for tool in listed_tools.tools}
        tool_descriptions = {
            tool.name: str(getattr(tool, "description", "") or "")
            for tool in listed_tools.tools
        }
        _validate_expected_tools(dataset, live_tool_set)

        router = load_router(router_name)
        reasoning_metadata = _reasoning_metadata(router, reasoning_mode)
        generation_metadata = _generation_metadata(router)
        hallucinated_tool = router.HALLUCINATED_TOOL
        model_name = router.MODEL_NAME
        prompt_template = router.PROMPT_TEMPLATE
        tool_pool_metadata = _tool_pool_metadata(
            live_tools,
            tool_schemas,
            tool_descriptions,
        )

        print(f"Discovered MCP tools: {', '.join(live_tools)}")
        print(f"Evaluation protocol: {SINGLE_STEP_EVALUATION_PROTOCOL}")

        with _open_samples_exclusive(samples_path) as sample_handle:
            for sample in tqdm(dataset):
                query = sample.query
                routed_query = _query_with_context(query, sample.prompt_context)
                expected = sample.expected_tool

                start = time.perf_counter()
                (
                    selected_tool,
                    selected_args,
                    raw_model_output,
                    parse_status,
                    attempted_tool,
                    parse_diagnostic,
                ) = _route_sample(
                    router,
                    sample,
                    live_tools,
                    tool_schemas,
                    tool_descriptions,
                    reasoning_mode,
                )
                latency = time.perf_counter() - start

                latencies.append(latency)

                no_tool_call = _is_no_tool_call(
                    selected_tool,
                    hallucinated_tool,
                    live_tool_set,
                )

                print(f"\nQuery: {query}")
                print(f"Expected: {expected}")
                print(f"Selected: {selected_tool}")
                print(f"Selected args: {selected_args}")
                if parse_status not in {"ok", "legacy_router"}:
                    print(f"Parse status: {parse_status}")
                    if parse_diagnostic:
                        print(f"Parse diagnostic: {parse_diagnostic}")
                    print(f"Raw model output: {raw_model_output[:1000]!r}")

                called_tool = None
                tool_result = None
                tool_result_value = None
                result_extraction_diagnostic = None
                tool_error = None
                execution_success = False
                execution_attempted = False

                if call_predicted_tools and not no_tool_call:
                    called_tool = selected_tool
                    execution_attempted = True
                    try:
                        call_result = await _call_tool_with_sample_isolation(
                            session,
                            server_path,
                            selected_tool,
                            selected_args,
                        )
                        executed_tool_calls += 1
                        tool_result = _summarize_tool_result(call_result)
                        extracted_result = _extract_structured_tool_result(call_result)
                        tool_result_value = extracted_result.value
                        result_extraction_diagnostic = extracted_result.diagnostic
                        execution_success = not bool(
                            getattr(call_result, "isError", False)
                        )
                        if execution_success:
                            print(f"Tool call: {tool_result}")
                        else:
                            tool_error = tool_result
                            if hasattr(router, "repair_tool_call"):
                                print(
                                    "Initial tool call failed; requesting "
                                    f"one correction: {tool_error}"
                                )
                                repair_start = time.perf_counter()
                                corrected = router.repair_tool_call(
                                    query,
                                    available_tools,
                                    prediction,
                                    tool_error,
                                    available_schemas,
                                    {
                                        tool: tool_descriptions.get(tool, "")
                                        for tool in available_tools
                                    },
                                )
                                latency += time.perf_counter() - repair_start
                                latencies[-1] = latency
                                corrected_no_tool_call = (
                                    corrected.selected_tool == hallucinated_tool
                                    or corrected.selected_tool not in live_tool_set
                                    or corrected.selected_tool not in available_tools
                                )
                                if not corrected_no_tool_call:
                                    selected_tool = corrected.selected_tool
                                    selected_args = corrected.selected_args
                                    raw_model_output = corrected.raw_output
                                    prediction = corrected
                                    called_tool = selected_tool
                                    print(
                                        "Retrying tool call: "
                                        f"{selected_tool} {selected_args}"
                                    )
                                    call_result = (
                                        await _call_tool_with_sample_isolation(
                                            session,
                                            server_path,
                                            selected_tool,
                                            selected_args,
                                        )
                                    )
                                    executed_tool_calls += 1
                                    tool_result = _summarize_tool_result(
                                        call_result
                                    )
                                    execution_success = not bool(
                                        getattr(call_result, "isError", False)
                                    )
                                    if execution_success:
                                        tool_error = None
                                        print(f"Tool call: {tool_result}")
                                    else:
                                        tool_error = tool_result
                                        print(
                                            "Tool call retry error: "
                                            f"{tool_error}"
                                        )
                            if not execution_success:
                                errors_count += 1
                                if not hasattr(router, "repair_tool_call"):
                                    print(f"Tool call error: {tool_error}")
                    except Exception as exc:  # pragma: no cover - exercised by integration runs
                        errors_count += 1
                        tool_error = str(exc)
                        print(f"Tool call error: {tool_error}")

                score = _score_sample(
                    expected_tool=expected,
                    selected_tool=None if no_tool_call else selected_tool,
                    expected_args=sample.expected_args,
                    selected_args=selected_args,
                    execution_success=execution_success,
                    execution_attempted=execution_attempted,
                )
                final_outcome = _score_final_outcome(
                    expected_answer=sample.expected_answer,
                    tool_result_value=tool_result_value,
                    result_extraction_diagnostic=result_extraction_diagnostic,
                    domain=sample.domain,
                    call_predicted_tools=call_predicted_tools,
                    no_tool_call=no_tool_call,
                    execution_success=execution_success,
                    expected_tool=sample.expected_tool,
                    called_tool=called_tool,
                )
                record = {
                    "sample_id": sample.id,
                    "benchmark_path": str(benchmark_path),
                    "domain": sample.domain,
                    "benchmark_mode": sample.benchmark_mode,
                    "evaluation_protocol": SINGLE_STEP_EVALUATION_PROTOCOL,
                    "evaluation_protocol_description": (
                        EVALUATION_PROTOCOL_DESCRIPTIONS[
                            SINGLE_STEP_EVALUATION_PROTOCOL
                        ]
                    ),
                    "query": query,
                    "prompt_context": sample.prompt_context,
                    "routed_query": routed_query,
                    "expected_tool": expected,
                    "selected_tool": None if no_tool_call else selected_tool,
                    "expected_args": sample.expected_args,
                    "expected_answer": sample.expected_answer,
                    "selected_args": selected_args,
                    "tool_selection_correct": score.tool_selection_correct,
                    "argument_match_correct": score.argument_match_correct,
                    "execution_success": score.execution_success,
                    "failure_category": score.failure_category,
                    "raw_model_output": raw_model_output,
                    "parse_status": parse_status,
                    "attempted_tool": attempted_tool,
                    "parse_diagnostic": parse_diagnostic,
                    "task_type": sample.task_type,
                    "difficulty": sample.difficulty,
                    "source": sample.source,
                    **tool_pool_metadata,
                    "latency_seconds": latency,
                    "model_name": model_name,
                    "router_id": getattr(router, "ROUTER_ID", router_name),
                    "router_backend": getattr(router, "ROUTER_BACKEND", "unknown"),
                    "architecture_source": getattr(
                        router,
                        "ARCHITECTURE_SOURCE",
                        "unknown",
                    ),
                    "weight_source": getattr(router, "WEIGHT_SOURCE", "unknown"),
                    "prompt_template": prompt_template,
                    **reasoning_metadata,
                    **generation_metadata,
                    "called_tool": called_tool,
                    "tool_result": tool_result,
                    "tool_result_value": tool_result_value,
                    "tool_error": tool_error,
                    **_final_outcome_record_fields(final_outcome),
                }
                records.append(record)
                sample_handle.write(json.dumps(record, ensure_ascii=True) + "\n")

    metrics = _build_aggregate_metrics(records)
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    summary = {
        "timestamp": timestamp,
        "benchmark_path": str(benchmark_path),
        "model_name": model_name,
        "router_id": getattr(router, "ROUTER_ID", router_name),
        "router_backend": getattr(router, "ROUTER_BACKEND", "unknown"),
        "architecture_source": getattr(router, "ARCHITECTURE_SOURCE", "unknown"),
        "weight_source": getattr(router, "WEIGHT_SOURCE", "unknown"),
        "prompt_template": prompt_template,
        **reasoning_metadata,
        **generation_metadata,
        "evaluation_protocol": SINGLE_STEP_EVALUATION_PROTOCOL,
        "evaluation_protocol_description": EVALUATION_PROTOCOL_DESCRIPTIONS[
            SINGLE_STEP_EVALUATION_PROTOCOL
        ],
        **tool_pool_metadata,
        "average_latency_seconds": avg_latency,
        "executed_tool_calls": executed_tool_calls,
        "errors_count": errors_count,
        **metrics,
    }
    _write_summary_exclusive(summary_path, summary)

    print("\n===================")
    print(f"Total: {metrics['total_samples']}")
    print(f"Tool selection accuracy: {metrics['tool_selection_accuracy']:.2%}")
    print(f"Exact argument match accuracy: {metrics['exact_argument_match_accuracy']:.2%}")
    print(f"Execution success rate: {metrics['execution_success_rate']:.2%}")
    print(f"No tool call rate: {metrics['no_tool_call_rate']:.2%}")
    final_outcome_accuracy = metrics["final_outcome_accuracy"]
    print(
        "Final outcome accuracy: "
        + (
            f"{final_outcome_accuracy:.2%}"
            if final_outcome_accuracy is not None
            else "not scored"
        )
    )
    print(f"Final outcome gold coverage: {metrics['final_outcome_gold_coverage']:.2%}")
    for mode, mode_metrics in metrics["metrics_by_benchmark_mode"].items():
        print(
            f"Benchmark mode {mode}: samples={mode_metrics['total_samples']}, "
            "tool-selection accuracy="
            f"{mode_metrics['tool_selection_accuracy']:.2%}, "
            "exact argument match accuracy="
            f"{mode_metrics['exact_argument_match_accuracy']:.2%}"
        )
    print(f"Avg Latency: {avg_latency:.2f}s")
    print(f"Executed tool calls: {executed_tool_calls}")
    print(f"Results: {samples_path}")
    print(f"Summary: {summary_path}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a tool-routing model against the LayerMCP benchmark."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=BENCHMARK_PATH,
        help="Path to the benchmark JSON dataset.",
    )
    parser.add_argument(
        "--benchmark",
        type=Path,
        help="Alias for --dataset. Path to the benchmark JSON dataset.",
    )
    parser.add_argument(
        "--server",
        type=Path,
        default=SERVER_PATH,
        help="Path to the MCP server entrypoint.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "Write samples.jsonl and summary.json inside this directory. "
            "Existing artifacts are never overwritten."
        ),
    )
    parser.add_argument(
        "--call-predicted-tools",
        action="store_true",
        help="Call the predicted MCP tool using sample.tool_args when present.",
    )
    parser.add_argument(
        "--router",
        default="qwen-hf",
        help=(
            "Router backend to evaluate. Use qwen-hf for the Hugging Face "
            "Qwen baseline or gpt-oss-local for the local GPT-OSS PyTorch router."
        ),
    )
    parser.add_argument(
        "--reasoning-mode",
        choices=REASONING_MODES,
        default="direct",
        help=(
            "Inference condition. 'direct' requests an immediate tool call; "
            "'reasoning' enables the router's supported reasoning mechanism."
        ),
    )
    return parser


async def _async_main(args: argparse.Namespace) -> None:
    benchmark_path = args.benchmark or args.dataset
    dataset = load_benchmark(benchmark_path)
    await _evaluate_with_server(
        dataset=dataset,
        benchmark_path=benchmark_path,
        server_path=args.server,
        call_predicted_tools=args.call_predicted_tools,
        router_name=args.router,
        reasoning_mode=args.reasoning_mode,
        output_dir=args.output_dir,
    )


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    asyncio.run(_async_main(args))


if __name__ == "__main__":
    main()
