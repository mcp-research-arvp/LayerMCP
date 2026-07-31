from __future__ import annotations

import argparse
import asyncio
from collections import Counter, defaultdict
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

BENCHMARK_PATH = PROJECT_ROOT / "benchmark" / "tool_routing.json"
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
    expected_args = sample.get("expected_args")
    if expected_args is None:
        expected_args = sample.get("tool_args", {})
    if expected_args is None:
        expected_args = {}
    if not isinstance(expected_args, dict):
        raise ValueError(f"Sample {index} expected_args/tool_args must be an object.")

    sample_id = sample.get("id") or f"sample_{index + 1:04d}"

    return BenchmarkSample(
        id=str(sample_id),
        domain=str(sample.get("domain", "unknown")),
        task_type=str(sample.get("task_type", "tool_routing")),
        difficulty=str(sample.get("difficulty", "unspecified")),
        source=str(sample.get("source", "unspecified")),
        query=str(sample["query"]),
        expected_tool=str(sample["expected_tool"]),
        expected_args=expected_args,
        expected_answer=sample.get("expected_answer"),
        perturbation_type=str(sample.get("perturbation_type", "none")),
        notes=str(sample.get("notes", "")),
    )


def load_benchmark(path: Path) -> list[BenchmarkSample]:
    if not path.exists():
        raise FileNotFoundError(
            f"Benchmark dataset not found: {path}. "
            "Create benchmark/tool_routing.json or update --dataset."
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


def _score_final_outcome(
    *,
    expected_answer: Any,
    tool_result_value: Any,
    result_extraction_diagnostic: str | None,
    domain: str,
    call_predicted_tools: bool,
    no_tool_call: bool,
    execution_success: bool,
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

    match = _match_expected_answer(
        tool_result_value,
        expected_answer,
        domain=domain,
    )
    return FinalOutcomeScore(
        match.matched,
        "correct" if match.matched else "mismatch",
        FINAL_OUTCOME_MATCHER,
        match.diagnostic,
    )


def _final_outcome_record_fields(score: FinalOutcomeScore) -> dict[str, Any]:
    return {
        "final_outcome_correct": score.correct,
        "final_outcome_status": score.status,
        "final_outcome_matcher": score.matcher,
        "final_outcome_diagnostic": score.diagnostic,
    }


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


def _build_aggregate_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
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

    return {
        "total_samples": total,
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


def _tool_schema(tool: Any) -> dict[str, Any]:
    schema = getattr(tool, "inputSchema", None)
    if schema is None:
        schema = getattr(tool, "parameters", None)
    return schema if isinstance(schema, dict) else {}


def _validate_expected_tools(
    dataset: list[BenchmarkSample],
    live_tool_set: set[str],
) -> None:
    missing = sorted(
        {
            sample.expected_tool
            for sample in dataset
            if sample.expected_tool not in live_tool_set
        }
    )
    if missing:
        raise ValueError(
            "Benchmark expected_tool values are not registered by the MCP server: "
            + ", ".join(missing)
        )


def _route_sample(
    router: Any,
    sample: BenchmarkSample,
    live_tools: list[str],
    tool_schemas: dict[str, dict[str, Any]],
    tool_descriptions: dict[str, str],
) -> tuple[str | None, dict[str, Any], str, str, str | None, str | None]:
    if hasattr(router, "choose_tool_call"):
        if getattr(router, "SUPPORTS_STRUCTURED_TOOL_DESCRIPTIONS", False):
            prediction = router.choose_tool_call(
                sample.query,
                live_tools,
                tool_schemas,
                tool_descriptions,
            )
        else:
            prediction = router.choose_tool_call(
                sample.query,
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
            sample.query,
            live_tools,
            tool_descriptions,
        )
    else:
        selected_tool = router.choose_tool(sample.query, live_tools)
    return selected_tool, {}, selected_tool, "legacy_router", selected_tool, None


def _is_no_tool_call(
    selected_tool: str | None,
    hallucinated_tool: str,
    live_tool_set: set[str],
) -> bool:
    return selected_tool == hallucinated_tool or selected_tool not in live_tool_set


def _tool_pool_metadata(live_tools: list[str]) -> dict[str, Any]:
    return {
        "tool_pool": "full_mcp_registry",
        "tool_count": len(live_tools),
    }


async def _evaluate_with_server(
    dataset: list[BenchmarkSample],
    benchmark_path: Path,
    server_path: Path,
    call_predicted_tools: bool,
    router_name: str,
) -> None:
    from models.routers.registry import load_router

    latencies: list[float] = []
    executed_tool_calls = 0
    errors_count = 0
    records: list[dict[str, Any]] = []
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    RESULTS_DIR.mkdir(exist_ok=True)
    samples_path = RESULTS_DIR / f"{timestamp}_samples.jsonl"
    summary_path = RESULTS_DIR / f"{timestamp}_summary.json"

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
        hallucinated_tool = router.HALLUCINATED_TOOL
        model_name = router.MODEL_NAME
        prompt_template = router.PROMPT_TEMPLATE
        tool_pool_metadata = _tool_pool_metadata(live_tools)

        print(f"Discovered MCP tools: {', '.join(live_tools)}")

        with samples_path.open("w", encoding="utf-8") as sample_handle:
            for sample in tqdm(dataset):
                query = sample.query
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
                            errors_count += 1
                            tool_error = tool_result
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
                )
                record = {
                    "sample_id": sample.id,
                    "domain": sample.domain,
                    "query": query,
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
        **tool_pool_metadata,
        "average_latency_seconds": avg_latency,
        "executed_tool_calls": executed_tool_calls,
        "errors_count": errors_count,
        **metrics,
    }
    with summary_path.open("w", encoding="utf-8") as summary_handle:
        json.dump(summary, summary_handle, ensure_ascii=True, indent=2)

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
    )


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    asyncio.run(_async_main(args))


if __name__ == "__main__":
    main()
