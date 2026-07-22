from __future__ import annotations

import argparse
import asyncio
from collections import Counter, defaultdict
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import json
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
    available_tools: list[str] | None
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


def _normalize_sample(sample: dict[str, Any], index: int) -> BenchmarkSample:
    expected_args = sample.get("expected_args")
    if expected_args is None:
        expected_args = sample.get("tool_args", {})
    if expected_args is None:
        expected_args = {}
    if not isinstance(expected_args, dict):
        raise ValueError(f"Sample {index} expected_args/tool_args must be an object.")

    available_tools = sample.get("available_tools")
    if available_tools is not None:
        if not isinstance(available_tools, list) or not all(
            isinstance(tool, str) for tool in available_tools
        ):
            raise ValueError(f"Sample {index} available_tools must be a list of strings.")

    sample_id = sample.get("id") or f"sample_{index + 1:04d}"

    return BenchmarkSample(
        id=str(sample_id),
        domain=str(sample.get("domain", "unknown")),
        task_type=str(sample.get("task_type", "tool_routing")),
        difficulty=str(sample.get("difficulty", "unspecified")),
        source=str(sample.get("source", "unspecified")),
        query=str(sample["query"]),
        available_tools=available_tools,
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


async def _evaluate_with_server(
    dataset: list[BenchmarkSample],
    benchmark_path: Path,
    server_path: Path,
    call_predicted_tools: bool,
    router_name: str,
) -> None:
    from models.routers.registry import load_router

    router = load_router(router_name)
    hallucinated_tool = router.HALLUCINATED_TOOL
    model_name = router.MODEL_NAME
    prompt_template = router.PROMPT_TEMPLATE

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

        print(f"Discovered MCP tools: {', '.join(live_tools)}")

        with samples_path.open("w", encoding="utf-8") as sample_handle:
            for sample in tqdm(dataset):
                available_tools = sample.available_tools or live_tools
                query = sample.query
                expected = sample.expected_tool

                start = time.perf_counter()
                if hasattr(router, "choose_tool_call"):
                    available_schemas = {
                        tool: tool_schemas.get(tool, {}) for tool in available_tools
                    }
                    if getattr(
                        router, "SUPPORTS_STRUCTURED_TOOL_DESCRIPTIONS", False
                    ):
                        prediction = router.choose_tool_call(
                            query,
                            available_tools,
                            available_schemas,
                            {
                                tool: tool_descriptions.get(tool, "")
                                for tool in available_tools
                            },
                        )
                    else:
                        prediction = router.choose_tool_call(
                            query,
                            available_tools,
                            available_schemas,
                        )
                    selected_tool = prediction.selected_tool
                    selected_args = prediction.selected_args
                    raw_model_output = prediction.raw_output
                else:
                    if getattr(router, "SUPPORTS_TOOL_DESCRIPTIONS", False):
                        selected_tool = router.choose_tool(
                            query,
                            available_tools,
                            {
                                tool: tool_descriptions.get(tool, "")
                                for tool in available_tools
                            },
                        )
                    else:
                        selected_tool = router.choose_tool(query, available_tools)
                    selected_args = {}
                    raw_model_output = selected_tool
                latency = time.perf_counter() - start

                latencies.append(latency)

                no_tool_call = (
                    selected_tool == hallucinated_tool
                    or selected_tool not in live_tool_set
                    or selected_tool not in available_tools
                )

                print(f"\nQuery: {query}")
                print(f"Expected: {expected}")
                print(f"Selected: {selected_tool}")
                print(f"Selected args: {selected_args}")
                if selected_tool == hallucinated_tool:
                    print(f"Raw model output: {raw_model_output[:1000]!r}")

                called_tool = None
                tool_result = None
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
                record = {
                    "sample_id": sample.id,
                    "domain": sample.domain,
                    "query": query,
                    "expected_tool": expected,
                    "selected_tool": None if no_tool_call else selected_tool,
                    "expected_args": sample.expected_args,
                    "selected_args": selected_args,
                    "tool_selection_correct": score.tool_selection_correct,
                    "argument_match_correct": score.argument_match_correct,
                    "execution_success": score.execution_success,
                    "failure_category": score.failure_category,
                    "raw_model_output": raw_model_output,
                    "task_type": sample.task_type,
                    "difficulty": sample.difficulty,
                    "source": sample.source,
                    "available_tools": available_tools,
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
                    "tool_error": tool_error,
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
