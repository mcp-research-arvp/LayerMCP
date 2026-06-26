from __future__ import annotations

import argparse
import asyncio
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


async def _evaluate_with_server(
    dataset: list[BenchmarkSample],
    benchmark_path: Path,
    server_path: Path,
    call_predicted_tools: bool,
) -> None:
    from models.qwen_router import HALLUCINATED_TOOL, MODEL_NAME, PROMPT_TEMPLATE, choose_tool

    total = 0
    correct = 0
    hallucinations = 0
    latencies: list[float] = []
    executed_tool_calls = 0
    errors_count = 0
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    RESULTS_DIR.mkdir(exist_ok=True)
    samples_path = RESULTS_DIR / f"{timestamp}_samples.jsonl"
    summary_path = RESULTS_DIR / f"{timestamp}_summary.json"

    async for session in _run_server_session(server_path):
        listed_tools = await session.list_tools()
        live_tools = [tool.name for tool in listed_tools.tools]

        print(f"Discovered MCP tools: {', '.join(live_tools)}")

        with samples_path.open("w", encoding="utf-8") as sample_handle:
            for sample in tqdm(dataset):
                available_tools = sample.available_tools or live_tools
                query = sample.query
                expected = sample.expected_tool

                start = time.perf_counter()
                predicted = choose_tool(query, available_tools)
                latency = time.perf_counter() - start

                latencies.append(latency)
                total += 1

                is_correct = predicted == expected
                if is_correct:
                    correct += 1

                hallucinated = predicted == HALLUCINATED_TOOL
                if hallucinated:
                    hallucinations += 1

                print(f"\nQuery: {query}")
                print(f"Expected: {expected}")
                print(f"Predicted: {predicted}")

                called_tool = None
                tool_result = None
                tool_error = None
                tool_args_used = sample.expected_args

                if call_predicted_tools and not hallucinated:
                    called_tool = predicted
                    try:
                        call_result = await session.call_tool(predicted, tool_args_used)
                        executed_tool_calls += 1
                        tool_result = _summarize_tool_result(call_result)
                        print(f"Tool call: {tool_result}")
                    except Exception as exc:  # pragma: no cover - exercised by integration runs
                        errors_count += 1
                        tool_error = str(exc)
                        print(f"Tool call error: {tool_error}")

                record = {
                    "sample_id": sample.id,
                    "domain": sample.domain,
                    "task_type": sample.task_type,
                    "difficulty": sample.difficulty,
                    "source": sample.source,
                    "query": query,
                    "available_tools": available_tools,
                    "expected_tool": expected,
                    "predicted_tool": predicted,
                    "is_correct": is_correct,
                    "hallucinated": hallucinated,
                    "latency_seconds": latency,
                    "model_name": MODEL_NAME,
                    "prompt_template": PROMPT_TEMPLATE,
                    "called_tool": called_tool,
                    "expected_args": sample.expected_args,
                    "tool_args_used": tool_args_used,
                    "tool_result": tool_result,
                    "tool_error": tool_error,
                }
                sample_handle.write(json.dumps(record, ensure_ascii=True) + "\n")

    accuracy = correct / total if total else 0.0
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    summary = {
        "timestamp": timestamp,
        "benchmark_path": str(benchmark_path),
        "model_name": MODEL_NAME,
        "total_samples": total,
        "accuracy": accuracy,
        "hallucination_count": hallucinations,
        "average_latency_seconds": avg_latency,
        "executed_tool_calls": executed_tool_calls,
        "errors_count": errors_count,
    }
    with summary_path.open("w", encoding="utf-8") as summary_handle:
        json.dump(summary, summary_handle, ensure_ascii=True, indent=2)

    print("\n===================")
    print(f"Total: {total}")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Hallucinations: {hallucinations}")
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
    return parser


async def _async_main(args: argparse.Namespace) -> None:
    benchmark_path = args.benchmark or args.dataset
    dataset = load_benchmark(benchmark_path)
    await _evaluate_with_server(
        dataset=dataset,
        benchmark_path=benchmark_path,
        server_path=args.server,
        call_predicted_tools=args.call_predicted_tools,
    )


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    asyncio.run(_async_main(args))


if __name__ == "__main__":
    main()
