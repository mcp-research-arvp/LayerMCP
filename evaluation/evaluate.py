from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.dataset import BenchmarkSample, NO_TOOL_NAME, load_benchmark

BENCHMARK_PATH = PROJECT_ROOT / "benchmark" / "tool_routing.jsonl"
SERVER_PATH = PROJECT_ROOT / "mcp_server" / "server.py"


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


async def _evaluate_with_server(
    dataset: list[BenchmarkSample],
    server_path: Path,
    call_predicted_tools: bool,
) -> None:
    from models.qwen_router import choose_tool
    from tqdm import tqdm

    total = 0
    correct = 0
    hallucinations = 0
    latencies: list[float] = []
    executed_tool_calls = 0

    async for session in _run_server_session(server_path):
        listed_tools = await session.list_tools()
        available_tools = [tool.name for tool in listed_tools.tools]

        print(f"Discovered MCP tools: {', '.join(available_tools)}")

        for sample in tqdm(dataset):
            missing_tools = sorted(set(sample.tools) - set(available_tools))
            if missing_tools:
                raise ValueError(
                    f"Sample {sample.id} references tool(s) not exposed by the "
                    f"server: {', '.join(missing_tools)}"
                )

            start = time.perf_counter()
            predicted = choose_tool(sample.query, sample.tools)
            latency = time.perf_counter() - start

            latencies.append(latency)
            total += 1

            if predicted == sample.expected_tool:
                correct += 1

            if predicted == NO_TOOL_NAME:
                hallucinations += 1

            print(f"\nSample: {sample.id} ({sample.domain}, {sample.difficulty})")
            print(f"Query: {sample.query}")
            print(f"Expected: {sample.expected_tool}")
            print(f"Predicted: {predicted}")

            if call_predicted_tools and predicted != NO_TOOL_NAME:
                if not sample.expected_arguments:
                    print("Tool call: skipped (no expected_arguments in dataset)")
                else:
                    try:
                        call_result = await session.call_tool(
                            predicted,
                            sample.expected_arguments,
                        )
                    except Exception as exc:
                        print(f"Tool call failed: {type(exc).__name__}: {exc}")
                    else:
                        executed_tool_calls += 1
                        print(f"Tool call: {_summarize_tool_result(call_result)}")

    accuracy = correct / total if total else 0.0
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

    print("\n===================")
    print(f"Total: {total}")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Hallucinations: {hallucinations}")
    print(f"Avg Latency: {avg_latency:.2f}s")
    print(f"Executed tool calls: {executed_tool_calls}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a tool-routing model against the LayerMCP benchmark."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=BENCHMARK_PATH,
        help="Path to the benchmark JSONL dataset.",
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
        help=(
            "Call the predicted MCP tool using sample.expected_arguments when "
            "present."
        ),
    )
    return parser


async def _async_main(args: argparse.Namespace) -> None:
    dataset = load_benchmark(args.dataset)
    await _evaluate_with_server(
        dataset=dataset,
        server_path=args.server,
        call_predicted_tools=args.call_predicted_tools,
    )


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    asyncio.run(_async_main(args))


if __name__ == "__main__":
    main()
