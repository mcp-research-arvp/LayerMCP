from __future__ import annotations

import argparse
import asyncio
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


def _load_dataset(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Benchmark dataset not found: {path}. "
            "Create benchmark/tool_routing.json or update --dataset."
        )

    with path.open("r", encoding="utf-8") as handle:
        dataset = json.load(handle)

    if not isinstance(dataset, list):
        raise ValueError("Benchmark dataset must be a JSON list.")

    return dataset


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
    dataset: list[dict[str, Any]],
    server_path: Path,
    call_predicted_tools: bool,
) -> None:
    from models.qwen_router import HALLUCINATED_TOOL, choose_tool

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
            query = sample["query"]
            expected = sample["expected_tool"]

            start = time.perf_counter()
            predicted = choose_tool(query, available_tools)
            latency = time.perf_counter() - start

            latencies.append(latency)
            total += 1

            if predicted == expected:
                correct += 1

            if predicted == HALLUCINATED_TOOL:
                hallucinations += 1

            print(f"\nQuery: {query}")
            print(f"Expected: {expected}")
            print(f"Predicted: {predicted}")

            if call_predicted_tools and predicted != HALLUCINATED_TOOL:
                tool_args = sample.get("tool_args")
                if tool_args is None:
                    print("Tool call: skipped (no tool_args in dataset)")
                else:
                    call_result = await session.call_tool(predicted, tool_args)
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
        help="Path to the benchmark JSON dataset.",
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
    dataset = _load_dataset(args.dataset)
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
