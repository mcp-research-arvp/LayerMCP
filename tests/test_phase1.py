from __future__ import annotations

from pathlib import Path
import unittest

from evaluation.evaluate import load_benchmark
from mcp_server.tool_impls import (
    calculator,
    customer_lookup,
    github_search,
    read_code_file,
    stock_price_api,
    ticket_router,
    unit_converter,
)
from models.routers.qwen_hf_router import HALLUCINATED_TOOL, _extract_tool_name


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ToolFixtureTests(unittest.TestCase):
    def test_calculator_safe_arithmetic(self) -> None:
        self.assertEqual(calculator("2 + 3 * 4")["result"], 14)

    def test_calculator_rejects_unsafe_expressions(self) -> None:
        with self.assertRaises(ValueError):
            calculator("__import__('os').system('echo unsafe')")

    def test_stock_price_api_deterministic_output(self) -> None:
        self.assertEqual(stock_price_api("aapl")["price"], 214.35)

    def test_unit_converter_deterministic_output(self) -> None:
        result = unit_converter(10, "km", "miles")
        self.assertEqual(result["converted_value"], 6.2137)

    def test_customer_lookup_deterministic_output(self) -> None:
        self.assertEqual(customer_lookup("12345")["status"], "premium")

    def test_github_search_deterministic_output(self) -> None:
        result = github_search("authentication bugs")
        self.assertEqual(result["source"], "offline-fixture")
        self.assertIn("authentication bugs", result["results"][0]["title"])

    def test_read_code_file_deterministic_output(self) -> None:
        result = read_code_file("src/auth.py")
        self.assertIn("authenticate", result["content"])

    def test_ticket_router_deterministic_output(self) -> None:
        self.assertEqual(
            ticket_router("duplicate invoice charge")["category"],
            "billing",
        )


class BenchmarkLoaderTests(unittest.TestCase):
    def test_benchmark_loader_handles_old_schema(self) -> None:
        samples = load_benchmark(PROJECT_ROOT / "benchmark" / "tool_routing.json")
        self.assertEqual(samples[0].id, "sample_0001")
        self.assertEqual(samples[0].domain, "unknown")
        self.assertEqual(samples[0].expected_args, {"expression": "25 * 17"})

    def test_benchmark_loader_handles_new_schema(self) -> None:
        samples = load_benchmark(PROJECT_ROOT / "benchmark" / "tool_routing_smoke.json")
        self.assertEqual(samples[0].id, "smoke_finance_001")
        self.assertEqual(samples[0].domain, "finance")
        self.assertEqual(samples[0].expected_args, {"ticker": "AAPL"})

    def test_evaluator_can_load_all_benchmark_files(self) -> None:
        benchmark_paths = [
            PROJECT_ROOT / "benchmark" / "tool_routing.json",
            PROJECT_ROOT / "benchmark" / "tool_routing_smoke.json",
            PROJECT_ROOT / "benchmark" / "tool_routing_controlled.json",
        ]
        lengths = [len(load_benchmark(path)) for path in benchmark_paths]
        self.assertEqual(lengths, [4, 8, 40])


class RouterParserTests(unittest.TestCase):
    def test_router_parser_exact_tool_name_matching(self) -> None:
        self.assertEqual(
            _extract_tool_name("calculator", ["calculator", "github_search"]),
            "calculator",
        )

    def test_router_parser_json_like_tool_output(self) -> None:
        self.assertEqual(
            _extract_tool_name('{"tool": "calculator"}', ["calculator", "github_search"]),
            "calculator",
        )

    def test_router_parser_returns_hallucinated_tool_for_unknown_output(self) -> None:
        self.assertEqual(
            _extract_tool_name("use the calculator please", ["calculator", "github_search"]),
            HALLUCINATED_TOOL,
        )


if __name__ == "__main__":
    unittest.main()
