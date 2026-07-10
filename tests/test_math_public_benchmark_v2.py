from __future__ import annotations

from collections import Counter
import inspect
import json
import re
from pathlib import Path
import unittest

from evaluation.evaluate import load_benchmark
from mcp_server.math_tools import (
    base_arithmetic,
    convert_units,
    differentiate_expression,
    expand_expression,
    factor_expression,
    gcd_lcm,
    integer_factorization,
    modular_arithmetic,
    simplify_expression,
    solve_equation,
)
from mcp_server.server import mcp
from mcp_server.tool_impls import calculator


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PUBLIC_BENCHMARK_PATH = (
    PROJECT_ROOT / "benchmark" / "math" / "tool_routing_math_public_v2.json"
)
CONTROLLED_BENCHMARK_PATH = (
    PROJECT_ROOT / "benchmark" / "math" / "tool_routing_math_v2_controlled.json"
)
MATH_V2_MENU = [
    "calculator",
    "simplify_expression",
    "solve_equation",
    "factor_expression",
    "expand_expression",
    "differentiate_expression",
    "convert_units",
    "integer_factorization",
    "gcd_lcm",
    "modular_arithmetic",
    "base_arithmetic",
]
PUBLIC_EXPECTED_COUNTS = {
    "calculator": 10,
    "simplify_expression": 10,
    "solve_equation": 10,
    "factor_expression": 8,
    "expand_expression": 10,
    "integer_factorization": 8,
    "gcd_lcm": 8,
    "modular_arithmetic": 8,
    "base_arithmetic": 5,
}
CONTROLLED_EXPECTED_COUNTS = {
    "integer_factorization": 4,
    "gcd_lcm": 4,
    "modular_arithmetic": 4,
    "base_arithmetic": 4,
}
TOOL_FUNCTIONS = {
    "calculator": calculator,
    "simplify_expression": simplify_expression,
    "solve_equation": solve_equation,
    "factor_expression": factor_expression,
    "expand_expression": expand_expression,
    "differentiate_expression": differentiate_expression,
    "convert_units": convert_units,
    "integer_factorization": integer_factorization,
    "gcd_lcm": gcd_lcm,
    "modular_arithmetic": modular_arithmetic,
    "base_arithmetic": base_arithmetic,
}


def _contains(actual, expected) -> bool:
    if expected is None:
        return True
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(
            key in actual and _contains(actual[key], value)
            for key, value in expected.items()
        )
    if isinstance(expected, list):
        return isinstance(actual, list) and actual == expected
    return actual == expected


def _load_json(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


class MathPublicBenchmarkV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.public_samples = _load_json(PUBLIC_BENCHMARK_PATH)
        cls.controlled_samples = _load_json(CONTROLLED_BENCHMARK_PATH)

    def test_public_and_controlled_benchmarks_load_with_existing_loader(self) -> None:
        self.assertEqual(len(load_benchmark(PUBLIC_BENCHMARK_PATH)), 77)
        self.assertEqual(len(load_benchmark(CONTROLLED_BENCHMARK_PATH)), 16)

    def test_counts_and_unique_ids(self) -> None:
        all_samples = self.public_samples + self.controlled_samples
        ids = [sample["id"] for sample in all_samples]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(
            Counter(sample["expected_tool"] for sample in self.public_samples),
            PUBLIC_EXPECTED_COUNTS,
        )
        self.assertEqual(
            Counter(sample["expected_tool"] for sample in self.controlled_samples),
            CONTROLLED_EXPECTED_COUNTS,
        )

    def test_tools_arguments_and_expected_answers_are_valid(self) -> None:
        registered_tools = set(mcp._tool_manager._tools)
        for sample in self.public_samples + self.controlled_samples:
            expected_tool = sample["expected_tool"]
            self.assertIn(expected_tool, registered_tools)
            self.assertIn(expected_tool, sample["available_tools"])
            self.assertEqual(sample["available_tools"], MATH_V2_MENU)

            function = TOOL_FUNCTIONS[expected_tool]
            inspect.signature(function).bind(**sample["expected_args"])
            result = function(**sample["expected_args"])
            self.assertTrue(_contains(result, sample.get("expected_answer")))

    def test_public_provenance_fields_are_present_without_raw_dataset_dependency(self) -> None:
        for sample in self.public_samples:
            self.assertEqual(sample["source"], "public_math_derived")
            self.assertEqual(sample["source_dataset"], "math")
            self.assertIsInstance(sample["source_row_index"], int)
            self.assertIsInstance(sample["source_category"], str)
            self.assertIsInstance(sample["source_level"], str)
            self.assertGreater(len(sample["query"]), 0)

    def test_known_bad_mapping_patterns_are_not_present(self) -> None:
        for sample in self.public_samples:
            query = sample["query"].lower()
            expected_tool = sample["expected_tool"]
            args = sample["expected_args"]

            if re.search(r"\bmodulo\b|\bmod\b|units digit|inverse", query):
                self.assertEqual(expected_tool, "modular_arithmetic")
            if re.search(r"greatest common divisor|greatest common factor|\bgcd\b|\bgcf\b|least common multiple|\blcm\b", query):
                self.assertEqual(expected_tool, "gcd_lcm")
            if re.search(r"prime factor|prime-factor", query):
                self.assertEqual(expected_tool, "integer_factorization")
            if re.search(r"_\d|\bbase\s*\$?\d", query):
                self.assertEqual(expected_tool, "base_arithmetic")

            if expected_tool == "calculator":
                self.assertNotRegex(query, r"\bmodulo\b|\bgcd\b|\bgcf\b|\blcm\b|inverse|units digit|_\d")
                self.assertNotRegex(args["expression"].strip(), r"^\d+$")
            if expected_tool == "simplify_expression":
                expression = args["expression"].strip()
                self.assertFalse(
                    re.fullmatch(r"[A-Za-z]", expression)
                    and not re.search(r"simplify\s+\$?[A-Za-z]\$?", query),
                )
            if expected_tool == "factor_expression":
                self.assertNotRegex(
                    query,
                    r"gcd|gcf|lcm|greatest common|least common|prime factor|factorial|divisib",
                )


if __name__ == "__main__":
    unittest.main()
