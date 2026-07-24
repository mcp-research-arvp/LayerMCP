from __future__ import annotations

import inspect
import json
from pathlib import Path
import unittest

from evaluation.evaluate import load_benchmark
from mcp_server.finance_tools import finance_query_table
from mcp_server.tool_impls import calculator


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_PATH = (
    PROJECT_ROOT
    / "benchmark"
    / "finance"
    / "tool_routing_finance_convfinqa_multistep.json"
)
FIXTURE_PATH = (
    PROJECT_ROOT
    / "benchmark"
    / "finance"
    / "fixtures"
    / "convfinqa_dev_cells.json"
)
SOURCE_REVISION = "cf3eed2d5984960bf06bb8145bcea5e80b0222a6"
SOURCE_ARCHIVE_SHA256 = (
    "d764271fae60d81b62e6d58dfc481807ebc8cfbcd633811241723c4a2101072a"
)
TOOL_FUNCTIONS = {
    "finance_query_table": finance_query_table,
    "calculator": calculator,
}


def _contains(actual: object, expected: object) -> bool:
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


class FinanceMultistepBenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.raw_rows = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
        cls.samples = load_benchmark(BENCHMARK_PATH)
        cls.fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))

    def test_three_workflows_and_twelve_exact_conversation_turns_load(self) -> None:
        self.assertEqual(len(self.samples), 3)
        self.assertEqual(
            sum(len(sample.expected_steps) for sample in self.samples),
            12,
        )
        for sample in self.samples:
            self.assertEqual(sample.domain, "finance")
            self.assertEqual(sample.task_type, "multi_step_tool_routing")
            self.assertGreaterEqual(len(sample.expected_steps), 2)

    def test_all_expected_steps_bind_and_execute_against_local_tools(self) -> None:
        for sample in self.samples:
            completed: set[str] = set()
            for step in sample.expected_steps:
                with self.subTest(sample=sample.id, step=step.id):
                    self.assertTrue(set(step.depends_on).issubset(completed))
                    function = TOOL_FUNCTIONS[step.expected_tool]
                    inspect.signature(function).bind(**step.expected_args)
                    result = function(**step.expected_args)
                    self.assertTrue(
                        _contains(result, step.expected_answer),
                        f"Expected answer mismatch for {sample.id}/{step.id}: "
                        f"{result!r}",
                    )
                    completed.add(step.id)

    def test_queries_programs_and_execution_answers_have_public_provenance(
        self,
    ) -> None:
        expected_rows = {
            0: {
                "turns": [
                    "what was the weighted average exercise price per share in 2007?",
                    "and what was it in 2005?",
                    "what was, then, the change over the years?",
                    "what was the weighted average exercise price per share in 2005?",
                    "and how much does that change represent in relation to this 2005 weighted average exercise price?",
                ],
                "programs": [
                    "60.94",
                    "25.14",
                    "subtract(60.94, 25.14)",
                    "25.14",
                    "subtract(60.94, 25.14), divide(#0, 25.14)",
                ],
                "answers": [60.94, 25.14, 35.8, 25.14, 1.42403],
            },
            2: {
                "turns": [
                    "what is the ratio of discretionary company contributions to total expensed amounts for savings plans in 2009?",
                    "what is that times 100?",
                ],
                "programs": [
                    "divide(3.8, 35.1)",
                    "divide(3.8, 35.1), multiply(#0, const_100)",
                ],
                "answers": [0.10826, 10.82621],
            },
            3: {
                "turns": [
                    "what was the equipment rents payable in 2008?",
                    "and in 2007?",
                    "so what was the difference between the two years?",
                    "and the value for 2007 again?",
                    "so what was the percentage change during this time?",
                ],
                "programs": [
                    "93",
                    "103",
                    "subtract(93, 103)",
                    "103",
                    "subtract(93, 103), divide(#0, 103)",
                ],
                "answers": [93.0, 103.0, -10.0, 103.0, -0.09709],
            },
        }

        for row in self.raw_rows:
            source_index = row["source_row_index"]
            expected = expected_rows[source_index]
            self.assertEqual(
                [step["query"] for step in row["expected_steps"]],
                expected["turns"],
            )
            self.assertEqual(
                [step["source_program"] for step in row["expected_steps"]],
                expected["programs"],
            )
            self.assertEqual(row["source_execution_answers"], expected["answers"])
            self.assertEqual(row["source_revision"], SOURCE_REVISION)
            self.assertEqual(row["source_license"], "MIT")
            self.assertEqual(
                row["query_origin"], "exact_public_dataset_dialogue"
            )
            self.assertEqual(
                row["tool_sequence_origin"],
                "mechanical_adaptation_of_gold_turn_programs",
            )

    def test_fixture_is_narrow_pinned_and_allowlisted(self) -> None:
        self.assertEqual(self.fixture["dataset_id"], "convfinqa-dev-v1")
        self.assertEqual(len(self.fixture["rows"]), 6)
        provenance = self.fixture["provenance"]
        self.assertEqual(provenance["source_revision"], SOURCE_REVISION)
        self.assertEqual(
            provenance["source_archive_sha256"],
            SOURCE_ARCHIVE_SHA256,
        )
        self.assertEqual(provenance["source_license"], "MIT")


if __name__ == "__main__":
    unittest.main()
