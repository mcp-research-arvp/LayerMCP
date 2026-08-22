from __future__ import annotations

import inspect
import json
from pathlib import Path
import unittest

from evaluation.evaluate import load_benchmark
from mcp_server.finance_tools import finance_query_table
from mcp_server.server import mcp
from mcp_server.tool_impls import calculator


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_PATH = (
    PROJECT_ROOT
    / "benchmark"
    / "finance"
    / "finance_convfinqa_multistep.json"
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

    def test_ten_workflows_and_thirty_five_exact_conversation_turns_load(
        self,
    ) -> None:
        self.assertEqual(len(self.samples), 10)
        self.assertEqual(
            sum(len(sample.expected_steps) for sample in self.samples),
            35,
        )
        for sample in self.samples:
            self.assertEqual(sample.domain, "finance")
            self.assertEqual(sample.task_type, "multi_step_tool_routing")
            self.assertGreaterEqual(len(sample.expected_steps), 2)

    def test_expected_tools_are_in_the_full_registry_without_row_menus(self) -> None:
        registered_tools = set(mcp._tool_manager._tools)
        for row in self.raw_rows:
            with self.subTest(row=row["id"]):
                self.assertNotIn("available_tools", row)
                self.assertLessEqual(
                    {
                        step["expected_tool"]
                        for step in row["expected_steps"]
                    },
                    registered_tools,
                )

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
            6: {
                "turns": [
                    "what was the cash provided by operating activities in 2013?",
                    "and in 2012?",
                    "so what was the difference in this value between the years?",
                    "and the value for 2012 again?",
                    "so what was the percentage change during this time?",
                ],
                "programs": [
                    "6823",
                    "6161",
                    "subtract(6823, 6161)",
                    "6161",
                    "subtract(6823, 6161), divide(#0, 6161)",
                ],
                "answers": [6823.0, 6161.0, 662.0, 6161.0, 0.10745],
            },
            7: {
                "turns": [
                    "what is the amount of oil and gas mmboe from canada divided by the total?",
                    "what is that times 100?",
                ],
                "programs": [
                    "divide(60, 243)",
                    "divide(60, 243), multiply(#0, const_100)",
                ],
                "answers": [0.24691, 24.69136],
            },
            8: {
                "turns": [
                    "what was the total of risk and insurance brokerage services segment revenue in 2009?",
                    "and what was that in 2008?",
                    "what was, then, the change over the year?",
                    "and how much does this change represent in relation to the 2008 total, in percentage?",
                ],
                "programs": [
                    "6305",
                    "6197",
                    "subtract(6305, 6197)",
                    "subtract(6305, 6197), divide(#0, 6197)",
                ],
                "answers": [6305.0, 6197.0, 108.0, 0.01743],
            },
            9: {
                "turns": [
                    "what is the change in price of the s&p 500 from 2015 to 2016?",
                    "what is 100000 divided by 100?",
                    "what is the product of the change by the quotient?",
                ],
                "programs": [
                    "subtract(129.05, 110.28)",
                    "subtract(129.05, 110.28), divide(100000, const_100)",
                    "subtract(129.05, 110.28), divide(100000, const_100), multiply(#1, #0)",
                ],
                "answers": [18.77, 1000.0, 18770.0],
            },
            10: {
                "turns": [
                    "what is the ratio of fair value to carrying value?",
                    "what is that less 1?",
                ],
                "programs": [
                    "divide(5309, 4938)",
                    "divide(5309, 4938), subtract(#0, const_1)",
                ],
                "answers": [1.07513, 0.07513],
            },
            11: {
                "turns": [
                    "what was the number of gas customers in 2008?",
                    "and what was it in 2007?",
                    "what was, then, the change in that number over the year?",
                    "and how much does this change represent in relation to the 2007 number of customers, in percentage?",
                ],
                "programs": [
                    "93000",
                    "86000",
                    "subtract(93000, 86000)",
                    "subtract(93000, 86000), divide(#0, 86000)",
                ],
                "answers": [93000.0, 86000.0, 7000.0, 0.0814],
            },
            13: {
                "turns": [
                    "what was the difference in total shipment volume between 2010 and 2011?",
                    "and the specific value for 2010?",
                    "so what was the growth rate over this time?",
                ],
                "programs": [
                    "subtract(734.6, 724.4)",
                    "724.4",
                    "subtract(734.6, 724.4), divide(#0, 724.4)",
                ],
                "answers": [10.2, 724.4, 0.01408],
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
            self.assertEqual(
                row["final_program_execution_contract"],
                "convfinqa_program_execution",
            )
            self.assertEqual(
                row["expected_final_program_result"],
                expected["answers"][-1],
            )
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
        self.assertEqual(len(self.fixture["rows"]), 20)
        provenance = self.fixture["provenance"]
        self.assertEqual(provenance["source_revision"], SOURCE_REVISION)
        self.assertEqual(
            provenance["source_archive_sha256"],
            SOURCE_ARCHIVE_SHA256,
        )
        self.assertEqual(provenance["source_license"], "MIT")


if __name__ == "__main__":
    unittest.main()
