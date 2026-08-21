from __future__ import annotations

import inspect
import json
from collections import Counter
from pathlib import Path
import unittest

from benchmark.math.build_multistep_controlled import TOOLS, build_rows
from evaluation.evaluate import load_benchmark


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_PATH = (
    PROJECT_ROOT / "benchmark" / "math" / "math_multistep_controlled.json"
)


class MathMultistepBenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rows = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
        cls.samples = load_benchmark(BENCHMARK_PATH)

    def test_workflow_shape_and_counts(self) -> None:
        self.assertEqual(len(self.rows), 50)
        self.assertEqual(len(self.samples), 50)
        self.assertEqual(
            Counter(len(row["expected_steps"]) for row in self.rows),
            {2: 45, 3: 5},
        )
        self.assertEqual(
            sum(len(row["expected_steps"]) for row in self.rows),
            105,
        )

        for row in self.rows:
            with self.subTest(row=row["id"]):
                self.assertEqual(row["domain"], "mathematics")
                self.assertEqual(row["task_type"], "multi_step_tool_routing")
                self.assertEqual(row["source"], "controlled_synthetic")
                self.assertEqual(row["benchmark_mode"], "grounded_tool_execution")
                self.assertEqual(
                    row["workflow_final_answer_contract"],
                    "structured_tool_result_v1",
                )
                self.assertEqual(
                    row["workflow_final_answer_expected"],
                    row["expected_final_answer"],
                )
                self.assertGreaterEqual(len(row["expected_steps"]), 2)
                self.assertLessEqual(len(row["expected_steps"]), 5)
                self.assertTrue(row["query"].strip())
                self.assertNotIn("available_tools", row)

    def test_every_step_is_grounded_and_depends_on_prior_results(self) -> None:
        for row in self.rows:
            completed: set[str] = set()
            dependent_step_count = 0
            for index, step in enumerate(row["expected_steps"]):
                with self.subTest(row=row["id"], step=step["id"]):
                    self.assertEqual(step["id"], f"step_{index:02d}")
                    self.assertTrue(step["query"].strip())
                    self.assertIsInstance(step["expected_args"], dict)
                    self.assertIn("expected_answer", step)
                    self.assertIn(step["expected_tool"], TOOLS)
                    self.assertTrue(set(step["depends_on"]).issubset(completed))
                    if index == 0:
                        self.assertEqual(step["depends_on"], [])
                    if step["depends_on"]:
                        dependent_step_count += 1

                    context = json.loads(step["prompt_context"])
                    self.assertEqual(
                        context["kind"], "math_controlled_current_step_v1"
                    )
                    self.assertEqual(context["inputs"], step["expected_args"])
                    self.assertTrue(context["sequence_relation"].strip())
                    completed.add(step["id"])
            self.assertGreaterEqual(dependent_step_count, 1)

    def test_all_expected_calls_bind_and_execute_exactly(self) -> None:
        for row in self.rows:
            for step in row["expected_steps"]:
                with self.subTest(row=row["id"], step=step["id"]):
                    function = TOOLS[step["expected_tool"]]
                    inspect.signature(function).bind(**step["expected_args"])
                    self.assertEqual(
                        function(**step["expected_args"]),
                        step["expected_answer"],
                    )

    def test_tool_sequences_are_connected_and_varied(self) -> None:
        sequences = {
            tuple(step["expected_tool"] for step in row["expected_steps"])
            for row in self.rows
        }
        self.assertEqual(len(sequences), 10)
        self.assertIn(("calculator", "integer_factorization"), sequences)
        self.assertIn(("factor_expression", "solve_equation"), sequences)
        self.assertIn(("differentiate_expression", "solve_equation"), sequences)
        self.assertIn(("convert_units", "calculator"), sequences)
        self.assertIn(("base_arithmetic", "integer_factorization"), sequences)

    def test_committed_artifact_matches_builder(self) -> None:
        self.assertEqual(self.rows, build_rows())


if __name__ == "__main__":
    unittest.main()
