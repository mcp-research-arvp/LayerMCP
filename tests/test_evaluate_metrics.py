from __future__ import annotations

import unittest
from pathlib import Path

from evaluation.evaluate import (
    _answer_match,
    _build_aggregate_metrics,
    _exact_argument_match,
    _normalize_json,
    _score_sample,
    load_benchmark,
)


class EvaluateMetricTests(unittest.TestCase):
    def test_finance_fixture_context_is_loaded_without_gold_answer(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        samples = load_benchmark(
            project_root
            / "benchmark"
            / "finance"
            / "tool_routing_finance_tatqa_public_derived.json"
        )
        sample = next(
            item
            for item in samples
            if item.id == "finance_public_tatqa_query_table_010"
        )

        self.assertIsNotNone(sample.model_context)
        context = sample.model_context or ""
        self.assertIn("tatqa-public-test-gold-v1", context)
        self.assertIn("f1e5a9243865198ceddc74bd6a0b10b5", context)
        self.assertIn("PSAs granted", context)
        self.assertNotIn("136218", context)

    def test_correct_tool_and_correct_args(self) -> None:
        score = _score_sample(
            expected_tool="calculator",
            selected_tool="calculator",
            expected_args={"expression": "2 + 2"},
            selected_args={"expression": "2 + 2"},
            execution_success=True,
            execution_attempted=True,
        )

        self.assertTrue(score.tool_selection_correct)
        self.assertTrue(score.argument_match_correct)
        self.assertTrue(score.execution_success)
        self.assertEqual(score.failure_category, "correct")

    def test_correct_tool_wrong_args(self) -> None:
        score = _score_sample(
            expected_tool="calculator",
            selected_tool="calculator",
            expected_args={"expression": "2 + 2"},
            selected_args={"expression": "2 + 3"},
            execution_success=True,
            execution_attempted=True,
        )

        self.assertTrue(score.tool_selection_correct)
        self.assertFalse(score.argument_match_correct)
        self.assertTrue(score.execution_success)
        self.assertEqual(score.failure_category, "wrong_args")

    def test_wrong_tool(self) -> None:
        score = _score_sample(
            expected_tool="calculator",
            selected_tool="factor_expression",
            expected_args={"expression": "2 + 2"},
            selected_args={"expression": "2 + 2"},
            execution_success=True,
            execution_attempted=True,
        )

        self.assertFalse(score.tool_selection_correct)
        self.assertFalse(score.argument_match_correct)
        self.assertTrue(score.execution_success)
        self.assertEqual(score.failure_category, "wrong_tool")

    def test_no_tool_call(self) -> None:
        score = _score_sample(
            expected_tool="calculator",
            selected_tool=None,
            expected_args={"expression": "2 + 2"},
            selected_args={},
            execution_success=False,
            execution_attempted=False,
        )

        self.assertEqual(score.failure_category, "no_tool_call")
        self.assertFalse(score.tool_selection_correct)
        self.assertFalse(score.argument_match_correct)
        self.assertFalse(score.execution_success)

    def test_execution_error(self) -> None:
        score = _score_sample(
            expected_tool="calculator",
            selected_tool="calculator",
            expected_args={"expression": "2 + 2"},
            selected_args={"expression": "2 + 2"},
            execution_success=False,
            execution_attempted=True,
        )

        self.assertEqual(score.failure_category, "execution_error")
        self.assertTrue(score.tool_selection_correct)
        self.assertTrue(score.argument_match_correct)
        self.assertFalse(score.execution_success)

    def test_successful_execution_without_optional_execution_is_still_correct(self) -> None:
        score = _score_sample(
            expected_tool="calculator",
            selected_tool="calculator",
            expected_args={"expression": "2 + 2"},
            selected_args={"expression": "2 + 2"},
            execution_success=False,
            execution_attempted=False,
        )

        self.assertEqual(score.failure_category, "correct")
        self.assertFalse(score.execution_success)

    def test_confusion_matrix_and_per_tool_accuracy(self) -> None:
        records = [
            {
                "expected_tool": "calculator",
                "selected_tool": "calculator",
                "tool_selection_correct": True,
                "argument_match_correct": True,
                "execution_success": True,
                "failure_category": "correct",
            },
            {
                "expected_tool": "calculator",
                "selected_tool": "factor_expression",
                "tool_selection_correct": False,
                "argument_match_correct": False,
                "execution_success": True,
                "failure_category": "wrong_tool",
            },
            {
                "expected_tool": "factor_expression",
                "selected_tool": None,
                "tool_selection_correct": False,
                "argument_match_correct": False,
                "execution_success": False,
                "failure_category": "no_tool_call",
            },
        ]

        metrics = _build_aggregate_metrics(records)

        self.assertEqual(metrics["tool_selection_accuracy"], 1 / 3)
        self.assertEqual(metrics["exact_argument_match_accuracy"], 1 / 3)
        self.assertEqual(metrics["execution_success_rate"], 2 / 3)
        self.assertEqual(metrics["no_tool_call_rate"], 1 / 3)
        self.assertEqual(metrics["per_tool_accuracy"]["calculator"], 0.5)
        self.assertEqual(metrics["per_tool_accuracy"]["factor_expression"], 0.0)
        self.assertEqual(
            metrics["confusion_matrix"],
            {
                "calculator": {"calculator": 1, "factor_expression": 1},
                "factor_expression": {"no_tool_call": 1},
            },
        )

    def test_argument_normalization_ignores_object_key_order(self) -> None:
        left = {"b": 2, "a": {"y": 1, "x": 0}}
        right = {"a": {"x": 0, "y": 1}, "b": 2}

        self.assertEqual(_normalize_json(left), _normalize_json(right))
        self.assertTrue(_exact_argument_match(left, right))

    def test_answer_match_allows_extra_tool_metadata(self) -> None:
        actual = (
            '{"dataset_id":"finqa-public-test-v1","engine":"sqlite3",'
            '"columns":["result"],"rows":[[94.0]],"row_count":1,'
            '"truncated":false}'
        )
        expected = {
            "dataset_id": "finqa-public-test-v1",
            "columns": ["result"],
            "rows": [[94.0]],
            "row_count": 1,
            "truncated": False,
        }

        self.assertTrue(_answer_match(actual, expected))


if __name__ == "__main__":
    unittest.main()
