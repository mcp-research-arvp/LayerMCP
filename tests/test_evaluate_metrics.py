from __future__ import annotations

from dataclasses import fields
from types import SimpleNamespace
import unittest

from evaluation.evaluate import (
    BenchmarkSample,
    _build_aggregate_metrics,
    _exact_argument_match,
    _is_no_tool_call,
    _normalize_json,
    _route_sample,
    _score_sample,
    _tool_pool_metadata,
    _validate_expected_tools,
)


class EvaluateMetricTests(unittest.TestCase):
    @staticmethod
    def _sample(expected_tool: str = "calculator") -> BenchmarkSample:
        return BenchmarkSample(
            id="sample-1",
            domain="mathematics",
            task_type="single_tool_routing",
            difficulty="easy",
            source="test",
            query="Calculate 2 + 2.",
            expected_tool=expected_tool,
            expected_args={"expression": "2 + 2"},
            expected_answer={"result": 4},
            perturbation_type="none",
            notes="",
        )

    def test_benchmark_sample_has_no_row_level_tool_catalog(self) -> None:
        self.assertNotIn("available_tools", {field.name for field in fields(BenchmarkSample)})

    def test_router_receives_full_live_catalog_for_every_sample(self) -> None:
        live_tools = ["calculator", "factor_expression", "customer_lookup"]
        schemas = {tool: {"type": "object"} for tool in live_tools}
        descriptions = {tool: f"{tool} description" for tool in live_tools}

        class FakeRouter:
            SUPPORTS_STRUCTURED_TOOL_DESCRIPTIONS = True

            def __init__(self) -> None:
                self.calls = []

            def choose_tool_call(self, query, tools, tool_schemas, tool_descriptions):
                self.calls.append((query, tools, tool_schemas, tool_descriptions))
                return SimpleNamespace(
                    selected_tool="calculator",
                    selected_args={"expression": "2 + 2"},
                    raw_output='{"name":"calculator","arguments":{"expression":"2 + 2"}}',
                    parse_status="ok",
                    attempted_tool="calculator",
                    diagnostic=None,
                )

        router = FakeRouter()
        for sample in (self._sample(), self._sample()):
            _route_sample(router, sample, live_tools, schemas, descriptions)

        self.assertEqual(len(router.calls), 2)
        for _, tools, received_schemas, received_descriptions in router.calls:
            self.assertIs(tools, live_tools)
            self.assertEqual(received_schemas, schemas)
            self.assertEqual(received_descriptions, descriptions)

    def test_unregistered_predicted_tool_is_rejected(self) -> None:
        self.assertTrue(
            _is_no_tool_call(
                "invented_tool",
                "hallucinated_tool",
                {"calculator"},
            )
        )
        self.assertFalse(
            _is_no_tool_call(
                "calculator",
                "hallucinated_tool",
                {"calculator"},
            )
        )

    def test_expected_tool_must_be_registered(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing_tool"):
            _validate_expected_tools(
                [self._sample(expected_tool="missing_tool")],
                {"calculator"},
            )

    def test_full_registry_metadata_is_recorded_for_samples_and_summaries(self) -> None:
        metadata = _tool_pool_metadata(
            ["calculator", "factor_expression", "customer_lookup"]
        )
        sample_record = {"sample_id": "sample-1", **metadata}
        summary_record = {"total_samples": 1, **metadata}

        for record in (sample_record, summary_record):
            self.assertEqual(record["tool_pool"], "full_mcp_registry")
            self.assertEqual(record["tool_count"], 3)

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


if __name__ == "__main__":
    unittest.main()
