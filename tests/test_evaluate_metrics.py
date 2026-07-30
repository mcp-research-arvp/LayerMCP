from __future__ import annotations

from dataclasses import fields
from types import SimpleNamespace
import unittest

from evaluation.evaluate import (
    BenchmarkSample,
    FINAL_OUTCOME_MATCHER,
    _build_aggregate_metrics,
    _exact_argument_match,
    _extract_structured_tool_result,
    _final_outcome_record_fields,
    _is_no_tool_call,
    _match_expected_answer,
    _normalize_json,
    _route_sample,
    _score_sample,
    _score_final_outcome,
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

    def test_nested_json_subset_match_allows_extra_actual_fields(self) -> None:
        match = _match_expected_answer(
            {
                "order": {
                    "id": "ORD-1",
                    "status": "shipped",
                    "internal_note": "ignored",
                },
                "source": "fixture",
            },
            {"order": {"id": "ORD-1", "status": "shipped"}},
            domain="enterprise_automation",
        )

        self.assertTrue(match.matched)
        self.assertIsNone(match.diagnostic)

    def test_missing_expected_key_is_rejected_with_path_diagnostic(self) -> None:
        match = _match_expected_answer(
            {"order": {"id": "ORD-1"}},
            {"order": {"status": "shipped"}},
            domain="enterprise_automation",
        )

        self.assertFalse(match.matched)
        self.assertIn("$.order.status", match.diagnostic)
        self.assertIn("missing", match.diagnostic)

    def test_integer_match_is_exact(self) -> None:
        self.assertTrue(
            _match_expected_answer(4, 4, domain="mathematics").matched
        )
        self.assertFalse(
            _match_expected_answer(5, 4, domain="mathematics").matched
        )

    def test_float_numeric_tolerance(self) -> None:
        self.assertTrue(
            _match_expected_answer(
                1.0000005,
                1.0,
                domain="finance",
            ).matched
        )
        self.assertFalse(
            _match_expected_answer(
                1.00001,
                1.0,
                domain="finance",
            ).matched
        )

    def test_boolean_is_not_treated_as_numeric(self) -> None:
        self.assertFalse(
            _match_expected_answer(1, True, domain="enterprise_automation").matched
        )
        self.assertFalse(
            _match_expected_answer(True, 1, domain="enterprise_automation").matched
        )

    def test_nested_finance_table_uses_numeric_tolerance(self) -> None:
        match = _match_expected_answer(
            {
                "columns": ["result"],
                "rows": [[0.1446400001]],
                "row_count": 1,
                "truncated": False,
            },
            {
                "columns": ["result"],
                "rows": [[0.14464]],
                "row_count": 1,
                "truncated": False,
            },
            domain="finance",
        )

        self.assertTrue(match.matched)

    def test_symbolically_equivalent_math_field_is_accepted(self) -> None:
        match = _match_expected_answer(
            {"expanded": "(x + 1) * (x - 1)"},
            {"expanded": "x**2 - 1"},
            domain="mathematics",
        )

        self.assertTrue(match.matched)

    def test_different_math_expression_is_rejected(self) -> None:
        match = _match_expected_answer(
            {"derivative": "3*x"},
            {"derivative": "2*x"},
            domain="mathematics",
        )

        self.assertFalse(match.matched)
        self.assertIn("$.derivative", match.diagnostic)

    def test_missing_expected_answer_is_unscored(self) -> None:
        score = _score_final_outcome(
            expected_answer=None,
            tool_result_value={"result": 4},
            result_extraction_diagnostic=None,
            domain="mathematics",
            call_predicted_tools=True,
            no_tool_call=False,
            execution_success=True,
        )

        self.assertIsNone(score.correct)
        self.assertEqual(score.status, "missing_expected_answer")

    def test_execution_disabled_is_unscored(self) -> None:
        score = _score_final_outcome(
            expected_answer={"result": 4},
            tool_result_value=None,
            result_extraction_diagnostic=None,
            domain="mathematics",
            call_predicted_tools=False,
            no_tool_call=False,
            execution_success=False,
        )

        self.assertIsNone(score.correct)
        self.assertEqual(score.status, "execution_disabled")

    def test_no_tool_with_expected_answer_counts_incorrect(self) -> None:
        score = _score_final_outcome(
            expected_answer={"result": 4},
            tool_result_value=None,
            result_extraction_diagnostic=None,
            domain="mathematics",
            call_predicted_tools=True,
            no_tool_call=True,
            execution_success=False,
        )

        self.assertFalse(score.correct)
        self.assertEqual(score.status, "no_tool_call")

    def test_execution_error_with_expected_answer_counts_incorrect(self) -> None:
        score = _score_final_outcome(
            expected_answer={"result": 4},
            tool_result_value=None,
            result_extraction_diagnostic=None,
            domain="mathematics",
            call_predicted_tools=True,
            no_tool_call=False,
            execution_success=False,
        )

        self.assertFalse(score.correct)
        self.assertEqual(score.status, "execution_error")

    def test_result_extraction_error_counts_incorrect(self) -> None:
        score = _score_final_outcome(
            expected_answer={"result": 4},
            tool_result_value=None,
            result_extraction_diagnostic="No result content.",
            domain="mathematics",
            call_predicted_tools=True,
            no_tool_call=False,
            execution_success=True,
        )

        self.assertFalse(score.correct)
        self.assertEqual(score.status, "result_extraction_error")

    def test_summary_denominator_includes_no_call_and_execution_error(self) -> None:
        records = [
            self._aggregate_record(True),
            self._aggregate_record(False, status="no_tool_call"),
            self._aggregate_record(False, status="execution_error"),
        ]

        metrics = _build_aggregate_metrics(records)

        self.assertEqual(metrics["final_outcome_correct_samples"], 1)
        self.assertEqual(metrics["final_outcome_scored_samples"], 3)
        self.assertEqual(metrics["final_outcome_gold_samples"], 3)
        self.assertEqual(metrics["final_outcome_accuracy"], 1 / 3)
        self.assertEqual(metrics["final_outcome_gold_coverage"], 1.0)

    def test_summary_accuracy_is_null_when_nothing_is_scored(self) -> None:
        record = self._aggregate_record(None, status="execution_disabled")
        metrics = _build_aggregate_metrics([record])

        self.assertIsNone(metrics["final_outcome_accuracy"])
        self.assertEqual(metrics["final_outcome_scored_samples"], 0)
        self.assertEqual(metrics["final_outcome_gold_samples"], 1)

    def test_structured_mcp_result_extraction(self) -> None:
        extraction = _extract_structured_tool_result(
            SimpleNamespace(
                structuredContent={"result": 4},
                content=[SimpleNamespace(text='{"result": 999}')],
            )
        )

        self.assertEqual(extraction.value, {"result": 4})
        self.assertIsNone(extraction.diagnostic)

    def test_json_text_fallback_extraction(self) -> None:
        extraction = _extract_structured_tool_result(
            SimpleNamespace(
                structuredContent=None,
                content=[SimpleNamespace(text='{"result": 4}')],
            )
        )

        self.assertEqual(extraction.value, {"result": 4})
        self.assertIsNone(extraction.diagnostic)

    def test_plain_text_fallback_extraction(self) -> None:
        extraction = _extract_structured_tool_result(
            SimpleNamespace(
                structuredContent=None,
                content=[SimpleNamespace(text="plain file contents")],
            )
        )

        self.assertEqual(extraction.value, "plain file contents")
        self.assertIsNone(extraction.diagnostic)

    def test_sample_result_contains_all_final_outcome_metadata(self) -> None:
        expected_answer = {"result": 4}
        tool_result_value = {"result": 4, "source": "fixture"}
        sample_result = {
            "expected_answer": expected_answer,
            "tool_result_value": tool_result_value,
            **_final_outcome_record_fields(
                _score_final_outcome(
                    expected_answer=expected_answer,
                    tool_result_value=tool_result_value,
                    result_extraction_diagnostic=None,
                    domain="mathematics",
                    call_predicted_tools=True,
                    no_tool_call=False,
                    execution_success=True,
                )
            ),
        }

        self.assertTrue(
            {
                "expected_answer",
                "tool_result_value",
                "final_outcome_correct",
                "final_outcome_status",
                "final_outcome_matcher",
                "final_outcome_diagnostic",
            }.issubset(sample_result)
        )
        self.assertEqual(sample_result["expected_answer"], expected_answer)
        self.assertEqual(sample_result["tool_result_value"], tool_result_value)
        self.assertTrue(sample_result["final_outcome_correct"])
        self.assertEqual(sample_result["final_outcome_status"], "correct")
        self.assertEqual(
            sample_result["final_outcome_matcher"],
            FINAL_OUTCOME_MATCHER,
        )
        self.assertIsNone(sample_result["final_outcome_diagnostic"])

    @staticmethod
    def _aggregate_record(
        final_outcome_correct,
        *,
        status: str = "correct",
    ):
        return {
            "expected_tool": "calculator",
            "selected_tool": "calculator",
            "tool_selection_correct": True,
            "argument_match_correct": True,
            "execution_success": final_outcome_correct is True,
            "failure_category": status,
            "expected_answer": {"result": 4},
            "final_outcome_correct": final_outcome_correct,
            "final_outcome_status": status,
        }

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
