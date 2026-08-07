from __future__ import annotations

from dataclasses import fields, replace
import hashlib
import json
from types import SimpleNamespace
import unittest

from evaluation.evaluate import (
    DEFAULT_BENCHMARK_MODE,
    DEFAULT_WORKFLOW_EXECUTION_MODE,
    EVALUATION_PROTOCOL_DESCRIPTIONS,
    BenchmarkSample,
    BenchmarkStep,
    FINAL_OUTCOME_MATCHER,
    FINANCE_QUERY_TABLE_RESULT_MATCHER,
    MULTISTEP_EVALUATION_PROTOCOL,
    MULTISTEP_CURRENT_STEP_CHAR_LIMIT,
    MULTISTEP_HISTORY_ITEM_CHAR_LIMIT,
    MULTISTEP_HISTORY_STEP_LIMIT,
    MULTISTEP_OVERALL_TASK_CHAR_LIMIT,
    PROMPT_CONTEXT_CHAR_LIMIT,
    REFERENCE_PREFIX_REPLAY_MODE,
    SINGLE_STEP_EVALUATION_PROTOCOL,
    TOOL_REGISTRY_FINGERPRINT_VERSION,
    _build_aggregate_metrics,
    _build_multistep_metrics,
    _bounded_prompt_text,
    _exact_argument_match,
    _extract_structured_tool_result,
    _final_outcome_record_fields,
    _gold_history_item,
    _is_no_tool_call,
    _match_expected_answer,
    _multistep_query,
    _normalize_benchmark_mode,
    _normalize_json,
    _normalize_sample,
    _normalize_workflow_execution_mode,
    _query_with_context,
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

    def test_single_step_prompt_context_is_visible_to_the_router(self) -> None:
        sample = replace(
            self._sample(),
            prompt_context="dataset_id=finance-public-v1; schema=result REAL",
        )
        received_queries: list[str] = []

        class FakeRouter:
            SUPPORTS_STRUCTURED_TOOL_DESCRIPTIONS = True

            def choose_tool_call(
                self,
                query,
                tools,
                tool_schemas,
                tool_descriptions,
            ):
                received_queries.append(query)
                return SimpleNamespace(
                    selected_tool="calculator",
                    selected_args={"expression": "2 + 2"},
                    raw_output="",
                    parse_status="ok",
                    attempted_tool="calculator",
                    diagnostic=None,
                )

        routed = _query_with_context(sample.query, sample.prompt_context)
        _route_sample(
            FakeRouter(),
            sample,
            ["calculator"],
            {"calculator": {"type": "object"}},
            {"calculator": "Evaluate an expression."},
        )

        self.assertIn(sample.query, routed)
        self.assertIn("Grounding context:", routed)
        self.assertIn("dataset_id=finance-public-v1", routed)
        self.assertEqual(received_queries, [routed])

    def test_multistep_prompt_includes_overall_and_step_grounding(self) -> None:
        sample = replace(
            self._sample(),
            task_type="multi_step_tool_routing",
            prompt_context="repository_id=repo-1",
        )
        step = BenchmarkStep(
            id="step-1",
            query="open src/main.py",
            expected_tool="code_read_file",
            expected_args={"repo_id": "repo-1", "path": "src/main.py"},
            expected_answer=None,
            depends_on=(),
            source_program=None,
            prompt_context="read lines 10 through 20",
        )

        routed = _multistep_query(sample, step, [])

        self.assertIn("Overall grounding context: repository_id=repo-1", routed)
        self.assertIn(
            "Current-step grounding context: read lines 10 through 20",
            routed,
        )

    def test_prompt_context_must_be_a_bounded_string(self) -> None:
        raw_sample = {
            "id": "sample-with-context",
            "domain": "finance",
            "task_type": "single_tool_routing",
            "query": "Run the grounded lookup.",
            "expected_tool": "calculator",
            "expected_args": {"expression": "2 + 2"},
        }

        with self.assertRaisesRegex(ValueError, "prompt_context must be a string"):
            _normalize_sample({**raw_sample, "prompt_context": {"hidden": "id"}}, 0)
        with self.assertRaisesRegex(ValueError, "prompt_context exceeds"):
            _normalize_sample(
                {**raw_sample, "prompt_context": "x" * (PROMPT_CONTEXT_CHAR_LIMIT + 1)},
                0,
            )

    def test_benchmark_mode_defaults_for_backwards_compatibility(self) -> None:
        sample = _normalize_sample(
            {
                "id": "legacy-row",
                "domain": "coding",
                "task_type": "single_tool_routing",
                "query": "Read the file.",
                "expected_tool": "calculator",
            },
            0,
        )

        self.assertEqual(sample.benchmark_mode, DEFAULT_BENCHMARK_MODE)
        self.assertEqual(
            _normalize_benchmark_mode(None, "Sample 0"),
            "grounded_tool_execution",
        )

    def test_offline_replay_benchmark_mode_is_preserved(self) -> None:
        sample = _normalize_sample(
            {
                "id": "replay-row",
                "domain": "coding",
                "task_type": "single_tool_routing",
                "query": "Replay the recorded call.",
                "expected_tool": "calculator",
                "benchmark_mode": "offline_trace_replay",
            },
            0,
        )

        self.assertEqual(sample.benchmark_mode, "offline_trace_replay")

    def test_unknown_benchmark_mode_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "benchmark_mode must be one of"):
            _normalize_benchmark_mode("autonomous_agent", "Sample 0")
        with self.assertRaisesRegex(ValueError, "benchmark_mode must be a string"):
            _normalize_benchmark_mode({"mode": "replay"}, "Sample 0")

    def test_workflow_execution_mode_is_explicit_and_validated(self) -> None:
        self.assertEqual(
            _normalize_workflow_execution_mode(None, "Sample 0"),
            DEFAULT_WORKFLOW_EXECUTION_MODE,
        )
        sample = _normalize_sample(
            {
                "id": "retail-reference-workflow",
                "domain": "enterprise_automation",
                "task_type": "single_tool_routing",
                "query": "Route the current reference action.",
                "expected_tool": "get_order_details",
                "workflow_execution_mode": REFERENCE_PREFIX_REPLAY_MODE,
            },
            0,
        )
        self.assertEqual(
            sample.workflow_execution_mode,
            REFERENCE_PREFIX_REPLAY_MODE,
        )
        with self.assertRaisesRegex(
            ValueError,
            "workflow_execution_mode must be one of",
        ):
            _normalize_workflow_execution_mode("predicted_sequence", "Sample 0")

    def test_workflow_final_answer_gold_is_preserved_without_being_scored(self) -> None:
        sample = _normalize_sample(
            {
                "id": "workflow-answer-gold",
                "domain": "finance",
                "task_type": "single_tool_routing",
                "query": "Run the lookup.",
                "expected_tool": "calculator",
                "expected_args": {"expression": "6 * 7"},
                "expected_final_answer": "42",
            },
            0,
        )

        self.assertEqual(sample.expected_final_answer, "42")

    def test_multistep_prompt_keeps_old_declared_dependencies(self) -> None:
        sample = replace(self._sample(), task_type="multi_step_tool_routing")
        step = BenchmarkStep(
            id="step-5",
            query="Use the result from step 1.",
            expected_tool="calculator",
            expected_args={"expression": "40 + 2"},
            expected_answer={"result": 42},
            depends_on=("step-1",),
            source_program=None,
        )
        history = [
            {"step_id": f"step-{index}", "expected_answer": {"result": index}}
            for index in range(1, 5)
        ]

        routed = _multistep_query(sample, step, history)

        self.assertIn('"step_id": "step-1"', routed)
        self.assertNotIn('"step_id": "step-2"', routed)
        self.assertIn('"step_id": "step-3"', routed)
        self.assertIn('"step_id": "step-4"', routed)

    def test_gold_history_is_independent_of_model_predictions(self) -> None:
        step = BenchmarkStep(
            id="step-1",
            query="Look up the source row.",
            expected_tool="finance_query_table",
            expected_args={"dataset_id": "public-v1", "sql": "SELECT 1"},
            expected_answer={"rows": [{"1": 1}]},
            depends_on=(),
            source_program=None,
        )

        self.assertEqual(
            _gold_history_item(step),
            {
                "step_id": "step-1",
                "query": "Look up the source row.",
                "expected_tool": "finance_query_table",
                "expected_args": {
                    "dataset_id": "public-v1",
                    "sql": "SELECT 1",
                },
                "expected_answer": {"rows": [{"1": 1}]},
            },
        )

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

    def test_every_multistep_expected_tool_must_be_registered(self) -> None:
        sample = replace(
            self._sample(),
            task_type="multi_step_tool_routing",
            expected_steps=(
                BenchmarkStep(
                    id="step-1",
                    query="Calculate 2 + 2.",
                    expected_tool="calculator",
                    expected_args={"expression": "2 + 2"},
                    expected_answer={"result": 4},
                    depends_on=(),
                    source_program=None,
                ),
                BenchmarkStep(
                    id="step-2",
                    query="Use a missing tool.",
                    expected_tool="missing_tool",
                    expected_args={},
                    expected_answer=None,
                    depends_on=("step-1",),
                    source_program=None,
                ),
            ),
        )

        with self.assertRaisesRegex(ValueError, "missing_tool"):
            _validate_expected_tools([sample], {"calculator"})

    def test_full_registry_metadata_is_recorded_for_samples_and_summaries(self) -> None:
        live_tools = ["factor_expression", "calculator", "customer_lookup"]
        schemas = {
            "calculator": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
            },
            "customer_lookup": {"type": "object"},
            "factor_expression": {"type": "object"},
        }
        descriptions = {
            "calculator": "Evaluate an expression.",
            "customer_lookup": "Find a customer.",
            "factor_expression": "Factor an expression.",
        }
        metadata = _tool_pool_metadata(live_tools, schemas, descriptions)
        sample_record = {"sample_id": "sample-1", **metadata}
        summary_record = {"total_samples": 1, **metadata}

        for record in (sample_record, summary_record):
            self.assertEqual(record["tool_pool"], "full_mcp_registry")
            self.assertEqual(record["tool_count"], 3)
            self.assertEqual(
                record["tool_names"],
                ["calculator", "customer_lookup", "factor_expression"],
            )

        registry_payload = [
            {
                "name": name,
                "schema": schemas[name],
                "description": descriptions[name],
            }
            for name in sorted(live_tools)
        ]
        encoded_payload = json.dumps(
            registry_payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        expected_fingerprint = "sha256:" + hashlib.sha256(
            encoded_payload
        ).hexdigest()
        self.assertEqual(
            metadata["tool_registry_fingerprint"],
            expected_fingerprint,
        )
        self.assertEqual(
            metadata["tool_registry_fingerprint_version"],
            TOOL_REGISTRY_FINGERPRINT_VERSION,
        )
        self.assertEqual(
            metadata,
            _tool_pool_metadata(list(reversed(live_tools)), schemas, descriptions),
        )

        changed_descriptions = {**descriptions, "calculator": "Changed."}
        self.assertNotEqual(
            metadata["tool_registry_fingerprint"],
            _tool_pool_metadata(
                live_tools,
                schemas,
                changed_descriptions,
            )["tool_registry_fingerprint"],
        )

    def test_evaluation_protocol_labels_are_explicit(self) -> None:
        self.assertEqual(
            SINGLE_STEP_EVALUATION_PROTOCOL,
            "single_step_tool_routing_v1",
        )
        self.assertEqual(
            MULTISTEP_EVALUATION_PROTOCOL,
            "teacher_forced_step_routing_v1",
        )
        self.assertIn(
            "does not evaluate autonomous end-to-end planning",
            EVALUATION_PROTOCOL_DESCRIPTIONS[MULTISTEP_EVALUATION_PROTOCOL],
        )

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

    def test_finance_query_result_ignores_column_alias(self) -> None:
        score = _score_final_outcome(
            expected_answer={
                "dataset_id": "finqa-public-test-program-results-v1",
                "columns": ["result"],
                "rows": [[0.25]],
                "row_count": 1,
                "truncated": False,
            },
            tool_result_value={
                "dataset_id": "finqa-public-test-program-results-v1",
                "columns": ["numeric_result"],
                "rows": [[0.25]],
                "row_count": 1,
                "truncated": False,
            },
            result_extraction_diagnostic=None,
            domain="finance",
            call_predicted_tools=True,
            no_tool_call=False,
            execution_success=True,
            expected_tool="finance_query_table",
            called_tool="finance_query_table",
        )

        self.assertTrue(score.correct)
        self.assertEqual(score.matcher, FINANCE_QUERY_TABLE_RESULT_MATCHER)

    def test_finance_query_result_rejects_extra_rows(self) -> None:
        score = _score_final_outcome(
            expected_answer={
                "dataset_id": "finqa-public-test-program-results-v1",
                "columns": ["result"],
                "rows": [[0.25]],
                "row_count": 1,
                "truncated": False,
            },
            tool_result_value={
                "dataset_id": "finqa-public-test-program-results-v1",
                "columns": ["numeric_result"],
                "rows": [[0.25], [0.5]],
                "row_count": 2,
                "truncated": False,
            },
            result_extraction_diagnostic=None,
            domain="finance",
            call_predicted_tools=True,
            no_tool_call=False,
            execution_success=True,
            expected_tool="finance_query_table",
            called_tool="finance_query_table",
        )

        self.assertFalse(score.correct)
        self.assertIn("$.rows", score.diagnostic)

    def test_finance_query_result_rejects_missing_columns(self) -> None:
        score = _score_final_outcome(
            expected_answer={
                "dataset_id": "finqa-public-test-program-results-v1",
                "columns": ["result"],
                "rows": [],
                "row_count": 0,
                "truncated": False,
            },
            tool_result_value={
                "dataset_id": "finqa-public-test-program-results-v1",
                "rows": [],
                "row_count": 0,
                "truncated": False,
            },
            result_extraction_diagnostic=None,
            domain="finance",
            call_predicted_tools=True,
            no_tool_call=False,
            execution_success=True,
            expected_tool="finance_query_table",
            called_tool="finance_query_table",
        )

        self.assertFalse(score.correct)
        self.assertIn("$.columns", score.diagnostic)

    def test_finance_query_result_rejects_wrong_column_count(self) -> None:
        score = _score_final_outcome(
            expected_answer={
                "dataset_id": "finqa-public-test-program-results-v1",
                "columns": ["result"],
                "rows": [],
                "row_count": 0,
                "truncated": False,
            },
            tool_result_value={
                "dataset_id": "finqa-public-test-program-results-v1",
                "columns": ["first", "second"],
                "rows": [],
                "row_count": 0,
                "truncated": False,
            },
            result_extraction_diagnostic=None,
            domain="finance",
            call_predicted_tools=True,
            no_tool_call=False,
            execution_success=True,
            expected_tool="finance_query_table",
            called_tool="finance_query_table",
        )

        self.assertFalse(score.correct)
        self.assertIn("$.columns", score.diagnostic)

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

    def test_summary_reports_benchmark_modes_separately(self) -> None:
        grounded = self._aggregate_record(True)
        replay = {
            **self._aggregate_record(False, status="wrong_tool"),
            "benchmark_mode": "offline_trace_replay",
            "selected_tool": "factor_expression",
            "tool_selection_correct": False,
            "argument_match_correct": False,
        }

        metrics = _build_aggregate_metrics([grounded, replay, replay])

        self.assertEqual(
            metrics["benchmark_mode_counts"],
            {
                "grounded_tool_execution": 1,
                "offline_trace_replay": 2,
            },
        )
        by_mode = metrics["metrics_by_benchmark_mode"]
        self.assertEqual(
            by_mode["grounded_tool_execution"]["tool_selection_accuracy"],
            1.0,
        )
        self.assertEqual(
            by_mode["offline_trace_replay"]["tool_selection_accuracy"],
            0.0,
        )
        self.assertNotIn(
            "metrics_by_benchmark_mode",
            by_mode["grounded_tool_execution"],
        )

    def test_multistep_metrics_are_broken_down_by_benchmark_mode(self) -> None:
        workflow_records = [
            {
                "benchmark_mode": "grounded_tool_execution",
                "sequence_tool_selection_correct": True,
                "sequence_argument_match_correct": True,
                "sequence_semantic_output_correct": True,
                "expected_final_answer": "done",
            },
            {
                "benchmark_mode": "offline_trace_replay",
                "sequence_tool_selection_correct": False,
                "sequence_argument_match_correct": False,
                "sequence_semantic_output_correct": False,
                "expected_final_answer": None,
            },
        ]
        step_records = [
            {
                "benchmark_mode": "grounded_tool_execution",
                "tool_selection_correct": True,
                "argument_match_correct": True,
                "final_outcome_correct": True,
            },
            {
                "benchmark_mode": "grounded_tool_execution",
                "tool_selection_correct": False,
                "argument_match_correct": False,
                "final_outcome_correct": None,
            },
            {
                "benchmark_mode": "offline_trace_replay",
                "tool_selection_correct": False,
                "argument_match_correct": False,
                "final_outcome_correct": False,
            },
        ]

        metrics = _build_multistep_metrics(workflow_records, step_records)
        workflow_by_mode = metrics["workflow_metrics_by_benchmark_mode"]
        steps_by_mode = metrics["step_metrics_by_benchmark_mode"]

        self.assertEqual(metrics["workflow_exact_sequence_accuracy"], 0.5)
        self.assertEqual(metrics["step_tool_selection_accuracy"], 1 / 3)
        self.assertEqual(
            workflow_by_mode["grounded_tool_execution"][
                "workflow_exact_sequence_accuracy"
            ],
            1.0,
        )
        self.assertEqual(
            workflow_by_mode["offline_trace_replay"][
                "workflow_exact_sequence_accuracy"
            ],
            0.0,
        )
        self.assertEqual(
            steps_by_mode["grounded_tool_execution"][
                "step_tool_selection_accuracy"
            ],
            0.5,
        )
        self.assertEqual(
            steps_by_mode["offline_trace_replay"][
                "step_tool_selection_accuracy"
            ],
            0.0,
        )
        self.assertNotIn(
            "step_metrics_by_benchmark_mode",
            steps_by_mode["grounded_tool_execution"],
        )

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

    def test_multistep_prompt_bounds_are_explicit_and_hash_preserving(self) -> None:
        value = "0123456789" * 100
        bounded = _bounded_prompt_text(value, 240)

        self.assertEqual(len(bounded), 240)
        self.assertIn("original_chars=1000", bounded)
        self.assertIn(
            "sha256=ab6c5f3237f551d208fc2ca5225a4cca20b3fd63"
            "8794a804f0ed5549d5041734",
            bounded,
        )
        self.assertEqual(
            _bounded_prompt_text("short", 240),
            "short",
        )
        self.assertEqual(MULTISTEP_HISTORY_STEP_LIMIT, 2)
        self.assertGreater(MULTISTEP_OVERALL_TASK_CHAR_LIMIT, 0)
        self.assertGreater(MULTISTEP_CURRENT_STEP_CHAR_LIMIT, 0)
        self.assertGreater(MULTISTEP_HISTORY_ITEM_CHAR_LIMIT, 0)


if __name__ == "__main__":
    unittest.main()
