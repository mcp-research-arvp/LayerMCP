from __future__ import annotations

import inspect
import json
import re
import unittest
from collections import Counter
from pathlib import Path

from benchmark.finance.build_finqa_expansion import (
    FINQA_EXCLUSION_POLICY,
    INTENTIONALLY_EXCLUDED_LONG_CONTEXT_FINQA_SOURCE_INDICES,
)
from evaluation.evaluate import BenchmarkSample, load_benchmark
from mcp_server import finance_tools
from mcp_server.server import mcp
from mcp_server.tool_impls import calculator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FINANCE_BENCHMARK_ROOT = PROJECT_ROOT / "benchmark" / "finance"
FINQA_SINGLE_PATH = (
    FINANCE_BENCHMARK_ROOT / "finance_finqa_test_single.json"
)
FINQA_MULTISTEP_PATH = (
    FINANCE_BENCHMARK_ROOT / "finance_finqa_test_multistep.json"
)
FINRETRIEVAL_MULTISTEP_PATH = (
    FINANCE_BENCHMARK_ROOT / "finance_finretrieval_replay_multistep.json"
)
FINQA_FIXTURE_PATH = (
    FINANCE_BENCHMARK_ROOT / "fixtures" / "finqa_test_program_results_cells.json"
)
FINRETRIEVAL_FIXTURE_PATH = (
    FINANCE_BENCHMARK_ROOT / "fixtures" / "finretrieval_replay.json"
)

FINQA_DATASET_ID = "finqa-public-test-program-results-v1"
FINQA_SOURCE_REVISION = "0f16e2867befa6840783e58be38c9efb9229d742"
FINQA_TOTAL_TEST_ROWS = 1_147
FINRETRIEVAL_TOTAL_QUESTIONS = 500
UNSUPPORTED_FINRETRIEVAL_INDICES = {253, 455}
FINRETRIEVAL_OVER_STEP_LIMIT_INDICES = {
    12,
    29,
    38,
    75,
    108,
    122,
    321,
    322,
    359,
    377,
    466,
    480,
    486,
}
FINRETRIEVAL_MAX_BENCHMARK_STEPS = 5
FINRETRIEVAL_BENCHMARK_WORKFLOWS = 485
FINRETRIEVAL_BENCHMARK_CALLS = 1_490
FINRETRIEVAL_SELECTED_SOURCE_WORKFLOWS = 498
FINRETRIEVAL_SELECTED_SOURCE_CALLS = 1_608

FINRETRIEVAL_TOOLS = {
    "finance_discover_companies",
    "finance_discover_company_series",
    "finance_get_company_fundamentals",
    "finance_search_web_archive",
}
TOOL_FUNCTIONS = {
    "finance_query_table": finance_tools.finance_query_table,
    "calculator": calculator,
    **{name: getattr(finance_tools, name, None) for name in FINRETRIEVAL_TOOLS},
}


def _load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


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


def _program_operations(program: str) -> list[str]:
    return re.findall(r"([a-z_]+)\(", program)


def _assert_source_revision(test: unittest.TestCase, revision: object) -> str:
    test.assertIsInstance(revision, str)
    assert isinstance(revision, str)
    test.assertRegex(revision, r"^[0-9a-f]{40}$")
    return revision


class FinancePublicExpansionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.raw_finqa_single = _load_json(FINQA_SINGLE_PATH)
        cls.raw_finqa_multistep = _load_json(FINQA_MULTISTEP_PATH)
        cls.raw_finretrieval_multistep = _load_json(FINRETRIEVAL_MULTISTEP_PATH)
        cls.finqa_fixture = _load_json(FINQA_FIXTURE_PATH)
        cls.finretrieval_fixture = _load_json(FINRETRIEVAL_FIXTURE_PATH)

        cls.finqa_single = load_benchmark(FINQA_SINGLE_PATH)
        cls.finqa_multistep = load_benchmark(FINQA_MULTISTEP_PATH)
        cls.finretrieval_multistep = load_benchmark(FINRETRIEVAL_MULTISTEP_PATH)

    def test_expected_files_load_with_the_existing_evaluator(self) -> None:
        self.assertEqual(len(self.finqa_single), 642)
        self.assertEqual(len(self.finqa_multistep), 490)
        self.assertEqual(
            len(self.finretrieval_multistep),
            FINRETRIEVAL_BENCHMARK_WORKFLOWS,
        )
        self.assertEqual(
            len(self.finqa_single)
            + len(self.finqa_multistep)
            + len(self.finretrieval_multistep),
            1_617,
        )
        self.assertEqual(
            len(self.finqa_multistep) + len(self.finretrieval_multistep),
            975,
        )

        self.assertTrue(
            all(
                sample.task_type == "single_tool_routing" and not sample.expected_steps
                for sample in self.finqa_single
            )
        )
        for samples in (
            self.finqa_multistep,
            self.finretrieval_multistep,
        ):
            self.assertTrue(
                all(
                    sample.task_type == "multi_step_tool_routing"
                    and 2
                    <= len(sample.expected_steps)
                    <= FINRETRIEVAL_MAX_BENCHMARK_STEPS
                    for sample in samples
                )
            )

    def test_ids_are_unique_across_all_finance_benchmark_files(self) -> None:
        ids_by_path: dict[Path, list[str]] = {}
        benchmark_paths = sorted(FINANCE_BENCHMARK_ROOT.glob("finance_*.json"))
        self.assertEqual(
            [path.name for path in benchmark_paths],
            [
                "finance_controlled.json",
                "finance_convfinqa_multistep.json",
                "finance_finqa_test_multistep.json",
                "finance_finqa_test_single.json",
                "finance_finretrieval_replay_multistep.json",
                "finance_smoke.json",
                "finance_tatqa_public_derived.json",
                "finance_upstream_inspired.json",
            ],
        )
        for path in benchmark_paths:
            rows = _load_json(path)
            self.assertIsInstance(rows, list)
            assert isinstance(rows, list)
            ids_by_path[path] = [str(row["id"]) for row in rows]

        all_ids = [
            sample_id for path_ids in ids_by_path.values() for sample_id in path_ids
        ]
        duplicates = [
            sample_id for sample_id, count in Counter(all_ids).items() if count > 1
        ]
        self.assertEqual(duplicates, [])

    def test_finqa_active_scope_and_intentional_exclusions_partition_source(
        self,
    ) -> None:
        single_indices = {row["source_row_index"] for row in self.raw_finqa_single}
        multistep_indices = {
            row["source_row_index"] for row in self.raw_finqa_multistep
        }

        self.assertEqual(len(single_indices), len(self.raw_finqa_single))
        self.assertEqual(
            len(multistep_indices),
            len(self.raw_finqa_multistep),
        )
        self.assertTrue(single_indices.isdisjoint(multistep_indices))
        active_indices = single_indices | multistep_indices
        self.assertEqual(len(active_indices), 1_132)
        self.assertEqual(
            len(INTENTIONALLY_EXCLUDED_LONG_CONTEXT_FINQA_SOURCE_INDICES),
            15,
        )
        self.assertTrue(
            active_indices.isdisjoint(
                INTENTIONALLY_EXCLUDED_LONG_CONTEXT_FINQA_SOURCE_INDICES
            )
        )
        self.assertEqual(
            active_indices | INTENTIONALLY_EXCLUDED_LONG_CONTEXT_FINQA_SOURCE_INDICES,
            set(range(FINQA_TOTAL_TEST_ROWS)),
        )
        self.assertIn("intentionally", FINQA_EXCLUSION_POLICY.lower())
        self.assertIn("context", FINQA_EXCLUSION_POLICY.lower())

    def test_finqa_program_lengths_match_the_single_and_multicall_split(
        self,
    ) -> None:
        for row in self.raw_finqa_single:
            with self.subTest(row=row["id"]):
                operations = _program_operations(row["source_program"])
                self.assertEqual(len(operations), 1)
                self.assertEqual(
                    row["expected_tool"],
                    "finance_query_table",
                )
                self.assertEqual(
                    row["expected_args"]["dataset_id"],
                    FINQA_DATASET_ID,
                )

        operation_counts: Counter[int] = Counter()
        total_steps = 0
        for row in self.raw_finqa_multistep:
            with self.subTest(row=row["id"]):
                operations = _program_operations(row["source_program"])
                steps = row["expected_steps"]
                self.assertEqual(len(steps), len(operations))
                self.assertGreaterEqual(len(steps), 2)
                self.assertEqual(
                    row["final_program_execution_contract"],
                    "finqa_program_execution",
                )
                self.assertEqual(
                    row["expected_final_program_result"],
                    row["source_exe_answer"],
                )
                self.assertEqual(
                    steps[0]["expected_tool"],
                    "finance_query_table",
                )
                operation_counts[len(operations)] += 1
                total_steps += len(steps)

                for operation_index, (operation, step) in enumerate(
                    zip(operations, steps)
                ):
                    expected_tool = (
                        "finance_query_table"
                        if operation_index == 0
                        or operation == "greater"
                        or operation.startswith("table_")
                        else "calculator"
                    )
                    self.assertEqual(step["expected_tool"], expected_tool)
                    if expected_tool == "finance_query_table":
                        self.assertEqual(
                            step["expected_args"]["dataset_id"],
                            FINQA_DATASET_ID,
                        )

        self.assertEqual(
            operation_counts,
            Counter({2: 407, 3: 54, 4: 10, 5: 19}),
        )
        self.assertEqual(total_steps, 1_111)

    def test_expected_tools_are_in_the_full_registry_without_row_menus(self) -> None:
        registered_tools = set(mcp._tool_manager._tools)
        for row in [
            *self.raw_finqa_single,
            *self.raw_finqa_multistep,
            *self.raw_finretrieval_multistep,
        ]:
            with self.subTest(row=row["id"]):
                self.assertNotIn("available_tools", row)
                expected_tools = (
                    {row["expected_tool"]}
                    if "expected_tool" in row
                    else {
                        step["expected_tool"]
                        for step in row["expected_steps"]
                    }
                )
                self.assertLessEqual(expected_tools, registered_tools)

    def test_finretrieval_uses_each_supported_question_once(self) -> None:
        source_indices = [
            row["source_index"] for row in self.raw_finretrieval_multistep
        ]
        self.assertEqual(len(source_indices), len(set(source_indices)))
        self.assertEqual(
            set(source_indices),
            set(range(FINRETRIEVAL_TOTAL_QUESTIONS))
            - UNSUPPORTED_FINRETRIEVAL_INDICES
            - FINRETRIEVAL_OVER_STEP_LIMIT_INDICES,
        )
        self.assertTrue(UNSUPPORTED_FINRETRIEVAL_INDICES.isdisjoint(source_indices))
        self.assertTrue(
            FINRETRIEVAL_OVER_STEP_LIMIT_INDICES.isdisjoint(source_indices)
        )

        for row in self.raw_finretrieval_multistep:
            with self.subTest(row=row["id"]):
                self.assertEqual(
                    row["benchmark_mode"],
                    "offline_trace_replay",
                )
                steps = row["expected_steps"]
                self.assertGreaterEqual(len(steps), 2)
                self.assertLessEqual(
                    len(steps),
                    FINRETRIEVAL_MAX_BENCHMARK_STEPS,
                )
                self.assertEqual(
                    {step["expected_tool"] for step in steps} - FINRETRIEVAL_TOOLS,
                    set(),
                )
                self.assertEqual(
                    row["fixture_id"],
                    self.finretrieval_fixture["fixture_id"],
                )
                self.assertEqual(
                    row["source_num_tool_calls"],
                    len(steps),
                )
                self.assertTrue(row["source_trace_correct"])
                self.assertNotIn("final_program_execution_contract", row)
                self.assertNotIn("final_step_outcome_contract", row)
                for step_index, step in enumerate(steps):
                    self.assertFalse(step["source_is_error"])
                    self.assertEqual(
                        step["depends_on"],
                        ([] if step_index == 0 else [f"call_{step_index:03d}"]),
                    )
                    source_call = json.loads(step["source_program"])
                    self.assertEqual(
                        source_call["name"],
                        step["source_tool"],
                    )
                    self.assertIsInstance(source_call["input"], dict)

        total_steps = sum(
            len(row["expected_steps"]) for row in self.raw_finretrieval_multistep
        )
        replay_manifest = self.finretrieval_fixture["manifest"]
        self.assertEqual(
            replay_manifest["workflow_count"],
            FINRETRIEVAL_SELECTED_SOURCE_WORKFLOWS,
        )
        self.assertEqual(
            total_steps,
            FINRETRIEVAL_BENCHMARK_CALLS,
        )
        self.assertEqual(
            replay_manifest["selected_call_count"],
            FINRETRIEVAL_SELECTED_SOURCE_CALLS,
        )
        self.assertEqual(
            len(self.finretrieval_fixture["records"]),
            replay_manifest["replay_record_count"],
        )
        fixture_call_keys = {
            (
                record["tool"],
                json.dumps(
                    record["args"],
                    ensure_ascii=True,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            )
            for record in self.finretrieval_fixture["records"]
        }
        benchmark_call_keys = {
            (
                step["expected_tool"],
                json.dumps(
                    step["expected_args"],
                    ensure_ascii=True,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            )
            for row in self.raw_finretrieval_multistep
            for step in row["expected_steps"]
        }
        self.assertLessEqual(benchmark_call_keys, fixture_call_keys)

    def test_rows_and_fixtures_have_pinned_public_provenance(self) -> None:
        for row in [
            *self.raw_finqa_single,
            *self.raw_finqa_multistep,
        ]:
            with self.subTest(dataset="FinQA", row=row["id"]):
                self.assertEqual(row["source_dataset"], "FinQA")
                self.assertEqual(row["source_repository"], "czyssrs/FinQA")
                self.assertEqual(row["source_split"], "test")
                self.assertEqual(
                    row["source_revision"],
                    FINQA_SOURCE_REVISION,
                )
                self.assertEqual(row["source_license"], "MIT")
                self.assertEqual(
                    row["fixture_dataset_id"],
                    FINQA_DATASET_ID,
                )

        finretrieval_revisions = set()
        for row in self.raw_finretrieval_multistep:
            with self.subTest(dataset="FinRetrieval", row=row["id"]):
                self.assertEqual(row["source_dataset"], "FinRetrieval")
                self.assertEqual(
                    row["source_repository"],
                    "daloopa/finretrieval",
                )
                self.assertEqual(
                    row["source_split"],
                    "questions_and_selected_scored_tool_traces",
                )
                self.assertEqual(row["source_license"], "MIT")
                finretrieval_revisions.add(
                    _assert_source_revision(self, row["source_revision"])
                )
        self.assertEqual(len(finretrieval_revisions), 1)

        self.assertIsInstance(self.finqa_fixture, dict)
        self.assertEqual(
            self.finqa_fixture["dataset_id"],
            FINQA_DATASET_ID,
        )
        self.assertEqual(
            self.finqa_fixture["provenance"]["source_dataset"],
            "FinQA",
        )
        self.assertEqual(
            self.finqa_fixture["provenance"]["source_revision"],
            FINQA_SOURCE_REVISION,
        )
        self.assertEqual(
            self.finqa_fixture["provenance"]["source_license"],
            "MIT",
        )
        finqa_fixture_provenance = self.finqa_fixture["provenance"]
        self.assertEqual(
            finqa_fixture_provenance["excluded_source_row_indices"],
            sorted(INTENTIONALLY_EXCLUDED_LONG_CONTEXT_FINQA_SOURCE_INDICES),
        )
        self.assertEqual(
            finqa_fixture_provenance["source_example_count"],
            FINQA_TOTAL_TEST_ROWS,
        )
        self.assertEqual(
            finqa_fixture_provenance["included_example_count"],
            1_132,
        )
        self.assertEqual(
            finqa_fixture_provenance["single_operation_count"],
            642,
        )
        self.assertEqual(
            finqa_fixture_provenance["multi_operation_count"],
            490,
        )
        self.assertEqual(
            finqa_fixture_provenance["operation_row_count"],
            1_753,
        )
        self.assertEqual(len(self.finqa_fixture["rows"]), 1_753)
        self.assertEqual(
            [column["name"] for column in self.finqa_fixture["columns"]],
            [
                "source_split",
                "source_row_index",
                "source_id",
                "operation_index",
                "operation",
                "operator",
                "arg1",
                "arg2",
                "source_result",
                "numeric_result",
                "text_result",
                "is_final",
            ],
        )
        fixture_coordinates = [(row[1], row[3]) for row in self.finqa_fixture["rows"]]
        self.assertEqual(
            len(fixture_coordinates),
            len(set(fixture_coordinates)),
        )
        self.assertEqual(
            {source_index for source_index, _ in fixture_coordinates},
            {
                row["source_row_index"]
                for row in [
                    *self.raw_finqa_single,
                    *self.raw_finqa_multistep,
                ]
            },
        )

        self.assertIsInstance(self.finretrieval_fixture, dict)
        replay_manifest = self.finretrieval_fixture["manifest"]
        self.assertEqual(
            replay_manifest["source_dataset"],
            "FinRetrieval",
        )
        self.assertEqual(
            replay_manifest["source_repository"],
            "daloopa/finretrieval",
        )
        self.assertEqual(replay_manifest["source_license"], "MIT")
        self.assertEqual(
            replay_manifest["source_revision"],
            next(iter(finretrieval_revisions)),
        )
        self.assertEqual(
            replay_manifest["excluded_no_correct_trace_indices"],
            sorted(UNSUPPORTED_FINRETRIEVAL_INDICES),
        )
        self.assertFalse(replay_manifest["synthetic"])
        self.assertFalse(replay_manifest["network_access"])
        replay_records = self.finretrieval_fixture["records"]
        replay_keys = []
        replay_source_calls = []
        for record in replay_records:
            self.assertEqual(
                set(record),
                {"tool", "args", "result", "source_calls"},
            )
            self.assertIn(record["tool"], FINRETRIEVAL_TOOLS)
            self.assertIsInstance(record["args"], dict)
            self.assertIsInstance(record["source_calls"], list)
            self.assertTrue(record["source_calls"])
            replay_source_calls.extend(record["source_calls"])
            replay_keys.append(
                (
                    record["tool"],
                    json.dumps(
                        record["args"],
                        ensure_ascii=True,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                )
            )
        self.assertEqual(len(replay_keys), len(set(replay_keys)))
        self.assertEqual(
            len(replay_source_calls),
            replay_manifest["selected_call_count"],
        )
        source_call_coordinates = [
            (
                call["source_index"],
                call["step_index"],
                call["configuration"],
            )
            for call in replay_source_calls
        ]
        self.assertEqual(
            len(source_call_coordinates),
            len(set(source_call_coordinates)),
        )
        self.assertEqual(
            {call["source_index"] for call in replay_source_calls},
            set(range(FINRETRIEVAL_TOTAL_QUESTIONS))
            - UNSUPPORTED_FINRETRIEVAL_INDICES,
        )
        web_backed_indices = {
            call["source_index"]
            for record in replay_records
            if record["tool"] == "finance_search_web_archive"
            for call in record["source_calls"]
        }
        self.assertEqual(
            replay_manifest["web_backed_indices"],
            sorted(web_backed_indices),
        )

    def test_all_expected_calls_bind_and_execute_against_local_tools(
        self,
    ) -> None:
        missing_tools = sorted(
            name for name, function in TOOL_FUNCTIONS.items() if function is None
        )
        self.assertEqual(missing_tools, [])

        calls: list[tuple[str, str, dict[str, object], object]] = []
        for sample in self.finqa_single:
            calls.append(
                (
                    sample.id,
                    sample.expected_tool,
                    sample.expected_args,
                    sample.expected_answer,
                )
            )
        for sample in [
            *self.finqa_multistep,
            *self.finretrieval_multistep,
        ]:
            calls.extend(self._workflow_calls(sample))
        self.assertEqual(
            len(calls),
            1_753 + FINRETRIEVAL_BENCHMARK_CALLS,
        )

        for call_id, tool_name, arguments, expected_answer in calls:
            with self.subTest(call=call_id, tool=tool_name):
                function = TOOL_FUNCTIONS.get(tool_name)
                self.assertIsNotNone(function)
                assert function is not None
                inspect.signature(function).bind(**arguments)
                result = function(**arguments)
                self.assertTrue(
                    _contains(result, expected_answer),
                    f"Expected answer mismatch for {call_id}: {result!r}",
                )

    @staticmethod
    def _workflow_calls(
        sample: BenchmarkSample,
    ) -> list[tuple[str, str, dict[str, object], object]]:
        return [
            (
                f"{sample.id}/{step.id}",
                step.expected_tool,
                step.expected_args,
                step.expected_answer,
            )
            for step in sample.expected_steps
        ]


if __name__ == "__main__":
    unittest.main()
