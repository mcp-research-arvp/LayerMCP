from __future__ import annotations

import asyncio
from collections import Counter
from copy import deepcopy
import hashlib
import inspect
import json
from pathlib import Path
import re
import unittest

from benchmark.coding.build_nebius_trajectory_expansion import (
    MAX_WORKFLOW_STEPS,
    _build_steps_and_records,
    _retained_benchmark_workflows,
)
from evaluation.evaluate import (
    MULTISTEP_CURRENT_STEP_CHAR_LIMIT,
    _multistep_query,
    load_benchmark,
)
import mcp_server.coding_replay_tools as replay_tools
from mcp_server.coding_replay_state import (
    CODING_REPLAY_FIXTURE_ID,
    CODING_REPLAY_FIXTURE_PATH,
    CODING_REPLAY_FIXTURE_PATHS,
    CODING_REPLAY_FIXTURE_VERSION,
    CODING_REPLAY_RECORD_COUNT,
    CODING_REPLAY_TOOL_NAMES,
    replay_coding_call,
    snapshot_coding_replay_state,
)
from mcp_server.coding_replay_tools import (
    code_replay_sweagent_file_edit,
    code_replay_sweagent_file_search,
    code_replay_sweagent_file_view,
    code_replay_sweagent_shell,
    code_replay_sweagent_submit,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODING_BENCHMARK_ROOT = PROJECT_ROOT / "benchmark" / "coding"
SWEAGENT_BENCHMARK_PATH = (
    CODING_BENCHMARK_ROOT
    / "tool_routing_coding_nebius_sweagent_multistep.json"
)
OPENHANDS_BENCHMARK_PATH = (
    CODING_BENCHMARK_ROOT
    / "tool_routing_coding_nebius_swerebench_openhands_multistep.json"
)
FIXTURE_PATH = (
    CODING_BENCHMARK_ROOT
    / "fixtures"
    / "coding_nebius_sweagent_benchmark_replay.json"
)

EXPECTED_WORKFLOW_COUNT = 33
EXPECTED_CALL_COUNT = 139
EXPECTED_FIXTURE_ID = "nebius-sweagent-benchmark-replay-v1"
EXPECTED_FIXTURE_VERSION = "coding_nebius_sweagent_benchmark_replay_v1"
EXPECTED_FIXTURE_BYTES = 224_669
EXPECTED_FIXTURE_SHA256 = (
    "bc70545d323424cd83b7d88eb8e6c0732cf200359ec83b584e13189f4b69b340"
)
EXPECTED_SOURCE_DATASET = "nebius/SWE-agent-trajectories"
EXPECTED_SOURCE_REVISION = "68195a1450865274106246d0d0296a1d6807b88e"
EXPECTED_SOURCE_FILE_HASHES = {
    "data/train-00000-of-00012.parquet": (
        "5a395e8c7bb8ddc4b8f4d268506b3a0e2cf9b5ec3922600117322fe788067a13"
    ),
    "data/train-00001-of-00012.parquet": (
        "fca106cce0f09891c2fadc032fb304da9ae5a7c31d2a39eb7ec70a7bdd4a9882"
    ),
    "data/train-00002-of-00012.parquet": (
        "fc28c2ab014c6c90d72026dda9cf8753e37b4b7b128c05c4232980cdcc99f3f7"
    ),
    "data/train-00003-of-00012.parquet": (
        "b6a4e13118de1792b383c077bf6023881a9ff5d4c171a75e767ffb5ad7c037c5"
    ),
    "data/train-00004-of-00012.parquet": (
        "d1a8d8d3bafbfd589d32a42d4b0a321f107d9aab170e3dc2f749b0f252a958c5"
    ),
    "data/train-00005-of-00012.parquet": (
        "c507212cd721512240a1bb87a8554bb7b429a9c03649c1473eb3d9e4673e6aca"
    ),
    "data/train-00006-of-00012.parquet": (
        "fcad81ec5704cb2dc8502a70a1a6cd86a7032227d8de7a613d904636ee53337c"
    ),
    "data/train-00007-of-00012.parquet": (
        "65547ce464ae1bbff550eacfcfec251ee1cb9b439744389a66bb97a8d5aa1cdb"
    ),
    "data/train-00008-of-00012.parquet": (
        "bfbd9a58fc9494b49ebd73d5f5e6836a72ae557cbc7ee59d2b2e3f91e3d44027"
    ),
    "data/train-00009-of-00012.parquet": (
        "8b4c45d8811f0fbbbc5fc46157312ddccb010c439d9e18289ae3ca3e5387ff09"
    ),
    "data/train-00010-of-00012.parquet": (
        "419e02daa099343e73d45d298b64e574ee012398ebbe19e5798526e5f1336b12"
    ),
    "data/train-00011-of-00012.parquet": (
        "7c3ccb843bab5457e29a82d9a36a3d59b8e1778c1d97324643c5f85a0d3f9492"
    ),
}

TOOL_FUNCTIONS = {
    "code_replay_sweagent_shell": code_replay_sweagent_shell,
    "code_replay_sweagent_file_view": code_replay_sweagent_file_view,
    "code_replay_sweagent_file_search": code_replay_sweagent_file_search,
    "code_replay_sweagent_file_edit": code_replay_sweagent_file_edit,
    "code_replay_sweagent_submit": code_replay_sweagent_submit,
}
OPENHANDS_TOOL_NAMES = {
    "code_replay_openhands_execute_bash",
    "code_replay_openhands_finish",
    "code_replay_openhands_task_tracker",
    "code_replay_openhands_str_replace_editor",
}
COORDINATE_ARGUMENTS = ("record_id", "trajectory_id", "step_index")
COORDINATE_ARGUMENT_SET = frozenset(COORDINATE_ARGUMENTS)
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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


class CodingPublicExpansionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        raw_benchmark = _load_json(SWEAGENT_BENCHMARK_PATH)
        raw_openhands = _load_json(OPENHANDS_BENCHMARK_PATH)
        raw_fixture = _load_json(FIXTURE_PATH)
        if not isinstance(raw_benchmark, list):
            raise TypeError("The SWE-agent benchmark must be a JSON list.")
        if not isinstance(raw_openhands, list):
            raise TypeError("The OpenHands benchmark must be a JSON list.")
        if not isinstance(raw_fixture, dict):
            raise TypeError("The SWE-agent replay fixture must be a JSON object.")
        raw_records = raw_fixture.get("records")
        if not isinstance(raw_records, list):
            raise TypeError("The SWE-agent replay fixture requires records.")

        cls.raw_benchmark = raw_benchmark
        cls.raw_openhands = raw_openhands
        cls.raw_fixture = raw_fixture
        cls.raw_records = raw_records
        cls.samples = load_benchmark(SWEAGENT_BENCHMARK_PATH)
        cls.openhands_samples = load_benchmark(OPENHANDS_BENCHMARK_PATH)
        cls.rows_by_id = {
            str(row["id"]): row
            for row in raw_benchmark
            if isinstance(row, dict)
        }
        cls.records_by_id = {
            str(record["record_id"]): record
            for record in raw_records
            if isinstance(record, dict)
        }

    def test_reduced_fixture_is_the_only_active_replay_source(self) -> None:
        self.assertEqual(CODING_REPLAY_FIXTURE_ID, EXPECTED_FIXTURE_ID)
        self.assertEqual(CODING_REPLAY_FIXTURE_VERSION, EXPECTED_FIXTURE_VERSION)
        self.assertEqual(CODING_REPLAY_RECORD_COUNT, EXPECTED_CALL_COUNT)
        self.assertEqual(CODING_REPLAY_FIXTURE_PATH.resolve(), FIXTURE_PATH.resolve())
        self.assertEqual(
            tuple(path.resolve() for path in CODING_REPLAY_FIXTURE_PATHS),
            (FIXTURE_PATH.resolve(),),
        )
        fixture_bytes = FIXTURE_PATH.read_bytes()
        self.assertEqual(len(fixture_bytes), EXPECTED_FIXTURE_BYTES)
        self.assertEqual(
            hashlib.sha256(fixture_bytes).hexdigest(),
            EXPECTED_FIXTURE_SHA256,
        )
        self.assertEqual(self.raw_openhands, [])
        self.assertEqual(self.openhands_samples, [])
        self.assertEqual(CODING_REPLAY_TOOL_NAMES, frozenset(TOOL_FUNCTIONS))
        self.assertTrue(CODING_REPLAY_TOOL_NAMES.isdisjoint(OPENHANDS_TOOL_NAMES))

    def test_only_bounded_multicall_workflows_are_retained(self) -> None:
        self.assertEqual(len(self.samples), EXPECTED_WORKFLOW_COUNT)
        self.assertEqual(
            sum(len(sample.expected_steps) for sample in self.samples),
            EXPECTED_CALL_COUNT,
        )
        self.assertEqual(
            Counter(len(sample.expected_steps) for sample in self.samples),
            Counter({3: 4, 4: 18, 5: 11}),
        )
        for sample in self.samples:
            with self.subTest(sample=sample.id):
                self.assertEqual(sample.domain, "coding")
                self.assertEqual(sample.task_type, "multi_step_tool_routing")
                self.assertGreaterEqual(len(sample.expected_steps), 2)
                self.assertLessEqual(len(sample.expected_steps), MAX_WORKFLOW_STEPS)

    def test_builder_applies_the_step_limit_without_mutating_inputs(self) -> None:
        source_workflows = [
            {"id": "two", "expected_steps": [{}, {}]},
            {"id": "five", "expected_steps": [{}] * 5},
            {"id": "six", "expected_steps": [{}] * 6},
        ]
        retained = _retained_benchmark_workflows(source_workflows)
        self.assertEqual([row["id"] for row in retained], ["two", "five"])
        self.assertEqual(len(source_workflows), 3)

    def test_builder_keeps_source_arguments_only_in_replay_records(self) -> None:
        source_args = {"command": "generated edit body\n" * 2_000}
        steps, records = _build_steps_and_records(
            dataset_prefix="sweagent",
            source_label="SWE-agent action",
            trajectory_id="trajectory-fixture",
            instance_id="instance-fixture",
            calls=[
                {
                    "source_message_index": 7,
                    "source_tool": "edit",
                    "normalized_tool": "code_replay_sweagent_file_edit",
                    "source_args": source_args,
                    "observation": {"full_sha256": "0" * 64},
                    "observation_redacted": False,
                }
            ],
        )
        self.assertEqual(len(steps), 1)
        self.assertEqual(len(records), 1)
        self.assertEqual(set(steps[0]["expected_args"]), COORDINATE_ARGUMENT_SET)
        self.assertNotIn("arguments=", steps[0]["query"])
        self.assertNotIn("arguments", steps[0]["expected_answer"])
        self.assertEqual(records[0]["args"], source_args)
        self.assertEqual(
            steps[0]["source_input_canonical_sha256"],
            _canonical_sha256(source_args),
        )

    def test_workflow_rows_and_steps_are_well_formed_and_public(self) -> None:
        workflow_ids: list[str] = []
        trajectory_ids: list[str] = []
        instance_ids: list[str] = []
        observed_tools: set[str] = set()

        for row, sample in zip(
            self.raw_benchmark,
            self.samples,
            strict=True,
        ):
            self.assertIsInstance(row, dict)
            assert isinstance(row, dict)
            workflow_ids.append(str(row["id"]))
            trajectory_ids.append(str(row["source_trajectory_id"]))
            instance_ids.append(str(row["source_instance_id"]))
            with self.subTest(sample=sample.id):
                self.assertEqual(row["id"], sample.id)
                self.assertEqual(row["source_dataset"], EXPECTED_SOURCE_DATASET)
                self.assertEqual(row["source_revision"], EXPECTED_SOURCE_REVISION)
                self.assertEqual(row["source_license"], "CC-BY-4.0")
                self.assertIs(row["source_success"], True)
                self.assertEqual(row["fixture_id"], EXPECTED_FIXTURE_ID)
                self.assertEqual(row["fixture_version"], EXPECTED_FIXTURE_VERSION)
                self.assertEqual(row["query_origin"], "extracted_public_issue_prompt")
                self.assertEqual(
                    row["tool_sequence_origin"],
                    "successful_released_sweagent_trajectory_actions",
                )
                self.assertNotIn("available_tools", row)
                self._assert_sha256(row["source_trajectory_sha256"])
                self._assert_sha256(row["source_full_prompt_sha256"])
                self._assert_sha256(row["source_issue_sha256"])

            raw_steps = row["expected_steps"]
            self.assertIsInstance(raw_steps, list)
            assert isinstance(raw_steps, list)
            completed: set[str] = set()
            source_step_indexes: list[int] = []
            for raw_step, step in zip(
                raw_steps,
                sample.expected_steps,
                strict=True,
            ):
                self.assertIsInstance(raw_step, dict)
                assert isinstance(raw_step, dict)
                with self.subTest(sample=sample.id, step=step.id):
                    self.assertTrue(set(step.depends_on).issubset(completed))
                    self.assertIn(step.expected_tool, CODING_REPLAY_TOOL_NAMES)
                    inspect.signature(TOOL_FUNCTIONS[step.expected_tool]).bind(
                        **step.expected_args
                    )
                    self.assertEqual(
                        set(step.expected_args),
                        COORDINATE_ARGUMENT_SET,
                    )
                    self.assertLessEqual(
                        len(step.query),
                        MULTISTEP_CURRENT_STEP_CHAR_LIMIT,
                    )
                    routed_query = _multistep_query(sample, step, [])
                    self.assertIn(f"Current step: {step.query}", routed_query)
                    for coordinate in COORDINATE_ARGUMENTS:
                        self.assertIn(
                            f"{coordinate}={step.expected_args[coordinate]}",
                            routed_query,
                        )
                    self.assertNotIn("arguments", raw_step["expected_answer"])
                    self.assertEqual(
                        step.expected_args["trajectory_id"],
                        row["source_trajectory_id"],
                    )
                    step_index = step.expected_args["step_index"]
                    self.assertIsInstance(step_index, int)
                    self.assertNotIsInstance(step_index, bool)
                    self.assertGreaterEqual(step_index, 0)
                    source_step_indexes.append(step_index)
                    self.assertIsInstance(raw_step["source_message_index"], int)
                    self.assertNotIsInstance(
                        raw_step["source_message_index"],
                        bool,
                    )
                    self.assertIsInstance(raw_step["source_tool"], str)
                    self._assert_sha256(
                        raw_step["source_input_canonical_sha256"]
                    )
                    self._assert_sha256(
                        raw_step["source_output_canonical_sha256"]
                    )
                    completed.add(step.id)
                    observed_tools.add(step.expected_tool)

            self.assertEqual(source_step_indexes, sorted(source_step_indexes))
            self.assertEqual(len(source_step_indexes), len(set(source_step_indexes)))

        self.assertEqual(len(workflow_ids), len(set(workflow_ids)))
        self.assertEqual(len(trajectory_ids), len(set(trajectory_ids)))
        self.assertEqual(len(instance_ids), len(set(instance_ids)))
        self.assertEqual(observed_tools, set(TOOL_FUNCTIONS))

    def test_fixture_records_exactly_match_benchmark_steps(self) -> None:
        benchmark_steps = [
            (row, raw_step, step)
            for row, sample in zip(
                self.raw_benchmark,
                self.samples,
                strict=True,
            )
            for raw_step, step in zip(
                row["expected_steps"],
                sample.expected_steps,
                strict=True,
            )
        ]
        benchmark_record_ids = [
            str(step.expected_args["record_id"])
            for _, _, step in benchmark_steps
        ]
        fixture_record_ids = [
            str(record["record_id"]) for record in self.raw_records
        ]

        self.assertEqual(len(benchmark_record_ids), EXPECTED_CALL_COUNT)
        self.assertEqual(len(benchmark_record_ids), len(set(benchmark_record_ids)))
        self.assertEqual(len(fixture_record_ids), EXPECTED_CALL_COUNT)
        self.assertEqual(len(fixture_record_ids), len(set(fixture_record_ids)))
        self.assertEqual(fixture_record_ids, benchmark_record_ids)
        self.assertEqual(set(fixture_record_ids), set(benchmark_record_ids))

        for (row, raw_step, step), record in zip(
            benchmark_steps,
            self.raw_records,
            strict=True,
        ):
            self.assertIsInstance(record, dict)
            assert isinstance(row, dict)
            assert isinstance(raw_step, dict)
            assert isinstance(record, dict)
            with self.subTest(sample=row["id"], record=record["record_id"]):
                self.assertEqual(record["record_id"], step.expected_args["record_id"])
                self.assertEqual(record["tool"], step.expected_tool)
                self.assertEqual(
                    record["trajectory_id"],
                    step.expected_args["trajectory_id"],
                )
                self.assertEqual(record["trajectory_id"], row["source_trajectory_id"])
                self.assertEqual(record["step_index"], step.expected_args["step_index"])
                self.assertIsInstance(record["args"], dict)
                self.assertEqual(
                    _canonical_sha256(record["args"]),
                    raw_step["source_input_canonical_sha256"],
                )
                self._assert_record_observation(record["observation"])
                source = record["source"]
                self.assertIsInstance(source, dict)
                assert isinstance(source, dict)
                self.assertEqual(
                    source["source_instance_id"],
                    row["source_instance_id"],
                )
                self.assertEqual(
                    source["source_message_index"],
                    raw_step["source_message_index"],
                )
                self.assertEqual(source["source_tool"], raw_step["source_tool"])
                self.assertEqual(
                    source["source_input_canonical_sha256"],
                    raw_step["source_input_canonical_sha256"],
                )
                self.assertEqual(
                    source["source_output_canonical_sha256"],
                    raw_step["source_output_canonical_sha256"],
                )

    def test_fixture_manifest_pins_source_and_reduced_scope(self) -> None:
        self.assertEqual(self.raw_fixture["fixture_id"], EXPECTED_FIXTURE_ID)
        self.assertEqual(
            self.raw_fixture["fixture_version"],
            EXPECTED_FIXTURE_VERSION,
        )
        manifest = self.raw_fixture["manifest"]
        self.assertIsInstance(manifest, dict)
        assert isinstance(manifest, dict)

        self.assertEqual(manifest["source_dataset"], EXPECTED_SOURCE_DATASET)
        self.assertIn(EXPECTED_SOURCE_DATASET, manifest["source_repository"])
        self.assertEqual(manifest["source_revision"], EXPECTED_SOURCE_REVISION)
        self.assertEqual(manifest["source_config"], "default")
        self.assertEqual(manifest["source_split"], "train")
        self.assertEqual(manifest["source_license"], "CC-BY-4.0")
        self.assertEqual(manifest["source_total_trajectory_count"], 80_036)
        self.assertEqual(manifest["source_successful_trajectory_count"], 13_389)
        self.assertEqual(manifest["source_unique_successful_instance_count"], 838)
        self.assertEqual(manifest["source_success_field"], "target")
        self.assertIs(manifest["source_success_value"], True)
        self.assertEqual(manifest["source_candidate_workflow_count"], 500)
        self.assertEqual(manifest["source_candidate_call_count"], 7_005)
        self.assertEqual(manifest["workflow_count"], EXPECTED_WORKFLOW_COUNT)
        self.assertEqual(manifest["selected_call_count"], EXPECTED_CALL_COUNT)
        self.assertEqual(manifest["replay_record_count"], EXPECTED_CALL_COUNT)
        self.assertEqual(manifest["benchmark_max_workflow_steps"], 5)
        self.assertEqual(
            manifest["record_scope"],
            "exact_benchmark_expected_step_record_ids",
        )
        self.assertIn("at most five calls", manifest["selection_rule"])
        self.assertEqual(
            manifest["selected_model_distribution"],
            {"swe-agent-llama-70b": 33},
        )
        self.assertEqual(
            manifest["selected_tool_distribution"],
            dict(Counter(record["tool"] for record in self.raw_records)),
        )
        self.assertEqual(manifest["omitted_reasoning_tools"], ["discussion"])
        self.assertIs(manifest["synthetic"], False)
        self.assertIs(manifest["network_access"], False)
        self.assertIs(manifest["process_execution"], False)
        self.assertIs(manifest["mutation_applied"], False)
        self.assertIs(manifest["teacher_forced_routing"], True)

        source_files = manifest["source_files"]
        self.assertIsInstance(source_files, list)
        self.assertEqual(
            {entry["path"]: entry["sha256"] for entry in source_files},
            EXPECTED_SOURCE_FILE_HASHES,
        )
        for entry in source_files:
            self._assert_sha256(entry["sha256"])

    def test_all_expected_calls_execute_as_inert_offline_replays(self) -> None:
        call_count = 0
        for row, sample in zip(
            self.raw_benchmark,
            self.samples,
            strict=True,
        ):
            assert isinstance(row, dict)
            for step in sample.expected_steps:
                with self.subTest(sample=sample.id, step=step.id):
                    function = TOOL_FUNCTIONS[step.expected_tool]
                    result = function(**step.expected_args)
                    self.assertTrue(_contains(result, step.expected_answer))
                    record = self.records_by_id[str(step.expected_args["record_id"])]
                    self.assertEqual(result["record_id"], record["record_id"])
                    self.assertEqual(result["tool"], record["tool"])
                    self.assertEqual(result["arguments"], record["args"])
                    self.assertEqual(result["observation"], record["observation"])
                    self.assertIs(result["offline_replay"], True)
                    self.assertIs(result["network_access"], False)
                    self.assertIs(result["process_executed"], False)
                    self.assertIs(result["mutation_applied"], False)
                    self.assertEqual(
                        result["provenance"],
                        {
                            "fixture_id": EXPECTED_FIXTURE_ID,
                            "fixture_version": EXPECTED_FIXTURE_VERSION,
                            "source": record["source"],
                        },
                    )
                    call_count += 1
        self.assertEqual(call_count, EXPECTED_CALL_COUNT)

    def test_unknown_or_mismatched_replay_coordinates_are_rejected(self) -> None:
        step = self.samples[0].expected_steps[0]
        expected_args = deepcopy(step.expected_args)
        tool = step.expected_tool
        invalid_calls = [
            lambda: replay_coding_call(
                tool,
                "unknown-record-id",
                str(expected_args["trajectory_id"]),
                int(expected_args["step_index"]),
            ),
            lambda: replay_coding_call(
                tool,
                str(expected_args["record_id"]),
                str(expected_args["trajectory_id"]) + "-mismatch",
                int(expected_args["step_index"]),
            ),
            lambda: replay_coding_call(
                tool,
                str(expected_args["record_id"]),
                str(expected_args["trajectory_id"]),
                int(expected_args["step_index"]) + 1,
            ),
            lambda: replay_coding_call(
                "code_replay_openhands_execute_bash",
                str(expected_args["record_id"]),
                str(expected_args["trajectory_id"]),
                int(expected_args["step_index"]),
            ),
        ]
        other_tool = next(name for name in TOOL_FUNCTIONS if name != tool)
        invalid_calls.append(
            lambda: replay_coding_call(
                other_tool,
                str(expected_args["record_id"]),
                str(expected_args["trajectory_id"]),
                int(expected_args["step_index"]),
            )
        )
        for call in invalid_calls:
            with self.subTest(call=call), self.assertRaises(ValueError):
                call()

    def test_replay_results_and_snapshots_are_deterministic_and_detached(self) -> None:
        record = self.raw_records[0]
        assert isinstance(record, dict)
        arguments = {
            "record_id": record["record_id"],
            "trajectory_id": record["trajectory_id"],
            "step_index": record["step_index"],
        }
        function = TOOL_FUNCTIONS[str(record["tool"])]
        expected_result = function(**arguments)
        mutated_result = function(**arguments)
        mutated_result["observation"]["excerpt"] = "mutated by test"
        mutated_result["provenance"]["source"]["test_mutation"] = True
        self.assertEqual(function(**arguments), expected_result)

        expected_snapshot = snapshot_coding_replay_state()
        mutated_snapshot = snapshot_coding_replay_state()
        mutated_snapshot["fixtures"][0]["manifest"]["workflow_count"] = -1
        mutated_snapshot["replay_tools"].append("mutated")
        self.assertEqual(snapshot_coding_replay_state(), expected_snapshot)

    def test_replay_snapshot_matches_the_reduced_fixture(self) -> None:
        snapshot = snapshot_coding_replay_state()
        self.assertEqual(snapshot["record_count"], EXPECTED_CALL_COUNT)
        self.assertEqual(snapshot["replay_tools"], sorted(TOOL_FUNCTIONS))
        self.assertEqual(
            snapshot["fixtures"],
            [
                {
                    "fixture_id": EXPECTED_FIXTURE_ID,
                    "fixture_version": EXPECTED_FIXTURE_VERSION,
                    "record_count": EXPECTED_CALL_COUNT,
                    "manifest": self.raw_fixture["manifest"],
                }
            ],
        )

    def test_tool_catalog_signatures_and_server_registration(self) -> None:
        for name, function in TOOL_FUNCTIONS.items():
            with self.subTest(tool=name):
                self.assertEqual(
                    list(inspect.signature(function).parameters),
                    list(COORDINATE_ARGUMENTS),
                )
        for name in OPENHANDS_TOOL_NAMES:
            self.assertFalse(hasattr(replay_tools, name))

        from mcp_server.server import mcp

        registered_tools = set(mcp._tool_manager._tools)
        self.assertTrue(set(TOOL_FUNCTIONS) <= registered_tools)
        self.assertTrue(OPENHANDS_TOOL_NAMES.isdisjoint(registered_tools))

        representatives: dict[str, dict[str, object]] = {}
        for record in self.raw_records:
            assert isinstance(record, dict)
            representatives.setdefault(
                str(record["tool"]),
                {
                    "record_id": record["record_id"],
                    "trajectory_id": record["trajectory_id"],
                    "step_index": record["step_index"],
                },
            )
        self.assertEqual(set(representatives), set(TOOL_FUNCTIONS))
        for tool_name, arguments in representatives.items():
            with self.subTest(tool=tool_name):
                result = asyncio.run(
                    mcp._tool_manager._tools[tool_name].run(arguments)
                )
                self.assertIsNotNone(result)

    def _assert_sha256(self, value: object) -> str:
        self.assertIsInstance(value, str)
        assert isinstance(value, str)
        self.assertRegex(value, SHA256_PATTERN)
        return value

    def _assert_record_observation(self, value: object) -> None:
        self.assertIsInstance(value, dict)
        assert isinstance(value, dict)
        self.assertTrue(
            {
                "available",
                "excerpt",
                "full_sha256",
                "original_chars",
                "truncated",
            }.issubset(value)
        )
        self.assertIsInstance(value["available"], bool)
        self.assertIsInstance(value["excerpt"], str)
        self._assert_sha256(value["full_sha256"])
        self.assertIsInstance(value["original_chars"], int)
        self.assertNotIsInstance(value["original_chars"], bool)
        self.assertIsInstance(value["truncated"], bool)
        excerpt = value["excerpt"]
        assert isinstance(excerpt, str)
        self.assertLessEqual(len(excerpt), value["original_chars"])
        if not value["available"]:
            self.assertEqual(excerpt, "")
            self.assertEqual(value["original_chars"], 0)
            self.assertFalse(value["truncated"])
        if not value["truncated"]:
            self.assertEqual(len(excerpt), value["original_chars"])
            self.assertEqual(
                hashlib.sha256(excerpt.encode("utf-8")).hexdigest(),
                value["full_sha256"],
            )


if __name__ == "__main__":
    unittest.main()
