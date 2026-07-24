from __future__ import annotations

import inspect
import json
from pathlib import Path
import unittest

from evaluation.evaluate import load_benchmark
from mcp_server.coding_state import (
    SWEAGENT_CODING_FIXTURE_VERSION,
    SWEAGENT_CODING_REPOSITORY_ID,
    SWEAGENT_MARSHMALLOW_BASE_COMMIT,
    SWEAGENT_SOURCE_REVISION,
    get_coding_repository,
)
from mcp_server.coding_tools import code_list_files, code_read_file


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_PATH = (
    PROJECT_ROOT
    / "benchmark"
    / "coding"
    / "tool_routing_coding_sweagent_multistep.json"
)
FIXTURE_PATH = (
    PROJECT_ROOT
    / "benchmark"
    / "coding"
    / "fixtures"
    / "sweagent_marshmallow_1867.json"
)
TRAJECTORY_SHA256 = (
    "8856076ec31832f20aefa7f0a2714e3ad6bc752f14815d94d2e852e50213a459"
)
TOOL_FUNCTIONS = {
    "code_list_files": code_list_files,
    "code_read_file": code_read_file,
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


class CodingMultistepBenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.raw_rows = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
        cls.samples = load_benchmark(BENCHMARK_PATH)
        cls.fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))

    def test_exact_three_step_read_only_trajectory_loads(self) -> None:
        self.assertEqual(len(self.samples), 1)
        sample = self.samples[0]
        self.assertEqual(sample.domain, "coding")
        self.assertEqual(sample.task_type, "multi_step_tool_routing")
        self.assertEqual(
            [step.query for step in sample.expected_steps],
            [
                "ls -F",
                'find_file "fields.py" src',
                "open src/marshmallow/fields.py 1474",
            ],
        )
        self.assertEqual(
            [step.id for step in sample.expected_steps],
            [
                "trajectory_action_000",
                "trajectory_action_007",
                "trajectory_action_008",
            ],
        )

    def test_all_expected_steps_bind_and_execute_against_fixture(self) -> None:
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

    def test_fixture_and_benchmark_have_pinned_research_provenance(self) -> None:
        row = self.raw_rows[0]
        provenance = self.fixture["provenance"]
        self.assertEqual(
            row["source_instance_id"],
            "marshmallow-code__marshmallow-1867",
        )
        self.assertEqual(row["source_trajectory_revision"], SWEAGENT_SOURCE_REVISION)
        self.assertEqual(row["source_trajectory_sha256"], TRAJECTORY_SHA256)
        self.assertEqual(
            row["source_repository_base_commit"],
            SWEAGENT_MARSHMALLOW_BASE_COMMIT,
        )
        self.assertEqual(
            row["source_trajectory_step_indexes"],
            [0, 7, 8],
        )
        self.assertEqual(row["query_origin"], "exact_swebench_issue")
        self.assertEqual(
            row["step_query_origin"],
            "exact_official_sweagent_actions",
        )
        self.assertEqual(provenance["source_revision"], SWEAGENT_SOURCE_REVISION)
        self.assertEqual(provenance["trajectory_sha256"], TRAJECTORY_SHA256)
        self.assertEqual(
            provenance["repository_base_commit"],
            SWEAGENT_MARSHMALLOW_BASE_COMMIT,
        )

    def test_fixture_is_registered_as_an_allowlisted_repository(self) -> None:
        repository = get_coding_repository(SWEAGENT_CODING_REPOSITORY_ID)
        self.assertEqual(repository.repo_id, SWEAGENT_CODING_REPOSITORY_ID)
        self.assertEqual(
            repository.fixture_version,
            SWEAGENT_CODING_FIXTURE_VERSION,
        )
        self.assertFalse(repository.provenance.get("synthetic", False))
        self.assertEqual(
            repository.provenance["trajectory_origin"],
            "official_sweagent_demonstration",
        )


if __name__ == "__main__":
    unittest.main()
