from __future__ import annotations

import inspect
import json
from pathlib import Path
import unittest

from evaluation.evaluate import (
    MULTISTEP_CURRENT_STEP_CHAR_LIMIT,
    _multistep_query,
    load_benchmark,
)
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
    / "coding_sweagent_multistep.json"
)
FIXTURE_DIRECTORY = (
    PROJECT_ROOT
    / "benchmark"
    / "coding"
    / "fixtures"
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
        cls.fixtures = {}
        for path in FIXTURE_DIRECTORY.glob("sweagent_*.json"):
            fixture = json.loads(path.read_text(encoding="utf-8"))
            cls.fixtures[fixture["repo_id"]] = fixture

    def test_five_workflows_and_eleven_exact_read_only_actions_load(self) -> None:
        self.assertEqual(len(self.samples), 5)
        self.assertEqual(
            sum(len(sample.expected_steps) for sample in self.samples),
            11,
        )
        for row in self.raw_rows:
            self.assertNotIn("available_tools", row)
        sample = next(
            sample
            for sample in self.samples
            if sample.id == "coding_multistep_sweagent_marshmallow_1867"
        )
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

    def test_every_exact_action_has_visible_live_repository_grounding(self) -> None:
        grounded_steps = 0
        raw_rows_by_id = {row["id"]: row for row in self.raw_rows}
        for sample in self.samples:
            raw_steps = raw_rows_by_id[sample.id]["expected_steps"]
            for raw_step, step in zip(
                raw_steps,
                sample.expected_steps,
                strict=True,
            ):
                with self.subTest(sample=sample.id, step=step.id):
                    context = json.loads(raw_step["prompt_context"])
                    self.assertEqual(
                        context["kind"],
                        "coding_live_repository_call_v1",
                    )
                    self.assertEqual(
                        context["execution_mode"],
                        "live_allowlisted_repository",
                    )
                    self.assertEqual(
                        context["operation"],
                        "list_files"
                        if step.expected_tool == "code_list_files"
                        else "read_file",
                    )
                    self.assertEqual(context["arguments"], step.expected_args)
                    self.assertEqual(step.prompt_context, raw_step["prompt_context"])
                    self.assertLessEqual(
                        len(step.query),
                        MULTISTEP_CURRENT_STEP_CHAR_LIMIT,
                    )
                    self.assertLessEqual(
                        len(step.prompt_context),
                        MULTISTEP_CURRENT_STEP_CHAR_LIMIT,
                    )

                    routed_query = _multistep_query(sample, step, [])
                    self.assertIn(f"Current step: {step.query}", routed_query)
                    self.assertIn(
                        "Current-step grounding context: " + step.prompt_context,
                        routed_query,
                    )
                    self.assertNotIn("inert", step.prompt_context.lower())
                    grounded_steps += 1

        self.assertEqual(grounded_steps, 11)

    def test_fixture_and_benchmark_have_pinned_research_provenance(self) -> None:
        row = next(
            row
            for row in self.raw_rows
            if row["id"] == "coding_multistep_sweagent_marshmallow_1867"
        )
        provenance = self.fixtures[SWEAGENT_CODING_REPOSITORY_ID]["provenance"]
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
        for row in self.raw_rows:
            with self.subTest(row=row["id"]):
                fixture = self.fixtures[row["fixture_id"]]
                fixture_provenance = fixture["provenance"]
                self.assertEqual(
                    row["source_trajectory_revision"],
                    SWEAGENT_SOURCE_REVISION,
                )
                self.assertEqual(
                    row["source_trajectory_sha256"],
                    fixture_provenance["trajectory_sha256"],
                )
                self.assertEqual(
                    row["source_trajectory_step_indexes"],
                    [
                        action["trajectory_step_index"]
                        for action in fixture["source_actions"]
                    ],
                )
                self.assertEqual(
                    row["step_query_origin"],
                    "exact_official_sweagent_actions",
                )

    def test_fixtures_are_registered_as_allowlisted_repositories(self) -> None:
        for repo_id, fixture in self.fixtures.items():
            with self.subTest(repo_id=repo_id):
                repository = get_coding_repository(repo_id)
                self.assertEqual(repository.repo_id, repo_id)
                self.assertEqual(
                    repository.fixture_version,
                    fixture["fixture_version"],
                )
                self.assertFalse(repository.provenance.get("synthetic", False))
                self.assertIn(
                    repository.provenance["trajectory_origin"],
                    {
                        "official_sweagent_demonstration",
                        "official_sweagent_trajectory",
                    },
                )

        marshmallow = get_coding_repository(SWEAGENT_CODING_REPOSITORY_ID)
        self.assertEqual(
            marshmallow.fixture_version,
            SWEAGENT_CODING_FIXTURE_VERSION,
        )


if __name__ == "__main__":
    unittest.main()
