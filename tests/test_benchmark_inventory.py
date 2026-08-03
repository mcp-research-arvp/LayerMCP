from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from analysis.benchmark_inventory import (
    PROJECT_ROOT,
    build_inventory,
    discover_benchmark_files,
    infer_benchmark_class,
    summarize_file,
)
from evaluation.evaluate import DEFAULT_BENCHMARK_MODE, DEFAULT_WORKFLOW_EXECUTION_MODE


class BenchmarkInventoryTests(unittest.TestCase):
    def test_classification_logic(self) -> None:
        cases = [
            ("coding_replay.json", [{"benchmark_mode": "offline_trace_replay"}], "replay/offline"),
            ("math_smoke.json", [{}], "smoke"),
            ("math_controlled.json", [{"source": "controlled_synthetic"}], "controlled"),
            ("enterprise_tau2.json", [{"source": "public_adapted"}], "diagnostic/adapted"),
            ("finance_workflow.json", [{"task_type": "multi_step_tool_routing"}], "workflow"),
            ("math_public.json", [{"source": "public_math_derived"}], "public/source-derived"),
            ("math_misc.json", [{}], "unknown"),
        ]
        for name, rows, expected in cases:
            with self.subTest(name=name):
                self.assertEqual(infer_benchmark_class(Path(name), rows), expected)

    def test_classification_ignores_checkout_directory_names(self) -> None:
        rows = [{"source": "manual", "task_type": "single_tool_routing"}]
        smoke_checkout = Path("/tmp/smoke-review/benchmark/math/math_misc.json")
        public_checkout = Path("/tmp/public-review/benchmark/math/math_misc.json")
        self.assertEqual(
            infer_benchmark_class(smoke_checkout, rows),
            infer_benchmark_class(public_checkout, rows),
        )
        self.assertEqual(infer_benchmark_class(smoke_checkout, rows), "unknown")

    def test_counting_logic_on_synthetic_dataset(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            path = root / "benchmark" / "math" / "math_synthetic.json"
            path.parent.mkdir(parents=True)
            path.write_text(
                json.dumps(
                    [
                        {
                            "domain": "mathematics",
                            "task_type": "single_tool_routing",
                            "source": "public_derived",
                            "expected_tool": "calculator",
                            "expected_answer": {"result": 4},
                            "prompt_context": "grounding",
                        },
                        {
                            "domain": "mathematics",
                            "task_type": "multi_step_tool_routing",
                            "source": "public_derived",
                            "expected_answer": None,
                            "expected_steps": [
                                {
                                    "expected_tool": "calculator",
                                    "expected_answer": {"result": 2},
                                    "prompt_context": "step grounding",
                                    "depends_on": [],
                                },
                                {
                                    "expected_tool": "factor_expression",
                                    "depends_on": ["step_00"],
                                },
                            ],
                        },
                    ]
                ),
                encoding="utf-8",
            )
            summary = summarize_file(path, project_root=root)

            self.assertEqual(summary["row_count"], 2)
            self.assertEqual(summary["single_step_count"], 1)
            self.assertEqual(summary["workflow_count"], 1)
            self.assertEqual(summary["expected_step_count"], 2)
            self.assertEqual(
                summary["benchmark_mode_values"],
                [DEFAULT_BENCHMARK_MODE],
            )
            self.assertEqual(
                summary["workflow_execution_mode_values"],
                [DEFAULT_WORKFLOW_EXECUTION_MODE],
            )
            self.assertEqual(
                summary["top_level_expected_answer"],
                {"populated": 1, "null": 1, "missing": 0},
            )
            self.assertEqual(
                summary["step_level_expected_answer"],
                {"populated": 1, "null": 0, "missing": 1},
            )
            self.assertEqual(summary["depends_on_length_distribution"], {0: 1, 1: 1})
            self.assertEqual(
                summary["expected_tool_distribution"],
                {"calculator": 2, "factor_expression": 1},
            )

    def test_archive_fixtures_and_pycache_are_excluded(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            included = root / "benchmark" / "math" / "math_public.json"
            excluded = [
                root / "benchmark" / "archive" / "math" / "old.json",
                root / "benchmark" / "math" / "fixtures" / "fixture.json",
                root / "benchmark" / "math" / "__pycache__" / "cache.json",
            ]
            for path in [included, *excluded]:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("[]", encoding="utf-8")

            self.assertEqual(discover_benchmark_files(root), [included])

    def test_mixed_file_aggregates_each_row_once(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            path = root / "benchmark" / "coding" / "mixed.json"
            path.parent.mkdir(parents=True)
            path.write_text(
                json.dumps(
                    [
                        {
                            "domain": "coding",
                            "task_type": "single_tool_routing",
                            "source": "controlled_synthetic",
                            "expected_tool": "code_search_text",
                        },
                        {
                            "domain": "finance",
                            "task_type": "multi_step_tool_routing",
                            "source": "public_finance_workflow",
                            "expected_steps": [
                                {
                                    "expected_tool": "calculator",
                                    "expected_args": {"expression": "2 + 2"},
                                }
                            ],
                        },
                    ]
                ),
                encoding="utf-8",
            )

            inventory = build_inventory(root)
            aggregates = inventory["aggregates"]
            self.assertEqual(aggregates["total_active_rows"], 2)
            self.assertEqual(
                sum(value["row_count"] for value in aggregates["by_domain"].values()),
                2,
            )
            self.assertEqual(aggregates["by_domain"]["coding"]["row_count"], 1)
            self.assertEqual(aggregates["by_domain"]["finance"]["row_count"], 1)
            self.assertEqual(
                aggregates["by_task_type"]["single_tool_routing"]["row_count"],
                1,
            )
            self.assertEqual(
                aggregates["by_task_type"]["multi_step_tool_routing"]["row_count"],
                1,
            )

    def test_empty_placeholder_is_reported_but_not_active(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            active = root / "benchmark" / "coding" / "coding_smoke.json"
            placeholder = (
                root
                / "benchmark"
                / "coding"
                / "coding_nebius_swerebench_openhands_replay_multistep.json"
            )
            active.parent.mkdir(parents=True)
            active.write_text(
                json.dumps(
                    [
                        {
                            "domain": "coding",
                            "task_type": "single_tool_routing",
                            "expected_tool": "code_search_text",
                        }
                    ]
                ),
                encoding="utf-8",
            )
            placeholder.write_text("[]", encoding="utf-8")

            inventory = build_inventory(root)
            aggregates = inventory["aggregates"]
            self.assertEqual(aggregates["total_active_files"], 1)
            self.assertEqual(aggregates["total_placeholder_files"], 1)
            self.assertEqual(aggregates["total_active_rows"], 1)
            self.assertNotIn("<missing>", aggregates["by_task_type"])
            self.assertEqual(
                inventory["placeholders"][0]["path"],
                "benchmark/coding/coding_nebius_swerebench_openhands_replay_multistep.json",
            )

    def test_real_repository_inventory_loads(self) -> None:
        inventory = build_inventory(PROJECT_ROOT)
        self.assertGreater(inventory["aggregates"]["total_active_files"], 0)
        self.assertGreater(inventory["aggregates"]["total_active_rows"], 0)
        for item in inventory["files"]:
            self.assertNotIn("/archive/", item["path"])
            self.assertNotIn("/fixtures/", item["path"])
        self.assertNotIn("<missing>", inventory["aggregates"]["by_task_type"])
        placeholder_paths = {item["path"] for item in inventory["placeholders"]}
        self.assertIn(
            "benchmark/coding/coding_nebius_swerebench_openhands_replay_multistep.json",
            placeholder_paths,
        )

    def test_enterprise_public_workflows_counts_when_present(self) -> None:
        path = (
            PROJECT_ROOT
            / "benchmark"
            / "enterprise"
            / "enterprise_public_workflows.json"
        )
        if not path.exists():
            self.skipTest("enterprise_public_workflows.json is not present")
        summary = summarize_file(path)
        self.assertEqual(summary["workflow_count"], 69)
        self.assertEqual(summary["expected_step_count"], 350)


if __name__ == "__main__":
    unittest.main()
