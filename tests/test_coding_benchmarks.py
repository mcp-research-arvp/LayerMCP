from __future__ import annotations

from collections import Counter
import inspect
import json
from pathlib import Path
import unittest

from evaluation.evaluate import load_benchmark
from mcp_server.coding_tools import (
    CODING_TOOL_NAMES,
    code_list_files,
    code_read_file,
    code_search_text,
    git_diff,
    git_log,
    git_show,
    git_status,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODING_BENCHMARK_ROOT = PROJECT_ROOT / "benchmark" / "coding"
BENCHMARK_PATHS = {
    "smoke": CODING_BENCHMARK_ROOT / "tool_routing_coding_smoke.json",
    "controlled": CODING_BENCHMARK_ROOT / "tool_routing_coding_controlled.json",
    "upstream": CODING_BENCHMARK_ROOT
    / "tool_routing_coding_upstream_inspired.json",
    "codesearchnet": CODING_BENCHMARK_ROOT
    / "tool_routing_coding_codesearchnet_public_derived.json",
}
CODING_TOOL_MENU = [
    "code_list_files",
    "code_read_file",
    "code_search_text",
    "git_log",
    "git_show",
    "git_diff",
    "git_status",
]
TOOL_FUNCTIONS = {
    "code_list_files": code_list_files,
    "code_read_file": code_read_file,
    "code_search_text": code_search_text,
    "git_log": git_log,
    "git_show": git_show,
    "git_diff": git_diff,
    "git_status": git_status,
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


def _load_json(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


class CodingBenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.datasets = {
            name: _load_json(path) for name, path in BENCHMARK_PATHS.items()
        }

    def test_datasets_load_with_existing_evaluator(self) -> None:
        expected_lengths = {
            "smoke": 7,
            "controlled": 35,
            "upstream": 28,
            "codesearchnet": 15,
        }
        for name, path in BENCHMARK_PATHS.items():
            with self.subTest(dataset=name):
                self.assertEqual(len(load_benchmark(path)), expected_lengths[name])

    def test_ids_tool_menus_and_tool_counts_are_exact(self) -> None:
        all_rows = [row for rows in self.datasets.values() for row in rows]
        ids = [row["id"] for row in all_rows]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(frozenset(CODING_TOOL_MENU), CODING_TOOL_NAMES)

        expected_per_tool = {"smoke": 1, "controlled": 5, "upstream": 4}
        for dataset, per_tool_count in expected_per_tool.items():
            self.assertEqual(
                Counter(
                    row["expected_tool"] for row in self.datasets[dataset]
                ),
                Counter({tool: per_tool_count for tool in CODING_TOOL_MENU}),
            )
        self.assertEqual(
            Counter(
                row["expected_tool"]
                for row in self.datasets["codesearchnet"]
            ),
            Counter({"code_search_text": 15}),
        )

        for row in all_rows:
            self.assertEqual(row["domain"], "coding")
            self.assertEqual(row["task_type"], "single_tool_routing")
            self.assertEqual(row["available_tools"], CODING_TOOL_MENU)
            self.assertIn(row["expected_tool"], row["available_tools"])

    def test_all_arguments_bind_and_expected_answers_execute(self) -> None:
        for dataset, rows in self.datasets.items():
            for row in rows:
                with self.subTest(dataset=dataset, row=row["id"]):
                    function = TOOL_FUNCTIONS[row["expected_tool"]]
                    inspect.signature(function).bind(**row["expected_args"])
                    result = function(**row["expected_args"])
                    self.assertTrue(
                        _contains(result, row.get("expected_answer")),
                        f"Expected answer mismatch for {row['id']}: {result!r}",
                    )

    def test_controlled_dataset_has_five_perturbations_per_tool(self) -> None:
        expected_perturbations = {
            "easy_direct",
            "same_domain_distractor",
            "parameter_specific",
            "paraphrase_robustness",
            "difficult_indirect",
        }
        for tool in CODING_TOOL_MENU:
            rows = [
                row
                for row in self.datasets["controlled"]
                if row["expected_tool"] == tool
            ]
            self.assertEqual(
                {row["perturbation_type"] for row in rows},
                expected_perturbations,
            )
            self.assertTrue(
                all(row["source"] == "controlled_synthetic" for row in rows)
            )
            self.assertTrue(
                all(row["fixture_id"] == "example/research-mcp" for row in rows)
            )
            self.assertTrue(
                all(row["fixture_version"] == "coding_fixture_v1" for row in rows)
            )

    def test_upstream_inspired_rows_are_generated_and_attributed(self) -> None:
        expected_perturbations = {
            "upstream_usage_adaptation",
            "developer_workflow",
            "argument_composition",
            "confusable_operation",
        }
        for tool in CODING_TOOL_MENU:
            rows = [
                row
                for row in self.datasets["upstream"]
                if row["expected_tool"] == tool
            ]
            self.assertEqual(
                {row["perturbation_type"] for row in rows},
                expected_perturbations,
            )
            for row in rows:
                self.assertEqual(
                    row["query_origin"], "generated_from_upstream_documentation"
                )
                self.assertEqual(row["provenance_type"], "controlled_fixture")
                self.assertTrue(row["inspiration_repository"])
                self.assertTrue(row["inspiration_url"].startswith("https://"))
                self.assertTrue(row["inspiration_reference"])

    def test_codesearchnet_rows_have_exact_pinned_provenance_and_indexes(
        self,
    ) -> None:
        rows = self.datasets["codesearchnet"]
        expected_coordinates = [
            (20, 1635, "k means clustering"),
            (13, 1666, "write csv"),
            (28, 1680, "get executable path"),
            (29, 1721, "httpclient post json"),
            (43, 1801, "how to make the checkbox checked"),
            (12, 1832, "socket recv timeout"),
            (74, 1929, "how to extract zip file recursively"),
            (25, 1952, "get current ip address"),
            (19, 2028, "replace in file"),
            (39, 2071, "encode url"),
            (57, 2144, "get current process id"),
            (79, 2259, "randomly extract x items from a list"),
            (24, 2264, "parse binary file to custom class"),
            (8, 2344, "group by count"),
            (11, 2398, "linear regression"),
        ]
        self.assertEqual(
            [
                (
                    row["source_query_index_zero_based"],
                    row["source_annotation_index_zero_based"],
                    row["original_query"],
                )
                for row in rows
            ],
            expected_coordinates,
        )
        self.assertEqual(
            [row["id"] for row in rows],
            [
                f"coding_public_codesearchnet_search_text_{index:03d}"
                for index in range(1, 16)
            ],
        )

        for line_number, row in enumerate(rows, start=1):
            self.assertEqual(row["source"], "public_coding_research_derived")
            self.assertEqual(
                row["source_dataset"],
                "CodeSearchNet Challenge human evaluation",
            )
            self.assertEqual(row["source_repository"], "github/CodeSearchNet")
            self.assertEqual(
                row["source_revision"],
                "bb121a53a559e99a6849409355ee5c83803f2e87",
            )
            self.assertEqual(
                row["verified_repository_tip"],
                "106e827405c968597da938f6b373d30183918869",
            )
            self.assertEqual(
                row["source_query_sha256"],
                "037509c717c2e164721f0fd3ea45cb05f36669551af643f53930a92b76b146cf",
            )
            self.assertEqual(
                row["source_query_url"],
                "https://github.com/github/CodeSearchNet/blob/"
                "bb121a53a559e99a6849409355ee5c83803f2e87/resources/queries.csv",
            )
            self.assertEqual(
                row["source_annotation_sha256"],
                "0340af32b551ceadb74fec147f97642b7fedf3ff039e38fb86baff49ee899846",
            )
            self.assertEqual(
                row["source_annotation_url"],
                "https://github.com/github/CodeSearchNet/blob/"
                "bb121a53a559e99a6849409355ee5c83803f2e87/"
                "resources/annotationStore.csv",
            )
            self.assertEqual(row["source_license"], "MIT")
            self.assertEqual(
                row["source_license_sha256"],
                "5ba1fd8a344040f2698ed3234aeb8f4b3e85211aa54a37048021f3eb0043be22",
            )
            self.assertEqual(
                row["source_license_url"],
                "https://github.com/github/CodeSearchNet/blob/"
                "bb121a53a559e99a6849409355ee5c83803f2e87/LICENSE",
            )
            self.assertEqual(
                row["source_paper_url"], "https://arxiv.org/abs/1909.09436"
            )
            self.assertEqual(
                row["query_origin"],
                "generated_wrapper_around_codesearchnet_query",
            )
            self.assertEqual(
                row["original_query_origin"],
                "codesearchnet_published_query",
            )
            self.assertEqual(
                row["query_wrapper_id"],
                "codesearchnet_annotation_lookup_v1",
            )
            self.assertEqual(
                row["provenance_type"], "research_dataset_adaptation"
            )
            self.assertEqual(row["perturbation_type"], "none")
            self.assertEqual(row["fixture_id"], "codesearchnet-public-v1")
            self.assertEqual(
                row["fixture_version"], "coding_codesearchnet_fixture_v1"
            )
            self.assertEqual(
                row["fixture_file"],
                "benchmark/coding/fixtures/codesearchnet_public_annotations.json",
            )
            self.assertEqual(
                row["license_file"],
                "benchmark/coding/fixtures/CODESEARCHNET_LICENSE.txt",
            )
            self.assertEqual(
                row["attribution_file"],
                "benchmark/coding/fixtures/CODESEARCHNET_ATTRIBUTION.md",
            )
            self.assertEqual(row["source_language"], "Python")
            self.assertEqual(row["source_relevance"], 3)
            self.assertEqual(row["source_annotation_pair_multiplicity"], 1)
            self.assertGreaterEqual(row["source_query_index_zero_based"], 0)
            self.assertGreaterEqual(row["source_annotation_index_zero_based"], 0)

            self.assertEqual(
                row["query"],
                "In repository codesearchnet-public-v1, search only "
                "resources/annotationStore_selected.jsonl for the exact text "
                f'"{row["original_query"]}". Match case exactly and return at '
                "most one result.",
            )
            self.assertEqual(
                row["source_annotation_record"],
                {
                    "Language": "Python",
                    "Query": row["original_query"],
                    "GitHubUrl": row["source_github_url"],
                    "Relevance": "3",
                    "Notes": "",
                },
            )
            self.assertEqual(
                row["expected_args"],
                {
                    "repo_id": "codesearchnet-public-v1",
                    "pattern": row["original_query"],
                    "path_glob": "resources/annotationStore_selected.jsonl",
                    "case_sensitive": True,
                    "max_results": 1,
                },
            )
            self.assertEqual(
                row["expected_answer"]["matches"][0]["line"], line_number
            )


if __name__ == "__main__":
    unittest.main()
