from __future__ import annotations

from collections import Counter
import inspect
import json
from pathlib import Path
import unittest

from evaluation.evaluate import load_benchmark
from mcp_server.finance_tools import (
    FINANCE_TOOL_NAMES,
    finance_extract_pdf_tables,
    finance_get_company_facts,
    finance_get_filing_section,
    finance_get_financial_statement,
    finance_get_market_quote,
    finance_get_market_time_series,
    finance_lookup_company,
    finance_parse_xbrl,
    finance_query_table,
    finance_search_filings,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FINANCE_BENCHMARK_ROOT = PROJECT_ROOT / "benchmark" / "finance"
BENCHMARK_PATHS = {
    "smoke": FINANCE_BENCHMARK_ROOT / "tool_routing_finance_smoke.json",
    "controlled": FINANCE_BENCHMARK_ROOT / "tool_routing_finance_controlled.json",
    "upstream": FINANCE_BENCHMARK_ROOT
    / "tool_routing_finance_upstream_inspired.json",
    "public": FINANCE_BENCHMARK_ROOT
    / "tool_routing_finance_public_derived.json",
    "tatqa_public": FINANCE_BENCHMARK_ROOT
    / "tool_routing_finance_tatqa_public_derived.json",
}
FINANCE_TOOL_MENU = [
    "finance_lookup_company",
    "finance_search_filings",
    "finance_get_filing_section",
    "finance_get_company_facts",
    "finance_get_financial_statement",
    "finance_parse_xbrl",
    "finance_query_table",
    "finance_extract_pdf_tables",
    "finance_get_market_quote",
    "finance_get_market_time_series",
]
TOOL_FUNCTIONS = {
    "finance_lookup_company": finance_lookup_company,
    "finance_search_filings": finance_search_filings,
    "finance_get_filing_section": finance_get_filing_section,
    "finance_get_company_facts": finance_get_company_facts,
    "finance_get_financial_statement": finance_get_financial_statement,
    "finance_parse_xbrl": finance_parse_xbrl,
    "finance_query_table": finance_query_table,
    "finance_extract_pdf_tables": finance_extract_pdf_tables,
    "finance_get_market_quote": finance_get_market_quote,
    "finance_get_market_time_series": finance_get_market_time_series,
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


class FinanceBenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.datasets = {
            name: _load_json(path) for name, path in BENCHMARK_PATHS.items()
        }

    def test_datasets_load_with_existing_evaluator(self) -> None:
        expected_lengths = {
            "smoke": 10,
            "controlled": 50,
            "upstream": 40,
            "public": 15,
            "tatqa_public": 15,
        }
        for name, path in BENCHMARK_PATHS.items():
            with self.subTest(dataset=name):
                self.assertEqual(len(load_benchmark(path)), expected_lengths[name])

    def test_ids_menus_and_balanced_tool_counts(self) -> None:
        all_rows = [row for rows in self.datasets.values() for row in rows]
        ids = [row["id"] for row in all_rows]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(frozenset(FINANCE_TOOL_MENU), FINANCE_TOOL_NAMES)

        expected_per_tool = {"smoke": 1, "controlled": 5, "upstream": 4}
        for dataset, per_tool_count in expected_per_tool.items():
            counts = Counter(
                row["expected_tool"] for row in self.datasets[dataset]
            )
            self.assertEqual(
                counts,
                Counter({tool: per_tool_count for tool in FINANCE_TOOL_MENU}),
            )
        self.assertEqual(
            Counter(row["expected_tool"] for row in self.datasets["public"]),
            Counter({"finance_query_table": 15}),
        )
        self.assertEqual(
            Counter(
                row["expected_tool"] for row in self.datasets["tatqa_public"]
            ),
            Counter({"finance_query_table": 15}),
        )

        for row in all_rows:
            self.assertEqual(row["domain"], "finance")
            self.assertEqual(row["task_type"], "single_tool_routing")
            self.assertEqual(row["available_tools"], FINANCE_TOOL_MENU)
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

    def test_controlled_dataset_has_five_routing_perturbations_per_tool(self) -> None:
        expected = {
            "easy_direct",
            "same_domain_distractor",
            "parameter_specific",
            "paraphrase_robustness",
            "difficult_indirect",
        }
        for tool in FINANCE_TOOL_MENU:
            rows = [
                row
                for row in self.datasets["controlled"]
                if row["expected_tool"] == tool
            ]
            self.assertEqual({row["perturbation_type"] for row in rows}, expected)
            self.assertTrue(all(row["source"] == "controlled_synthetic" for row in rows))
            self.assertTrue(
                all(row["fixture_id"] == "example/finance-research" for row in rows)
            )
            self.assertTrue(
                all(row["fixture_version"] == "finance_fixture_v1" for row in rows)
            )

    def test_upstream_inspired_rows_are_generated_and_attributed(self) -> None:
        expected_perturbations = {
            "upstream_usage_adaptation",
            "developer_workflow",
            "argument_composition",
            "confusable_operation",
        }
        for tool in FINANCE_TOOL_MENU:
            rows = [
                row
                for row in self.datasets["upstream"]
                if row["expected_tool"] == tool
            ]
            self.assertEqual(
                {row["perturbation_type"] for row in rows}, expected_perturbations
            )
            for row in rows:
                self.assertEqual(
                    row["query_origin"], "generated_from_upstream_documentation"
                )
                self.assertEqual(row["provenance_type"], "controlled_fixture")
                self.assertTrue(row["inspiration_repository"])
                self.assertTrue(row["inspiration_url"].startswith("https://"))
                self.assertTrue(row["inspiration_reference"])

    def test_public_finqa_rows_have_pinned_provenance(self) -> None:
        rows = self.datasets["public"]
        source_indices = []
        for row in rows:
            self.assertEqual(row["source"], "public_finance_derived")
            self.assertEqual(row["source_dataset"], "FinQA")
            self.assertEqual(row["source_split"], "test")
            self.assertEqual(
                row["source_revision"],
                "0f16e2867befa6840783e58be38c9efb9229d742",
            )
            self.assertEqual(row["source_license"], "MIT")
            self.assertEqual(row["fixture_dataset_id"], "finqa-public-test-v1")
            self.assertEqual(row["provenance_type"], "public_dataset_adaptation")
            source_indices.append(row["source_row_index"])
        self.assertEqual(len(source_indices), len(set(source_indices)))

    def test_public_tatqa_rows_are_exact_paper_queries_with_pinned_provenance(
        self,
    ) -> None:
        rows = self.datasets["tatqa_public"]
        expected_source_coordinates = [
            (3, 3, 21, "cac10d43fde9e07342fa7144876e77e7"),
            (6, 5, 41, "3bff27eb2944cb199f863d6eb50bb06d"),
            (8, 3, 51, "62afbd1d4e987a0cb3193b11523d0dfd"),
            (9, 3, 57, "25d2e81cd8e2d8ab81c5796d6e89c4ec"),
            (12, 4, 76, "fc99efc2ca485dc3677f3e998dea801b"),
            (25, 3, 153, "4cc89377aa421d8d6eece5a7dce4de2d"),
            (29, 3, 177, "a79311cf198fe3c0b698543ae20ca9b9"),
            (33, 5, 203, "6769cc0cee475e77d8eb292b9bdefd81"),
            (36, 5, 222, "7f5203a09a509ad96e94ca5d3d7cb647"),
            (43, 4, 263, "5745196398b3689e34ad6098849b0269"),
            (47, 5, 288, "6cf8ec5b621c2ad7848f6a65ac0063d5"),
            (56, 2, 339, "f6647f46037f82005ef3f04cda1052d3"),
            (57, 2, 345, "bd11d656bda1738c42b39479ace8fe80"),
            (63, 4, 383, "53f5f57955697fb79d7a20d605a5abd5"),
            (30, 5, 185, "7e2290f656cbeef6d9a61c02767b752c"),
        ]
        self.assertEqual(
            [
                (
                    row["source_context_index"],
                    row["source_question_index"],
                    row["source_flat_question_index"],
                    row["source_id"],
                )
                for row in rows
            ],
            expected_source_coordinates,
        )
        self.assertEqual(
            [row["id"] for row in rows],
            [
                f"finance_public_tatqa_query_table_{index:03d}"
                for index in range(1, 16)
            ],
        )

        for row in rows:
            self.assertEqual(row["source"], "public_finance_paper_derived")
            self.assertEqual(row["source_dataset"], "TAT-QA")
            self.assertEqual(row["source_repository"], "NExTplusplus/TAT-QA")
            self.assertEqual(row["source_split"], "test_gold")
            self.assertEqual(
                row["source_revision"],
                "870accc41953dcde885aabeb963d94aabdc0fbc3",
            )
            self.assertEqual(row["source_license"], "CC BY 4.0")
            self.assertEqual(
                row["source_paper_url"],
                "https://aclanthology.org/2021.acl-long.254/",
            )
            self.assertEqual(
                row["source_paper_doi"], "10.18653/v1/2021.acl-long.254"
            )
            self.assertEqual(
                row["query_origin"], "official_research_dataset_question"
            )
            self.assertEqual(
                row["provenance_type"],
                "research_paper_dataset_adaptation",
            )
            self.assertEqual(
                row["fixture_dataset_id"], "tatqa-public-test-gold-v1"
            )
            self.assertEqual(
                row["fixture_file"],
                "benchmark/finance/fixtures/tatqa_public_test_gold_cells.json",
            )
            self.assertEqual(
                row["expected_args"]["dataset_id"],
                "tatqa-public-test-gold-v1",
            )
            self.assertEqual(
                row["expected_answer"]["dataset_id"],
                "tatqa-public-test-gold-v1",
            )
            self.assertTrue(row["source_derivation"])
            self.assertTrue(row["source_tree_derivation"])
            self.assertIn("source_answer", row)


if __name__ == "__main__":
    unittest.main()
