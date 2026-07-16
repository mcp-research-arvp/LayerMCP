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


if __name__ == "__main__":
    unittest.main()
