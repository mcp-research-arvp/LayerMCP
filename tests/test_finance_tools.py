from __future__ import annotations

import inspect
import unittest

from mcp_server.finance_state import (
    DEFAULT_FINANCE_FIXTURE_ID,
    FINANCE_FIXTURE_VERSION,
    snapshot_finance_state,
)
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
from mcp_server.server import mcp


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


class FinanceToolTests(unittest.TestCase):
    def assert_fixture_provenance(self, result: dict) -> None:
        self.assertEqual(result["source"], "deterministic_offline_finance_fixture")
        provenance = result["provenance"]
        self.assertEqual(provenance["fixture_id"], DEFAULT_FINANCE_FIXTURE_ID)
        self.assertEqual(provenance["fixture_version"], FINANCE_FIXTURE_VERSION)
        self.assertFalse(provenance["network_access"])

    def test_exact_tool_catalog_is_registered(self) -> None:
        self.assertEqual(FINANCE_TOOL_NAMES, frozenset(TOOL_FUNCTIONS))
        registered = set(mcp._tool_manager._tools)
        self.assertTrue(FINANCE_TOOL_NAMES <= registered)

        expected_parameters = {
            "finance_lookup_company": ["query", "max_results"],
            "finance_search_filings": [
                "company_identifier",
                "form_type",
                "fiscal_year",
                "max_results",
            ],
            "finance_get_filing_section": [
                "accession_number",
                "section",
                "max_chars",
            ],
            "finance_get_company_facts": [
                "company_identifier",
                "concept",
                "unit",
                "fiscal_year",
                "max_results",
            ],
            "finance_get_financial_statement": [
                "company_identifier",
                "statement",
                "fiscal_year",
                "fiscal_period",
            ],
            "finance_parse_xbrl": ["accession_number", "concepts", "max_facts"],
            "finance_query_table": ["dataset_id", "sql", "max_rows"],
            "finance_extract_pdf_tables": [
                "document_id",
                "pages",
                "flavor",
                "max_tables",
            ],
            "finance_get_market_quote": ["symbol"],
            "finance_get_market_time_series": [
                "symbol",
                "start_date",
                "end_date",
                "interval",
                "max_points",
            ],
        }
        for name, function in TOOL_FUNCTIONS.items():
            self.assertEqual(
                list(inspect.signature(function).parameters),
                expected_parameters[name],
            )

    def test_fixture_inventory_is_deterministic_and_explicitly_synthetic(self) -> None:
        state = snapshot_finance_state()
        self.assertEqual(state["fixture_id"], "example/finance-research")
        self.assertEqual(state["fixture_version"], "finance_fixture_v1")
        self.assertEqual(
            [company["ticker"] for company in state["companies"]],
            ["LMCP", "TBLR"],
        )
        self.assertTrue(all(company["fictional"] for company in state["companies"]))
        self.assertEqual(
            state["table_datasets"],
            ["annual_metrics", "finqa-public-test-v1", "quarterly_metrics"],
        )

    def test_filing_retrieval_tools(self) -> None:
        lookup = finance_lookup_company("Layer Manufacturing")
        self.assertEqual(lookup["results"][0]["ticker"], "LMCP")
        self.assertEqual(lookup["count"], 1)
        self.assert_fixture_provenance(lookup)

        filings = finance_search_filings("0001900001", "10-K", 2024, 1)
        self.assertEqual(filings["count"], 1)
        self.assertEqual(filings["filings"][0]["accession_number"], "LMCP-2024-10K")
        self.assert_fixture_provenance(filings)

        section = finance_get_filing_section("LMCP-2024-10K", "Item 1A", 80)
        self.assertEqual(section["section"], "risk_factors")
        self.assertEqual(section["character_count"], 80)
        self.assertTrue(section["truncated"])
        self.assert_fixture_provenance(section)

    def test_fact_statement_and_xbrl_tools(self) -> None:
        facts = finance_get_company_facts(
            "LMCP", "us-gaap:Revenue", "usd", 2024, 5
        )
        self.assertEqual(facts["count"], 1)
        self.assertEqual(facts["facts"][0]["value"], 125_000_000)
        self.assert_fixture_provenance(facts)

        statement = finance_get_financial_statement(
            "LMCP", "balance sheet", 2024
        )
        values = {row["concept"]: row["value"] for row in statement["rows"]}
        self.assertEqual(values["Assets"], 210_000_000)
        self.assertEqual(
            values["Assets"],
            values["Liabilities"] + values["StockholdersEquity"],
        )
        self.assert_fixture_provenance(statement)

        parsed = finance_parse_xbrl(
            "LMCP-2024-10K", ["Revenue", "NetIncomeLoss"], 10
        )
        self.assertEqual(parsed["fact_count"], 2)
        parsed_values = {fact["concept"]: fact["value"] for fact in parsed["facts"]}
        self.assertEqual(parsed_values, {"NetIncomeLoss": 14_000_000, "Revenue": 125_000_000})
        self.assertEqual(parsed["facts"][0]["context"]["end_date"], "2024-12-31")
        self.assert_fixture_provenance(parsed)

    def test_table_query_is_bounded_and_read_only(self) -> None:
        result = finance_query_table(
            "quarterly_metrics",
            "SELECT ticker, SUM(revenue) AS revenue FROM data GROUP BY ticker ORDER BY ticker",
        )
        self.assertEqual(result["engine"], "sqlite3")
        self.assertEqual(result["columns"], ["ticker", "revenue"])
        self.assertEqual(
            result["rows"],
            [["LMCP", 125_000_000], ["TBLR", 112_000_000]],
        )
        self.assert_fixture_provenance(result)

        limited = finance_query_table(
            "quarterly_metrics", "SELECT ticker FROM data ORDER BY ticker", 2
        )
        self.assertEqual(limited["row_count"], 2)
        self.assertTrue(limited["truncated"])

        rejected = [
            "DELETE FROM data",
            "SELECT * FROM data; SELECT * FROM data",
            "SELECT * FROM sqlite_master",
            "SELECT random() FROM data",
            "SELECT * FROM data -- comment",
        ]
        for sql in rejected:
            with self.subTest(sql=sql), self.assertRaises(ValueError):
                finance_query_table("quarterly_metrics", sql)

    def test_pdf_and_market_tools_are_explicit_offline_snapshots(self) -> None:
        tables = finance_extract_pdf_tables(
            "lmcp-2024-annual-report", "42", "lattice", 2
        )
        self.assertEqual(tables["table_count"], 1)
        self.assertEqual(tables["tables"][0]["title"], "Consolidated Statements of Income")
        self.assertFalse(tables["live_pdf_parsing"])
        self.assertFalse(tables["flavor_applied"])
        self.assert_fixture_provenance(tables)

        quote = finance_get_market_quote("lmcp")
        self.assertEqual(quote["as_of"], "2025-01-10")
        self.assertEqual(quote["price"], 42.75)
        self.assertTrue(quote["synthetic"])
        self.assert_fixture_provenance(quote)

        series = finance_get_market_time_series(
            "TBLR", "2025-01-03", "2025-01-08", "daily", 10
        )
        self.assertEqual(series["point_count"], 4)
        self.assertEqual(series["points"][0]["date"], "2025-01-03")
        self.assertEqual(series["points"][-1]["date"], "2025-01-08")
        self.assert_fixture_provenance(series)

    def test_invalid_identifiers_ranges_and_modes_are_rejected(self) -> None:
        invalid_calls = [
            lambda: finance_search_filings("UNKNOWN"),
            lambda: finance_get_filing_section("LMCP-2024-10K", "Item 99"),
            lambda: finance_get_company_facts("LMCP", max_results=0),
            lambda: finance_get_financial_statement("LMCP", "income", 2022),
            lambda: finance_parse_xbrl("LMCP-2024-10K", []),
            lambda: finance_extract_pdf_tables("lmcp-2024-annual-report", "0"),
            lambda: finance_extract_pdf_tables(
                "lmcp-2024-annual-report", "42", "unsupported"
            ),
            lambda: finance_get_market_quote("REAL"),
            lambda: finance_get_market_time_series(
                "LMCP", "2025-01-10", "2025-01-02"
            ),
            lambda: finance_get_market_time_series("LMCP", interval="intraday"),
        ]
        for call in invalid_calls:
            with self.subTest(call=call), self.assertRaises(ValueError):
                call()


if __name__ == "__main__":
    unittest.main()
