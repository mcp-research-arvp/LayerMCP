from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any


DEFAULT_FINANCE_FIXTURE_ID = "example/finance-research"
FINANCE_FIXTURE_VERSION = "finance_fixture_v1"


_COMPANIES: dict[str, dict[str, Any]] = {
    "LMCP": {
        "ticker": "LMCP",
        "cik": "0001900001",
        "name": "Layer Manufacturing Concepts, Inc.",
        "aliases": ["Layer Manufacturing", "Layer Manufacturing Concepts"],
        "industry": "Industrial machinery",
        "fiscal_year_end": "12-31",
        "fictional": True,
    },
    "TBLR": {
        "ticker": "TBLR",
        "cik": "0001900002",
        "name": "Tabular Cloud Labs, Inc.",
        "aliases": ["Tabular Cloud", "Tabular Cloud Labs"],
        "industry": "Cloud data services",
        "fiscal_year_end": "12-31",
        "fictional": True,
    },
}


_FILINGS: list[dict[str, Any]] = [
    {
        "accession_number": "LMCP-2024-10K",
        "ticker": "LMCP",
        "form_type": "10-K",
        "filing_date": "2025-02-20",
        "period_end": "2024-12-31",
        "fiscal_year": 2024,
        "fiscal_period": "FY",
        "document_id": "lmcp-2024-annual-report",
        "sections": {
            "business": (
                "Layer Manufacturing Concepts, Inc. is a fictional manufacturer "
                "of modular production equipment. During 2024, it sold equipment "
                "and maintenance plans to synthetic customers in North America."
            ),
            "risk_factors": (
                "The fictional company is exposed to component availability, "
                "customer concentration, warranty costs, and foreign-exchange "
                "volatility. These statements are synthetic research evidence."
            ),
            "management_discussion_and_analysis": (
                "Revenue increased to $125 million in 2024 from $108 million in "
                "2023. Gross profit was $53 million and operating income was $18 "
                "million in 2024. Management attributes the synthetic increase to "
                "higher equipment volume and maintenance-plan renewals."
            ),
            "financial_statements": (
                "For 2024, the fictional company reported revenue of $125 million, "
                "net income of $14 million, assets of $210 million, liabilities of "
                "$82 million, and stockholders' equity of $128 million."
            ),
        },
    },
    {
        "accession_number": "LMCP-2024-Q3-10Q",
        "ticker": "LMCP",
        "form_type": "10-Q",
        "filing_date": "2024-11-05",
        "period_end": "2024-09-30",
        "fiscal_year": 2024,
        "fiscal_period": "Q3",
        "document_id": None,
        "sections": {
            "business": "The fictional company continued its modular equipment operations.",
            "risk_factors": "Component supply and customer concentration remained the principal synthetic risks.",
            "management_discussion_and_analysis": "Third-quarter revenue was $32 million and net income was $3.6 million.",
            "financial_statements": "At September 30, 2024, synthetic assets were $207 million and liabilities were $80 million.",
        },
    },
    {
        "accession_number": "LMCP-2023-10K",
        "ticker": "LMCP",
        "form_type": "10-K",
        "filing_date": "2024-02-22",
        "period_end": "2023-12-31",
        "fiscal_year": 2023,
        "fiscal_period": "FY",
        "document_id": None,
        "sections": {
            "business": "The fictional company manufactured modular production equipment in 2023.",
            "risk_factors": "Synthetic risks included input costs, warranties, and customer concentration.",
            "management_discussion_and_analysis": "Revenue was $108 million in 2023, compared with $92 million in 2022.",
            "financial_statements": "For 2023, synthetic revenue was $108 million and net income was $10 million.",
        },
    },
    {
        "accession_number": "TBLR-2025-Q1-10Q",
        "ticker": "TBLR",
        "form_type": "10-Q",
        "filing_date": "2025-05-08",
        "period_end": "2025-03-31",
        "fiscal_year": 2025,
        "fiscal_period": "Q1",
        "document_id": "tblr-2025-q1-report",
        "sections": {
            "business": "Tabular Cloud Labs, Inc. is a fictional provider of cloud analytics services.",
            "risk_factors": "Synthetic risks include service availability, data security, and customer retention.",
            "management_discussion_and_analysis": "First-quarter 2025 revenue was $26 million and net income was $3.4 million.",
            "financial_statements": "At March 31, 2025, synthetic assets were $164 million and liabilities were $62 million.",
        },
    },
    {
        "accession_number": "TBLR-2024-10K",
        "ticker": "TBLR",
        "form_type": "10-K",
        "filing_date": "2025-02-27",
        "period_end": "2024-12-31",
        "fiscal_year": 2024,
        "fiscal_period": "FY",
        "document_id": None,
        "sections": {
            "business": "The fictional company provides hosted analytics and table-processing services.",
            "risk_factors": "Service reliability, competition, and data-security incidents are synthetic risks.",
            "management_discussion_and_analysis": "Revenue increased to $86 million in 2024 from $70 million in 2023.",
            "financial_statements": "For 2024, synthetic revenue was $86 million and net income was $9 million.",
        },
    },
    {
        "accession_number": "TBLR-2023-10K",
        "ticker": "TBLR",
        "form_type": "10-K",
        "filing_date": "2024-02-29",
        "period_end": "2023-12-31",
        "fiscal_year": 2023,
        "fiscal_period": "FY",
        "document_id": None,
        "sections": {
            "business": "The fictional company supplied cloud analytics services in 2023.",
            "risk_factors": "Synthetic risks included outages, competition, and customer churn.",
            "management_discussion_and_analysis": "Revenue was $70 million and operating income was $6 million in 2023.",
            "financial_statements": "For 2023, synthetic net income was $4 million and assets were $130 million.",
        },
    },
]


def _annual_facts(
    ticker: str,
    accession_number: str,
    fiscal_year: int,
    filed: str,
    values: dict[str, int],
) -> list[dict[str, Any]]:
    return [
        {
            "ticker": ticker,
            "accession_number": accession_number,
            "concept": concept,
            "label": concept,
            "value": value,
            "unit": "USD",
            "fiscal_year": fiscal_year,
            "fiscal_period": "FY",
            "period_end": f"{fiscal_year}-12-31",
            "filed": filed,
        }
        for concept, value in values.items()
    ]


_FACTS: list[dict[str, Any]] = []
_FACTS.extend(
    _annual_facts(
        "LMCP",
        "LMCP-2024-10K",
        2024,
        "2025-02-20",
        {
            "Revenue": 125_000_000,
            "CostOfRevenue": 72_000_000,
            "GrossProfit": 53_000_000,
            "OperatingIncomeLoss": 18_000_000,
            "NetIncomeLoss": 14_000_000,
            "Assets": 210_000_000,
            "Liabilities": 82_000_000,
            "StockholdersEquity": 128_000_000,
            "CashAndCashEquivalentsAtCarryingValue": 35_000_000,
            "NetCashProvidedByUsedInOperatingActivities": 22_000_000,
        },
    )
)
_FACTS.extend(
    _annual_facts(
        "LMCP",
        "LMCP-2023-10K",
        2023,
        "2024-02-22",
        {
            "Revenue": 108_000_000,
            "CostOfRevenue": 65_000_000,
            "GrossProfit": 43_000_000,
            "OperatingIncomeLoss": 14_000_000,
            "NetIncomeLoss": 10_000_000,
            "Assets": 180_000_000,
            "Liabilities": 74_000_000,
            "StockholdersEquity": 106_000_000,
            "CashAndCashEquivalentsAtCarryingValue": 28_000_000,
            "NetCashProvidedByUsedInOperatingActivities": 18_000_000,
        },
    )
)
_FACTS.extend(
    _annual_facts(
        "TBLR",
        "TBLR-2024-10K",
        2024,
        "2025-02-27",
        {
            "Revenue": 86_000_000,
            "CostOfRevenue": 24_000_000,
            "GrossProfit": 62_000_000,
            "OperatingIncomeLoss": 12_000_000,
            "NetIncomeLoss": 9_000_000,
            "Assets": 155_000_000,
            "Liabilities": 60_000_000,
            "StockholdersEquity": 95_000_000,
            "CashAndCashEquivalentsAtCarryingValue": 42_000_000,
            "NetCashProvidedByUsedInOperatingActivities": 17_000_000,
        },
    )
)
_FACTS.extend(
    _annual_facts(
        "TBLR",
        "TBLR-2023-10K",
        2023,
        "2024-02-29",
        {
            "Revenue": 70_000_000,
            "CostOfRevenue": 21_000_000,
            "GrossProfit": 49_000_000,
            "OperatingIncomeLoss": 6_000_000,
            "NetIncomeLoss": 4_000_000,
            "Assets": 130_000_000,
            "Liabilities": 52_000_000,
            "StockholdersEquity": 78_000_000,
            "CashAndCashEquivalentsAtCarryingValue": 34_000_000,
            "NetCashProvidedByUsedInOperatingActivities": 11_000_000,
        },
    )
)


_STATEMENTS: dict[tuple[str, int, str, str], list[dict[str, Any]]] = {}


def _add_statements(
    ticker: str,
    fiscal_year: int,
    fiscal_period: str,
    income_values: tuple[int, int, int, int, int],
    balance_values: tuple[int, int, int, int],
    cash_flow_values: tuple[int, int, int],
) -> None:
    revenue, cost, gross_profit, operating_income, net_income = income_values
    assets, liabilities, equity, cash = balance_values
    operating_cash, investing_cash, financing_cash = cash_flow_values
    _STATEMENTS[(ticker, fiscal_year, fiscal_period, "income_statement")] = [
        {"concept": "Revenue", "label": "Revenue", "value": revenue, "unit": "USD"},
        {"concept": "CostOfRevenue", "label": "Cost of revenue", "value": cost, "unit": "USD"},
        {"concept": "GrossProfit", "label": "Gross profit", "value": gross_profit, "unit": "USD"},
        {"concept": "OperatingIncomeLoss", "label": "Operating income", "value": operating_income, "unit": "USD"},
        {"concept": "NetIncomeLoss", "label": "Net income", "value": net_income, "unit": "USD"},
    ]
    _STATEMENTS[(ticker, fiscal_year, fiscal_period, "balance_sheet")] = [
        {"concept": "Assets", "label": "Total assets", "value": assets, "unit": "USD"},
        {"concept": "Liabilities", "label": "Total liabilities", "value": liabilities, "unit": "USD"},
        {"concept": "StockholdersEquity", "label": "Stockholders' equity", "value": equity, "unit": "USD"},
        {"concept": "CashAndCashEquivalentsAtCarryingValue", "label": "Cash and cash equivalents", "value": cash, "unit": "USD"},
    ]
    _STATEMENTS[(ticker, fiscal_year, fiscal_period, "cash_flow_statement")] = [
        {"concept": "NetCashProvidedByUsedInOperatingActivities", "label": "Net cash from operating activities", "value": operating_cash, "unit": "USD"},
        {"concept": "NetCashProvidedByUsedInInvestingActivities", "label": "Net cash used in investing activities", "value": investing_cash, "unit": "USD"},
        {"concept": "NetCashProvidedByUsedInFinancingActivities", "label": "Net cash from financing activities", "value": financing_cash, "unit": "USD"},
    ]


_add_statements("LMCP", 2024, "FY", (125_000_000, 72_000_000, 53_000_000, 18_000_000, 14_000_000), (210_000_000, 82_000_000, 128_000_000, 35_000_000), (22_000_000, -17_000_000, -2_000_000))
_add_statements("LMCP", 2023, "FY", (108_000_000, 65_000_000, 43_000_000, 14_000_000, 10_000_000), (180_000_000, 74_000_000, 106_000_000, 28_000_000), (18_000_000, -14_000_000, -1_000_000))
_add_statements("LMCP", 2024, "Q3", (32_000_000, 18_500_000, 13_500_000, 4_700_000, 3_600_000), (207_000_000, 80_000_000, 127_000_000, 33_000_000), (5_800_000, -4_200_000, -500_000))
_add_statements("TBLR", 2024, "FY", (86_000_000, 24_000_000, 62_000_000, 12_000_000, 9_000_000), (155_000_000, 60_000_000, 95_000_000, 42_000_000), (17_000_000, -9_000_000, 2_000_000))
_add_statements("TBLR", 2023, "FY", (70_000_000, 21_000_000, 49_000_000, 6_000_000, 4_000_000), (130_000_000, 52_000_000, 78_000_000, 34_000_000), (11_000_000, -8_000_000, 1_000_000))
_add_statements("TBLR", 2025, "Q1", (26_000_000, 7_000_000, 19_000_000, 4_500_000, 3_400_000), (164_000_000, 62_000_000, 102_000_000, 46_000_000), (5_500_000, -2_500_000, 900_000))


def _build_xbrl_instance(
    ticker: str,
    cik: str,
    accession_number: str,
    period_end: str,
    facts: list[dict[str, Any]],
) -> str:
    fact_xml = "\n".join(
        f'  <us-gaap:{fact["concept"]} contextRef="FY" unitRef="USD" decimals="-3">{fact["value"]}</us-gaap:{fact["concept"]}>'
        for fact in facts
    )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<xbrli:xbrl xmlns:xbrli="http://www.xbrl.org/2003/instance" '
        'xmlns:us-gaap="http://fasb.org/us-gaap/2024" '
        'xmlns:iso4217="http://www.xbrl.org/2003/iso4217">\n'
        '  <xbrli:context id="FY">\n'
        '    <xbrli:entity><xbrli:identifier scheme="https://fixture.invalid/cik">'
        f'{cik}</xbrli:identifier></xbrli:entity>\n'
        '    <xbrli:period><xbrli:startDate>'
        f'{period_end[:4]}-01-01</xbrli:startDate><xbrli:endDate>{period_end}'
        '</xbrli:endDate></xbrli:period>\n'
        '  </xbrli:context>\n'
        '  <xbrli:unit id="USD"><xbrli:measure>iso4217:USD</xbrli:measure></xbrli:unit>\n'
        f'{fact_xml}\n'
        f'  <!-- synthetic fixture: {ticker} {accession_number} -->\n'
        '</xbrli:xbrl>\n'
    )


_XBRL_INSTANCES: dict[str, str] = {}
for _filing in _FILINGS:
    _filing_facts = [
        fact for fact in _FACTS if fact["accession_number"] == _filing["accession_number"]
    ]
    if not _filing_facts:
        statement_key = (
            _filing["ticker"],
            _filing["fiscal_year"],
            _filing["fiscal_period"],
            "income_statement",
        )
        _filing_facts = [
            {
                "concept": row["concept"],
                "value": row["value"],
            }
            for row in _STATEMENTS.get(statement_key, [])
        ]
    _XBRL_INSTANCES[_filing["accession_number"]] = _build_xbrl_instance(
        _filing["ticker"],
        _COMPANIES[_filing["ticker"]]["cik"],
        _filing["accession_number"],
        _filing["period_end"],
        _filing_facts,
    )


# Each entry is loaded into a private in-memory SQLite table named ``data``.
# Keeping schemas and rows declarative makes additional licensed table snapshots easy
# to add without widening the SQL tool's filesystem or network authority.
FINANCE_TABLES: dict[str, dict[str, Any]] = {
    "quarterly_metrics": {
        "description": "Synthetic quarterly metrics for the two fictional fixture companies.",
        "columns": [
            {"name": "ticker", "type": "TEXT"},
            {"name": "company", "type": "TEXT"},
            {"name": "fiscal_year", "type": "INTEGER"},
            {"name": "fiscal_period", "type": "TEXT"},
            {"name": "revenue", "type": "INTEGER"},
            {"name": "gross_profit", "type": "INTEGER"},
            {"name": "operating_income", "type": "INTEGER"},
            {"name": "net_income", "type": "INTEGER"},
            {"name": "assets", "type": "INTEGER"},
            {"name": "liabilities", "type": "INTEGER"},
        ],
        "rows": [
            ["LMCP", "Layer Manufacturing Concepts, Inc.", 2024, "Q1", 29_000_000, 12_000_000, 4_000_000, 3_000_000, 198_000_000, 79_000_000],
            ["LMCP", "Layer Manufacturing Concepts, Inc.", 2024, "Q2", 31_000_000, 13_000_000, 4_500_000, 3_500_000, 203_000_000, 80_000_000],
            ["LMCP", "Layer Manufacturing Concepts, Inc.", 2024, "Q3", 32_000_000, 13_500_000, 4_700_000, 3_600_000, 207_000_000, 80_000_000],
            ["LMCP", "Layer Manufacturing Concepts, Inc.", 2024, "Q4", 33_000_000, 14_500_000, 4_800_000, 3_900_000, 210_000_000, 82_000_000],
            ["TBLR", "Tabular Cloud Labs, Inc.", 2024, "Q1", 19_000_000, 14_000_000, 2_000_000, 1_500_000, 139_000_000, 55_000_000],
            ["TBLR", "Tabular Cloud Labs, Inc.", 2024, "Q2", 21_000_000, 15_000_000, 3_000_000, 2_000_000, 145_000_000, 57_000_000],
            ["TBLR", "Tabular Cloud Labs, Inc.", 2024, "Q3", 22_000_000, 16_000_000, 3_200_000, 2_400_000, 149_000_000, 59_000_000],
            ["TBLR", "Tabular Cloud Labs, Inc.", 2024, "Q4", 24_000_000, 17_000_000, 3_800_000, 3_100_000, 155_000_000, 60_000_000],
            ["TBLR", "Tabular Cloud Labs, Inc.", 2025, "Q1", 26_000_000, 19_000_000, 4_500_000, 3_400_000, 164_000_000, 62_000_000],
        ],
        "provenance": {
            "origin": "LayerMCP synthetic fixture",
            "license": "repository license",
            "synthetic": True,
        },
    },
    "annual_metrics": {
        "description": "Synthetic annual metrics in U.S. dollars.",
        "columns": [
            {"name": "ticker", "type": "TEXT"},
            {"name": "fiscal_year", "type": "INTEGER"},
            {"name": "revenue", "type": "INTEGER"},
            {"name": "net_income", "type": "INTEGER"},
            {"name": "assets", "type": "INTEGER"},
            {"name": "liabilities", "type": "INTEGER"},
            {"name": "operating_cash_flow", "type": "INTEGER"},
        ],
        "rows": [
            ["LMCP", 2023, 108_000_000, 10_000_000, 180_000_000, 74_000_000, 18_000_000],
            ["LMCP", 2024, 125_000_000, 14_000_000, 210_000_000, 82_000_000, 22_000_000],
            ["TBLR", 2023, 70_000_000, 4_000_000, 130_000_000, 52_000_000, 11_000_000],
            ["TBLR", 2024, 86_000_000, 9_000_000, 155_000_000, 60_000_000, 17_000_000],
        ],
        "provenance": {
            "origin": "LayerMCP synthetic fixture",
            "license": "repository license",
            "synthetic": True,
        },
    },
}


_FINQA_PUBLIC_FIXTURE_PATH = (
    Path(__file__).resolve().parents[1]
    / "benchmark"
    / "finance"
    / "fixtures"
    / "finqa_public_test_cells.json"
)
_FINQA_PUBLIC_TABLE = json.loads(_FINQA_PUBLIC_FIXTURE_PATH.read_text(encoding="utf-8"))
FINANCE_TABLES[_FINQA_PUBLIC_TABLE["dataset_id"]] = _FINQA_PUBLIC_TABLE


_PDF_DOCUMENTS: dict[str, dict[str, Any]] = {
    "lmcp-2024-annual-report": {
        "document_id": "lmcp-2024-annual-report",
        "accession_number": "LMCP-2024-10K",
        "page_count": 50,
        "tables": [
            {
                "table_id": "lmcp-income-2024",
                "page": 42,
                "title": "Consolidated Statements of Income",
                "unit": "USD millions",
                "columns": ["Line item", "2024", "2023"],
                "rows": [
                    ["Revenue", 125, 108],
                    ["Gross profit", 53, 43],
                    ["Operating income", 18, 14],
                    ["Net income", 14, 10],
                ],
            },
            {
                "table_id": "lmcp-balance-sheet-2024",
                "page": 45,
                "title": "Consolidated Balance Sheets",
                "unit": "USD millions",
                "columns": ["Line item", "2024", "2023"],
                "rows": [
                    ["Total assets", 210, 180],
                    ["Total liabilities", 82, 74],
                    ["Stockholders' equity", 128, 106],
                ],
            },
        ],
    },
    "tblr-2025-q1-report": {
        "document_id": "tblr-2025-q1-report",
        "accession_number": "TBLR-2025-Q1-10Q",
        "page_count": 15,
        "tables": [
            {
                "table_id": "tblr-income-2025-q1",
                "page": 8,
                "title": "Condensed Statements of Operations",
                "unit": "USD millions",
                "columns": ["Line item", "Q1 2025", "Q1 2024"],
                "rows": [
                    ["Revenue", 26.0, 19.0],
                    ["Gross profit", 19.0, 14.0],
                    ["Operating income", 4.5, 2.0],
                    ["Net income", 3.4, 1.5],
                ],
            },
            {
                "table_id": "tblr-balance-sheet-2025-q1",
                "page": 11,
                "title": "Condensed Balance Sheets",
                "unit": "USD millions",
                "columns": ["Line item", "March 31, 2025", "December 31, 2024"],
                "rows": [
                    ["Total assets", 164, 155],
                    ["Total liabilities", 62, 60],
                    ["Stockholders' equity", 102, 95],
                ],
            },
        ],
    },
}


_MARKET_SERIES: dict[str, list[dict[str, Any]]] = {
    "LMCP": [
        {"date": "2025-01-02", "open": 40.00, "high": 41.00, "low": 39.50, "close": 40.50, "volume": 120_000},
        {"date": "2025-01-03", "open": 40.60, "high": 41.40, "low": 40.20, "close": 41.10, "volume": 128_000},
        {"date": "2025-01-06", "open": 41.00, "high": 41.80, "low": 40.70, "close": 41.50, "volume": 133_000},
        {"date": "2025-01-07", "open": 41.60, "high": 42.20, "low": 41.10, "close": 41.80, "volume": 126_000},
        {"date": "2025-01-08", "open": 41.90, "high": 42.60, "low": 41.70, "close": 42.40, "volume": 144_000},
        {"date": "2025-01-09", "open": 42.30, "high": 42.80, "low": 41.90, "close": 42.10, "volume": 137_000},
        {"date": "2025-01-10", "open": 42.20, "high": 43.00, "low": 42.00, "close": 42.75, "volume": 151_000},
    ],
    "TBLR": [
        {"date": "2025-01-02", "open": 26.50, "high": 27.10, "low": 26.20, "close": 26.90, "volume": 210_000},
        {"date": "2025-01-03", "open": 26.95, "high": 27.40, "low": 26.70, "close": 27.20, "volume": 205_000},
        {"date": "2025-01-06", "open": 27.15, "high": 27.80, "low": 27.00, "close": 27.65, "volume": 222_000},
        {"date": "2025-01-07", "open": 27.70, "high": 28.10, "low": 27.35, "close": 27.90, "volume": 218_000},
        {"date": "2025-01-08", "open": 27.95, "high": 28.50, "low": 27.80, "close": 28.30, "volume": 230_000},
        {"date": "2025-01-09", "open": 28.20, "high": 28.55, "low": 27.95, "close": 28.25, "volume": 225_000},
        {"date": "2025-01-10", "open": 28.30, "high": 28.90, "low": 28.10, "close": 28.60, "volume": 241_000},
    ],
}


_FINANCE_FIXTURE: dict[str, Any] = {
    "fixture_id": DEFAULT_FINANCE_FIXTURE_ID,
    "fixture_version": FINANCE_FIXTURE_VERSION,
    "description": "Deterministic offline finance research fixture containing only fictional companies and synthetic data.",
    "companies": _COMPANIES,
    "filings": _FILINGS,
    "facts": _FACTS,
    "statements": _STATEMENTS,
    "xbrl_instances": _XBRL_INSTANCES,
    "tables": FINANCE_TABLES,
    "pdf_documents": _PDF_DOCUMENTS,
    "market_series": _MARKET_SERIES,
}


def get_finance_fixture() -> dict[str, Any]:
    """Return the server-owned deterministic fixture used by all finance tools."""
    return _FINANCE_FIXTURE


def snapshot_finance_state() -> dict[str, Any]:
    """Return a detached inventory suitable for diagnostics and tests."""
    return {
        "fixture_id": DEFAULT_FINANCE_FIXTURE_ID,
        "fixture_version": FINANCE_FIXTURE_VERSION,
        "companies": deepcopy(list(_COMPANIES.values())),
        "filing_accessions": [item["accession_number"] for item in _FILINGS],
        "table_datasets": sorted(FINANCE_TABLES),
        "pdf_documents": sorted(_PDF_DOCUMENTS),
        "market_symbols": sorted(_MARKET_SERIES),
    }
