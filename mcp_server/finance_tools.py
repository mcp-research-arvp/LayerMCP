from __future__ import annotations

import re
import sqlite3
import xml.etree.ElementTree as ET
from copy import deepcopy
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Any

from mcp_server.finance_state import (
    DEFAULT_FINANCE_FIXTURE_ID,
    FINANCE_FIXTURE_VERSION,
    get_finance_fixture,
)


FINANCE_TOOL_NAMES = frozenset(
    {
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
    }
)

_SOURCE = "deterministic_offline_finance_fixture"
_MAX_QUERY_LENGTH = 500
_MAX_SQL_LENGTH = 4_000
_MAX_LOOKUP_RESULTS = 25
_MAX_FILING_RESULTS = 100
_MAX_SECTION_CHARS = 50_000
_MAX_FACT_RESULTS = 500
_MAX_XBRL_FACTS = 500
_MAX_SQL_ROWS = 500
_MAX_PDF_TABLES = 50
_MAX_TIME_SERIES_POINTS = 500
_MAX_PAGE_SELECTION = 100
_SQL_PROGRESS_CALLBACK_LIMIT = 250
_SQL_PROGRESS_INSTRUCTION_INTERVAL = 1_000


def _result(
    payload: dict[str, Any],
    *,
    origin: str,
    classification: str = "fictional_synthetic_fixture",
    extra_provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    provenance: dict[str, Any] = {
        "fixture_id": DEFAULT_FINANCE_FIXTURE_ID,
        "fixture_version": FINANCE_FIXTURE_VERSION,
        "origin": origin,
        "classification": classification,
        "network_access": False,
    }
    if extra_provenance:
        provenance["dataset"] = deepcopy(extra_provenance)
    return {
        **payload,
        "source": _SOURCE,
        "provenance": provenance,
    }


def _required_text(
    value: str,
    field: str,
    *,
    maximum: int = _MAX_QUERY_LENGTH,
) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string.")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field} must not be empty.")
    if len(normalized) > maximum:
        raise ValueError(f"{field} must contain at most {maximum} characters.")
    if "\x00" in normalized:
        raise ValueError(f"{field} must not contain NUL characters.")
    return normalized


def _bounded_integer(
    value: int,
    field: str,
    minimum: int,
    maximum: int,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer.")
    if value < minimum or value > maximum:
        raise ValueError(f"{field} must be between {minimum} and {maximum}.")
    return value


def _optional_year(value: int | None, field: str = "fiscal_year") -> int | None:
    if value is None:
        return None
    return _bounded_integer(value, field, 1900, 2200)


def _parse_iso_date(value: str | None, field: str) -> str | None:
    if value is None:
        return None
    normalized = _required_text(value, field, maximum=10)
    try:
        parsed = date.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError(f"{field} must use YYYY-MM-DD format.") from exc
    if parsed.isoformat() != normalized:
        raise ValueError(f"{field} must use YYYY-MM-DD format.")
    return normalized


def _searchable_company_text(company: dict[str, Any]) -> list[str]:
    return [
        company["ticker"],
        company["cik"],
        company["name"],
        *company.get("aliases", []),
    ]


def _company_score(company: dict[str, Any], query: str) -> int:
    normalized = query.casefold()
    values = [value.casefold() for value in _searchable_company_text(company)]
    if normalized == company["ticker"].casefold():
        return 100
    if normalized == company["cik"].casefold():
        return 98
    if normalized == company["name"].casefold():
        return 96
    if normalized in values:
        return 94
    if any(normalized in value for value in values):
        return 70
    terms = set(re.findall(r"[a-z0-9]+", normalized))
    company_terms = set(
        re.findall(r"[a-z0-9]+", " ".join(values).casefold())
    )
    overlap = len(terms & company_terms)
    return 30 + overlap if overlap else 0


def _company_payload(company: dict[str, Any]) -> dict[str, Any]:
    return {
        "ticker": company["ticker"],
        "cik": company["cik"],
        "name": company["name"],
        "aliases": list(company.get("aliases", [])),
        "industry": company["industry"],
        "fiscal_year_end": company["fiscal_year_end"],
        "fictional": True,
    }


def _resolve_company(identifier: str) -> dict[str, Any]:
    normalized = _required_text(identifier, "company_identifier").casefold()
    fixture = get_finance_fixture()
    exact = [
        company
        for company in fixture["companies"].values()
        if normalized
        in {value.casefold() for value in _searchable_company_text(company)}
    ]
    if len(exact) == 1:
        return exact[0]

    partial = [
        company
        for company in fixture["companies"].values()
        if any(
            normalized in value.casefold()
            for value in _searchable_company_text(company)
        )
    ]
    if len(partial) == 1:
        return partial[0]

    available = ", ".join(sorted(fixture["companies"]))
    if len(partial) > 1:
        raise ValueError("company_identifier is ambiguous; use a ticker or CIK.")
    raise ValueError(f"Unknown company_identifier. Available tickers: {available}")


def _resolve_filing(accession_number: str) -> dict[str, Any]:
    normalized = _required_text(
        accession_number, "accession_number", maximum=80
    ).casefold()
    fixture = get_finance_fixture()
    for filing in fixture["filings"]:
        if filing["accession_number"].casefold() == normalized:
            return filing
    available = ", ".join(
        sorted(item["accession_number"] for item in fixture["filings"])
    )
    raise ValueError(f"Unknown accession_number. Available values: {available}")


def _filing_summary(filing: dict[str, Any]) -> dict[str, Any]:
    return {
        "accession_number": filing["accession_number"],
        "ticker": filing["ticker"],
        "form_type": filing["form_type"],
        "filing_date": filing["filing_date"],
        "period_end": filing["period_end"],
        "fiscal_year": filing["fiscal_year"],
        "fiscal_period": filing["fiscal_period"],
        "document_id": filing["document_id"],
    }


def finance_lookup_company(query: str, max_results: int = 5) -> dict[str, Any]:
    """Look up fictional fixture companies by ticker, CIK, name, or alias."""
    normalized = _required_text(query, "query")
    maximum = _bounded_integer(
        max_results, "max_results", 1, _MAX_LOOKUP_RESULTS
    )
    fixture = get_finance_fixture()
    matches: list[tuple[int, dict[str, Any]]] = []
    for company in fixture["companies"].values():
        score = _company_score(company, normalized)
        if score:
            result = _company_payload(company)
            result["match_score"] = score
            matches.append((score, result))
    matches.sort(key=lambda item: (-item[0], item[1]["ticker"]))
    total = len(matches)
    results = [item[1] for item in matches[:maximum]]
    return _result(
        {
            "query": normalized,
            "companies": results,
            "results": results,
            "count": len(results),
            "truncated": total > maximum,
        },
        origin="fictional_company_registry",
    )


def finance_search_filings(
    company_identifier: str,
    form_type: str | None = None,
    fiscal_year: int | None = None,
    max_results: int = 10,
) -> dict[str, Any]:
    """Search bounded synthetic filing metadata for one fictional company."""
    company = _resolve_company(company_identifier)
    maximum = _bounded_integer(
        max_results, "max_results", 1, _MAX_FILING_RESULTS
    )
    year = _optional_year(fiscal_year)
    normalized_form: str | None = None
    if form_type is not None:
        form_text = _required_text(form_type, "form_type", maximum=20)
        compact = re.sub(r"[^A-Za-z0-9]", "", form_text).upper()
        form_map = {"10K": "10-K", "10Q": "10-Q", "8K": "8-K"}
        normalized_form = form_map.get(compact)
        if normalized_form is None:
            raise ValueError("form_type must be one of: 10-K, 10-Q, 8-K.")

    fixture = get_finance_fixture()
    filings = [
        filing
        for filing in fixture["filings"]
        if filing["ticker"] == company["ticker"]
        and (normalized_form is None or filing["form_type"] == normalized_form)
        and (year is None or filing["fiscal_year"] == year)
    ]
    filings.sort(
        key=lambda item: (item["filing_date"], item["accession_number"]),
        reverse=True,
    )
    total = len(filings)
    results = [_filing_summary(item) for item in filings[:maximum]]
    return _result(
        {
            "ticker": company["ticker"],
            "company": _company_payload(company),
            "filters": {"form_type": normalized_form, "fiscal_year": year},
            "filings": results,
            "count": len(results),
            "truncated": total > maximum,
        },
        origin="synthetic_filing_index",
    )


_SECTION_ALIASES = {
    "business": "business",
    "item1": "business",
    "1": "business",
    "riskfactors": "risk_factors",
    "item1a": "risk_factors",
    "1a": "risk_factors",
    "managementdiscussionandanalysis": "management_discussion_and_analysis",
    "mdanda": "management_discussion_and_analysis",
    "mda": "management_discussion_and_analysis",
    "item7": "management_discussion_and_analysis",
    "7": "management_discussion_and_analysis",
    "financialstatements": "financial_statements",
    "item8": "financial_statements",
    "8": "financial_statements",
}


def _normalize_section(section: str) -> str:
    text = _required_text(section, "section", maximum=100)
    compact = re.sub(r"[^a-z0-9]", "", text.casefold())
    canonical = _SECTION_ALIASES.get(compact)
    if canonical is None:
        available = ", ".join(
            [
                "business",
                "risk_factors",
                "management_discussion_and_analysis",
                "financial_statements",
            ]
        )
        raise ValueError(f"section must be one of: {available}")
    return canonical


def finance_get_filing_section(
    accession_number: str,
    section: str,
    max_chars: int = 12_000,
) -> dict[str, Any]:
    """Retrieve one bounded section from a synthetic filing fixture."""
    filing = _resolve_filing(accession_number)
    canonical_section = _normalize_section(section)
    maximum = _bounded_integer(max_chars, "max_chars", 1, _MAX_SECTION_CHARS)
    full_content = filing["sections"].get(canonical_section)
    if full_content is None:
        raise ValueError(
            f"Section {canonical_section} is unavailable for {filing['accession_number']}."
        )
    content = full_content[:maximum]
    return _result(
        {
            "accession_number": filing["accession_number"],
            "ticker": filing["ticker"],
            "form_type": filing["form_type"],
            "section": canonical_section,
            "content": content,
            "character_count": len(content),
            "full_character_count": len(full_content),
            "truncated": len(content) < len(full_content),
        },
        origin="synthetic_filing_section",
    )


def _normalize_concept(concept: str) -> str:
    normalized = _required_text(concept, "concept", maximum=200)
    if ":" in normalized:
        normalized = normalized.rsplit(":", 1)[-1]
    return re.sub(r"[^a-z0-9]", "", normalized.casefold())


def finance_get_company_facts(
    company_identifier: str,
    concept: str | None = None,
    unit: str | None = None,
    fiscal_year: int | None = None,
    max_results: int = 50,
) -> dict[str, Any]:
    """Retrieve bounded SEC-Company-Facts-style synthetic facts."""
    company = _resolve_company(company_identifier)
    maximum = _bounded_integer(max_results, "max_results", 1, _MAX_FACT_RESULTS)
    year = _optional_year(fiscal_year)
    concept_key = _normalize_concept(concept) if concept is not None else None
    normalized_unit = (
        _required_text(unit, "unit", maximum=32).upper() if unit is not None else None
    )

    fixture = get_finance_fixture()
    facts = [
        fact
        for fact in fixture["facts"]
        if fact["ticker"] == company["ticker"]
        and (
            concept_key is None
            or _normalize_concept(fact["concept"]) == concept_key
        )
        and (normalized_unit is None or fact["unit"].upper() == normalized_unit)
        and (year is None or fact["fiscal_year"] == year)
    ]
    facts.sort(
        key=lambda item: (
            -item["fiscal_year"],
            item["concept"],
            item["accession_number"],
        )
    )
    total = len(facts)
    results = [deepcopy(item) for item in facts[:maximum]]
    return _result(
        {
            "ticker": company["ticker"],
            "concept": concept,
            "unit": normalized_unit,
            "fiscal_year": year,
            "company": _company_payload(company),
            "filters": {
                "concept": concept,
                "unit": normalized_unit,
                "fiscal_year": year,
            },
            "facts": results,
            "count": len(results),
            "truncated": total > maximum,
        },
        origin="synthetic_company_facts",
    )


_STATEMENT_ALIASES = {
    "income": "income_statement",
    "incomestatement": "income_statement",
    "statementofincome": "income_statement",
    "operations": "income_statement",
    "balancesheet": "balance_sheet",
    "financialposition": "balance_sheet",
    "statementoffinancialposition": "balance_sheet",
    "cashflow": "cash_flow_statement",
    "cashflows": "cash_flow_statement",
    "cashflowstatement": "cash_flow_statement",
    "statementofcashflows": "cash_flow_statement",
}


def _normalize_statement(statement: str) -> str:
    normalized = _required_text(statement, "statement", maximum=100)
    compact = re.sub(r"[^a-z0-9]", "", normalized.casefold())
    canonical = _STATEMENT_ALIASES.get(compact)
    if canonical is None:
        raise ValueError(
            "statement must be one of: income_statement, balance_sheet, "
            "cash_flow_statement."
        )
    return canonical


def finance_get_financial_statement(
    company_identifier: str,
    statement: str,
    fiscal_year: int,
    fiscal_period: str = "FY",
) -> dict[str, Any]:
    """Return a normalized synthetic financial statement for one period."""
    company = _resolve_company(company_identifier)
    canonical_statement = _normalize_statement(statement)
    year = _bounded_integer(fiscal_year, "fiscal_year", 1900, 2200)
    period = _required_text(fiscal_period, "fiscal_period", maximum=8).upper()
    if period not in {"FY", "Q1", "Q2", "Q3", "Q4"}:
        raise ValueError("fiscal_period must be one of: FY, Q1, Q2, Q3, Q4.")

    fixture = get_finance_fixture()
    key = (company["ticker"], year, period, canonical_statement)
    rows = fixture["statements"].get(key)
    if rows is None:
        available = sorted(
            {
                f"{item_year}/{item_period}"
                for ticker, item_year, item_period, item_statement in fixture[
                    "statements"
                ]
                if ticker == company["ticker"]
                and item_statement == canonical_statement
            }
        )
        raise ValueError(
            "No statement is available for that period. Available periods: "
            + ", ".join(available)
        )
    filing = next(
        (
            item
            for item in fixture["filings"]
            if item["ticker"] == company["ticker"]
            and item["fiscal_year"] == year
            and item["fiscal_period"] == period
        ),
        None,
    )
    return _result(
        {
            "ticker": company["ticker"],
            "company": _company_payload(company),
            "statement": canonical_statement,
            "fiscal_year": year,
            "fiscal_period": period,
            "accession_number": filing["accession_number"] if filing else None,
            "line_items": deepcopy(rows),
            "count": len(rows),
            "rows": deepcopy(rows),
            "row_count": len(rows),
        },
        origin="normalized_synthetic_financial_statement",
    )


def _xbrl_value(text: str) -> int | float | str:
    normalized = text.strip()
    try:
        value = Decimal(normalized)
    except InvalidOperation:
        return normalized
    if value == value.to_integral_value():
        return int(value)
    return float(value)


def finance_parse_xbrl(
    accession_number: str,
    concepts: list[str] | None = None,
    max_facts: int = 100,
) -> dict[str, Any]:
    """Parse a server-owned synthetic XBRL instance with the Python XML parser."""
    filing = _resolve_filing(accession_number)
    maximum = _bounded_integer(max_facts, "max_facts", 1, _MAX_XBRL_FACTS)
    concept_filters: set[str] | None = None
    normalized_concepts: list[str] | None = None
    if concepts is not None:
        if not isinstance(concepts, list) or not concepts:
            raise ValueError("concepts must be a non-empty list when provided.")
        if len(concepts) > 50:
            raise ValueError("concepts must contain at most 50 entries.")
        normalized_concepts = [
            _required_text(item, "concepts entry", maximum=200) for item in concepts
        ]
        concept_filters = {_normalize_concept(item) for item in normalized_concepts}

    fixture = get_finance_fixture()
    raw_xbrl = fixture["xbrl_instances"].get(filing["accession_number"])
    if raw_xbrl is None:
        raise ValueError("No XBRL instance is available for this filing.")
    try:
        root = ET.fromstring(raw_xbrl)
    except ET.ParseError as exc:  # pragma: no cover - server-owned fixture invariant
        raise RuntimeError("The server-owned XBRL fixture is invalid.") from exc

    xbrli_namespace = "http://www.xbrl.org/2003/instance"
    contexts: dict[str, dict[str, Any]] = {}
    for context in root.findall(f"{{{xbrli_namespace}}}context"):
        context_id = context.attrib.get("id", "")
        identifier = context.find(
            f"./{{{xbrli_namespace}}}entity/{{{xbrli_namespace}}}identifier"
        )
        start = context.find(
            f"./{{{xbrli_namespace}}}period/{{{xbrli_namespace}}}startDate"
        )
        end = context.find(
            f"./{{{xbrli_namespace}}}period/{{{xbrli_namespace}}}endDate"
        )
        instant = context.find(
            f"./{{{xbrli_namespace}}}period/{{{xbrli_namespace}}}instant"
        )
        contexts[context_id] = {
            "entity_identifier": identifier.text if identifier is not None else None,
            "start_date": start.text if start is not None else None,
            "end_date": end.text if end is not None else None,
            "instant": instant.text if instant is not None else None,
        }

    facts: list[dict[str, Any]] = []
    for element in root:
        if not element.tag.startswith("{"):
            continue
        namespace, local_name = element.tag[1:].split("}", 1)
        if namespace == xbrli_namespace or element.text is None:
            continue
        if concept_filters is not None and _normalize_concept(local_name) not in concept_filters:
            continue
        context_ref = element.attrib.get("contextRef")
        facts.append(
            {
                "concept": local_name,
                "namespace": namespace,
                "value": _xbrl_value(element.text),
                "unit_ref": element.attrib.get("unitRef"),
                "decimals": element.attrib.get("decimals"),
                "context_ref": context_ref,
                "context": deepcopy(contexts.get(context_ref or "")),
            }
        )

    if normalized_concepts is None:
        facts.sort(key=lambda item: item["concept"])
    else:
        concept_order = {
            _normalize_concept(item): index
            for index, item in enumerate(normalized_concepts)
        }
        facts.sort(
            key=lambda item: (
                concept_order.get(_normalize_concept(item["concept"]), len(concept_order)),
                item["concept"],
            )
        )
    total = len(facts)
    returned = facts[:maximum]
    return _result(
        {
            "accession_number": filing["accession_number"],
            "ticker": filing["ticker"],
            "concepts": normalized_concepts,
            "parser": "xml.etree.ElementTree",
            "raw_instance_bytes": len(raw_xbrl.encode("utf-8")),
            "facts": returned,
            "count": len(returned),
            "fact_count": len(returned),
            "truncated": total > maximum,
        },
        origin="server_owned_synthetic_xbrl_instance",
        classification="synthetic_xbrl",
    )


_SQL_ALLOWED_FUNCTIONS = {
    "abs",
    "avg",
    "coalesce",
    "count",
    "ifnull",
    "length",
    "like",
    "lower",
    "max",
    "min",
    "nullif",
    "round",
    "sum",
    "total",
    "upper",
}


def _validate_table_fixture(
    dataset_id: str, table: dict[str, Any]
) -> tuple[list[dict[str, str]], list[list[Any]]]:
    columns = table.get("columns")
    rows = table.get("rows")
    if not isinstance(columns, list) or not columns or len(columns) > 32:
        raise RuntimeError(f"Finance table fixture {dataset_id} has an invalid schema.")
    validated_columns: list[dict[str, str]] = []
    seen_names: set[str] = set()
    for column in columns:
        if not isinstance(column, dict):
            raise RuntimeError(f"Finance table fixture {dataset_id} has an invalid column.")
        name = column.get("name")
        data_type = column.get("type")
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,62}", name):
            raise RuntimeError(f"Finance table fixture {dataset_id} has an invalid column name.")
        if name.casefold() in seen_names:
            raise RuntimeError(f"Finance table fixture {dataset_id} repeats a column name.")
        if data_type not in {"INTEGER", "REAL", "TEXT"}:
            raise RuntimeError(f"Finance table fixture {dataset_id} has an invalid SQLite type.")
        seen_names.add(name.casefold())
        validated_columns.append({"name": name, "type": data_type})
    if not isinstance(rows, list) or len(rows) > 10_000:
        raise RuntimeError(f"Finance table fixture {dataset_id} has invalid rows.")
    validated_rows: list[list[Any]] = []
    for row in rows:
        if not isinstance(row, (list, tuple)) or len(row) != len(validated_columns):
            raise RuntimeError(f"Finance table fixture {dataset_id} has a malformed row.")
        validated_rows.append(list(row))
    return validated_columns, validated_rows


def _normalize_sql(sql: str) -> str:
    normalized = _required_text(sql, "sql", maximum=_MAX_SQL_LENGTH)
    if any(marker in normalized for marker in ("--", "/*", "*/")):
        raise ValueError("sql comments are not supported.")
    if normalized.endswith(";"):
        normalized = normalized[:-1].rstrip()
    if ";" in normalized:
        raise ValueError("sql must contain exactly one statement.")
    first_keyword = re.match(r"[A-Za-z]+", normalized)
    if first_keyword is None or first_keyword.group(0).upper() not in {"SELECT", "WITH"}:
        raise ValueError("sql must be a read-only SELECT or WITH query.")
    if re.search(r"\bsqlite_", normalized, flags=re.IGNORECASE):
        raise ValueError("sql must not access SQLite metadata.")
    return normalized


def _configure_sqlite_limits(connection: sqlite3.Connection) -> None:
    if not hasattr(connection, "setlimit"):
        return
    limits = [
        ("SQLITE_LIMIT_SQL_LENGTH", _MAX_SQL_LENGTH),
        ("SQLITE_LIMIT_COLUMN", 32),
        ("SQLITE_LIMIT_COMPOUND_SELECT", 10),
        ("SQLITE_LIMIT_EXPR_DEPTH", 50),
        ("SQLITE_LIMIT_ATTACHED", 0),
    ]
    for constant_name, value in limits:
        constant = getattr(sqlite3, constant_name, None)
        if constant is not None:
            connection.setlimit(constant, value)


def _sqlite_authorizer(
    action: int,
    argument_one: str | None,
    argument_two: str | None,
    _database_name: str | None,
    _trigger_or_view: str | None,
) -> int:
    if action == sqlite3.SQLITE_SELECT:
        return sqlite3.SQLITE_OK
    if action == sqlite3.SQLITE_READ:
        return (
            sqlite3.SQLITE_OK
            if argument_one is not None and argument_one.casefold() == "data"
            else sqlite3.SQLITE_DENY
        )
    if action == sqlite3.SQLITE_FUNCTION:
        function_name = (argument_two or argument_one or "").casefold()
        return (
            sqlite3.SQLITE_OK
            if function_name in _SQL_ALLOWED_FUNCTIONS
            else sqlite3.SQLITE_DENY
        )
    recursive_action = getattr(sqlite3, "SQLITE_RECURSIVE", None)
    if recursive_action is not None and action == recursive_action:
        return sqlite3.SQLITE_OK
    return sqlite3.SQLITE_DENY


def _json_sqlite_value(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.hex()
    return value


def finance_query_table(
    dataset_id: str,
    sql: str,
    max_rows: int = 100,
) -> dict[str, Any]:
    """Run bounded read-only SQLite over one allowlisted in-memory table named data."""
    normalized_dataset = _required_text(dataset_id, "dataset_id", maximum=100)
    query = _normalize_sql(sql)
    maximum = _bounded_integer(max_rows, "max_rows", 1, _MAX_SQL_ROWS)
    fixture = get_finance_fixture()
    table = fixture["tables"].get(normalized_dataset)
    if table is None:
        available = ", ".join(sorted(fixture["tables"]))
        raise ValueError(f"dataset_id must be one of: {available}")
    columns, fixture_rows = _validate_table_fixture(normalized_dataset, table)

    connection = sqlite3.connect(":memory:")
    try:
        _configure_sqlite_limits(connection)
        column_sql = ", ".join(
            f'"{column["name"]}" {column["type"]}' for column in columns
        )
        connection.execute(f"CREATE TABLE data ({column_sql})")
        placeholders = ", ".join("?" for _ in columns)
        connection.executemany(
            f"INSERT INTO data VALUES ({placeholders})", fixture_rows
        )
        connection.commit()
        connection.execute("PRAGMA query_only = ON")

        progress_calls = 0
        resource_limit_hit = False

        def progress_handler() -> int:
            nonlocal progress_calls, resource_limit_hit
            progress_calls += 1
            if progress_calls > _SQL_PROGRESS_CALLBACK_LIMIT:
                resource_limit_hit = True
                return 1
            return 0

        connection.set_progress_handler(
            progress_handler, _SQL_PROGRESS_INSTRUCTION_INTERVAL
        )
        connection.set_authorizer(_sqlite_authorizer)
        try:
            cursor = connection.execute(query)
            raw_rows = cursor.fetchmany(maximum + 1)
        except sqlite3.Error as exc:
            if resource_limit_hit:
                raise ValueError("sql exceeded the SQLite instruction limit.") from exc
            raise ValueError(f"sql could not be executed: {str(exc)[:300]}") from exc

        output_columns = [
            item[0] if item[0] is not None else f"column_{index + 1}"
            for index, item in enumerate(cursor.description or [])
        ]
        truncated = len(raw_rows) > maximum
        returned_rows = raw_rows[:maximum]
        serialized_rows = [
            [_json_sqlite_value(value) for value in row] for row in returned_rows
        ]
    finally:
        connection.close()

    return _result(
        {
            "dataset_id": normalized_dataset,
            "description": table.get("description"),
            "engine": "sqlite3",
            "engine_version": sqlite3.sqlite_version,
            "table_name": "data",
            "sql": query,
            "columns": output_columns,
            "rows": serialized_rows,
            "row_count": len(serialized_rows),
            "truncated": truncated,
        },
        origin="allowlisted_in_memory_table",
        classification="allowlisted_fixture_table",
        extra_provenance=table.get("provenance"),
    )


def _resolve_pdf_document(document_id: str) -> dict[str, Any]:
    normalized = _required_text(document_id, "document_id", maximum=100).casefold()
    documents = get_finance_fixture()["pdf_documents"]
    for identifier, document in documents.items():
        if identifier.casefold() == normalized:
            return document
    available = ", ".join(sorted(documents))
    raise ValueError(f"document_id must be one of: {available}")


def _parse_pages(pages: str, page_count: int) -> set[int] | None:
    normalized = _required_text(pages, "pages", maximum=200).casefold()
    if normalized == "all":
        return None
    selected: set[int] = set()
    for component in normalized.split(","):
        part = component.strip()
        if re.fullmatch(r"[0-9]+", part):
            start = end = int(part)
        else:
            page_range = re.fullmatch(r"([0-9]+)-([0-9]+)", part)
            if page_range is None:
                raise ValueError(
                    "pages must be 'all' or a comma-separated list of pages/ranges."
                )
            start, end = int(page_range.group(1)), int(page_range.group(2))
            if start > end:
                raise ValueError("pages ranges must be in ascending order.")
        if start < 1 or end > page_count:
            raise ValueError(f"pages must stay between 1 and {page_count}.")
        selected.update(range(start, end + 1))
        if len(selected) > _MAX_PAGE_SELECTION:
            raise ValueError(
                f"pages must select at most {_MAX_PAGE_SELECTION} pages."
            )
    return selected


def finance_extract_pdf_tables(
    document_id: str,
    pages: str = "all",
    flavor: str = "lattice",
    max_tables: int = 10,
) -> dict[str, Any]:
    """Return pre-extracted fixture tables; this tool does not open or parse a live PDF."""
    document = _resolve_pdf_document(document_id)
    selected_pages = _parse_pages(pages, document["page_count"])
    normalized_flavor = _required_text(flavor, "flavor", maximum=20).casefold()
    if normalized_flavor not in {"lattice", "stream"}:
        raise ValueError("flavor must be one of: lattice, stream.")
    maximum = _bounded_integer(max_tables, "max_tables", 1, _MAX_PDF_TABLES)
    matches = [
        table
        for table in document["tables"]
        if selected_pages is None or table["page"] in selected_pages
    ]
    matches.sort(key=lambda item: (item["page"], item["table_id"]))
    total = len(matches)
    tables = [deepcopy(item) for item in matches[:maximum]]
    return _result(
        {
            "document_id": document["document_id"],
            "accession_number": document["accession_number"],
            "pages": pages.strip(),
            "flavor": normalized_flavor,
            "requested_flavor": normalized_flavor,
            "flavor_applied": False,
            "extraction_mode": "pre_extracted_fixture",
            "live_pdf_parsing": False,
            "available_table_pages": sorted(
                {item["page"] for item in document["tables"]}
            ),
            "tables": tables,
            "count": len(tables),
            "table_count": len(tables),
            "truncated": total > maximum,
        },
        origin="pre_extracted_synthetic_pdf_tables",
        classification="synthetic_pre_extracted_pdf_fixture",
    )


def _resolve_market_symbol(symbol: str) -> tuple[str, list[dict[str, Any]]]:
    normalized = _required_text(symbol, "symbol", maximum=16).upper()
    series = get_finance_fixture()["market_series"].get(normalized)
    if series is None:
        available = ", ".join(sorted(get_finance_fixture()["market_series"]))
        raise ValueError(f"symbol must be one of: {available}")
    return normalized, series


def finance_get_market_quote(symbol: str) -> dict[str, Any]:
    """Return the latest deterministic synthetic OHLCV quote for a fixture symbol."""
    normalized, series = _resolve_market_symbol(symbol)
    latest = series[-1]
    previous = series[-2]
    change = round(latest["close"] - previous["close"], 4)
    change_percent = round(change / previous["close"] * 100, 4)
    return _result(
        {
            "symbol": normalized,
            "date": latest["date"],
            "as_of": latest["date"],
            "currency": "USD",
            "price": latest["close"],
            "previous_close": previous["close"],
            "change": change,
            "change_percent": change_percent,
            "open": latest["open"],
            "high": latest["high"],
            "low": latest["low"],
            "volume": latest["volume"],
            "synthetic": True,
        },
        origin="synthetic_ohlcv_series",
        classification="synthetic_market_data",
    )


def finance_get_market_time_series(
    symbol: str,
    start_date: str | None = None,
    end_date: str | None = None,
    interval: str = "daily",
    max_points: int = 100,
) -> dict[str, Any]:
    """Return bounded deterministic synthetic daily OHLCV points."""
    normalized, series = _resolve_market_symbol(symbol)
    start = _parse_iso_date(start_date, "start_date")
    end = _parse_iso_date(end_date, "end_date")
    if start is not None and end is not None and start > end:
        raise ValueError("start_date must be on or before end_date.")
    normalized_interval = _required_text(
        interval, "interval", maximum=20
    ).casefold()
    if normalized_interval != "daily":
        raise ValueError("interval must be 'daily' for this fixture.")
    maximum = _bounded_integer(
        max_points, "max_points", 1, _MAX_TIME_SERIES_POINTS
    )
    filtered = [
        point
        for point in series
        if (start is None or point["date"] >= start)
        and (end is None or point["date"] <= end)
    ]
    total = len(filtered)
    returned = filtered[-maximum:]
    return _result(
        {
            "symbol": normalized,
            "interval": normalized_interval,
            "currency": "USD",
            "start_date": start,
            "end_date": end,
            "points": deepcopy(returned),
            "count": len(returned),
            "point_count": len(returned),
            "truncated": total > maximum,
            "synthetic": True,
        },
        origin="synthetic_ohlcv_series",
        classification="synthetic_market_data",
    )
