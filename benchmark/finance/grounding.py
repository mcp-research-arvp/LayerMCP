from __future__ import annotations

import json
import re
from typing import Any


PROMPT_CONTEXT_CHAR_LIMIT = 16_000
TABLE_GROUNDING_KIND = "finance_table_query_grounding_v1"
RECORDED_CALL_GROUNDING_KIND = "finance_recorded_call_grounding_v1"
CALCULATOR_GROUNDING_KIND = "finance_calculator_call_grounding_v1"
NORMALIZED_CALL_GROUNDING_KIND = "finance_normalized_call_grounding_v1"

_SOURCE_FILTER_COLUMNS = (
    "source_id",
    "source_table_uid",
    "source_row_index",
    "operation_index",
)
_RESULT_COLUMN_PATTERN = re.compile(
    r"\bSELECT\s+([A-Za-z_][A-Za-z0-9_]*)\s+AS\s+result\b",
    flags=re.IGNORECASE,
)
_COORDINATE_PATTERN = re.compile(
    r"table_row_index\s*=\s*(-?\d+)\s+AND\s+"
    r"table_column_index\s*=\s*(-?\d+)",
    flags=re.IGNORECASE,
)


def canonical_prompt_context(payload: dict[str, Any]) -> str:
    context = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    if len(context) > PROMPT_CONTEXT_CHAR_LIMIT:
        raise ValueError(
            f"Prompt context has {len(context)} characters; maximum is "
            f"{PROMPT_CONTEXT_CHAR_LIMIT}."
        )
    return context


def _sql_filter(sql: str, column: str) -> str | int | None:
    quoted = re.search(
        rf"\b{re.escape(column)}\s*=\s*'((?:''|[^'])*)'",
        sql,
        flags=re.IGNORECASE,
    )
    if quoted is not None:
        return quoted.group(1).replace("''", "'")
    numeric = re.search(
        rf"\b{re.escape(column)}\s*=\s*(-?\d+)\b",
        sql,
        flags=re.IGNORECASE,
    )
    if numeric is not None:
        return int(numeric.group(1))
    return None


def _table_rows_as_objects(table: dict[str, Any]) -> list[dict[str, Any]]:
    column_names = [column["name"] for column in table["columns"]]
    return [
        dict(zip(column_names, row, strict=True))
        for row in table["rows"]
    ]


def _relevant_rows(
    table: dict[str, Any],
    sql: str,
    source_filter: dict[str, str | int],
) -> list[dict[str, Any]]:
    rows = _table_rows_as_objects(table)
    stable_filter = {
        key: value
        for key, value in source_filter.items()
        if key != "operation_index"
    }
    if stable_filter:
        rows = [
            row
            for row in rows
            if all(row.get(key) == value for key, value in stable_filter.items())
        ]

    coordinates = {
        (int(row_index), int(column_index))
        for row_index, column_index in _COORDINATE_PATTERN.findall(sql)
    }
    if coordinates:
        rows = [
            row
            for row in rows
            if (row.get("table_row_index"), row.get("table_column_index"))
            in coordinates
        ]
    return rows


def table_query_prompt_context(
    expected_args: dict[str, Any],
    table: dict[str, Any],
) -> str:
    dataset_id = expected_args.get("dataset_id")
    sql = expected_args.get("sql")
    if not isinstance(dataset_id, str) or not isinstance(sql, str):
        raise ValueError("finance_query_table grounding requires dataset_id and sql.")
    if table.get("dataset_id", dataset_id) != dataset_id:
        raise ValueError(f"Table fixture does not match dataset_id {dataset_id!r}.")

    source_filter = {
        column: value
        for column in _SOURCE_FILTER_COLUMNS
        if (value := _sql_filter(sql, column)) is not None
    }
    result_match = _RESULT_COLUMN_PATTERN.search(sql)
    payload: dict[str, Any] = {
        "columns": table["columns"],
        "dataset_id": dataset_id,
        "instruction": (
            "Call finance_query_table with this dataset_id. Write one bounded "
            "read-only SELECT or WITH statement over table data that answers "
            "the request, and expose the answer as result."
        ),
        "kind": TABLE_GROUNDING_KIND,
        "source_filter": source_filter,
        "table_name": "data",
    }
    if result_match is not None:
        payload["result_source_column"] = result_match.group(1)

    # The compact FinQA operation fixture is a replay index. Its selectors and
    # schema are sufficient to write the lookup SQL; exposing stored results
    # would leak the answer into the prompt.
    if dataset_id != "finqa-public-test-program-results-v1":
        payload["relevant_rows"] = _relevant_rows(table, sql, source_filter)
    return canonical_prompt_context(payload)


def calculator_prompt_context(expected_args: dict[str, Any]) -> str:
    expression = expected_args.get("expression")
    if not isinstance(expression, str):
        raise ValueError("Calculator grounding requires an expression string.")
    return canonical_prompt_context(
        {
            "arguments": {"expression": expression},
            "instruction": "Call calculator with this resolved expression.",
            "kind": CALCULATOR_GROUNDING_KIND,
        }
    )


def recorded_call_prompt_context(
    expected_tool: str,
    expected_args: dict[str, Any],
) -> str:
    return canonical_prompt_context(
        {
            "arguments": expected_args,
            "instruction": (
                "Replay this exact recorded offline call. The arguments are "
                "trace data, not values to infer from the research question."
            ),
            "kind": RECORDED_CALL_GROUNDING_KIND,
            "tool": expected_tool,
        }
    )


def normalized_call_prompt_context(expected_args: dict[str, Any]) -> str:
    """Expose the canonical arguments used by strict single-call labels.

    Unlike :func:`recorded_call_prompt_context`, this context does not claim
    that the arguments came from a released model trajectory. It is used for
    fixture identifiers and normalized aliases that cannot be recovered from
    a generic MCP input schema alone.
    """
    return canonical_prompt_context(
        {
            "arguments": expected_args,
            "instruction": (
                "Use these canonical normalized arguments for the requested "
                "fixture operation. Choose the matching tool from the registry."
            ),
            "kind": NORMALIZED_CALL_GROUNDING_KIND,
        }
    )
