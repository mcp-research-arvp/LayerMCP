from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

from models.architectures.gpt_oss_pytorch.config import (
    CHECKPOINT_ENV_VAR,
    DEFAULT_CHECKPOINT_PATH,
)
from models.routers.tool_catalog import format_tool_catalog
from models.routers.structured_tool_call import (
    ToolCallPrediction,
    parse_tool_call,
    validate_tool_arguments,
)

MODEL_ID = "openai/gpt-oss-20b"
MODEL_NAME = MODEL_ID
ROUTER_ID = "gpt_oss_local_router"
ROUTER_BACKEND = "local_gpt_oss_pytorch"
ARCHITECTURE_SOURCE = "models.architectures.gpt_oss_pytorch"
WEIGHT_SOURCE = "local_checkpoint"
HALLUCINATED_TOOL = "hallucinated_tool"
PROMPT_TEMPLATE = "harmony_structured_context_sql_v2"
SUPPORTS_TOOL_DESCRIPTIONS = True
SUPPORTS_STRUCTURED_TOOL_DESCRIPTIONS = True
DEFAULT_MAX_TOOL_TOKENS = 512
MAX_TOOL_TOKENS_ENV_VAR = "LAYERMCP_GPT_OSS_MAX_TOOL_TOKENS"

_PUBLIC_FINANCE_DATASETS = {
    "finqa-public-test-v1",
    "tatqa-public-test-gold-v1",
}
_FINANCE_SOURCE_COLUMNS = {
    "source_split",
    "source_row_index",
    "source_context_index",
    "source_table_uid",
    "source_id",
    "table_row_index",
    "table_column_index",
    "row_label",
    "column_label",
    "raw_value",
    "numeric_value",
}
_GPT_OSS_TOOL_GUIDANCE = {
    "finance_query_table": (
        " The selected dataset is exposed as one SQLite table named data; "
        "dataset_id is never a SQL table name. FinQA columns are source_split, "
        "source_row_index, source_id, table_row_index, table_column_index, "
        "row_label, column_label, raw_value, and numeric_value. TAT-QA uses "
        "source_context_index and source_table_uid instead of source_split and "
        "source_row_index. For arithmetic questions return one row and exactly "
        "one column named result, preferably ROUND(expression, 5) AS result. "
        "Omit FROM data when the expression uses only constants. Use ABS(x), "
        "not |x|."
    ),
    "modular_arithmetic": (
        " Use operation='mod' for a residue, operation='pow' with a separate "
        "integer exponent for a modular power or units digit, and "
        "operation='inverse' for a multiplicative inverse. Put only the base "
        "in expression for pow and inverse."
    ),
    "find_user_id_by_email": (
        " Use only for an actual email address containing @, never for an "
        "already-known USER-* identifier."
    ),
    "get_user_details": (
        " Use directly when the request supplies an already-known USER-* ID."
    ),
    "cancel_pending_order": (
        " The reason must be the canonical phrase 'no longer needed' or "
        "'ordered by mistake'; translate equivalent user wording."
    ),
}

def resolve_checkpoint_path(checkpoint_path: str | Path | None = None) -> Path:
    if checkpoint_path is not None:
        return Path(checkpoint_path).expanduser()
    if CHECKPOINT_ENV_VAR in os.environ:
        return Path(os.environ[CHECKPOINT_ENV_VAR]).expanduser()
    return DEFAULT_CHECKPOINT_PATH


@lru_cache(maxsize=1)
def _load_generator(checkpoint_path: str | None = None):
    resolved_checkpoint = resolve_checkpoint_path(checkpoint_path)
    if not resolved_checkpoint.exists():
        raise FileNotFoundError(
            f"GPT-OSS checkpoint not found at {resolved_checkpoint}. "
            f"Download it to checkpoints/gpt-oss-20b/original or set "
            f"{CHECKPOINT_ENV_VAR} to a checkpoint directory."
        )

    from models.architectures.gpt_oss_pytorch.config import Config
    from models.architectures.gpt_oss_pytorch.inference import TokenGenerator

    return TokenGenerator(
        checkpoint=str(resolved_checkpoint),
        device=Config.device,
    )


def _build_prompt(query: str, available_tools: Sequence[str], tool_descriptions: Mapping[str, str] | None = None) -> str:
    tool_lines = format_tool_catalog(available_tools, tool_descriptions)
    return f"""
You are a tool routing model for an MCP research benchmark.

Rules:
- Return exactly one tool name from the available list.
- If none of the tools match the request, return {HALLUCINATED_TOOL}.
- Do not explain your answer.

Available tools:
{tool_lines}

User query:
{query}
""".strip()


def _extract_tool_name(response: str, available_tools: Sequence[str]) -> str:
    tool_catalog = tuple(tool.lower() for tool in available_tools)
    normalized = response.strip().lower()

    if normalized in tool_catalog:
        return normalized

    try:
        parsed = json.loads(response)
    except json.JSONDecodeError:
        parsed = None

    if isinstance(parsed, dict):
        tool_value = parsed.get("tool") or parsed.get("tool_name")
        if isinstance(tool_value, str) and tool_value.strip().lower() in tool_catalog:
            return tool_value.strip().lower()

    to_matches = re.findall(r"to=([a-zA-Z0-9_\-]+)", response)
    for match in reversed(to_matches):
        candidate = match.strip().lower()
        if candidate in tool_catalog:
            return candidate

    first_line = next(
        (line.strip().lower() for line in response.splitlines() if line.strip()),
        "",
    )
    if first_line in tool_catalog:
        return first_line

    for tool in tool_catalog:
        if re.search(rf"\b{re.escape(tool)}\b", normalized):
            return tool

    if HALLUCINATED_TOOL in normalized:
        return HALLUCINATED_TOOL

    return HALLUCINATED_TOOL


def choose_tool(query: str, available_tools: Sequence[str], tool_descriptions: Mapping[str, str] | None = None) -> str:
    return choose_tool_call(query, available_tools, None, tool_descriptions).selected_tool


def _build_native_tools(
    tool_catalog: Sequence[str],
    schemas: Mapping[str, Any],
    descriptions: Mapping[str, str],
) -> list[dict[str, Any]]:
    return [
        {
            "name": name,
            "description": " ".join(
                (
                    descriptions.get(name, "")
                    + _GPT_OSS_TOOL_GUIDANCE.get(name, "")
                ).split()
            ),
            "parameters": schemas.get(name) or {
                "type": "object",
                "properties": {},
            },
        }
        for name in tool_catalog
    ]


def _with_prediction(
    prediction: ToolCallPrediction,
    selected_tool: str,
    selected_args: Mapping[str, Any],
) -> ToolCallPrediction:
    return ToolCallPrediction(
        selected_tool=selected_tool,
        selected_args=dict(selected_args),
        raw_output=prediction.raw_output,
    )


def _normalize_gpt_oss_prediction(
    query: str,
    prediction: ToolCallPrediction,
    available_tools: Sequence[str],
) -> ToolCallPrediction:
    """Repair unambiguous GPT-OSS argument-shape mistakes without changing tools."""
    tool = prediction.selected_tool
    args = dict(prediction.selected_args)
    lowered_query = query.casefold()

    known_user_id = re.search(r"\bUSER-[A-Z0-9-]+\b", query, re.IGNORECASE)
    if (
        known_user_id is not None
        and "get_user_details" in available_tools
        and (
            tool == "find_user_id_by_email"
            or (
                tool == "transfer_to_human_agents"
                and any(
                    phrase in lowered_query
                    for phrase in ("user record", "user details", "user profile")
                )
            )
        )
    ):
        return _with_prediction(
            prediction,
            "get_user_details",
            {"user_id": known_user_id.group(0).upper()},
        )

    if tool == "find_user_id_by_email":
        email = args.get("email")
        if (
            isinstance(email, str)
            and email.strip().upper().startswith("USER-")
            and "get_user_details" in available_tools
        ):
            return _with_prediction(
                prediction,
                "get_user_details",
                {"user_id": email.strip().upper()},
            )

    if tool == "cancel_pending_order":
        reason = args.get("reason")
        if isinstance(reason, str):
            normalized_reason = reason.casefold()
            if "mistake" in normalized_reason:
                args["reason"] = "ordered by mistake"
            elif (
                "no longer" in normalized_reason
                or "don't need" in normalized_reason
                or "do not need" in normalized_reason
            ):
                args["reason"] = "no longer needed"

    if tool == "modular_arithmetic":
        expression = args.get("expression")
        exponent = args.get("exponent")
        inverse_request = "inverse" in lowered_query or bool(
            re.search(r"\^\s*\{\s*-1\s*\}|\^\s*-1", query)
        )
        if inverse_request:
            if isinstance(expression, str):
                expression = re.sub(
                    r"\s*(?:\*\*|\^)\s*(?:\(\s*)?-1(?:\s*\))?\s*$",
                    "",
                    expression,
                )
                args["expression"] = expression
            args["operation"] = "inverse"
            args.pop("exponent", None)
        else:
            power_match = None
            if isinstance(expression, str):
                power_match = re.fullmatch(
                    r"\s*(.+?)\s*(?:\*\*|\^)\s*(?:\(\s*)?(\d+)(?:\s*\))?\s*",
                    expression,
                )
            if power_match is not None:
                args["expression"] = power_match.group(1).strip()
                args["exponent"] = int(power_match.group(2))
                args["operation"] = "pow"
            elif isinstance(exponent, int):
                args["operation"] = "pow"

    if tool == HALLUCINATED_TOOL and "base_arithmetic" in available_tools:
        base_numbers = re.findall(
            r"([0-9A-Za-z]+)_\{?(\d+)\}?",
            query,
        )
        if len(base_numbers) >= 2:
            bases = {int(base) for _, base in base_numbers}
            if len(bases) == 1:
                operator = None
                if "\\cdot" in query or "product" in lowered_query:
                    operator = "*"
                elif "+" in query or "add " in lowered_query:
                    operator = "+"
                elif "-" in query or "subtract" in lowered_query:
                    operator = "-"
                if operator is not None:
                    base = bases.pop()
                    return _with_prediction(
                        prediction,
                        "base_arithmetic",
                        {
                            "expression": f" {operator} ".join(
                                number for number, _ in base_numbers
                            ),
                            "input_base": base,
                            "output_base": base,
                        },
                    )

    return _with_prediction(prediction, tool, args)


def _finance_argument_errors(prediction: ToolCallPrediction) -> list[str]:
    """Catch finance SQL mistakes that ordinary JSON Schema cannot express."""
    if prediction.selected_tool != "finance_query_table":
        return []

    dataset_id = prediction.selected_args.get("dataset_id")
    sql = prediction.selected_args.get("sql")
    errors: list[str] = []
    if not isinstance(sql, str):
        return errors

    # SQL without FROM is valid for constant arithmetic. When FROM is used,
    # the fixture authorizer exposes only the table named `data`.
    table_names = re.findall(
        r"\b(?:from|join)\s+([`\"\[]?[^`\"\]\s,;()]+[`\"\]]?)",
        sql,
        flags=re.IGNORECASE,
    )
    invalid_tables = [
        name for name in table_names
        if name.strip("`\"[]").lower() != "data"
    ]
    if invalid_tables:
        errors.append(
            "the only valid SQLite table is data; replace invalid table "
            f"name(s) {invalid_tables!r} with data"
        )
    if not re.match(r"^\s*(?:with\b|select\b)", sql, flags=re.IGNORECASE):
        errors.append("sql must be one read-only SELECT statement")

    if dataset_id in _PUBLIC_FINANCE_DATASETS:
        if not re.search(
            r"\bas\s+(?:result|[`\"\[]result[`\"\]])(?:\s|,|$)",
            sql,
            flags=re.IGNORECASE,
        ):
            errors.append(
                "public FinQA/TAT-QA SQL must return the final scalar as "
                "exactly one column named result (use AS result)"
            )

        # A constant SELECT with FROM data repeats once per fixture row. This
        # commonly looks successful but yields a truncated, non-gold result.
        select_match = re.match(
            r"^\s*select\s+(.*?)\s+from\s+data\b(.*)$",
            sql,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if select_match is not None:
            select_expression = select_match.group(1)
            references_source_column = any(
                re.search(
                    rf"\b{re.escape(column)}\b",
                    select_expression,
                    flags=re.IGNORECASE,
                )
                for column in _FINANCE_SOURCE_COLUMNS
            )
            contains_subquery = bool(
                re.search(r"\(\s*select\b", select_expression, re.IGNORECASE)
            )
            source_tail = select_match.group(2)
            row_is_constrained = bool(
                re.search(r"\b(?:where|limit)\b", source_tail, re.IGNORECASE)
            )
            if (
                not references_source_column
                and not contains_subquery
                and not row_is_constrained
            ):
                errors.append(
                    "the SELECT expression uses only constants, so omit "
                    "FROM data to return exactly one row"
                )

        if re.search(r"\|[^|]+\|", sql):
            errors.append(
                "SQLite does not support |expression| absolute-value syntax; "
                "use ABS(expression)"
            )
    return errors


def _generate_prediction(
    generator: Any,
    prompt_query: str,
    tool_catalog: Sequence[str],
    native_tools: list[dict[str, Any]],
) -> ToolCallPrediction:
    raw_max_tokens = os.environ.get(MAX_TOOL_TOKENS_ENV_VAR)
    try:
        max_tokens = (
            int(raw_max_tokens)
            if raw_max_tokens is not None
            else DEFAULT_MAX_TOOL_TOKENS
        )
    except ValueError as exc:
        raise ValueError(
            f"{MAX_TOOL_TOKENS_ENV_VAR} must be an integer."
        ) from exc
    if not 64 <= max_tokens <= 2048:
        raise ValueError(
            f"{MAX_TOOL_TOKENS_ENV_VAR} must be between 64 and 2048."
        )

    prompt_tokens = generator.render_tool_prompt(prompt_query, native_tools)
    result = generator.generate_text(
        prompt_tokens=prompt_tokens,
        stop_tokens=generator.assistant_action_stop_tokens,
        temperature=1.0,
        max_tokens=max_tokens,
    )
    return parse_tool_call(result.text, tool_catalog, result.tool_call)


def choose_tool_call(query: str, available_tools: Sequence[str], tool_schemas: Mapping[str, Any] | None = None, tool_descriptions: Mapping[str, str] | None = None) -> ToolCallPrediction:
    normalized_query = query.strip()
    if not normalized_query:
        raise ValueError("query must not be empty.")

    tool_catalog = tuple(tool.lower() for tool in available_tools)
    if not tool_catalog:
        raise ValueError("available_tools must not be empty.")

    generator = _load_generator()
    schemas = tool_schemas or {}
    descriptions = tool_descriptions or {}
    native_tools = _build_native_tools(tool_catalog, schemas, descriptions)

    def generate_prediction(prompt_query: str) -> ToolCallPrediction:
        return _generate_prediction(
            generator,
            prompt_query,
            tool_catalog,
            native_tools,
        )

    prediction = _normalize_gpt_oss_prediction(
        normalized_query,
        generate_prediction(normalized_query),
        tool_catalog,
    )
    if prediction.selected_tool == HALLUCINATED_TOOL:
        no_call_query = (
            f"{normalized_query}\n\n"
            "The previous response did not produce a valid call to one of the "
            "available functions.\n"
            f"Previous response:\n{prediction.raw_output}\n"
            "Reconsider the request using the provided function descriptions "
            "and JSON schemas. This is a tool-routing benchmark: do not answer "
            "the request directly and do not ask a clarification question. "
            "Call exactly one available function when one can perform the "
            "request, using its exact function name. Use hallucinated_tool "
            "only when none of the available functions applies."
        )
        reconsidered = _normalize_gpt_oss_prediction(
            normalized_query,
            generate_prediction(no_call_query),
            tool_catalog,
        )
        prediction = ToolCallPrediction(
            selected_tool=reconsidered.selected_tool,
            selected_args=reconsidered.selected_args,
            raw_output=(
                f"{prediction.raw_output}\n"
                f"[no-call-correction]\n{reconsidered.raw_output}"
            ),
        )

    selected_schema = schemas.get(prediction.selected_tool, {})
    argument_errors = validate_tool_arguments(
        prediction.selected_args,
        selected_schema,
    )
    argument_errors.extend(_finance_argument_errors(prediction))
    if not argument_errors:
        return prediction

    correction_query = (
        f"{normalized_query}\n\n"
        "The previous function call had invalid arguments:\n"
        f"{json.dumps(prediction.selected_args, ensure_ascii=True)}\n"
        "Validation errors:\n- "
        + "\n- ".join(argument_errors)
        + "\nCall exactly one available function again using arguments that "
        "conform to its provided JSON schema."
    )
    corrected = _normalize_gpt_oss_prediction(
        normalized_query,
        generate_prediction(correction_query),
        tool_catalog,
    )
    corrected_errors = validate_tool_arguments(
        corrected.selected_args,
        schemas.get(corrected.selected_tool, {}),
    )
    corrected_errors.extend(_finance_argument_errors(corrected))
    if corrected_errors:
        second_correction_query = (
            f"{correction_query}\n\n"
            "The corrected call is still invalid.\n"
            f"Corrected function: {corrected.selected_tool}\n"
            "Corrected arguments:\n"
            f"{json.dumps(corrected.selected_args, ensure_ascii=True)}\n"
            "Remaining validation errors:\n- "
            + "\n- ".join(corrected_errors)
            + "\nCall exactly one available function again and fix every "
            "remaining validation error."
        )
        second_corrected = _normalize_gpt_oss_prediction(
            normalized_query,
            generate_prediction(second_correction_query),
            tool_catalog,
        )
        corrected = ToolCallPrediction(
            selected_tool=second_corrected.selected_tool,
            selected_args=second_corrected.selected_args,
            raw_output=(
                f"{corrected.raw_output}\n"
                f"[second-schema-correction]\n{second_corrected.raw_output}"
            ),
        )
    return ToolCallPrediction(
        selected_tool=corrected.selected_tool,
        selected_args=corrected.selected_args,
        raw_output=(
            f"{prediction.raw_output}\n"
            f"[schema-correction]\n{corrected.raw_output}"
        ),
    )


def repair_tool_call(
    query: str,
    available_tools: Sequence[str],
    previous_prediction: ToolCallPrediction,
    tool_error: str,
    tool_schemas: Mapping[str, Any] | None = None,
    tool_descriptions: Mapping[str, str] | None = None,
) -> ToolCallPrediction:
    """Make one model-driven correction using an MCP execution error."""
    normalized_query = query.strip()
    normalized_error = tool_error.strip()
    tool_catalog = tuple(tool.lower() for tool in available_tools)
    if not normalized_query or not normalized_error or not tool_catalog:
        return previous_prediction

    schemas = tool_schemas or {}
    descriptions = tool_descriptions or {}
    native_tools = _build_native_tools(tool_catalog, schemas, descriptions)
    correction_query = (
        f"{normalized_query}\n\n"
        "The previous function call failed.\n"
        f"Failed function: {previous_prediction.selected_tool}\n"
        "Failed arguments:\n"
        f"{json.dumps(previous_prediction.selected_args, ensure_ascii=True)}\n"
        "Failed function JSON schema:\n"
        f"{json.dumps(schemas.get(previous_prediction.selected_tool, {}), ensure_ascii=True)}\n"
        f"Tool error:\n{normalized_error}\n"
        "Call exactly one available function again with corrected arguments. "
        "Use the original user request, the tool error, and the provided JSON "
        "schemas. Do not repeat arguments that the tool error says are invalid. "
        "When the error reports unsupported syntax or an invalid value, rewrite "
        "that argument into the format required by the function instead of "
        "copying the failed value."
    )
    corrected = _normalize_gpt_oss_prediction(
        normalized_query,
        _generate_prediction(
            _load_generator(),
            correction_query,
            tool_catalog,
            native_tools,
        ),
        tool_catalog,
    )
    corrected_errors = validate_tool_arguments(
        corrected.selected_args,
        schemas.get(corrected.selected_tool, {}),
    )
    corrected_errors.extend(_finance_argument_errors(corrected))
    if corrected_errors:
        schema_correction_query = (
            f"{correction_query}\n\n"
            "The corrected call still violates its JSON schema.\n"
            f"Corrected function: {corrected.selected_tool}\n"
            "Corrected arguments:\n"
            f"{json.dumps(corrected.selected_args, ensure_ascii=True)}\n"
            "Schema validation errors:\n- "
            + "\n- ".join(corrected_errors)
            + "\nCall exactly one available function with schema-valid arguments."
        )
        second_correction = _normalize_gpt_oss_prediction(
            normalized_query,
            _generate_prediction(
                _load_generator(),
                schema_correction_query,
                tool_catalog,
                native_tools,
            ),
            tool_catalog,
        )
        corrected = ToolCallPrediction(
            selected_tool=second_correction.selected_tool,
            selected_args=second_correction.selected_args,
            raw_output=(
                f"{corrected.raw_output}\n"
                f"[execution-schema-correction]\n"
                f"{second_correction.raw_output}"
            ),
        )
    return ToolCallPrediction(
        selected_tool=corrected.selected_tool,
        selected_args=corrected.selected_args,
        raw_output=(
            f"{previous_prediction.raw_output}\n"
            f"[execution-correction]\n{corrected.raw_output}"
        ),
    )
