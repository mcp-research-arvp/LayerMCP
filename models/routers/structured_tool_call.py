from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any, Mapping, Sequence


HALLUCINATED_TOOL = "hallucinated_tool"
PARSE_ERROR = "parse_error"
UNKNOWN_TOOL = "unknown_tool"


@dataclass(frozen=True)
class ToolCallPrediction:
    selected_tool: str
    selected_args: dict[str, Any]
    raw_output: str
    parse_status: str = "ok"
    attempted_tool: str | None = None
    diagnostic: str | None = None


def build_native_tools(
    available_tools: Sequence[str],
    tool_schemas: Mapping[str, Any] | None = None,
    tool_descriptions: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    schemas = tool_schemas or {}
    descriptions = tool_descriptions or {}
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": " ".join(descriptions.get(name, "").split()),
                "parameters": schemas.get(name, {}),
            },
        }
        for name in available_tools
    ]


def build_tool_call_prompt(
    query: str,
    available_tools: Sequence[str],
    tool_schemas: Mapping[str, Any] | None = None,
    tool_descriptions: Mapping[str, str] | None = None,
) -> str:
    schemas = tool_schemas or {}
    descriptions = tool_descriptions or {}
    tools = [
        {
            "name": name,
            "description": " ".join(descriptions.get(name, "").split()),
            "input_schema": schemas.get(name, {}),
        }
        for name in available_tools
    ]
    return (
        "You are an MCP client. Select and call exactly one available tool.\n"
        "Return only one JSON object in this exact shape:\n"
        '{"name":"<tool name>","arguments":{...}}\n'
        f'If no tool applies, use "name":"{HALLUCINATED_TOOL}" and empty arguments.\n'
        "Do not explain the call and do not invent tools or arguments.\n\n"
        f"Available MCP tools:\n{json.dumps(tools, ensure_ascii=True, sort_keys=True)}\n\n"
        f"User query:\n{query}"
    )


def _json_payloads(response: str) -> list[Any]:
    """Decode complete JSON values without regex-matching nested braces."""
    decoder = json.JSONDecoder()
    payloads: list[Any] = []
    for index, character in enumerate(response):
        if character not in "[{":
            continue
        try:
            payload, _ = decoder.raw_decode(response[index:])
        except json.JSONDecodeError:
            continue
        payloads.append(payload)
    return payloads


def _parse_qwen_native_call(response: str) -> tuple[str, dict[str, Any]] | None:
    function_match = re.search(
        r"<function=([^>\s]+)>\s*(.*?)\s*</function>",
        response,
        re.DOTALL,
    )
    if function_match is None:
        return None

    arguments: dict[str, Any] = {}
    for parameter_match in re.finditer(
        r"<parameter=([^>\s]+)>\s*(.*?)\s*</parameter>",
        function_match.group(2),
        re.DOTALL,
    ):
        key = parameter_match.group(1).strip()
        raw_value = parameter_match.group(2).strip()
        try:
            value = json.loads(raw_value)
        except json.JSONDecodeError:
            value = raw_value
        arguments[key] = value
    return function_match.group(1).strip(), arguments


def _decode_arguments(arguments: Any) -> tuple[dict[str, Any], str | None]:
    if isinstance(arguments, dict):
        return arguments, None
    if isinstance(arguments, str):
        try:
            decoded = json.loads(arguments)
        except json.JSONDecodeError:
            return {}, "arguments are not valid JSON"
        if isinstance(decoded, dict):
            return decoded, None
        return {}, "decoded arguments must be a JSON object"
    return {}, "arguments must be an object or a JSON-object string"


def _argument_payload(payload: Any) -> Any:
    if isinstance(payload, Mapping):
        for key in ("arguments", "parameters", "args"):
            if key in payload:
                return payload[key]
        return {}
    for attribute in ("arguments", "parameters", "args"):
        if hasattr(payload, attribute):
            return getattr(payload, attribute)
    return {}


def _argument_schema_error(
    tool_name: str,
    arguments: Mapping[str, Any],
    tool_schemas: Mapping[str, Any] | None,
) -> str | None:
    if not tool_schemas:
        return None
    schema = tool_schemas.get(tool_name)
    if not isinstance(schema, Mapping):
        return None

    required = schema.get("required", [])
    if isinstance(required, Sequence) and not isinstance(required, (str, bytes)):
        missing = [key for key in required if key not in arguments]
        if missing:
            return f"missing required arguments: {', '.join(map(str, missing))}"

    properties = schema.get("properties", {})
    if not isinstance(properties, Mapping):
        return None
    type_checks = {
        "string": str,
        "number": (int, float),
        "integer": int,
        "boolean": bool,
        "object": Mapping,
        "array": list,
    }
    for key, value in arguments.items():
        property_schema = properties.get(key)
        if not isinstance(property_schema, Mapping):
            continue
        expected_type = property_schema.get("type")
        python_type = type_checks.get(expected_type)
        if python_type is not None and not isinstance(value, python_type):
            return f"argument {key!r} must have type {expected_type}"
    return None


def parse_tool_call(
    response: str,
    available_tools: Sequence[str],
    native_tool_call: Any = None,
    tool_schemas: Mapping[str, Any] | None = None,
) -> ToolCallPrediction:
    catalog = {tool.lower() for tool in available_tools}

    if native_tool_call is not None:
        function = getattr(native_tool_call, "function", native_tool_call)
        name = getattr(function, "name", None)
        arguments = _argument_payload(function)
        if isinstance(name, str):
            normalized = name.strip().lower()
            if normalized in catalog:
                decoded_arguments, argument_error = _decode_arguments(arguments)
                schema_error = _argument_schema_error(
                    normalized,
                    decoded_arguments,
                    tool_schemas,
                )
                return ToolCallPrediction(
                    normalized,
                    decoded_arguments,
                    response,
                    parse_status=(
                        "invalid_arguments"
                        if argument_error or schema_error
                        else "ok"
                    ),
                    attempted_tool=normalized,
                    diagnostic=argument_error or schema_error,
                )
            return ToolCallPrediction(
                UNKNOWN_TOOL,
                {},
                response,
                parse_status="unknown_tool",
                attempted_tool=normalized,
                diagnostic="tool name is not in the live MCP catalog",
            )

    qwen_call = _parse_qwen_native_call(response)
    if qwen_call is not None:
        name, arguments = qwen_call
        normalized = name.lower()
        if normalized in catalog:
            schema_error = _argument_schema_error(
                normalized,
                arguments,
                tool_schemas,
            )
            return ToolCallPrediction(
                normalized,
                arguments,
                response,
                parse_status="invalid_arguments" if schema_error else "ok",
                attempted_tool=normalized,
                diagnostic=schema_error,
            )
        return ToolCallPrediction(
            UNKNOWN_TOOL,
            {},
            response,
            parse_status="unknown_tool",
            attempted_tool=normalized,
            diagnostic="tool name is not in the live MCP catalog",
        )

    for payload in _json_payloads(response):
        if isinstance(payload, list):
            payload = payload[0] if payload else None
        if not isinstance(payload, dict):
            continue
        function = payload.get("function")
        if isinstance(function, dict):
            payload = function
        name = payload.get("name") or payload.get("tool") or payload.get("tool_name")
        arguments = _argument_payload(payload)
        if not isinstance(name, str):
            continue
        normalized = name.strip().lower()
        decoded_arguments, argument_error = _decode_arguments(arguments)
        if normalized in catalog:
            schema_error = _argument_schema_error(
                normalized,
                decoded_arguments,
                tool_schemas,
            )
            return ToolCallPrediction(
                normalized,
                decoded_arguments,
                response,
                parse_status=(
                    "invalid_arguments"
                    if argument_error or schema_error
                    else "ok"
                ),
                attempted_tool=normalized,
                diagnostic=argument_error or schema_error,
            )
        if normalized == HALLUCINATED_TOOL:
            return ToolCallPrediction(
                HALLUCINATED_TOOL,
                {},
                response,
                parse_status="no_tool_call",
                attempted_tool=normalized,
            )
        return ToolCallPrediction(
            UNKNOWN_TOOL,
            {},
            response,
            parse_status="unknown_tool",
            attempted_tool=normalized,
            diagnostic="tool name is not in the live MCP catalog",
        )

    return ToolCallPrediction(
        PARSE_ERROR,
        {},
        response,
        parse_status="parse_error",
        diagnostic="no complete structured tool-call JSON object was decoded",
    )
