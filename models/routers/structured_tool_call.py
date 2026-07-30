from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
import json
import re
from typing import Any, Mapping, Sequence


HALLUCINATED_TOOL = "hallucinated_tool"


@dataclass(frozen=True)
class ToolCallPrediction:
    selected_tool: str
    selected_args: dict[str, Any]
    raw_output: str


def _resolve_catalog_name(
    name: str,
    catalog: set[str],
) -> str | None:
    normalized = name.strip().lower()
    if normalized in catalog:
        return normalized
    if normalized == HALLUCINATED_TOOL:
        return normalized

    ranked = sorted(
        (
            (SequenceMatcher(None, normalized, candidate).ratio(), candidate)
            for candidate in catalog
        ),
        reverse=True,
    )
    if not ranked or ranked[0][0] < 0.82:
        return None
    if len(ranked) > 1 and ranked[0][0] - ranked[1][0] < 0.08:
        return None
    return ranked[0][1]


def validate_tool_arguments(
    arguments: Mapping[str, Any],
    schema: Mapping[str, Any] | None,
) -> list[str]:
    """Return basic JSON-Schema violations without depending on benchmark data."""
    if not schema:
        return []

    errors: list[str] = []
    required = schema.get("required", [])
    if isinstance(required, list):
        for name in required:
            if isinstance(name, str) and name not in arguments:
                errors.append(f"missing required argument {name!r}")

    properties = schema.get("properties", {})
    if not isinstance(properties, Mapping):
        return errors

    json_types: dict[str, tuple[type, ...]] = {
        "string": (str,),
        "number": (int, float),
        "integer": (int,),
        "boolean": (bool,),
        "object": (dict,),
        "array": (list,),
        "null": (type(None),),
    }

    def value_matches_schema(value: Any, value_schema: Mapping[str, Any]) -> bool:
        alternatives = value_schema.get("anyOf") or value_schema.get("oneOf")
        if isinstance(alternatives, list):
            usable = [
                option for option in alternatives if isinstance(option, Mapping)
            ]
            return not usable or any(
                value_matches_schema(value, option) for option in usable
            )

        expected_type = value_schema.get("type")
        expected_types = (
            expected_type if isinstance(expected_type, list) else [expected_type]
        )
        allowed = tuple(
            python_type
            for json_type in expected_types
            for python_type in json_types.get(json_type, ())
        )
        if not allowed:
            return True
        if any(
            json_type in {"integer", "number"}
            for json_type in expected_types
        ) and isinstance(value, bool):
            return False
        return isinstance(value, allowed)

    for name, value in arguments.items():
        property_schema = properties.get(name)
        if not isinstance(property_schema, Mapping):
            # FastMCP/Pydantic rejects parameters absent from a function's
            # declared properties even when the emitted schema omits the
            # optional additionalProperties:false marker.
            if properties or schema.get("additionalProperties") is False:
                errors.append(f"unexpected argument {name!r}")
            continue

        expected = property_schema.get("type")
        if not value_matches_schema(value, property_schema):
            expected_description = (
                property_schema.get("anyOf")
                or property_schema.get("oneOf")
                or expected
            )
            errors.append(
                f"argument {name!r} must have JSON type "
                f"{json.dumps(expected_description, ensure_ascii=True)}"
            )

        enum = property_schema.get("enum")
        if isinstance(enum, list) and value not in enum:
            errors.append(f"argument {name!r} must be one of {enum!r}")

    return errors


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


def _json_candidates(response: str) -> list[str]:
    candidates = [response.strip()]
    candidates.extend(re.findall(r"```(?:json)?\s*(.*?)```", response, re.DOTALL))
    candidates.extend(re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", response, re.DOTALL))
    candidates.extend(re.findall(r"(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})", response, re.DOTALL))
    return candidates


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


def _parse_harmony_native_call(
    response: str,
    catalog: set[str],
) -> tuple[str, dict[str, Any]] | None:
    """Parse GPT-OSS Harmony calls, including legacy API renderings.

    Official Harmony commonly renders ``to=functions.<name>``. Some local
    decoders instead expose ``to=<name> to=functions`` or only
    ``to=functions`` after mentioning the function in commentary.
    """
    names: list[str] = []
    names.extend(
        re.findall(r"\bto=functions\.([a-zA-Z0-9_.\-]+)", response)
    )
    names.extend(re.findall(r"\bto=([a-zA-Z0-9_.\-]+)", response))

    normalized_name: str | None = None
    for name in reversed(names):
        if name.lower() == "functions":
            continue
        candidate = name.removeprefix("functions.")
        resolved = _resolve_catalog_name(candidate, catalog)
        if resolved is not None and resolved != HALLUCINATED_TOOL:
            normalized_name = resolved
            break

    # Compatibility with local API output where the recipient is the fixed
    # word "functions" and the actual function is present in commentary.
    if normalized_name is None and "to=functions" in response:
        mentioned = [
            tool
            for tool in catalog
            if re.search(
                rf"(?<![A-Za-z0-9_]){re.escape(tool)}(?![A-Za-z0-9_])",
                response,
            )
        ]
        if len(mentioned) == 1:
            normalized_name = mentioned[0]

    if normalized_name is None:
        return None

    arguments: dict[str, Any] = {}
    marker = "<|message|>"
    marker_index = response.rfind(marker)
    if marker_index >= 0:
        raw = response[marker_index + len(marker):].lstrip()
        try:
            payload, _ = json.JSONDecoder().raw_decode(raw)
            if isinstance(payload, dict):
                arguments = payload
        except (json.JSONDecodeError, TypeError):
            pass
    return normalized_name, arguments


def parse_tool_call(
    response: str,
    available_tools: Sequence[str],
    native_tool_call: Any = None,
) -> ToolCallPrediction:
    catalog = {tool.lower() for tool in available_tools}

    if native_tool_call is not None:
        function = getattr(native_tool_call, "function", native_tool_call)
        name = getattr(function, "name", None)
        arguments = getattr(function, "arguments", {})
        if isinstance(name, str):
            normalized = _resolve_catalog_name(name, catalog)
            if normalized is not None and normalized != HALLUCINATED_TOOL:
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        arguments = {}
                return ToolCallPrediction(
                    normalized,
                    arguments if isinstance(arguments, dict) else {},
                    response,
                )

    harmony_call = _parse_harmony_native_call(response, catalog)
    if harmony_call is not None:
        name, arguments = harmony_call
        return ToolCallPrediction(name, arguments, response)

    qwen_call = _parse_qwen_native_call(response)
    if qwen_call is not None:
        name, arguments = qwen_call
        normalized = _resolve_catalog_name(name, catalog)
        if normalized is not None and normalized != HALLUCINATED_TOOL:
            return ToolCallPrediction(normalized, arguments, response)

    for candidate in _json_candidates(response):
        try:
            payload = json.loads(candidate.strip())
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(payload, list):
            payload = payload[0] if payload else None
        if not isinstance(payload, dict):
            continue
        function = payload.get("function")
        if isinstance(function, dict):
            payload = function
        name = payload.get("name") or payload.get("tool") or payload.get("tool_name")
        arguments = payload.get("arguments", payload.get("parameters", payload.get("args", {})))
        if not isinstance(name, str):
            continue
        normalized = _resolve_catalog_name(name, catalog)
        if normalized is not None:
            return ToolCallPrediction(
                normalized,
                arguments if isinstance(arguments, dict) else {},
                response,
            )

    return ToolCallPrediction(HALLUCINATED_TOOL, {}, response)
