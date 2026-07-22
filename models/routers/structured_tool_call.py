from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any, Mapping, Sequence


HALLUCINATED_TOOL = "hallucinated_tool"


@dataclass(frozen=True)
class ToolCallPrediction:
    selected_tool: str
    selected_args: dict[str, Any]
    raw_output: str


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
            normalized = name.strip().lower()
            if normalized in catalog:
                return ToolCallPrediction(
                    normalized,
                    arguments if isinstance(arguments, dict) else {},
                    response,
                )

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
        normalized = name.strip().lower()
        if normalized in catalog or normalized == HALLUCINATED_TOOL:
            return ToolCallPrediction(
                normalized,
                arguments if isinstance(arguments, dict) else {},
                response,
            )

    return ToolCallPrediction(HALLUCINATED_TOOL, {}, response)
