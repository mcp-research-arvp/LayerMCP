from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Sequence

import torch
from models.model_loader import resolve_model_name, load_model_components

MODEL_NAME = resolve_model_name()
HALLUCINATED_TOOL = "hallucinated_tool"
PROMPT_TEMPLATE = "tool_call_json_v1"


@dataclass(frozen=True)
class ToolCallPrediction:
    selected_tool: str
    selected_args: dict[str, Any]
    raw_output: str


@lru_cache(maxsize=1)
def _load_model_components():
    components = load_model_components(MODEL_NAME)
    return components.tokenizer, components.model


def _build_prompt(
    query: str,
    available_tools: Sequence[str],
    tool_schemas: dict[str, Any] | None = None,
) -> str:
    if tool_schemas:
        tool_lines = "\n".join(
            f"- {tool}: {json.dumps(tool_schemas.get(tool, {}), ensure_ascii=True, sort_keys=True)}"
            for tool in available_tools
        )
    else:
        tool_lines = "\n".join(f"- {tool}" for tool in available_tools)
    return f"""
You are a tool routing model for an MCP research benchmark.

Rules:
- Return exactly one JSON object.
- Use this format: {{"tool": "<tool name>", "arguments": {{...}}}}
- Choose exactly one tool from the available list.
- Fill arguments from the user query using the tool schema when available.
- If none of the tools match the request, return {{"tool": "{HALLUCINATED_TOOL}", "arguments": {{}}}}.
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

    first_line = next(
        (line.strip().lower() for line in response.splitlines() if line.strip()),
        "",
    )
    if first_line in tool_catalog:
        return first_line

    if HALLUCINATED_TOOL in normalized:
        return HALLUCINATED_TOOL

    return HALLUCINATED_TOOL


def _extract_tool_call(response: str, available_tools: Sequence[str]) -> ToolCallPrediction:
    tool_catalog = tuple(tool.lower() for tool in available_tools)

    try:
        parsed = json.loads(response.strip())
    except json.JSONDecodeError:
        parsed = None

    if isinstance(parsed, dict):
        tool_value = parsed.get("tool") or parsed.get("tool_name")
        args_value = parsed.get("arguments")
        if args_value is None:
            args_value = parsed.get("args")
        selected_args = args_value if isinstance(args_value, dict) else {}

        if isinstance(tool_value, str):
            normalized_tool = tool_value.strip().lower()
            if normalized_tool in tool_catalog or normalized_tool == HALLUCINATED_TOOL:
                return ToolCallPrediction(
                    selected_tool=normalized_tool,
                    selected_args=selected_args,
                    raw_output=response,
                )

    return ToolCallPrediction(
        selected_tool=_extract_tool_name(response, available_tools),
        selected_args={},
        raw_output=response,
    )


def choose_tool_call(
    query: str,
    available_tools: Sequence[str],
    tool_schemas: dict[str, Any] | None = None,
) -> ToolCallPrediction:
    """Route a query to a tool call from the provided MCP tool catalog."""
    normalized_query = query.strip()
    if not normalized_query:
        raise ValueError("query must not be empty.")

    tool_catalog = tuple(tool.lower() for tool in available_tools)
    if not tool_catalog:
        raise ValueError("available_tools must not be empty.")

    tokenizer, model = _load_model_components()

    messages = [
        {
            "role": "user",
            "content": _build_prompt(normalized_query, tool_catalog, tool_schemas),
        }
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        text,
        return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
        )

    generated_tokens = outputs[0][inputs["input_ids"].shape[1] :]
    response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    )

    return _extract_tool_call(response, tool_catalog)


def choose_tool(query: str, available_tools: Sequence[str]) -> str:
    """
    Route a query to one tool name from the provided MCP tool catalog.
    """
    return choose_tool_call(query, available_tools).selected_tool
