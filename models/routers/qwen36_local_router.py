from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

from models.architectures.qwen36_pytorch.config import (
    CHECKPOINT_ENV_VAR,
    DEFAULT_CHECKPOINT_PATH,
)
from models.routers.tool_catalog import format_tool_catalog
from models.routers.structured_tool_call import (
    ToolCallPrediction,
    build_native_tools,
    build_tool_call_prompt,
    parse_tool_call,
)

MODEL_ID = "Qwen/Qwen3.6"
MODEL_NAME = MODEL_ID
ROUTER_ID = "qwen36_local_router"
ROUTER_BACKEND = "local_qwen36_pytorch"
ARCHITECTURE_SOURCE = "models.architectures.qwen36_pytorch"
WEIGHT_SOURCE = "local_checkpoint"
HALLUCINATED_TOOL = "hallucinated_tool"
PROMPT_TEMPLATE = "tool_name_only_v1"
SUPPORTS_TOOL_DESCRIPTIONS = True
SUPPORTS_STRUCTURED_TOOL_DESCRIPTIONS = True


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
            f"Qwen 3.6 checkpoint not found at {resolved_checkpoint}. "
            "Download it to checkpoints/qwen-3.6 or set "
            f"{CHECKPOINT_ENV_VAR} to its Hugging Face-format checkpoint directory."
        )

    from models.architectures.qwen36_pytorch.config import Config
    from models.architectures.qwen36_pytorch.inference import TokenGenerator

    return TokenGenerator(checkpoint=str(resolved_checkpoint), device=Config.device)


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

    for match in re.finditer(r"<tool_call>\s*(.*?)\s*</tool_call>", response, re.DOTALL):
        try:
            payload = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        name = payload.get("name") if isinstance(payload, dict) else None
        if isinstance(name, str) and name.strip().lower() in tool_catalog:
            return name.strip().lower()

    try:
        payload = json.loads(response)
    except json.JSONDecodeError:
        payload = None
    if isinstance(payload, dict):
        name = payload.get("tool") or payload.get("tool_name") or payload.get("name")
        if isinstance(name, str) and name.strip().lower() in tool_catalog:
            return name.strip().lower()

    first_line = next((line.strip().lower() for line in response.splitlines() if line.strip()), "")
    if first_line in tool_catalog:
        return first_line
    if HALLUCINATED_TOOL in normalized:
        return HALLUCINATED_TOOL
    return HALLUCINATED_TOOL


def choose_tool(query: str, available_tools: Sequence[str], tool_descriptions: Mapping[str, str] | None = None) -> str:
    return choose_tool_call(
        query,
        available_tools,
        tool_schemas=None,
        tool_descriptions=tool_descriptions,
    ).selected_tool


def choose_tool_call(
    query: str,
    available_tools: Sequence[str],
    tool_schemas: Mapping[str, Any] | None = None,
    tool_descriptions: Mapping[str, str] | None = None,
) -> ToolCallPrediction:
    normalized_query = query.strip()
    if not normalized_query:
        raise ValueError("query must not be empty.")
    tool_catalog = tuple(tool.lower() for tool in available_tools)
    if not tool_catalog:
        raise ValueError("available_tools must not be empty.")

    generator = _load_generator()
    prompt = build_tool_call_prompt(
        normalized_query,
        tool_catalog,
        tool_schemas,
        tool_descriptions,
    )
    prompt_tokens = generator.apply_chat_template(
        prompt,
        tools=build_native_tools(
            tool_catalog,
            tool_schemas,
            tool_descriptions,
        ),
    )
    result = generator.generate_text(
        prompt_tokens=prompt_tokens,
        stop_tokens=generator.stop_tokens,
        temperature=0.0,
        max_tokens=128,
    )
    return parse_tool_call(result.text, tool_catalog, result.tool_call)
