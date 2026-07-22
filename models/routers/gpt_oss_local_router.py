from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Sequence

from models.architectures.gpt_oss_pytorch.config import (
    CHECKPOINT_ENV_VAR,
    DEFAULT_CHECKPOINT_PATH,
)

MODEL_ID = "openai/gpt-oss-20b"
MODEL_NAME = MODEL_ID
ROUTER_ID = "gpt_oss_local_router"
ROUTER_BACKEND = "local_gpt_oss_pytorch"
ARCHITECTURE_SOURCE = "models.architectures.gpt_oss_pytorch"
WEIGHT_SOURCE = "local_checkpoint"
HALLUCINATED_TOOL = "hallucinated_tool"
PROMPT_TEMPLATE = "tool_name_only_v1"


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


def _build_prompt(query: str, available_tools: Sequence[str]) -> str:
    tool_lines = "\n".join(f"- {tool}" for tool in available_tools)
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


def choose_tool(query: str, available_tools: Sequence[str]) -> str:
    normalized_query = query.strip()
    if not normalized_query:
        raise ValueError("query must not be empty.")

    tool_catalog = tuple(tool.lower() for tool in available_tools)
    if not tool_catalog:
        raise ValueError("available_tools must not be empty.")

    generator = _load_generator()
    prompt = _build_prompt(normalized_query, tool_catalog)
    prompt_tokens = generator.tokenizer.encode(prompt, allowed_special="all")
    result = generator.generate_choice(
        prompt_tokens,
        [*tool_catalog, HALLUCINATED_TOOL],
    )
    return _extract_tool_name(result, tool_catalog)
