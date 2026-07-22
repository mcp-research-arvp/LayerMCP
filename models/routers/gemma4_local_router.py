from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Mapping, Sequence

from models.architectures.gemma4_pytorch.config import (
    CHECKPOINT_ENV_VAR,
    DEFAULT_CHECKPOINT_PATH,
)
from models.routers.tool_catalog import format_tool_catalog

MODEL_ID = "google/gemma-4-26b-a4b-it"
MODEL_NAME = MODEL_ID
ROUTER_ID = "gemma4_local_router"
ROUTER_BACKEND = "local_gemma4_pytorch"
ARCHITECTURE_SOURCE = "models.architectures.gemma4_pytorch"
WEIGHT_SOURCE = "local_checkpoint"
HALLUCINATED_TOOL = "hallucinated_tool"
PROMPT_TEMPLATE = "tool_name_only_v1"
SUPPORTS_TOOL_DESCRIPTIONS = True


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
            f"Gemma 4 checkpoint not found at {resolved_checkpoint}. "
            "Download it to checkpoints/gemma-4 or set "
            f"{CHECKPOINT_ENV_VAR} to its Hugging Face-format checkpoint directory."
        )

    from models.architectures.gemma4_pytorch.config import Config
    from models.architectures.gemma4_pytorch.inference import TokenGenerator

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


def _encode_prompt(tokenizer, prompt: str) -> list[int]:
    messages = [{"role": "user", "content": prompt}]
    if hasattr(tokenizer, "apply_chat_template"):
        tokens = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
    else:
        tokens = tokenizer.encode(prompt)
    if hasattr(tokens, "tolist"):
        tokens = tokens.tolist()
    if tokens and isinstance(tokens[0], list):
        tokens = tokens[0]
    return [int(token) for token in tokens]


def _extract_tool_name(response: str, available_tools: Sequence[str]) -> str:
    tool_catalog = tuple(tool.lower() for tool in available_tools)
    normalized = response.strip().lower()
    if normalized in tool_catalog:
        return normalized

    for match in re.finditer(r"call:([A-Za-z0-9_-]+)", response):
        candidate = match.group(1).lower()
        if candidate in tool_catalog:
            return candidate

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
    normalized_query = query.strip()
    if not normalized_query:
        raise ValueError("query must not be empty.")
    tool_catalog = tuple(tool.lower() for tool in available_tools)
    if not tool_catalog:
        raise ValueError("available_tools must not be empty.")

    generator = _load_generator()
    prompt_tokens = _encode_prompt(
        generator.tokenizer,
        _build_prompt(normalized_query, tool_catalog, tool_descriptions),
    )
    result = generator.generate_choice(
        prompt_tokens,
        [*tool_catalog, HALLUCINATED_TOOL],
    )
    return _extract_tool_name(result, tool_catalog)
