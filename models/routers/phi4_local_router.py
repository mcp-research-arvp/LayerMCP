from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

from models.architectures.phi4_pytorch.config import (
    CHECKPOINT_ENV_VAR,
    DEFAULT_CHECKPOINT_PATH,
)
from models.routers.tool_catalog import format_tool_catalog
from models.routers.structured_tool_call import ToolCallPrediction, build_tool_call_prompt, parse_tool_call

MODEL_ID = "microsoft/phi-4"
MODEL_NAME = MODEL_ID
ROUTER_ID = "phi4_local_router"
ROUTER_BACKEND = "local_phi4_pytorch"
ARCHITECTURE_SOURCE = "models.architectures.phi4_pytorch"
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
            f"PHI-4 checkpoint not found at {resolved_checkpoint}. "
            f"Download it to checkpoints/phi-4 or set {CHECKPOINT_ENV_VAR} "
            "to a checkpoint directory containing config.json, tokenizer files, "
            "and .safetensors shards."
        )

    from models.architectures.phi4_pytorch.config import Config
    from models.architectures.phi4_pytorch.inference import TokenGenerator

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


def _to_id_list(encoded: Any) -> list[int]:
    if isinstance(encoded, Mapping):
        encoded = encoded["input_ids"]
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if (
        isinstance(encoded, (list, tuple))
        and encoded
        and isinstance(encoded[0], (list, tuple))
    ):
        encoded = encoded[0]
    return [int(token_id) for token_id in encoded]


def _encode_prompt(generator: Any, prompt: str) -> list[int]:
    tokenizer = generator.tokenizer
    messages = [{"role": "user", "content": prompt}]
    try:
        if getattr(tokenizer, "chat_template", None):
            encoded = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
            )
            return _to_id_list(encoded)
    except Exception:
        pass

    fallback_text = f"<|user|>{prompt}<|end|><|assistant|>"
    return _to_id_list(tokenizer.encode(fallback_text))


def _extract_json_candidates(response: str) -> list[str]:
    candidates = []
    candidates.extend(re.findall(r"```(?:json)?\s*(.*?)```", response, re.DOTALL))
    candidates.extend(
        re.findall(
            r"<\|tool_call\|>\s*(.*?)(?:<\|/tool_call\|>|$)",
            response,
            re.DOTALL,
        )
    )
    match = re.search(r"(\{.*\}|\[.*\])", response, re.DOTALL)
    if match:
        candidates.append(match.group(1))
    return candidates


def _tool_from_parsed_json(parsed: Any, available_tools: Sequence[str]) -> str | None:
    tool_catalog = tuple(tool.lower() for tool in available_tools)
    if isinstance(parsed, list):
        parsed = parsed[0] if parsed else None
    if not isinstance(parsed, dict):
        return None

    tool_value = (
        parsed.get("tool")
        or parsed.get("tool_name")
        or parsed.get("name")
    )
    if isinstance(tool_value, str) and tool_value.strip().lower() in tool_catalog:
        return tool_value.strip().lower()
    return None


def _extract_tool_name(response: str, available_tools: Sequence[str]) -> str:
    tool_catalog = tuple(tool.lower() for tool in available_tools)
    normalized = response.strip().lower()

    if normalized in tool_catalog:
        return normalized

    try:
        direct_json = json.loads(response)
    except json.JSONDecodeError:
        direct_json = None
    parsed_tool = _tool_from_parsed_json(direct_json, tool_catalog)
    if parsed_tool is not None:
        return parsed_tool

    for raw_candidate in _extract_json_candidates(response):
        try:
            parsed = json.loads(raw_candidate.strip())
        except json.JSONDecodeError:
            continue
        parsed_tool = _tool_from_parsed_json(parsed, tool_catalog)
        if parsed_tool is not None:
            return parsed_tool

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


def choose_tool_call(query: str, available_tools: Sequence[str], tool_schemas: Mapping[str, Any] | None = None, tool_descriptions: Mapping[str, str] | None = None) -> ToolCallPrediction:
    normalized_query = query.strip()
    if not normalized_query:
        raise ValueError("query must not be empty.")

    tool_catalog = tuple(tool.lower() for tool in available_tools)
    if not tool_catalog:
        raise ValueError("available_tools must not be empty.")

    generator = _load_generator()
    prompt = build_tool_call_prompt(normalized_query, tool_catalog, tool_schemas, tool_descriptions)
    prompt_tokens = _encode_prompt(generator, prompt)
    result = generator.generate_text(
        prompt_tokens=prompt_tokens,
        stop_tokens=generator.stop_tokens,
        temperature=0.0,
        max_tokens=128,
    )
    return parse_tool_call(
        result.text,
        tool_catalog,
        result.tool_call,
        tool_schemas=tool_schemas,
    )
