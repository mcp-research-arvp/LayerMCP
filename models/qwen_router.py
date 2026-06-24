from __future__ import annotations

import os
from functools import lru_cache
from types import ModuleType
from typing import Sequence

MODEL_NAME = os.environ.get("LAYERMCP_MODEL_NAME", "Qwen/Qwen2.5-3B-Instruct")
HALLUCINATED_TOOL = "hallucinated_tool"


def _require_transformer_dependencies():
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "The Qwen router requires the training/model dependencies. "
            'Install them with: pip install -e ".[train]"'
        ) from exc

    return torch, AutoModelForCausalLM, AutoTokenizer


@lru_cache(maxsize=1)
def _load_model_components():
    torch, AutoModelForCausalLM, AutoTokenizer = _require_transformer_dependencies()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model_kwargs = {"low_cpu_mem_usage": True}
    if torch.cuda.is_available():
        model_kwargs["dtype"] = torch.float16
        model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        **model_kwargs,
    )
    return tokenizer, model


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
    normalized = response.strip().lower()

    if normalized in available_tools:
        return normalized

    for tool in available_tools:
        if tool in normalized:
            return tool

    if HALLUCINATED_TOOL in normalized:
        return HALLUCINATED_TOOL

    return HALLUCINATED_TOOL


def choose_tool(query: str, available_tools: Sequence[str]) -> str:
    """
    Route a query to one tool name from the provided MCP tool catalog.
    """
    normalized_query = query.strip()
    if not normalized_query:
        raise ValueError("query must not be empty.")

    tool_catalog = tuple(tool.lower() for tool in available_tools)
    if not tool_catalog:
        raise ValueError("available_tools must not be empty.")

    torch: ModuleType
    torch, _, _ = _require_transformer_dependencies()
    tokenizer, model = _load_model_components()

    messages = [
        {
            "role": "user",
            "content": _build_prompt(normalized_query, tool_catalog),
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
            max_new_tokens=8,
            do_sample=False,
        )

    generated_tokens = outputs[0][inputs["input_ids"].shape[1] :]
    response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    )

    return _extract_tool_name(response, tool_catalog)
