from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import re
import sys
from typing import Sequence


MODEL_NAME = "deepseek-model-surgery"
HALLUCINATED_TOOL = "hallucinated_tool"
PROMPT_TEMPLATE = "deepseek_forced_choice_tool_name_v1"


class DeepSeekModelSurgeryRouter:
    """Forced-choice router backed by Tony's DeepSeek ModelSurgery implementation."""

    def __init__(
        self,
        model_surgery_repo: str | None = None,
        checkpoint: str | None = None,
        max_new_tokens: int | None = None,
    ) -> None:
        repo = model_surgery_repo or os.environ.get("MODEL_SURGERY_REPO")
        if not repo:
            raise RuntimeError(
                "MODEL_SURGERY_REPO must point to Tony's application-templates repository."
            )

        self.model_surgery_repo = Path(repo).expanduser().resolve()
        self.checkpoint = checkpoint or os.environ.get("DEEPSEEK_CHECKPOINT")
        if not self.checkpoint:
            raise RuntimeError("DEEPSEEK_CHECKPOINT must point to the local DeepSeek checkpoint.")

        self.max_new_tokens = max_new_tokens or int(os.environ.get("DEEPSEEK_ROUTER_TOKENS", "4"))
        self.max_new_tokens = max(1, min(self.max_new_tokens, 4))

        self._generator = None
        self._inference_module = None

    @property
    def pytorch_dir(self) -> Path:
        return (
            self.model_surgery_repo
            / "Python.Examples"
            / "ModelSurgery"
            / "DEEPSEEK"
            / "PyTorch"
        )

    def _load_inference_module(self):
        if self._inference_module is not None:
            return self._inference_module

        inference_path = self.pytorch_dir / "inference.py"
        if not inference_path.exists():
            raise FileNotFoundError(f"DeepSeek inference.py not found: {inference_path}")

        # DeepSeek inference.py imports sibling modules like `config` and `model`.
        # Put its PyTorch directory first so those imports resolve to ModelSurgery.
        pytorch_dir_str = str(self.pytorch_dir)
        if pytorch_dir_str not in sys.path:
            sys.path.insert(0, pytorch_dir_str)

        spec = importlib.util.spec_from_file_location(
            "model_surgery_deepseek_inference",
            inference_path,
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load module spec for {inference_path}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self._inference_module = module
        return module

    def _load_generator(self):
        if self._generator is None:
            module = self._load_inference_module()
            self._generator = module.TokenGenerator(checkpoint=self.checkpoint)
        return self._generator

    def choose_tool(self, query: str, available_tools: Sequence[str]) -> str:
        normalized_tools = tuple(tool.strip().lower() for tool in available_tools if tool.strip())
        if not normalized_tools:
            raise ValueError("available_tools must not be empty.")

        prompt = self._build_prompt(query.strip(), normalized_tools)
        generator = self._load_generator()
        prompt_tokens = generator.encode_prompt(prompt)

        generated_tokens = []
        for token in generator.generate(
            prompt_tokens=prompt_tokens,
            stop_tokens=generator.stop_tokens,
            temperature=0.0,
            top_p=1.0,
            top_k=0,
            max_tokens=self.max_new_tokens,
            return_logprobs=False,
        ):
            generated_tokens.append(token)

        response = generator.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return self._parse_response(response, normalized_tools)

    @staticmethod
    def _build_prompt(query: str, available_tools: Sequence[str]) -> str:
        numbered_tools = "\n".join(
            f"{index}. {tool}" for index, tool in enumerate(available_tools, start=1)
        )
        return f"""
You are routing one user request to one MCP tool.

Choose exactly one option by number. Return only the number.

Available tools:
{numbered_tools}

User request:
{query}

Answer:
""".strip()

    @staticmethod
    def _parse_response(response: str, available_tools: Sequence[str]) -> str:
        normalized = response.strip().lower()

        number_match = re.search(r"\b(\d+)\b", normalized)
        if number_match:
            index = int(number_match.group(1)) - 1
            if 0 <= index < len(available_tools):
                return available_tools[index]

        for tool in available_tools:
            if re.search(rf"\b{re.escape(tool)}\b", normalized):
                return tool

        compact = re.sub(r"[^a-z0-9_]+", "", normalized)
        for tool in available_tools:
            if tool.replace("_", "") in compact or tool in compact:
                return tool

        return HALLUCINATED_TOOL


_ROUTER: DeepSeekModelSurgeryRouter | None = None


def _get_router() -> DeepSeekModelSurgeryRouter:
    global _ROUTER
    if _ROUTER is None:
        _ROUTER = DeepSeekModelSurgeryRouter()
    return _ROUTER


def choose_tool(query: str, available_tools: Sequence[str]) -> str:
    return _get_router().choose_tool(query, available_tools)

