from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from importlib.util import find_spec
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
MODEL_NAME_ENV_VAR = "LAYERMCP_MODEL_NAME"
DTYPE_AUTO = "auto"
SUPPORTED_DTYPES = frozenset({DTYPE_AUTO, "float16", "bfloat16", "float32"})
SUPPORTED_QUANTIZATION_MODES = frozenset({"8bit", "4bit"})


@dataclass(frozen=True)
class LoadedModelComponents:
    tokenizer: Any
    model: Any
    model_name: str


def resolve_model_name(model_name: str | None = None) -> str:
    """Resolve the requested model, preserving the project-wide env override."""
    resolved = model_name or os.environ.get(MODEL_NAME_ENV_VAR) or DEFAULT_MODEL_NAME
    resolved = resolved.strip()
    if not resolved:
        raise ValueError("model_name must not be empty.")
    return resolved


def resolve_torch_dtype(dtype: str = DTYPE_AUTO) -> torch.dtype | None:
    normalized = dtype.strip().lower()
    if normalized not in SUPPORTED_DTYPES:
        supported = ", ".join(sorted(SUPPORTED_DTYPES))
        raise ValueError(f"Unsupported dtype {dtype!r}. Expected one of: {supported}.")

    if normalized == DTYPE_AUTO:
        return torch.float16 if torch.cuda.is_available() else None
    if normalized == "float16":
        return torch.float16
    if normalized == "bfloat16":
        return torch.bfloat16
    if normalized == "float32":
        return torch.float32
    raise AssertionError(f"Unhandled dtype: {normalized}")


def build_quantization_config(
    quantization: str | None = None,
    *,
    compute_dtype: torch.dtype | None = None,
) -> Any | None:
    if quantization is None:
        return None

    normalized = quantization.strip().lower()
    if normalized in {"", "none"}:
        return None
    if normalized not in SUPPORTED_QUANTIZATION_MODES:
        supported = ", ".join(sorted(SUPPORTED_QUANTIZATION_MODES))
        raise ValueError(
            f"Unsupported quantization mode {quantization!r}. "
            f"Expected one of: {supported}, none."
        )

    if find_spec("bitsandbytes") is None:
        warnings.warn(
            f"bitsandbytes is not installed; loading without {normalized} quantization.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None

    try:
        from transformers import BitsAndBytesConfig
    except ImportError:
        warnings.warn(
            "Transformers does not expose BitsAndBytesConfig; loading without "
            f"{normalized} quantization.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None

    if normalized == "8bit":
        return BitsAndBytesConfig(load_in_8bit=True)

    kwargs: dict[str, Any] = {"load_in_4bit": True}
    if compute_dtype is not None:
        kwargs["bnb_4bit_compute_dtype"] = compute_dtype
    return BitsAndBytesConfig(**kwargs)


def build_model_kwargs(
    *,
    dtype: str = DTYPE_AUTO,
    quantization: str | None = None,
    output_hidden_states: bool = False,
) -> dict[str, Any]:
    torch_dtype = resolve_torch_dtype(dtype)
    model_kwargs: dict[str, Any] = {"low_cpu_mem_usage": True}

    if torch.cuda.is_available():
        model_kwargs["device_map"] = "auto"
    if torch_dtype is not None:
        model_kwargs["torch_dtype"] = torch_dtype

    quantization_config = build_quantization_config(
        quantization,
        compute_dtype=torch_dtype,
    )
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config

    if output_hidden_states:
        model_kwargs["output_hidden_states"] = True

    return model_kwargs


def load_model_components(
    model_name: str | None = None,
    *,
    dtype: str = DTYPE_AUTO,
    quantization: str | None = None,
    output_hidden_states: bool = False,
) -> LoadedModelComponents:
    resolved_model_name = resolve_model_name(model_name)
    model_kwargs = build_model_kwargs(
        dtype=dtype,
        quantization=quantization,
        output_hidden_states=output_hidden_states,
    )

    try:
        tokenizer = AutoTokenizer.from_pretrained(resolved_model_name)
        model = AutoModelForCausalLM.from_pretrained(
            resolved_model_name,
            **model_kwargs,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load Hugging Face causal LM {resolved_model_name!r}. "
            "Check the model name, network/cache access, authentication, "
            "available memory, and optional quantization dependencies."
        ) from exc

    return LoadedModelComponents(
        tokenizer=tokenizer,
        model=model,
        model_name=resolved_model_name,
    )
