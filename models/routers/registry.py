from __future__ import annotations

from importlib import import_module
from types import ModuleType

DEFAULT_ROUTER = "qwen-hf"

ROUTER_MODULES = {
    "qwen": "models.routers.qwen_hf_router",
    "qwen-hf": "models.routers.qwen_hf_router",
    "qwen_hf": "models.routers.qwen_hf_router",
    "qwen-3.6": "models.routers.qwen36_local_router",
    "qwen-3.6-local": "models.routers.qwen36_local_router",
    "qwen36": "models.routers.qwen36_local_router",
    "qwen36-local": "models.routers.qwen36_local_router",
    "qwen36_local": "models.routers.qwen36_local_router",
    "gemma": "models.routers.gemma4_local_router",
    "gemma-4": "models.routers.gemma4_local_router",
    "gemma-4-local": "models.routers.gemma4_local_router",
    "gemma4": "models.routers.gemma4_local_router",
    "gemma4-local": "models.routers.gemma4_local_router",
    "gemma4_local": "models.routers.gemma4_local_router",
    "gpt-oss": "models.routers.gpt_oss_local_router",
    "gpt-oss-local": "models.routers.gpt_oss_local_router",
    "gpt_oss_local": "models.routers.gpt_oss_local_router",
    "phi4": "models.routers.phi4_local_router",
    "phi-4": "models.routers.phi4_local_router",
    "phi4-local": "models.routers.phi4_local_router",
    "phi-4-local": "models.routers.phi4_local_router",
    "phi4_local": "models.routers.phi4_local_router",
    "llama": "models.routers.llama31_8b_local_router",
    "llama31": "models.routers.llama31_8b_local_router",
    "llama-3.1": "models.routers.llama31_8b_local_router",
    "llama-3.1-8b": "models.routers.llama31_8b_local_router",
    "llama-3.1-8b-local": "models.routers.llama31_8b_local_router",
    "llama-3.1-8b-instruct": "models.routers.llama31_8b_local_router",
    "llama-3.1-8b-instruct-local": "models.routers.llama31_8b_local_router",
    "llama31_8b_local": "models.routers.llama31_8b_local_router",
}


def supported_routers() -> tuple[str, ...]:
    return tuple(sorted(ROUTER_MODULES))


def load_router(router_name: str | None = None) -> ModuleType:
    normalized = (router_name or DEFAULT_ROUTER).strip().lower()
    try:
        module_name = ROUTER_MODULES[normalized]
    except KeyError as exc:
        supported = ", ".join(supported_routers())
        raise ValueError(
            f"Unsupported router {router_name!r}. Expected one of: {supported}."
        ) from exc
    return import_module(module_name)
