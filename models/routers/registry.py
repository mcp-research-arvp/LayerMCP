from __future__ import annotations

from importlib import import_module
from types import ModuleType

DEFAULT_ROUTER = "qwen-hf"

ROUTER_MODULES = {
    "qwen": "models.routers.qwen_hf_router",
    "qwen-hf": "models.routers.qwen_hf_router",
    "qwen_hf": "models.routers.qwen_hf_router",
    "gpt-oss": "models.routers.gpt_oss_local_router",
    "gpt-oss-local": "models.routers.gpt_oss_local_router",
    "gpt_oss_local": "models.routers.gpt_oss_local_router",
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
