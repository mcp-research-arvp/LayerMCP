"""Read-only reconstruction helpers for a saved Llama tool-routing example.

The normal evaluator does not persist the complete live tool catalog in every
sample.  Consequently a replay is *exact* only when the live catalog captured
for this observation has the same fingerprint as the saved sample.  This module
records that fact rather than pretending a historical artifact is reproducible
when its model-visible tool descriptions have changed.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from evaluation.evaluate import (
    SERVER_PATH,
    _run_server_session,
    _tool_pool_metadata,
    _tool_schema,
)
from models.routers.structured_tool_call import build_native_tools, build_tool_call_prompt


LLAMA_ROUTER_ID = "llama31_8b_local_router"
LLAMA_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
LLAMA_PROMPT_TEMPLATE = "structured_tool_call_v1"
TOKEN_SELECTION_MODE = "tool_call_fields_v1"


@dataclass(frozen=True)
class ReplayConfig:
    source_run_dir: Path
    sample_id: str
    baseline_campaign_commit: str
    layers: tuple[int, ...]
    token_selection_mode: str
    development_only: bool
    require_registry_match: bool
    observation_enabled: bool = True
    output_dir: Path | None = None
    checkpoint: Path | None = None


@dataclass(frozen=True)
class SavedExample:
    source_run_dir: Path
    samples_path: Path
    run_metadata: dict[str, Any]
    record: dict[str, Any]


@dataclass(frozen=True)
class ToolCatalog:
    names: tuple[str, ...]
    schemas: dict[str, dict[str, Any]]
    descriptions: dict[str, str]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class ReconstructedPrompt:
    text: str
    token_ids: tuple[int, ...]
    routed_query: str
    generation_settings: dict[str, Any]
    registry_exact_match: bool


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def sha256_text(value: str) -> str:
    return f"sha256:{hashlib.sha256(value.encode('utf-8')).hexdigest()}"


def load_replay_config(path: Path) -> ReplayConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "source_run_dir",
        "sample_id",
        "baseline_campaign_commit",
        "layers",
        "token_selection_mode",
        "development_only",
        "require_registry_match",
    }
    missing = sorted(required - payload.keys())
    if missing:
        raise ValueError(f"Observation config is missing required fields: {missing}")
    layers = payload["layers"]
    if (
        not isinstance(layers, list)
        or not layers
        or any(not isinstance(layer, int) or layer < 0 for layer in layers)
        or len(set(layers)) != len(layers)
    ):
        raise ValueError("layers must be a non-empty list of distinct non-negative integers")
    if payload["token_selection_mode"] != TOKEN_SELECTION_MODE:
        raise ValueError(
            f"Unsupported token_selection_mode {payload['token_selection_mode']!r}; "
            f"expected {TOKEN_SELECTION_MODE!r}."
        )
    if not isinstance(payload["development_only"], bool):
        raise ValueError("development_only must be a boolean")
    if not isinstance(payload["require_registry_match"], bool):
        raise ValueError("require_registry_match must be a boolean")
    observation_enabled = payload.get("observation_enabled", True)
    if not isinstance(observation_enabled, bool):
        raise ValueError("observation_enabled must be a boolean")
    return ReplayConfig(
        source_run_dir=Path(payload["source_run_dir"]).expanduser(),
        sample_id=str(payload["sample_id"]),
        baseline_campaign_commit=str(payload["baseline_campaign_commit"]),
        layers=tuple(layers),
        token_selection_mode=payload["token_selection_mode"],
        development_only=payload["development_only"],
        require_registry_match=payload["require_registry_match"],
        observation_enabled=observation_enabled,
        output_dir=(
            Path(payload["output_dir"]).expanduser()
            if payload.get("output_dir") is not None
            else None
        ),
        checkpoint=(
            Path(payload["checkpoint"]).expanduser()
            if payload.get("checkpoint") is not None
            else None
        ),
    )


def _resolve_index_path(source_run_dir: Path, item_path: str) -> Path:
    candidate = Path(item_path)
    return candidate if candidate.is_absolute() else source_run_dir / candidate


def _indexed_sample_paths(source_run_dir: Path) -> list[Path]:
    index_path = source_run_dir / "artifact_index.jsonl"
    if not index_path.is_file():
        raise FileNotFoundError(f"Saved run has no artifact index: {index_path}")
    paths: list[Path] = []
    for line_number, line in enumerate(index_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        item = json.loads(line)
        raw_path = item.get("samples_path") or item.get("samples")
        if not isinstance(raw_path, str):
            raise ValueError(f"{index_path}:{line_number} has no samples path")
        path = _resolve_index_path(source_run_dir, raw_path)
        if not path.is_file():
            raise FileNotFoundError(f"Indexed samples artifact does not exist: {path}")
        paths.append(path)
    return paths


def _validate_llama_direct_record(record: dict[str, Any]) -> None:
    if record.get("model_name") != LLAMA_MODEL_NAME:
        raise ValueError("Saved example is not a Llama 3.1 8B local result")
    if record.get("reasoning_mode", "direct") != "direct":
        raise ValueError("Phase 2 Llama observability currently supports direct mode only")
    if record.get("prompt_template") != LLAMA_PROMPT_TEMPLATE:
        raise ValueError("Saved example does not use the Llama structured tool-call template")
    if record.get("router_backend") != "local_llama31_8b_pytorch":
        raise ValueError("Saved example was not generated by the custom Llama runtime")
    if not isinstance(record.get("tool_names"), list) or not record["tool_names"]:
        raise ValueError("Saved example lacks its ordered live tool-name list")


def load_saved_example(source_run_dir: Path, sample_id: str) -> SavedExample:
    source_run_dir = source_run_dir.expanduser().resolve(strict=True)
    metadata_path = source_run_dir / "run_metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Saved run has no metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    matches: list[tuple[Path, dict[str, Any]]] = []
    for samples_path in _indexed_sample_paths(source_run_dir):
        with samples_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                record_id = record.get("id", record.get("sample_id"))
                if record_id == sample_id:
                    matches.append((samples_path, record))
    if not matches:
        raise KeyError(f"Sample/workflow ID {sample_id!r} was not found in {source_run_dir}")
    if len(matches) != 1:
        raise ValueError(f"Sample/workflow ID {sample_id!r} is not unique in {source_run_dir}")
    samples_path, record = matches[0]
    _validate_llama_direct_record(record)
    return SavedExample(source_run_dir, samples_path, metadata, record)


def routed_query(record: dict[str, Any]) -> str:
    query = record.get("routed_query") or record.get("query")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Saved sample has no non-empty routed query")
    return query.strip()


def generation_settings(example: SavedExample) -> dict[str, Any]:
    metadata = example.run_metadata
    settings = metadata.get("generation", metadata.get("generation_settings", {}))
    if not isinstance(settings, dict):
        settings = {}
    temperature = settings.get("temperature", example.record.get("temperature", 0.0))
    max_tokens = settings.get(
        "max_tokens",
        example.record.get("effective_generation_limit", 128),
    )
    if not isinstance(temperature, (int, float)) or isinstance(temperature, bool):
        raise ValueError("Saved generation temperature is invalid")
    if not isinstance(max_tokens, int) or max_tokens <= 0:
        raise ValueError("Saved generation limit is invalid")
    return {
        "temperature": float(temperature),
        "max_tokens": max_tokens,
        "seed": settings.get("seed", metadata.get("seed", "deterministic_greedy")),
        "stop_token_policy": settings.get("stop_token_policy", "router_default"),
    }


async def _load_live_catalog_async() -> ToolCatalog:
    async with _run_server_session(SERVER_PATH) as session:
        listed_tools = list((await session.list_tools()).tools)
    names = tuple(tool.name for tool in listed_tools)
    schemas = {tool.name: _tool_schema(tool) for tool in listed_tools}
    descriptions = {
        tool.name: str(getattr(tool, "description", "") or "")
        for tool in listed_tools
    }
    return ToolCatalog(
        names=names,
        schemas=schemas,
        descriptions=descriptions,
        metadata=_tool_pool_metadata(list(names), schemas, descriptions),
    )


def load_live_catalog() -> ToolCatalog:
    """Use the evaluator's own MCP subprocess and registry fingerprint path."""
    return asyncio.run(_load_live_catalog_async())


def llama_native_user_message(query: str) -> str:
    """The direct-mode message in llama31_8b_local_router.choose_tool_call."""
    return (
        "You are an MCP client in a tool-routing benchmark. "
        "Call exactly one of the tools supplied by the chat template. "
        "Do not answer the request directly and do not explain the call.\n\n"
        f"User query:\n{query}"
    )


def _token_ids_from_text(tokenizer: Any, prompt: str) -> tuple[int, ...]:
    encoded = tokenizer(prompt, return_tensors=None)["input_ids"]
    if hasattr(encoded, "keys") and "input_ids" in encoded:
        encoded = encoded["input_ids"]
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if encoded and isinstance(encoded[0], list):
        encoded = encoded[0]
    return tuple(int(token) for token in encoded)


def render_llama_prompt(
    generator: Any,
    example: SavedExample,
    catalog: ToolCatalog,
) -> ReconstructedPrompt:
    ordered_names = tuple(example.record["tool_names"])
    missing = sorted(set(ordered_names) - set(catalog.names))
    if missing:
        raise ValueError(f"Live registry is missing saved prompt tools: {missing}")
    native_message = {"role": "user", "content": llama_native_user_message(routed_query(example.record))}
    tools = build_native_tools(ordered_names, catalog.schemas, catalog.descriptions)
    fallback_message = {
        "role": "user",
        "content": build_tool_call_prompt(
            routed_query(example.record),
            ordered_names,
            catalog.schemas,
            catalog.descriptions,
        ),
    }
    # Delegate tokenization and native-template fallback to the same public
    # method the router calls.  The separately rendered text is checked against
    # those returned token IDs, so its hash describes the actual prompt.
    encoded = tuple(
        int(token)
        for token in generator.encode_chat(
            [native_message],
            tools=tools,
            fallback_messages=[fallback_message],
        )
    )
    try:
        prompt = generator.tokenizer.apply_chat_template(
            [native_message], tools=tools, add_generation_prompt=True, tokenize=False
        )
    except (TypeError, ValueError):
        prompt = generator.tokenizer.apply_chat_template(
            [fallback_message], add_generation_prompt=True, tokenize=False
        )
    if _token_ids_from_text(generator.tokenizer, prompt) != encoded:
        raise RuntimeError("Rendered Llama prompt tokens differ from the router replay tokens")
    registry_exact_match = all(
        example.record.get(key) == catalog.metadata.get(key)
        for key in (
            "tool_pool",
            "tool_count",
            "tool_registry_fingerprint",
            "tool_registry_fingerprint_version",
        )
    )
    return ReconstructedPrompt(
        text=prompt,
        token_ids=encoded,
        routed_query=routed_query(example.record),
        generation_settings=generation_settings(example),
        registry_exact_match=registry_exact_match,
    )


def config_provenance(config: ReplayConfig) -> dict[str, Any]:
    return {
        "baseline_campaign_commit": config.baseline_campaign_commit,
        "observation_layers": list(config.layers),
        "token_selection_mode": config.token_selection_mode,
        "development_only": config.development_only,
        "require_registry_match": config.require_registry_match,
        "observation_enabled": config.observation_enabled,
        "config": {
            "source_run_dir": str(config.source_run_dir),
            "sample_id": config.sample_id,
            "baseline_campaign_commit": config.baseline_campaign_commit,
            "layers": list(config.layers),
            "token_selection_mode": config.token_selection_mode,
            "development_only": config.development_only,
            "require_registry_match": config.require_registry_match,
            "observation_enabled": config.observation_enabled,
            "output_dir": str(config.output_dir) if config.output_dir else None,
            "checkpoint": str(config.checkpoint) if config.checkpoint else None,
        },
    }
