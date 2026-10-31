"""Paired GPT-OSS attention-intervention evaluation.

This runner reuses the local GPT-OSS Harmony rendering/generation/parsing path
and the baseline evaluator's canonical single-step tool execution, argument
scoring, and final-outcome scoring path.
"""

from __future__ import annotations

import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import math
from numbers import Real
import os
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Literal, Sequence
import uuid

from analysis.benchmark_inventory import infer_benchmark_class

from evaluation.evaluate import (
    SERVER_PATH,
    BenchmarkSample,
    RoutedToolCall,
    query_with_context,
    open_evaluation_session,
    load_tool_catalog,
    EvaluationToolCatalog as ToolCatalog,
    evaluate_single_step_prediction,
    load_benchmark,
)
from models.routers import gpt_oss_local_router as gpt_oss_router
from models.routers.structured_tool_call import ToolCallPrediction
from research.phase2.intervention import (
    AttentionTarget,
    StrengthMode,
    SUPPORTED_ATTENTION_TARGETS,
    SUPPORTED_STRENGTH_MODES,
    available_attention_layers,
    available_attention_targets,
    temporary_attention_intervention,
)
from research.phase2.layer_screen import LayerRequest, LayerScreenPlan, LayerScreenPlanError, plan_layer_screen
from research.phase2.experiment_integrity import (
    checkpoint_identity, verify_identity_inputs, runtime_identity,
    compare_saved_control, paired_diagnostics, native_generator_identity,
)
from research.phase2.control_reference import (
    REFERENCE_KIND, canonical_bytes as _reference_bytes, load_reference, preflight_reference,
    sha256_bytes as _reference_sha256, write_reference,
)


InterventionMethod = Literal["noise", "replace"]
# Fresh condition sessions/provenance are a new protocol, not retroactive
# certification of v1 runs. Old strict combiners must reject it until upgraded.
RUN_KIND = "phase2_gpt_oss_attention_intervention_eval_v3"
RUN_MANIFEST_KIND = "phase2_paired_run_manifest_v1"
INTERVENTION_ONLY_MANIFEST_KIND = "phase2_intervention_only_run_manifest_v1"
SCREEN_PLAN_KIND = "phase2_layer_screen_plan_v1"


@dataclass(frozen=True)
class EvaluationConfig:
    layers: LayerRequest
    seeds: tuple[int, ...]
    method: InterventionMethod
    strength: float
    example_limit: int | None
    strength_mode: StrengthMode = "absolute"
    target: AttentionTarget = "attention_all"
    sample_ids: tuple[str, ...] | None = None
    require_registry_match: bool = True

    def __post_init__(self) -> None:
        if self.example_limit is not None and self.example_limit <= 0:
            raise ValueError("example_limit must be positive")
        # Reject invalid strengths before checkpoint hashing or GPU loading.
        if isinstance(self.strength, bool) or not isinstance(self.strength, Real):
            raise TypeError("strength must be a finite non-negative number")
        if not math.isfinite(float(self.strength)) or self.strength < 0:
            raise ValueError("strength must be a finite non-negative number")
        if self.strength_mode not in SUPPORTED_STRENGTH_MODES:
            raise ValueError("strength_mode must be 'absolute' or 'tensor_rms'")
        if self.strength_mode == "tensor_rms" and self.method != "noise":
            raise ValueError("tensor_rms strength mode is currently defined only for noise interventions")


@dataclass(frozen=True)
class SavedGptOssExample:
    sample_id: str
    benchmark_path: str
    domain: str
    task_type: str
    difficulty: str
    source: str
    query: str
    prompt_context: str
    expected_tool: str
    expected_args: dict[str, Any]
    expected_answer: Any
    benchmark_mode: str
    tool_names: tuple[str, ...]
    registry_metadata: dict[str, Any]
    saved_prediction: dict[str, Any] = field(default_factory=dict)
    benchmark_sample: BenchmarkSample | None = None


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _index_paths(source_run_dir: Path) -> list[Path]:
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
        candidate = Path(raw_path)
        path = candidate if candidate.is_absolute() else source_run_dir / candidate
        if not path.is_file():
            raise FileNotFoundError(f"Indexed samples artifact does not exist: {path}")
        paths.append(path)
    if not paths:
        raise ValueError(f"Saved run has no indexed sample artifacts: {index_path}")
    return paths


def _validate_gpt_oss_run_metadata(metadata: dict[str, Any]) -> None:
    if metadata.get("model") != "gpt-oss-local":
        raise ValueError("Saved run is not a gpt-oss-local evaluation")
    if metadata.get("expected_model_name") != gpt_oss_router.MODEL_NAME:
        raise ValueError("Saved run does not identify the local GPT-OSS model")
    if metadata.get("prompt_template_id") != gpt_oss_router.PROMPT_TEMPLATE:
        raise ValueError("Saved run does not use the GPT-OSS Harmony prompt template")
    if metadata.get("reasoning_mode") != "reasoning":
        raise ValueError("Saved run does not use GPT-OSS Harmony reasoning mode")
    if metadata.get("reasoning_effort") != "low":
        raise ValueError("Saved run does not use GPT-OSS low reasoning effort")


def _saved_example(record: dict[str, Any], source: Path, line_number: int) -> SavedGptOssExample:
    sample_id = record.get("sample_id", record.get("id"))
    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError(f"{source}:{line_number} has no sample ID")
    if record.get("model_name") != gpt_oss_router.MODEL_NAME:
        raise ValueError(f"{source}:{line_number} is not a GPT-OSS result")
    if record.get("router_backend") != gpt_oss_router.ROUTER_BACKEND:
        raise ValueError(f"{source}:{line_number} was not generated by the local GPT-OSS runtime")
    if record.get("prompt_template") != gpt_oss_router.PROMPT_TEMPLATE:
        raise ValueError(f"{source}:{line_number} does not use the GPT-OSS Harmony prompt")
    if record.get("reasoning_mode") != "reasoning":
        raise ValueError(f"{source}:{line_number} does not use reasoning mode")
    if record.get("reasoning_effort") != "low":
        raise ValueError(f"{source}:{line_number} does not use low reasoning effort")
    query = record.get("query")
    expected_tool = record.get("expected_tool")
    expected_args = record.get("expected_args")
    tool_names = record.get("tool_names")
    if not isinstance(query, str) or not query.strip():
        raise ValueError(f"{source}:{line_number} has no non-empty query")
    if not isinstance(expected_tool, str) or not expected_tool:
        raise ValueError(f"{source}:{line_number} has no expected tool")
    if not isinstance(expected_args, dict):
        raise ValueError(f"{source}:{line_number} has no expected argument object")
    if not isinstance(tool_names, list) or not tool_names or not all(isinstance(name, str) for name in tool_names):
        raise ValueError(f"{source}:{line_number} has no ordered tool list")
    return SavedGptOssExample(
        sample_id=sample_id,
        benchmark_path=str(record.get("benchmark_path", "<saved_gpt_oss_sample>")),
        domain=str(record.get("domain", "unspecified")),
        task_type=str(record.get("task_type", "single_tool_routing")),
        difficulty=str(record.get("difficulty", "unspecified")),
        source=str(record.get("source", "saved_gpt_oss_run")),
        query=query,
        prompt_context=str(record.get("prompt_context", "")),
        expected_tool=expected_tool,
        expected_args=expected_args,
        expected_answer=record.get("expected_answer"),
        benchmark_mode=str(record.get("benchmark_mode", "grounded_tool_execution")),
        tool_names=tuple(tool_names),
        registry_metadata={
            key: record.get(key)
            for key in (
                "tool_pool",
                "tool_count",
                "tool_registry_fingerprint",
                "tool_registry_fingerprint_version",
            )
        },
        saved_prediction={key: record[key] for key in (
            "raw_model_output", "selected_tool", "selected_args", "parse_status",
        ) if key in record},
    )


def load_saved_gpt_oss_examples(
    source_run_dir: Path,
    example_limit: int,
    sample_ids: Sequence[str] | None = None,
) -> tuple[dict[str, Any], tuple[SavedGptOssExample, ...]]:
    """Load saved GPT-OSS examples by explicit ID or bounded saved-run order."""
    if example_limit <= 0:
        raise ValueError("example_limit must be positive")
    requested_ids = tuple(sample_ids) if sample_ids is not None else None
    if requested_ids is not None:
        if not requested_ids or any(not isinstance(item, str) or not item for item in requested_ids):
            raise ValueError("sample_ids must be a non-empty sequence of non-empty strings")
        if len(set(requested_ids)) != len(requested_ids):
            raise ValueError("sample_ids must not contain duplicates")
    source = source_run_dir.expanduser().resolve(strict=True)
    metadata_path = source / "run_metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Saved run has no metadata: {metadata_path}")
    metadata = _read_json(metadata_path)
    _validate_gpt_oss_run_metadata(metadata)
    examples: list[SavedGptOssExample] = []
    examples_by_id: dict[str, SavedGptOssExample] = {}
    seen: set[str] = set()
    for samples_path in _index_paths(source):
        for line_number, line in enumerate(samples_path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            example = _saved_example(json.loads(line), samples_path, line_number)
            if example.sample_id in seen:
                raise ValueError(f"Saved run has duplicate sample ID: {example.sample_id}")
            seen.add(example.sample_id)
            if requested_ids is None:
                examples.append(example)
            else:
                examples_by_id[example.sample_id] = example
            if requested_ids is None and len(examples) == example_limit:
                return metadata, tuple(examples)
    if requested_ids is not None:
        missing = [sample_id for sample_id in requested_ids if sample_id not in examples_by_id]
        if missing:
            raise ValueError(f"Saved run has no requested sample IDs: {missing}")
        return metadata, tuple(examples_by_id[sample_id] for sample_id in requested_ids)
    if not examples:
        raise ValueError(f"Saved run has no GPT-OSS examples: {source}")
    return metadata, tuple(examples)


def _registry_exact_match(example: SavedGptOssExample, catalog: ToolCatalog) -> bool:
    return all(
        example.registry_metadata.get(key) == catalog.metadata.get(key)
        for key in example.registry_metadata
    )


def _benchmark_sample(example: SavedGptOssExample) -> BenchmarkSample:
    """Adapt a saved GPT-OSS baseline row to the canonical evaluator input."""
    if example.benchmark_sample is not None:
        return example.benchmark_sample
    return BenchmarkSample(
        id=example.sample_id,
        domain=example.domain,
        task_type=example.task_type,
        difficulty=example.difficulty,
        source=example.source,
        query=example.query,
        expected_tool=example.expected_tool,
        expected_args=example.expected_args,
        expected_answer=example.expected_answer,
        perturbation_type="saved_baseline_replay",
        notes="",
        prompt_context=example.prompt_context,
        benchmark_mode=example.benchmark_mode,
    )


def load_direct_benchmark_examples(
    paths: Sequence[Path], catalog: ToolCatalog,
) -> tuple[SavedGptOssExample, ...]:
    """Use the canonical evaluator's benchmark parser, not historical outputs."""
    if not paths:
        raise ValueError("at least one benchmark path is required")
    examples: list[SavedGptOssExample] = []
    seen: set[str] = set()
    for raw_path in paths:
        path = raw_path.expanduser().resolve(strict=True)
        samples = load_benchmark(path)
        if not samples:
            raise ValueError(f"Benchmark has no runnable samples: {path}")
        for sample in samples:
            if sample.expected_steps or sample.task_type == "multi_step_tool_routing":
                raise ValueError(f"Multi-step sample {sample.id} requires the multi-step runner: {path}")
            if sample.id in seen:
                raise ValueError(f"Duplicate sample ID across direct benchmarks: {sample.id}")
            seen.add(sample.id)
            examples.append(SavedGptOssExample(
                sample_id=sample.id,
                benchmark_path=str(raw_path),
                domain=sample.domain,
                task_type=sample.task_type,
                difficulty=sample.difficulty,
                source=sample.source,
                query=sample.query,
                prompt_context=sample.prompt_context,
                expected_tool=sample.expected_tool,
                expected_args=sample.expected_args,
                expected_answer=sample.expected_answer,
                benchmark_mode=sample.benchmark_mode,
                tool_names=tuple(catalog.names),
                registry_metadata=dict(catalog.metadata),
                benchmark_sample=sample,
            ))
    return tuple(examples)


def _routed_tool_call(prediction: ToolCallPrediction) -> RoutedToolCall:
    """Copy native Harmony parser output into the evaluator-neutral format."""
    return RoutedToolCall(
        selected_tool=prediction.selected_tool,
        selected_args=prediction.selected_args,
        raw_model_output=prediction.raw_output,
        parse_status=prediction.parse_status,
        attempted_tool=prediction.attempted_tool,
        parse_diagnostic=prediction.diagnostic,
    )


def _condition_record(record: dict[str, Any]) -> dict[str, Any]:
    """Preserve concise paired-run aliases beside the canonical evaluator record."""
    condition = dict(record)
    condition["chosen_tool"] = condition["selected_tool"]
    condition["tool_choice_correct"] = condition["tool_selection_correct"]
    condition["invalid_output"] = condition["parse_status"] != "ok"
    condition["no_call_outcome"] = condition["selected_tool"] is None
    return condition


def _evidence_category(example: SavedGptOssExample) -> str:
    """Use the standard reporter's classifier on one canonical sample at a time."""
    return infer_benchmark_class(Path(example.benchmark_path), [asdict(_benchmark_sample(example))])


def _benchmark_hashes(paths: Sequence[Path]) -> dict[str, str]:
    return {
        str(path.expanduser().resolve(strict=True)): hashlib.sha256(path.expanduser().read_bytes()).hexdigest()
        for path in paths
    }


def _selected_sample_manifest(
    paths: Sequence[Path], examples: Sequence[SavedGptOssExample],
) -> list[dict[str, Any]]:
    selected: dict[str, list[str]] = {
        str(path.expanduser().resolve()): [] for path in paths
    }
    for example in examples:
        selected[str(Path(example.benchmark_path).expanduser().resolve())].append(example.sample_id)
    return [
        {"benchmark_path": path, "selected_sample_ids": ids, "selected_sample_count": len(ids)}
        for path, ids in selected.items()
    ]


def _verify_benchmark_inputs(expected: dict[str, str]) -> None:
    for filename, digest in expected.items():
        path = Path(filename)
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Benchmark input changed during evaluation: {filename}")


def _direct_plan_selection(
    paths: Sequence[Path], example_limit: int | None,
) -> tuple[tuple[str, ...], list[dict[str, Any]]]:
    """Resolve the exact direct-input sample order without opening an MCP session."""
    selected: list[str] = []
    by_path: dict[str, list[str]] = {
        str(path.expanduser().resolve(strict=True)): [] for path in paths
    }
    seen: set[str] = set()
    for raw_path in paths:
        path = raw_path.expanduser().resolve(strict=True)
        samples = load_benchmark(path)
        if not samples:
            raise ValueError(f"Benchmark has no runnable samples: {path}")
        for sample in samples:
            if sample.expected_steps or sample.task_type == "multi_step_tool_routing":
                raise ValueError(f"Multi-step sample {sample.id} requires the multi-step runner: {path}")
            if sample.id in seen:
                raise ValueError(f"Duplicate sample ID across direct benchmarks: {sample.id}")
            seen.add(sample.id)
            if example_limit is None or len(selected) < example_limit:
                selected.append(sample.id)
                by_path[str(path)].append(sample.id)
    if not selected:
        raise ValueError("Direct benchmark selection produced no samples")
    return tuple(selected), [
        {"benchmark_path": path, "selected_sample_ids": sample_ids, "selected_sample_count": len(sample_ids)}
        for path, sample_ids in by_path.items()
    ]


def _canonical_json_bytes(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _job_details() -> dict[str, Any]:
    """Record scheduler facts if present; absent values are explicitly portable."""
    keys = (
        "SLURM_JOB_ID", "SLURM_JOB_NAME", "SLURM_JOB_ACCOUNT", "SLURM_JOB_PARTITION",
        "SLURM_JOB_NODELIST", "SLURM_CPUS_PER_TASK", "SLURM_MEM_PER_NODE",
        "SLURM_TIMELIMIT", "SLURM_JOB_GPUS", "SLURM_SUBMIT_DIR",
    )
    return {
        "actual": {key.lower(): os.environ.get(key) for key in keys},
        "requested": {
            key: os.environ.get(key) for key in (
                "LAYERMCP_SLURM_ACCOUNT", "LAYERMCP_SLURM_PARTITION", "LAYERMCP_SLURM_GRES",
                "LAYERMCP_SLURM_CPUS_PER_TASK", "LAYERMCP_SLURM_MEM", "LAYERMCP_SLURM_TIME",
            )
        },
    }


def _resolve_screen_config(
    generator: Any,
    sample_ids: Sequence[str],
    config: EvaluationConfig,
) -> tuple[EvaluationConfig, LayerScreenPlan]:
    available_layers = available_attention_layers(generator.model)
    targets = {
        layer: available_attention_targets(generator.model, layer)
        for layer in available_layers
    }
    try:
        plan = plan_layer_screen(
            layer_request=config.layers,
            available_layers=available_layers,
            available_targets_by_layer=targets,
            sample_ids=sample_ids,
            seeds=config.seeds,
            target=config.target,
            method=config.method,
            strength_mode=config.strength_mode,
            strength=config.strength,
        )
    except LayerScreenPlanError as error:
        raise ValueError(str(error)) from error
    return EvaluationConfig(
        layers=plan.selected_layers, seeds=config.seeds, method=config.method,
        strength=config.strength, example_limit=config.example_limit,
        strength_mode=config.strength_mode, target=config.target,
        sample_ids=config.sample_ids, require_registry_match=config.require_registry_match,
    ), plan


def plan_evaluation(
    *,
    source_run_dir: Path | None = None,
    benchmark_paths: Sequence[Path] | None = None,
    checkpoint_dir: Path,
    config: EvaluationConfig,
    generator_loader: Callable[[str], Any] = gpt_oss_router.load_generator,
) -> dict[str, Any]:
    """Discover layers and freeze a plan without generation or MCP execution."""
    if (source_run_dir is None) == (benchmark_paths is None):
        raise ValueError("select exactly one of source_run_dir or benchmark_paths")
    checkpoint = checkpoint_dir.expanduser().resolve(strict=True)
    if source_run_dir is not None:
        if config.example_limit is None:
            raise ValueError("saved source runs require an explicit example limit")
        metadata, examples = load_saved_gpt_oss_examples(source_run_dir, config.example_limit, config.sample_ids)
        input_identity: dict[str, Any] = {
            "input_kind": "saved_source_run", "source_run_directory": str(source_run_dir.expanduser().resolve()),
            "source_run_metadata": metadata,
        }
        sample_ids = tuple(example.sample_id for example in examples)
    else:
        paths = tuple(path.expanduser().resolve(strict=True) for path in benchmark_paths or ())
        sample_ids, selected_manifest = _direct_plan_selection(paths, config.example_limit)
        input_identity = {
            "input_kind": "benchmark_files", "benchmark_paths": [str(path) for path in paths],
            "benchmark_sha256": _benchmark_hashes(paths), "selected_sample_manifest": selected_manifest,
        }
    generator = generator_loader(str(checkpoint))
    resolved, screen_plan = _resolve_screen_config(generator, sample_ids, config)
    return {
        "kind": SCREEN_PLAN_KIND,
        "created_at": _utc_now(),
        "checkpoint_path": str(checkpoint),
        **input_identity,
        "requested_config": asdict(config), "resolved_config": asdict(resolved),
        "screen_plan": screen_plan.as_dict(),
        "valid_layer_indices": list(available_attention_layers(generator.model)),
        "available_attention_targets": {
            str(layer): list(available_attention_targets(generator.model, layer))
            for layer in available_attention_layers(generator.model)
        },
    }


def write_plan_manifest(path: Path, plan: dict[str, Any]) -> str:
    """Write a new frozen plan and return the SHA-256 recorded by job chunks."""
    destination = path.expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite plan manifest: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    content = _canonical_json_bytes(plan)
    destination.write_bytes(content)
    return _sha256_bytes(content)


def _parent_plan_reference(
    path: Path | None, plan: LayerScreenPlan, input_identity: dict[str, Any],
) -> dict[str, Any] | None:
    if path is None:
        return None
    content = path.expanduser().resolve(strict=True).read_bytes()
    parent = json.loads(content)
    if parent.get("kind") != SCREEN_PLAN_KIND:
        raise ValueError(f"{path}: not a {SCREEN_PLAN_KIND} manifest")
    expected = parent.get("screen_plan", {})
    for field, actual in (("sample_ids", list(plan.sample_ids)), ("seeds", list(plan.seeds)),
                          ("target", plan.target), ("method", plan.method),
                          ("strength_mode", plan.strength_mode), ("strength", plan.strength)):
        if expected.get(field) != actual:
            raise ValueError(f"{path}: frozen plan {field} differs from this chunk")
    expected_layers = expected.get("selected_layers")
    if not isinstance(expected_layers, list) or not set(plan.selected_layers).issubset(set(expected_layers)):
        raise ValueError(f"{path}: chunk layers are not contained in the frozen plan")
    if parent.get("input_kind") != input_identity.get("input_kind"):
        raise ValueError(f"{path}: frozen plan input kind differs from this chunk")
    if input_identity.get("input_kind") == "benchmark_files":
        for field in ("benchmark_sha256", "selected_sample_manifest"):
            if parent.get(field) != input_identity.get(field):
                raise ValueError(f"{path}: frozen plan {field} differs from this chunk")
    return {"path": str(path.expanduser().resolve()), "sha256": _sha256_bytes(content)}


def _predict(
    generator: Any,
    example: SavedGptOssExample,
    catalog: ToolCatalog,
    *,
    layer_index: int,
    enabled: bool,
    method: InterventionMethod,
    target: AttentionTarget,
    strength: float,
    strength_mode: StrengthMode,
    seed: int,
) -> tuple[ToolCallPrediction, dict[str, Any]]:
    missing_tools = sorted(set(example.tool_names) - set(catalog.names))
    if missing_tools:
        raise ValueError(f"Live registry is missing saved prompt tools: {missing_tools}")
    native_tools = gpt_oss_router.build_native_tools(
        example.tool_names,
        catalog.schemas,
        catalog.descriptions,
    )
    query = query_with_context(example.query, example.prompt_context)
    with temporary_attention_intervention(
        generator.model,
        layer_index,
        method=method,
        target=target,
        strength=strength,
        strength_mode=strength_mode,
        seed=seed,
        enabled=enabled,
    ) as intervention:
        prediction = gpt_oss_router.generate_prediction(
            generator,
            query,
            example.tool_names,
            native_tools,
            catalog.schemas,
            "low",
        )
    verification = {
        "enabled": enabled,
        "changed_attention_parameter_names": list(intervention.changed_parameter_names),
        "restored_exactly": intervention.restored_exactly,
        "perturbation": intervention.perturbation,
    }
    if enabled and not intervention.restored_exactly:
        raise RuntimeError(f"Attention weights did not restore exactly: {verification}")
    return prediction, verification


async def paired_records(
    generator: Any,
    examples: Sequence[SavedGptOssExample],
    catalog: ToolCatalog,
    config: EvaluationConfig,
    session: Any,
    *,
    session_factory: Callable[[], Any] | None = None,
    provenance: dict[str, Any] | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """Yield one complete unchanged/intervention pair per example, layer, and seed."""
    if not examples:
        raise ValueError("at least one saved example is required")
    if not isinstance(config.layers, tuple) or not config.layers or not config.seeds:
        raise ValueError("layers and seeds must be non-empty and distinct")
    open_session = session_factory or (lambda: open_evaluation_session(SERVER_PATH))
    provenance = provenance or {"checkpoint_fingerprint": None, "model_binding": "caller_supplied_unverified"}

    async def evaluate_condition(
        example: SavedGptOssExample, prediction: ToolCallPrediction,
    ) -> dict[str, Any]:
        # A new server for EACH condition, including non-retail tools. The
        # canonical evaluator still owns execution and scoring unchanged.
        async with open_session() as condition_session:
            condition_catalog = await load_tool_catalog(condition_session)
            if condition_catalog.metadata != catalog.metadata:
                raise ValueError("Tool registry drifted between evaluation conditions")
            result = await evaluate_single_step_prediction(
                sample=_benchmark_sample(example),
                benchmark_path=Path(example.benchmark_path),
                router=gpt_oss_router, prediction=_routed_tool_call(prediction),
                session=condition_session, server_path=SERVER_PATH,
                live_tools=list(catalog.names), tool_schemas=catalog.schemas,
                tool_descriptions=catalog.descriptions, call_predicted_tools=True,
                reasoning_mode="reasoning", reasoning_effort="low",
            )
            return _condition_record(result.record)

    valid_layers = available_attention_layers(generator.model)
    invalid_layers = sorted(set(config.layers) - set(valid_layers))
    if invalid_layers:
        raise ValueError(f"Invalid layer indices {invalid_layers}; valid indices: {list(valid_layers)}")
    unavailable_targets = {
        layer_index: available_attention_targets(generator.model, layer_index)
        for layer_index in config.layers
        if config.target not in available_attention_targets(generator.model, layer_index)
    }
    if unavailable_targets:
        details = "; ".join(
            f"layer {layer_index}: {list(targets)}"
            for layer_index, targets in sorted(unavailable_targets.items())
        )
        raise ValueError(
            f"Attention target {config.target!r} is unavailable for selected layer(s); {details}"
        )
    for layer_index in config.layers:
        for seed in config.seeds:
            for example in examples:
                registry_exact_match = _registry_exact_match(example, catalog)
                if config.require_registry_match and not registry_exact_match:
                    raise ValueError(
                        f"Live registry does not exactly match saved example {example.sample_id}; "
                        "use a matching registry rather than comparing a changed prompt."
                    )
                control, control_verification = _predict(
                    generator,
                    example,
                    catalog,
                    layer_index=layer_index,
                    enabled=False,
                    method=config.method,
                    target=config.target,
                    strength=config.strength,
                    strength_mode=config.strength_mode,
                    seed=seed,
                )
                intervention, intervention_verification = _predict(
                    generator,
                    example,
                    catalog,
                    layer_index=layer_index,
                    enabled=True,
                    method=config.method,
                    target=config.target,
                    strength=config.strength,
                    strength_mode=config.strength_mode,
                    seed=seed,
                )
                control_record = await evaluate_condition(example, control)
                intervention_record = await evaluate_condition(example, intervention)
                for name, record in (("control", control_record), ("intervention", intervention_record)):
                    record.update({"experiment_kind": RUN_KIND, "condition": name})
                yield {
                    "kind": RUN_KIND,
                    "experiment_kind": RUN_KIND,
                    **provenance,
                    "sample_id": example.sample_id,
                    "benchmark_path": example.benchmark_path,
                    "source": example.source,
                    "evidence_category": _evidence_category(example),
                    "benchmark_mode": example.benchmark_mode,
                    "layer_index": layer_index,
                    "seed": seed,
                    "method": config.method,
                    "target": config.target,
                    "strength": config.strength,
                    "strength_mode": config.strength_mode,
                    "registry_exact_match": registry_exact_match,
                    "condition_isolation": "fresh_mcp_server_per_condition",
                    "control": control_record,
                    "intervention": intervention_record,
                    "saved_control_comparison": compare_saved_control(example.saved_prediction, control_record),
                    "control_verification": control_verification,
                    "intervention_verification": intervention_verification,
                }


async def evaluate_one_saved_example_async(
    *,
    generator: Any,
    source_run_dir: Path,
    sample_id: str,
    layer_index: int,
    seed: int,
    method: InterventionMethod,
    target: AttentionTarget,
    strength: float,
    strength_mode: StrengthMode = "absolute",
    session_factory: Callable[[], Any] | None = None,
    require_registry_match: bool = True,
) -> dict[str, Any]:
    """Evaluate one saved GPT-OSS sample as a visible control/intervention pair.

    This is the interactive counterpart to :func:`run_evaluation_async`: it
    accepts an already-loaded generator, executes the same native Harmony and
    baseline evaluator path, and returns one complete pair without creating an
    output directory.  It is intended for a notebook inspection, not a sweep.
    """
    metadata, examples = load_saved_gpt_oss_examples(
        source_run_dir,
        example_limit=1,
        sample_ids=(sample_id,),
    )
    config = EvaluationConfig(
        layers=(layer_index,),
        seeds=(seed,),
        method=method,
        strength=strength,
        strength_mode=strength_mode,
        example_limit=1,
        target=target,
        sample_ids=(sample_id,),
        require_registry_match=require_registry_match,
    )
    identity = native_generator_identity(generator)
    open_session = session_factory or (lambda: open_evaluation_session(SERVER_PATH))
    async with open_session() as session:
        catalog = await load_tool_catalog(session)
        records = [
            record
            async for record in paired_records(
                generator,
                examples,
                catalog,
                config,
                session,
                session_factory=open_session,
                provenance={
                    "source_run_id": source_run_dir.resolve().name,
                    "checkpoint_fingerprint": identity["fingerprint"],
                    "model_binding": "native_loaded_inputs" if identity["status"] == "complete" else "caller_supplied_unverified",
                    "source_checkpoint_equivalence": (
                        "matches" if metadata["checkpoint_fingerprint"] == identity["fingerprint"] else "differs"
                    ) if metadata.get("checkpoint_fingerprint") and identity["fingerprint"] else "unverifiable",
                    "runtime": runtime_identity(generator.model),
                },
            )
        ]
    if len(records) != 1:  # Defensive guard if paired-record iteration changes.
        raise RuntimeError(f"Expected exactly one paired record, got {len(records)}")
    if identity["status"] == "complete":
        verify_identity_inputs(identity)
    return records[0]


def evaluate_one_saved_example(**kwargs: Any) -> dict[str, Any]:
    """Synchronously run one paired sample in scripts and Jupyter notebooks.

    IPython kernels already own an event loop.  In that case the evaluation is
    run in one short-lived worker thread with its own loop; generation remains
    strictly sequential and the supplied model instance is never shared across
    simultaneous evaluations.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(evaluate_one_saved_example_async(**kwargs))

    with ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(
            lambda: asyncio.run(evaluate_one_saved_example_async(**kwargs))
        ).result()


def _safe_output_directory(path: Path) -> Path:
    target = path.expanduser().resolve()
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite evaluation output directory: {target}")
    if target.name in {"", ".", ".."}:
        raise ValueError("Evaluation output directory has an unsafe final component")
    return target


def _condition_summary(records: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    conditions: dict[str, dict[str, Any]] = {}
    for condition in ("control", "intervention"):
        outcomes = [record[condition] for record in records]
        final_outcome_scored = sum(
            outcome["final_outcome_correct"] is not None for outcome in outcomes
        )
        final_outcome_correct = sum(
            outcome["final_outcome_correct"] is True for outcome in outcomes
        )
        conditions[condition] = {
            "paired_records": len(outcomes),
            "correct_tool_choices": sum(outcome["tool_choice_correct"] for outcome in outcomes),
            "incorrect_tool_choices": sum(not outcome["tool_choice_correct"] for outcome in outcomes),
            "invalid_outputs": sum(outcome["invalid_output"] for outcome in outcomes),
            "no_call_outcomes": sum(outcome["no_call_outcome"] for outcome in outcomes),
            "exact_argument_matches": sum(outcome["argument_match_correct"] for outcome in outcomes),
            "execution_successes": sum(outcome["execution_success"] for outcome in outcomes),
            "final_outcome_scored": final_outcome_scored,
            "final_outcome_correct": final_outcome_correct,
            "final_outcome_accuracy": (
                final_outcome_correct / final_outcome_scored
                if final_outcome_scored
                else None
            ),
        }
    return conditions


def _summary(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    conditions = _condition_summary(records)
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault((record["layer_index"], record["seed"]), []).append(record)
    grouped_categories: dict[str, list[dict[str, Any]]] = {}
    grouped_benchmarks: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        grouped_categories.setdefault(record.get("evidence_category", "unclassified"), []).append(record)
        grouped_benchmarks.setdefault(record.get("benchmark_path", "unknown"), []).append(record)
    return {
        "paired_record_count": len(records), "conditions": conditions,
        "aggregation_note": "Overall totals mix provenance categories; compare by_evidence_category and by_benchmark separately.",
        "by_evidence_category": [
            {"category": category, "paired_record_count": len(pairs),
             "conditions": _condition_summary(pairs), "paired_diagnostics": paired_diagnostics(pairs)}
            for category, pairs in sorted(grouped_categories.items())
        ],
        "by_benchmark": [
            {"benchmark_path": path, "paired_record_count": len(pairs),
             "conditions": _condition_summary(pairs), "paired_diagnostics": paired_diagnostics(pairs)}
            for path, pairs in sorted(grouped_benchmarks.items())
        ],
        "paired_diagnostics": paired_diagnostics(records),
        "by_layer_seed": [
            {"layer_index": layer, "seed": seed, **paired_diagnostics(pairs)}
            for (layer, seed), pairs in sorted(grouped.items())
        ],
        "saved_control_comparison_counts": {
            status: sum(row["saved_control_comparison"]["status"] == status for row in records)
            for status in ("matches", "differs_new_control_baseline", "unavailable")
        },
        "no_op_interventions": sum(
            row["intervention_verification"]["perturbation"]["status"] == "no_op" for row in records
        ),
    }


def _direct_reference_compatibility(
    *,
    provenance: dict[str, Any], input_config: dict[str, Any], catalog: ToolCatalog,
) -> dict[str, Any]:
    """Fields that must not change when controls are reused across layers."""
    return {
        "workflow_type": "single_step",
        "experiment_kind": RUN_KIND,
        "checkpoint_fingerprint": provenance["checkpoint_fingerprint"],
        "model_binding": provenance["model_binding"],
        "runtime": provenance["runtime"],
        "inputs": input_config,
        "generation": {"reasoning_mode": "reasoning", "reasoning_effort": "low"},
        "evaluation": {"call_predicted_tools": True, "condition_isolation": "fresh_mcp_server_per_condition"},
        "tool_registry": {"metadata": catalog.metadata, "names": list(catalog.names)},
    }


async def _run_reference_or_intervention_direct(
    *,
    source_run_dir: Path | None,
    benchmark_paths: Sequence[Path] | None,
    checkpoint: Path,
    output_dir: Path,
    config: EvaluationConfig,
    plan_manifest: Path | None,
    generator: Any,
    identity: dict[str, Any],
    provenance: dict[str, Any],
    metadata: dict[str, Any] | None,
    saved_examples: tuple[SavedGptOssExample, ...] | None,
    selected_ids: tuple[str, ...],
    direct_selection: list[dict[str, Any]] | None,
    benchmark_hashes: dict[str, str],
    execution_mode: Literal["reference", "intervention_only"],
    control_reference: Path | None,
    session_factory: Callable[[], Any] | None,
) -> Path:
    """Persist a one-time control or a linked intervention-only direct chunk."""
    output_dir.mkdir(parents=True)
    in_progress = output_dir / "RUN_IN_PROGRESS"
    run_id = str(uuid.uuid4())
    started_at = _utc_now()
    records: list[dict[str, Any]] = []
    failure_stage = "setup"
    try:
        in_progress.write_text(json.dumps({"run_id": run_id, "started_at": started_at}) + "\n", encoding="utf-8")
        resolved_config, screen_plan = _resolve_screen_config(generator, selected_ids, config)
        parent_plan = _parent_plan_reference(plan_manifest, screen_plan, {
            "input_kind": "saved_source_run" if source_run_dir is not None else "benchmark_files",
            "benchmark_sha256": benchmark_hashes,
            "selected_sample_manifest": direct_selection,
        })
        open_session = session_factory or (lambda: open_evaluation_session(SERVER_PATH))
        async with open_session() as session:
            catalog = await load_tool_catalog(session)
            examples = saved_examples if saved_examples is not None else load_direct_benchmark_examples(benchmark_paths or (), catalog)
            if benchmark_paths is not None and resolved_config.example_limit is not None:
                examples = examples[:resolved_config.example_limit]
            if tuple(example.sample_id for example in examples) != selected_ids:
                raise RuntimeError("Direct sample selection changed after reference preflight")
            _verify_benchmark_inputs(benchmark_hashes)
            input_config = (
                {"source_run_directory": str(source_run_dir.expanduser().resolve()), "source_run_metadata": metadata,
                 "selected_sample_ids": list(selected_ids)}
                if source_run_dir is not None else
                {"benchmark_paths": [str(path.expanduser().resolve()) for path in benchmark_paths or ()],
                 "benchmark_sha256": benchmark_hashes,
                 "selected_sample_manifest": direct_selection or _selected_sample_manifest(benchmark_paths or (), examples),
                 "selected_sample_ids": list(selected_ids)}
            )
            compatibility = _direct_reference_compatibility(provenance=provenance, input_config=input_config, catalog=catalog)
            run_config = {
                "kind": RUN_KIND, "experiment_kind": RUN_KIND, "run_id": run_id,
                "execution_mode": execution_mode, **provenance, "checkpoint_identity": identity,
                "condition_isolation": "fresh_mcp_server_per_condition",
                "checkpoint_path": str(checkpoint), "requested_config": asdict(config),
                "resolved_config": asdict(resolved_config), "config": asdict(resolved_config),
                "screen_plan": screen_plan.as_dict(), "parent_plan": parent_plan,
                "example_count": len(examples), **input_config,
            }
            if execution_mode == "reference":
                failure_stage = "control"
                for example in examples:
                    registry_exact_match = _registry_exact_match(example, catalog)
                    if config.require_registry_match and not registry_exact_match:
                        raise ValueError(f"Live registry does not exactly match saved example {example.sample_id}")
                    # No selected layer is changed.  Use a validated layer only to retain
                    # the existing disabled-intervention verification record shape.
                    control, verification = _predict(
                        generator, example, catalog, layer_index=resolved_config.layers[0], enabled=False,
                        method=resolved_config.method, target=resolved_config.target,
                        strength=resolved_config.strength, strength_mode=resolved_config.strength_mode,
                        seed=resolved_config.seeds[0],
                    )
                    async with open_session() as condition_session:
                        condition_catalog = await load_tool_catalog(condition_session)
                        if condition_catalog.metadata != catalog.metadata:
                            raise ValueError("Tool registry drifted during canonical control")
                        result = await evaluate_single_step_prediction(
                            sample=_benchmark_sample(example), benchmark_path=Path(example.benchmark_path),
                            router=gpt_oss_router, prediction=_routed_tool_call(control), session=condition_session,
                            server_path=SERVER_PATH, live_tools=list(catalog.names), tool_schemas=catalog.schemas,
                            tool_descriptions=catalog.descriptions, call_predicted_tools=True,
                            reasoning_mode="reasoning", reasoning_effort="low",
                        )
                    control_record = _condition_record(result.record)
                    control_record.update({"experiment_kind": RUN_KIND, "condition": "control"})
                    records.append({
                        "kind": REFERENCE_KIND, "sample_id": example.sample_id,
                        "benchmark_path": example.benchmark_path, "source": example.source,
                        "benchmark_mode": example.benchmark_mode, "evidence_category": _evidence_category(example),
                        "registry_exact_match": registry_exact_match, "control": control_record,
                        "control_verification": verification,
                        "saved_control_comparison": compare_saved_control(example.saved_prediction, control_record),
                    })
            else:
                assert control_reference is not None
                reference, controls = load_reference(
                    control_reference, workflow_type="single_step", compatibility=compatibility,
                    expected_ids=[example.sample_id for example in examples],
                )
                run_config["control_reference"] = reference
                failure_stage = "intervention"
                with (output_dir / "paired_records.jsonl").open("x", encoding="utf-8") as handle:
                    for layer_index in resolved_config.layers:
                        for seed in resolved_config.seeds:
                            for example in examples:
                                registry_exact_match = _registry_exact_match(example, catalog)
                                if config.require_registry_match and not registry_exact_match:
                                    raise ValueError(f"Live registry does not exactly match saved example {example.sample_id}")
                                prediction, verification = _predict(
                                    generator, example, catalog, layer_index=layer_index, enabled=True,
                                    method=resolved_config.method, target=resolved_config.target,
                                    strength=resolved_config.strength, strength_mode=resolved_config.strength_mode, seed=seed,
                                )
                                async with open_session() as condition_session:
                                    condition_catalog = await load_tool_catalog(condition_session)
                                    if condition_catalog.metadata != catalog.metadata:
                                        raise ValueError("Tool registry drifted during intervention")
                                    result = await evaluate_single_step_prediction(
                                        sample=_benchmark_sample(example), benchmark_path=Path(example.benchmark_path),
                                        router=gpt_oss_router, prediction=_routed_tool_call(prediction), session=condition_session,
                                        server_path=SERVER_PATH, live_tools=list(catalog.names), tool_schemas=catalog.schemas,
                                        tool_descriptions=catalog.descriptions, call_predicted_tools=True,
                                        reasoning_mode="reasoning", reasoning_effort="low",
                                    )
                                intervention = _condition_record(result.record)
                                intervention.update({"experiment_kind": RUN_KIND, "condition": "intervention"})
                                record = {
                                    "kind": RUN_KIND, "experiment_kind": RUN_KIND, "execution_mode": execution_mode,
                                    "sample_id": example.sample_id, "benchmark_path": example.benchmark_path,
                                    "source": example.source, "evidence_category": _evidence_category(example),
                                    "benchmark_mode": example.benchmark_mode, "layer_index": layer_index, "seed": seed,
                                    "method": resolved_config.method, "target": resolved_config.target,
                                    "strength": resolved_config.strength, "strength_mode": resolved_config.strength_mode,
                                    "registry_exact_match": registry_exact_match, "condition_isolation": "fresh_mcp_server_per_condition",
                                    "control_reference": reference, "reference_control_id": example.sample_id,
                                    "intervention": intervention, "intervention_verification": verification,
                                }
                                handle.write(json.dumps(record, sort_keys=True) + "\n")
                                handle.flush()
                                records.append(record)
                materialized = [
                    {**record, "control": controls[record["sample_id"]]["control"],
                     "saved_control_comparison": controls[record["sample_id"]].get(
                         "saved_control_comparison", {"status": "unavailable"})}
                    for record in records
                ]

            (output_dir / "run_config.json").write_bytes(_canonical_json_bytes(run_config))
            if execution_mode == "reference":
                failure_stage = "completion"
                completed_at = _utc_now()
                reference = write_reference(
                    output_dir / "canonical_control", workflow_type="single_step", compatibility=compatibility,
                    controls=records, metadata={"run_id": run_id, "started_at": started_at, "completed_at": completed_at,
                                                "job_details": _job_details(), "source_run_manifest": None},
                )
                (output_dir / "summary.json").write_text(json.dumps({"kind": REFERENCE_KIND, "execution_mode": "reference",
                    "reference": reference, "control": _condition_summary([{"control": row["control"], "intervention": row["control"]} for row in records])["control"]}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            else:
                manifest = {**run_config, "manifest_kind": INTERVENTION_ONLY_MANIFEST_KIND, "started_at": started_at,
                            "job_details": _job_details(), "tool_registry": compatibility["tool_registry"],
                            "lifecycle_status_file": "run_status.json"}
                manifest_bytes = _canonical_json_bytes(manifest)
                (output_dir / "run_manifest.json").write_bytes(manifest_bytes)
                manifest_reference = {"run_id": run_id, "filename": "run_manifest.json", "sha256": _sha256_bytes(manifest_bytes)}
                path = output_dir / "paired_records.jsonl"
                lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
                path.write_text("".join(json.dumps({**line, "run_manifest": manifest_reference}, sort_keys=True) + "\n" for line in lines), encoding="utf-8")
                (output_dir / "summary.json").write_text(json.dumps(_summary(materialized), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        failure_stage = "completion"
        _verify_benchmark_inputs(benchmark_hashes)
        verify_identity_inputs(identity)
        (output_dir / "run_status.json").write_bytes(_canonical_json_bytes({
            "run_id": run_id, "status": "completed", "execution_mode": execution_mode,
            "started_at": started_at, "completed_at": _utc_now(), "record_count": len(records),
        }))
        in_progress.unlink()
        (output_dir / "RUN_COMPLETE").write_text("\n", encoding="utf-8")
    except BaseException as error:
        (output_dir / "RUN_FAILED.json").write_text(json.dumps({"kind": RUN_KIND, "run_id": run_id,
            "execution_mode": execution_mode, "error_type": type(error).__name__, "error": str(error),
            "failure_stage": failure_stage, "completed_record_count": len(records)}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        raise
    return output_dir


async def run_evaluation_async(
    *,
    source_run_dir: Path | None = None,
    benchmark_paths: Sequence[Path] | None = None,
    checkpoint_dir: Path,
    output_dir: Path,
    config: EvaluationConfig,
    plan_manifest: Path | None = None,
    execution_mode: Literal["paired", "reference", "intervention_only"] = "paired",
    control_reference: Path | None = None,
    generator_loader: Callable[[str], Any] = gpt_oss_router.load_generator,
    session_factory: Callable[[], Any] | None = None,
) -> Path:
    """Run a paired, canonical-control, or intervention-only evaluation."""
    if (source_run_dir is None) == (benchmark_paths is None):
        raise ValueError("select exactly one of source_run_dir or benchmark_paths")
    target = _safe_output_directory(output_dir)
    if benchmark_paths is not None:
        benchmark_paths = tuple(path.expanduser().resolve(strict=True) for path in benchmark_paths)
    benchmark_hashes = _benchmark_hashes(benchmark_paths) if benchmark_paths is not None else {}
    checkpoint = checkpoint_dir.expanduser().resolve(strict=True)
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"GPT-OSS checkpoint does not exist: {checkpoint}")
    if source_run_dir is not None:
        if config.example_limit is None:
            raise ValueError("saved source runs require an explicit example limit")
        metadata, saved_examples = load_saved_gpt_oss_examples(
            source_run_dir, config.example_limit, config.sample_ids
        )
        selected_ids = tuple(example.sample_id for example in saved_examples)
        direct_selection = None
        plan_input_identity = {"input_kind": "saved_source_run", "source_run_directory": str(source_run_dir.expanduser().resolve())}
    else:
        if config.sample_ids is not None:
            raise ValueError("sample IDs are supported only with a saved source run")
        metadata, saved_examples = None, None
        selected_ids, direct_selection = _direct_plan_selection(benchmark_paths or (), config.example_limit)
        plan_input_identity = {"input_kind": "benchmark_files", "benchmark_sha256": benchmark_hashes,
                               "selected_sample_manifest": direct_selection}
    if generator_loader is gpt_oss_router.load_generator:
        print("Fingerprinting checkpoint/config/Harmony files (full content read; cold storage may be slow)...", flush=True)
    identity = checkpoint_identity(checkpoint)
    if generator_loader is gpt_oss_router.load_generator and identity["status"] != "complete":
        raise ValueError("Checkpoint provenance requires weights, config.json and a local Harmony asset via TIKTOKEN_ENCODINGS_BASE")
    if execution_mode == "intervention_only":
        assert control_reference is not None
        preflight_inputs = (
            {"source_run_directory": str(source_run_dir.expanduser().resolve()), "source_run_metadata": metadata,
             "selected_sample_ids": list(selected_ids)}
            if source_run_dir is not None else
            {"benchmark_paths": [str(path.expanduser().resolve()) for path in benchmark_paths or ()],
             "benchmark_sha256": benchmark_hashes, "selected_sample_manifest": direct_selection,
             "selected_sample_ids": list(selected_ids)}
        )
        preflight_reference(control_reference, workflow_type="single_step", expected_ids=selected_ids,
                            required_compatibility={"workflow_type": "single_step", "experiment_kind": RUN_KIND,
                                "checkpoint_fingerprint": identity["fingerprint"], "model_binding": "native_loader" if generator_loader is gpt_oss_router.load_generator else "injected_loader_unverified",
                                "inputs": preflight_inputs})
    generator = generator_loader(str(checkpoint))
    verify_identity_inputs(identity)
    if generator_loader is gpt_oss_router.load_generator:
        native_generator_identity(generator, identity)
    source_fingerprint = metadata.get("checkpoint_fingerprint") if metadata else None
    provenance = {
        "input_kind": "saved_source_run" if source_run_dir is not None else "benchmark_files",
        "checkpoint_fingerprint": identity["fingerprint"],
        "model_binding": "native_loader" if generator_loader is gpt_oss_router.load_generator else "injected_loader_unverified",
        "runtime": runtime_identity(generator.model),
    }
    if source_run_dir is not None:
        provenance["source_run_id"] = source_run_dir.resolve().name
        provenance["source_checkpoint_equivalence"] = (
            "matches" if source_fingerprint == identity["fingerprint"] else "differs"
        ) if source_fingerprint and identity["fingerprint"] else "unverifiable"

    if execution_mode not in {"paired", "reference", "intervention_only"}:
        raise ValueError("execution_mode must be paired, reference, or intervention_only")
    if execution_mode == "intervention_only" and control_reference is None:
        raise ValueError("intervention_only execution requires --control-reference")
    if execution_mode != "intervention_only" and control_reference is not None:
        raise ValueError("--control-reference applies only to intervention_only execution")
    if execution_mode != "paired":
        return await _run_reference_or_intervention_direct(
            source_run_dir=source_run_dir, benchmark_paths=benchmark_paths,
            checkpoint=checkpoint, output_dir=target, config=config,
            plan_manifest=plan_manifest, generator=generator, identity=identity,
            provenance=provenance, metadata=metadata, saved_examples=saved_examples,
            selected_ids=selected_ids, direct_selection=direct_selection,
            benchmark_hashes=benchmark_hashes, execution_mode=execution_mode,
            control_reference=control_reference, session_factory=session_factory,
        )

    target.mkdir(parents=True)
    in_progress = target / "RUN_IN_PROGRESS"
    records: list[dict[str, Any]] = []
    failure_stage = "setup"
    try:
        run_id = str(uuid.uuid4())
        started_at = _utc_now()
        in_progress.write_text(json.dumps({"run_id": run_id, "started_at": started_at}) + "\n", encoding="utf-8")
        resolved_config, screen_plan = _resolve_screen_config(generator, selected_ids, config)
        parent_plan = _parent_plan_reference(plan_manifest, screen_plan, plan_input_identity)
        open_session = session_factory or (lambda: open_evaluation_session(SERVER_PATH))
        async with open_session() as session:
            catalog = await load_tool_catalog(session)
            examples = saved_examples if saved_examples is not None else load_direct_benchmark_examples(
                benchmark_paths or (), catalog
            )
            if benchmark_paths is not None and resolved_config.example_limit is not None:
                examples = examples[:resolved_config.example_limit]
            if tuple(example.sample_id for example in examples) != selected_ids:
                raise RuntimeError("Direct sample selection changed after plan resolution")
            _verify_benchmark_inputs(benchmark_hashes)
            input_config = (
                {"source_run_directory": str(source_run_dir.expanduser().resolve()), "source_run_metadata": metadata}
                if source_run_dir is not None else
                {
                    "benchmark_paths": [str(path.expanduser().resolve()) for path in benchmark_paths or ()],
                    "benchmark_sha256": benchmark_hashes,
                    "selected_sample_manifest": direct_selection or _selected_sample_manifest(benchmark_paths or (), examples),
                }
            )
            run_config = {
                "kind": RUN_KIND, "experiment_kind": RUN_KIND, "run_id": run_id, **provenance,
                "checkpoint_identity": identity, "condition_isolation": "fresh_mcp_server_per_condition",
                "strength_mode": resolved_config.strength_mode, **input_config,
                "checkpoint_path": str(checkpoint), "requested_config": asdict(config),
                "resolved_config": asdict(resolved_config), "config": asdict(resolved_config),
                "screen_plan": screen_plan.as_dict(), "parent_plan": parent_plan,
                "example_count": len(examples), "call_predicted_tools": True,
                "valid_layer_indices": list(available_attention_layers(generator.model),),
                "available_attention_targets": {
                    str(layer_index): list(available_attention_targets(generator.model, layer_index))
                    for layer_index in available_attention_layers(generator.model)
                },
            }
            (target / "run_config.json").write_text(_canonical_json_bytes(run_config).decode("utf-8"), encoding="utf-8")
            screen_payload = {"kind": SCREEN_PLAN_KIND, "screen_plan": screen_plan.as_dict(),
                              "parent_plan": parent_plan, **input_config}
            (target / "screen_plan.json").write_bytes(_canonical_json_bytes(screen_payload))
            manifest = {
                **run_config, "manifest_kind": RUN_MANIFEST_KIND, "run_id": run_id,
                "started_at": started_at, "job_details": _job_details(),
                "tool_registry": {"metadata": catalog.metadata, "names": list(catalog.names)},
                "lifecycle_status_file": "run_status.json",
            }
            manifest_bytes = _canonical_json_bytes(manifest)
            (target / "run_manifest.json").write_bytes(manifest_bytes)
            manifest_reference = {"run_id": run_id, "filename": "run_manifest.json", "sha256": _sha256_bytes(manifest_bytes)}
            (target / "run_status.json").write_bytes(_canonical_json_bytes({
                "run_id": run_id, "status": "running", "started_at": started_at,
            }))
            failure_stage = "evaluation"
            with (target / "paired_records.jsonl").open("x", encoding="utf-8") as handle:
                async for record in paired_records(
                    generator,
                    examples,
                    catalog,
                    resolved_config,
                    session,
                    session_factory=open_session,
                    provenance={**provenance, "run_manifest": manifest_reference},
                ):
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
                    handle.flush()
                    records.append(record)
        failure_stage = "completion"
        _verify_benchmark_inputs(benchmark_hashes)
        verify_identity_inputs(identity)
        (target / "summary.json").write_text(
            json.dumps(_summary(records), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (target / "run_status.json").write_bytes(_canonical_json_bytes({
            "run_id": run_id, "status": "completed", "started_at": started_at,
            "completed_at": _utc_now(), "paired_record_count": len(records),
        }))
        in_progress.unlink()
        (target / "RUN_COMPLETE").write_text("\n", encoding="utf-8")
    except BaseException as error:
        try:
            (target / "RUN_FAILED.json").write_text(
                json.dumps(
                    {
                        "kind": RUN_KIND, "run_id": locals().get("run_id"),
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "failure_stage": failure_stage,
                        "completed_pair_count": len(records),
                        "failed_at": _utc_now(),
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            (target / "run_status.json").write_bytes(_canonical_json_bytes({
                "run_id": locals().get("run_id"), "status": "failed", "failed_at": _utc_now(),
                "failure_stage": failure_stage, "completed_pair_count": len(records),
            }))
        except Exception as artifact_error:
            # An unwritable filesystem cannot guarantee a durable report.
            # Preserve the original failure and chain the reporting failure.
            raise error from artifact_error
        raise
    return target


def run_evaluation(
    *,
    source_run_dir: Path | None = None,
    benchmark_paths: Sequence[Path] | None = None,
    checkpoint_dir: Path,
    output_dir: Path,
    config: EvaluationConfig,
    plan_manifest: Path | None = None,
    execution_mode: Literal["paired", "reference", "intervention_only"] = "paired",
    control_reference: Path | None = None,
    generator_loader: Callable[[str], Any] = gpt_oss_router.load_generator,
    session_factory: Callable[[], Any] | None = None,
) -> Path:
    """Synchronous CLI wrapper around :func:`run_evaluation_async`."""
    return asyncio.run(
        run_evaluation_async(
            source_run_dir=source_run_dir,
            benchmark_paths=benchmark_paths,
            checkpoint_dir=checkpoint_dir,
            output_dir=output_dir,
            config=config,
            plan_manifest=plan_manifest,
            execution_mode=execution_mode,
            control_reference=control_reference,
            generator_loader=generator_loader,
            session_factory=session_factory,
        )
    )


def _parse_csv_ints(value: str, option: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"{option} must be comma-separated integers") from error
    if not parsed or len(set(parsed)) != len(parsed):
        raise argparse.ArgumentTypeError(f"{option} must be a non-empty list of distinct integers")
    return parsed


def _parse_layer_request(value: str) -> LayerRequest:
    normalized = value.strip().lower()
    if normalized in {"all", "representative"}:
        return normalized
    return _parse_csv_ints(value, "--layers")


def _parse_csv_sample_ids(value: str) -> tuple[str, ...]:
    parsed = tuple(item.strip() for item in value.split(",") if item.strip())
    if not parsed or len(set(parsed)) != len(parsed):
        raise argparse.ArgumentTypeError(
            "--sample-ids must be a non-empty comma-separated list of distinct saved sample IDs"
        )
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GPT-OSS paired, canonical-control, or intervention-only routing evaluations.")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--source-run-dir", type=Path)
    input_group.add_argument("--benchmarks", nargs="+", type=Path, help="Direct benchmark JSON files (all rows by default)")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--layers", default="0", type=_parse_layer_request,
                        help="Comma-separated IDs, or model-discovered 'representative'/'all'.")
    parser.add_argument("--seeds", default="1234", help="Comma-separated intervention seeds")
    parser.add_argument("--method", choices=("noise", "replace"), default="noise")
    parser.add_argument("--target", choices=sorted(SUPPORTED_ATTENTION_TARGETS), default="attention_all")
    parser.add_argument("--strength", type=float, default=0.01)
    parser.add_argument("--strength-mode", choices=sorted(SUPPORTED_STRENGTH_MODES), default="absolute")
    parser.add_argument("--example-limit", type=int, help="Optional row limit (saved-run default: 2; direct default: all)")
    parser.add_argument("--sample-ids", type=_parse_csv_sample_ids)
    parser.add_argument("--allow-registry-mismatch", action="store_true")
    parser.add_argument("--plan-only", action="store_true", help="Discover layers and print a frozen screen plan without generation.")
    parser.add_argument("--plan-output", type=Path, help="New JSON path to save a plan-only manifest.")
    parser.add_argument("--plan-manifest", type=Path, help="Frozen parent plan that this output chunk must match.")
    parser.add_argument("--execution-mode", choices=("paired", "reference", "intervention-only"), default="paired",
                        help="paired (default), reference (one unchanged control), or intervention-only (requires --control-reference).")
    parser.add_argument("--control-reference", type=Path,
                        help="Completed canonical_control directory used only by intervention-only execution.")
    args = parser.parse_args()
    if args.example_limit is not None and args.example_limit <= 0:
        parser.error("--example-limit must be positive")
    if args.strength < 0:
        parser.error("--strength must be non-negative")
    if not args.plan_only and args.output_dir is None:
        parser.error("--output-dir is required unless --plan-only is used")
    if args.plan_only and args.plan_manifest is not None:
        parser.error("--plan-manifest applies to an evaluation chunk, not --plan-only")
    if args.plan_only and args.execution_mode != "paired":
        parser.error("--plan-only is independent of control execution; use the default paired mode")
    if args.execution_mode == "intervention-only" and args.control_reference is None:
        parser.error("--control-reference is required with --execution-mode intervention-only")
    if args.execution_mode != "intervention-only" and args.control_reference is not None:
        parser.error("--control-reference applies only to intervention-only execution")
    config = EvaluationConfig(
        layers=args.layers,
        seeds=_parse_csv_ints(args.seeds, "--seeds"),
        method=args.method,
        strength=args.strength,
        strength_mode=args.strength_mode,
        example_limit=len(args.sample_ids) if args.sample_ids is not None else args.example_limit,
        target=args.target,
        sample_ids=args.sample_ids,
        require_registry_match=not args.allow_registry_mismatch,
    )
    if args.plan_only:
        plan = plan_evaluation(
            source_run_dir=args.source_run_dir, benchmark_paths=args.benchmarks,
            checkpoint_dir=args.checkpoint, config=config,
        )
        if args.plan_output is not None:
            digest = write_plan_manifest(args.plan_output, plan)
            plan["plan_manifest"] = {"path": str(args.plan_output.expanduser().resolve()), "sha256": digest}
        print(json.dumps(plan, indent=2, sort_keys=True))
        return
    destination = run_evaluation(
        source_run_dir=args.source_run_dir,
        benchmark_paths=args.benchmarks,
        checkpoint_dir=args.checkpoint,
        output_dir=args.output_dir,
        config=config,
        plan_manifest=args.plan_manifest,
        execution_mode=args.execution_mode.replace("-", "_"),
        control_reference=args.control_reference,
    )
    print(destination)


if __name__ == "__main__":
    main()
