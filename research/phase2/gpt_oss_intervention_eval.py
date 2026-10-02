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
import json
import math
from numbers import Real
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Literal, Sequence

from evaluation.evaluate import (
    SERVER_PATH,
    BenchmarkSample,
    RoutedToolCall,
    query_with_context,
    open_evaluation_session,
    load_tool_catalog,
    EvaluationToolCatalog as ToolCatalog,
    evaluate_single_step_prediction,
)
from models.routers import gpt_oss_local_router as gpt_oss_router
from models.routers.structured_tool_call import ToolCallPrediction
from research.phase2.intervention import (
    AttentionTarget,
    SUPPORTED_ATTENTION_TARGETS,
    available_attention_layers,
    available_attention_targets,
    temporary_attention_intervention,
)
from research.phase2.experiment_integrity import (
    checkpoint_identity, verify_identity_inputs, runtime_identity,
    compare_saved_control, paired_diagnostics, native_generator_identity,
)


InterventionMethod = Literal["noise", "replace"]
# Fresh condition sessions/provenance are a new protocol, not retroactive
# certification of v1 runs. Old strict combiners must reject it until upgraded.
RUN_KIND = "phase2_gpt_oss_attention_intervention_eval_v2"


@dataclass(frozen=True)
class EvaluationConfig:
    layers: tuple[int, ...]
    seeds: tuple[int, ...]
    method: InterventionMethod
    strength: float
    example_limit: int
    target: AttentionTarget = "attention_all"
    sample_ids: tuple[str, ...] | None = None
    require_registry_match: bool = True

    def __post_init__(self) -> None:
        # Reject invalid strengths before checkpoint hashing or GPU loading.
        if isinstance(self.strength, bool) or not isinstance(self.strength, Real):
            raise TypeError("strength must be a finite non-negative number")
        if not math.isfinite(float(self.strength)) or self.strength < 0:
            raise ValueError("strength must be a finite non-negative number")


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
    if (
        not config.layers or not config.seeds
        or len(set(config.layers)) != len(config.layers)
        or len(set(config.seeds)) != len(config.seeds)
    ):
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
                    "layer_index": layer_index,
                    "seed": seed,
                    "method": config.method,
                    "target": config.target,
                    "strength": config.strength,
                    "registry_exact_match": registry_exact_match,
                    "strength_mode": "absolute",
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


def _summary(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
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
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault((record["layer_index"], record["seed"]), []).append(record)
    return {
        "paired_record_count": len(records), "conditions": conditions,
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


async def run_evaluation_async(
    *,
    source_run_dir: Path,
    checkpoint_dir: Path,
    output_dir: Path,
    config: EvaluationConfig,
    generator_loader: Callable[[str], Any] = gpt_oss_router.load_generator,
    session_factory: Callable[[], Any] | None = None,
) -> Path:
    """Run and persist a paired full evaluation; only ``RUN_COMPLETE`` marks success."""
    if (
        not config.layers or not config.seeds
        or len(set(config.layers)) != len(config.layers)
        or len(set(config.seeds)) != len(config.seeds)
    ):
        raise ValueError("layers and seeds must both be non-empty and distinct")
    target = _safe_output_directory(output_dir)
    checkpoint = checkpoint_dir.expanduser().resolve(strict=True)
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"GPT-OSS checkpoint does not exist: {checkpoint}")
    metadata, examples = load_saved_gpt_oss_examples(
        source_run_dir, config.example_limit, config.sample_ids
    )
    if generator_loader is gpt_oss_router.load_generator:
        print("Fingerprinting checkpoint/config/Harmony files (full content read; cold storage may be slow)...", flush=True)
    identity = checkpoint_identity(checkpoint)
    if generator_loader is gpt_oss_router.load_generator and identity["status"] != "complete":
        raise ValueError("Checkpoint provenance requires weights, config.json and a local Harmony asset via TIKTOKEN_ENCODINGS_BASE")
    generator = generator_loader(str(checkpoint))
    verify_identity_inputs(identity)
    if generator_loader is gpt_oss_router.load_generator:
        native_generator_identity(generator, identity)
    source_fingerprint = metadata.get("checkpoint_fingerprint")
    provenance = {
        "source_run_id": source_run_dir.resolve().name,
        "checkpoint_fingerprint": identity["fingerprint"],
        "model_binding": "native_loader" if generator_loader is gpt_oss_router.load_generator else "injected_loader_unverified",
        "source_checkpoint_equivalence": (
            "matches" if source_fingerprint == identity["fingerprint"] else "differs"
        ) if source_fingerprint and identity["fingerprint"] else "unverifiable",
        "runtime": runtime_identity(generator.model),
    }

    target.mkdir(parents=True)
    in_progress = target / "RUN_IN_PROGRESS"
    in_progress.write_text("\n", encoding="utf-8")
    (target / "run_config.json").write_text(
        json.dumps(
            {
                "kind": RUN_KIND,
                "experiment_kind": RUN_KIND,
                **provenance,
                "checkpoint_identity": identity,
                "condition_isolation": "fresh_mcp_server_per_condition",
                "strength_mode": "absolute",
                "source_run_directory": str(source_run_dir.expanduser().resolve()),
                "checkpoint_path": str(checkpoint),
                "config": asdict(config),
                "source_run_metadata": metadata,
                "example_count": len(examples),
                "call_predicted_tools": True,
                "valid_layer_indices": list(available_attention_layers(generator.model)),
                "available_attention_targets": {
                    str(layer_index): list(available_attention_targets(generator.model, layer_index))
                    for layer_index in available_attention_layers(generator.model)
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    records: list[dict[str, Any]] = []
    try:
        open_session = session_factory or (lambda: open_evaluation_session(SERVER_PATH))
        async with open_session() as session:
            catalog = await load_tool_catalog(session)
            with (target / "paired_records.jsonl").open("x", encoding="utf-8") as handle:
                async for record in paired_records(
                    generator,
                    examples,
                    catalog,
                    config,
                    session,
                    session_factory=open_session,
                    provenance=provenance,
                ):
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
                    handle.flush()
                    records.append(record)
        verify_identity_inputs(identity)
        (target / "summary.json").write_text(
            json.dumps(_summary(records), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        in_progress.unlink()
        (target / "RUN_COMPLETE").write_text("\n", encoding="utf-8")
    except BaseException as error:
        (target / "RUN_FAILED.json").write_text(
            json.dumps(
                {
                    "kind": RUN_KIND,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "completed_pair_count": len(records),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        raise
    return target


def run_evaluation(
    *,
    source_run_dir: Path,
    checkpoint_dir: Path,
    output_dir: Path,
    config: EvaluationConfig,
    generator_loader: Callable[[str], Any] = gpt_oss_router.load_generator,
    session_factory: Callable[[], Any] | None = None,
) -> Path:
    """Synchronous CLI wrapper around :func:`run_evaluation_async`."""
    return asyncio.run(
        run_evaluation_async(
            source_run_dir=source_run_dir,
            checkpoint_dir=checkpoint_dir,
            output_dir=output_dir,
            config=config,
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


def _parse_csv_sample_ids(value: str) -> tuple[str, ...]:
    parsed = tuple(item.strip() for item in value.split(",") if item.strip())
    if not parsed or len(set(parsed)) != len(parsed):
        raise argparse.ArgumentTypeError(
            "--sample-ids must be a non-empty comma-separated list of distinct saved sample IDs"
        )
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Run paired GPT-OSS attention-intervention routing evaluations.")
    parser.add_argument("--source-run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layers", required=True, help="Comma-separated GPT-OSS layer indices, e.g. 0,12")
    parser.add_argument("--seeds", default="1234", help="Comma-separated intervention seeds")
    parser.add_argument("--method", choices=("noise", "replace"), default="noise")
    parser.add_argument("--target", choices=sorted(SUPPORTED_ATTENTION_TARGETS), default="attention_all")
    parser.add_argument("--strength", type=float, default=0.01)
    parser.add_argument("--example-limit", type=int, default=2)
    parser.add_argument("--sample-ids", type=_parse_csv_sample_ids)
    parser.add_argument("--allow-registry-mismatch", action="store_true")
    args = parser.parse_args()
    if args.example_limit <= 0:
        parser.error("--example-limit must be positive")
    if args.strength < 0:
        parser.error("--strength must be non-negative")
    config = EvaluationConfig(
        layers=_parse_csv_ints(args.layers, "--layers"),
        seeds=_parse_csv_ints(args.seeds, "--seeds"),
        method=args.method,
        strength=args.strength,
        example_limit=len(args.sample_ids) if args.sample_ids is not None else args.example_limit,
        target=args.target,
        sample_ids=args.sample_ids,
        require_registry_match=not args.allow_registry_mismatch,
    )
    destination = run_evaluation(
        source_run_dir=args.source_run_dir,
        checkpoint_dir=args.checkpoint,
        output_dir=args.output_dir,
        config=config,
    )
    print(destination)


if __name__ == "__main__":
    main()
