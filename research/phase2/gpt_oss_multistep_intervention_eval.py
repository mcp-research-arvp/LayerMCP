"""Paired GPT-OSS attention interventions over canonical multi-step rollouts."""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Callable
import uuid

from analysis.multi_step_run import DATASET_GROUPS, dataset_counts
from evaluation.evaluate import (
    DEFAULT_BENCHMARK_MODE,
    DEFAULT_WORKFLOW_EXECUTION_MODE,
    MULTISTEP_EVALUATION_PROTOCOL,
    PREDICTED_ROLLOUT_EXECUTION_MODE,
    SERVER_PATH,
    _evaluate_multistep_with_server,
    load_benchmark,
)
from models.routers import gpt_oss_local_router as router
from research.phase2.experiment_integrity import (
    checkpoint_identity,
    native_generator_identity,
    runtime_identity,
    verify_identity_inputs,
)
from research.phase2.intervention import (
    SUPPORTED_ATTENTION_TARGETS,
    SUPPORTED_STRENGTH_MODES,
    available_attention_layers,
    available_attention_targets,
    temporary_attention_intervention,
)
from research.phase2.layer_screen import (
    LayerScreenPlanError,
    plan_layer_screen,
)
from research.phase2.control_reference import REFERENCE_KIND, REFERENCE_MANIFEST, load_reference, write_reference


RUN_KIND = "phase2_gpt_oss_multistep_attention_intervention_v2"
RUN_MANIFEST_KIND = "phase2_multistep_paired_run_manifest_v1"
INTERVENTION_ONLY_MANIFEST_KIND = "phase2_multistep_intervention_only_run_manifest_v1"
SCREEN_PLAN_KIND = "phase2_multistep_layer_screen_plan_v1"
METRICS = {
    "tool_selection": "tool_selection_correct",
    "argument_match": "argument_match_correct",
    "step_outcome": "final_outcome_correct",
    "all_steps_correct": "all_steps_correct",
    "final_step_outcome": "final_step_outcome_correct",
    "final_program_execution": "final_program_execution_correct",
}


@dataclass(frozen=True)
class MultistepConfig:
    layer: int
    target: str
    method: str
    strength: float
    seed: int
    workflow_limit: int | None = None
    strength_mode: str = "absolute"
    plan_manifest: Path | None = None

    def __post_init__(self) -> None:
        if self.target not in SUPPORTED_ATTENTION_TARGETS:
            raise ValueError(f"Unsupported attention target: {self.target}")
        if self.method not in {"noise", "replace"}:
            raise ValueError(f"Unsupported intervention method: {self.method}")
        if self.strength_mode not in SUPPORTED_STRENGTH_MODES:
            raise ValueError("strength_mode must be 'absolute' or 'tensor_rms'")
        if self.strength_mode == "tensor_rms" and self.method != "noise":
            raise ValueError("tensor_rms strength mode is defined only for noise interventions")
        if isinstance(self.strength, bool) or not isinstance(self.strength, (float, int)) or not math.isfinite(self.strength) or self.strength < 0:
            raise ValueError("strength must be a finite non-negative number")
        if isinstance(self.layer, bool) or not isinstance(self.layer, int) or self.layer < 0:
            raise ValueError("layer must be a non-negative integer")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")
        if self.workflow_limit is not None and (isinstance(self.workflow_limit, bool) or not isinstance(self.workflow_limit, int) or self.workflow_limit <= 0):
            raise ValueError("workflow_limit must be a positive integer")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _read_workflows(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"Expected workflow JSON objects: {path}")
    return rows


def _canonical_json_bytes(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _job_details() -> dict[str, dict[str, str | None]]:
    """Record scheduler facts when present without requiring a particular scheduler."""
    actual_keys = (
        "SLURM_JOB_ID", "SLURM_JOB_NAME", "SLURM_JOB_ACCOUNT", "SLURM_JOB_PARTITION",
        "SLURM_JOB_NODELIST", "SLURM_CPUS_PER_TASK", "SLURM_MEM_PER_NODE",
        "SLURM_TIMELIMIT", "SLURM_JOB_GPUS", "SLURM_SUBMIT_DIR",
    )
    requested_keys = (
        "LAYERMCP_SLURM_ACCOUNT", "LAYERMCP_SLURM_PARTITION", "LAYERMCP_SLURM_GRES",
        "LAYERMCP_SLURM_CPUS_PER_TASK", "LAYERMCP_SLURM_MEM", "LAYERMCP_SLURM_TIME",
    )
    return {
        "actual": {key.lower(): os.environ.get(key) for key in actual_keys},
        "requested": {key: os.environ.get(key) for key in requested_keys},
    }


def _selected_dataset(dataset_group: str, workflow_limit: int | None) -> tuple[Path, list[dict[str, Any]], list[Any]]:
    if dataset_group not in DATASET_GROUPS:
        raise ValueError(f"Unknown active multi-step dataset group: {dataset_group}")
    benchmark = DATASET_GROUPS[dataset_group].resolve(strict=True)
    dataset_counts(benchmark)
    raw_rows = json.loads(benchmark.read_text(encoding="utf-8"))
    dataset = load_benchmark(benchmark)
    if workflow_limit is not None:
        dataset = dataset[:workflow_limit]
        raw_rows = raw_rows[:workflow_limit]
    if not dataset or not all(sample.expected_steps for sample in dataset):
        raise ValueError("Selected dataset has no runnable multi-step workflows")
    return benchmark, raw_rows, dataset


def _write_plan(path: Path, payload: dict[str, Any]) -> None:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as handle:
        handle.write(_canonical_json_bytes(payload))


def _load_parent_plan(path: Path, *, dataset_group: str, benchmark: Path, sample_ids: list[str], config: MultistepConfig) -> dict[str, Any]:
    content = path.expanduser().resolve(strict=True).read_bytes()
    payload = json.loads(content)
    if payload.get("kind") != SCREEN_PLAN_KIND:
        raise ValueError("Invalid multi-step screen plan kind")
    if payload.get("dataset_group") != dataset_group:
        raise ValueError("Screen plan dataset group differs from this run")
    if payload.get("benchmark_sha256") != hashlib.sha256(benchmark.read_bytes()).hexdigest():
        raise ValueError("Screen plan benchmark content differs from this run")
    plan = payload.get("screen_plan", {})
    if plan.get("sample_ids") != sample_ids:
        raise ValueError("Screen plan selected workflow IDs differ from this run")
    for field, value in (
        ("target", config.target), ("method", config.method),
        ("strength_mode", config.strength_mode), ("strength", float(config.strength)),
    ):
        if plan.get(field) != value:
            raise ValueError(f"Screen plan {field} differs from this run")
    if config.layer not in plan.get("selected_layers", ()) or config.seed not in plan.get("seeds", ()):
        raise ValueError("Run layer or seed is absent from the frozen screen plan")
    return {
        "filename": str(path.expanduser().resolve()),
        "sha256": _sha256_bytes(content),
        "kind": payload["kind"],
    }


def plan_evaluation(
    *,
    dataset_group: str,
    checkpoint_dir: Path,
    layer_request: tuple[int, ...] | str,
    seeds: tuple[int, ...],
    target: str,
    method: str,
    strength: float,
    strength_mode: str,
    workflow_limit: int | None = None,
    generator_loader: Callable[[str], Any] = router.load_generator,
) -> dict[str, Any]:
    """Discover safe layers and freeze a multi-step screen without generation."""
    benchmark, _raw_rows, dataset = _selected_dataset(dataset_group, workflow_limit)
    checkpoint = checkpoint_dir.expanduser().resolve(strict=True)
    generator = generator_loader(str(checkpoint))
    layers = available_attention_layers(generator.model)
    targets = {layer: available_attention_targets(generator.model, layer) for layer in layers}
    try:
        screen_plan = plan_layer_screen(
            layer_request=layer_request, available_layers=layers, available_targets_by_layer=targets,
            sample_ids=[sample.id for sample in dataset], seeds=seeds, target=target,
            method=method, strength_mode=strength_mode, strength=strength,
        )
    except LayerScreenPlanError as error:
        raise ValueError(str(error)) from error
    return {
        "kind": SCREEN_PLAN_KIND,
        "dataset_group": dataset_group,
        "benchmark_path": str(benchmark),
        "benchmark_sha256": hashlib.sha256(benchmark.read_bytes()).hexdigest(),
        "selected_workflow_count": len(dataset),
        "selected_step_count": sum(len(sample.expected_steps) for sample in dataset),
        "workflow_limit": workflow_limit,
        "screen_plan": screen_plan.as_dict(),
    }


def _transition(control: Any, intervention: Any) -> str:
    if control is None or intervention is None:
        return "unscored"
    return ("correct" if control is True else "wrong") + "_" + ("correct" if intervention is True else "wrong")


def _paired_results(
    controls: list[dict[str, Any]],
    interventions: list[dict[str, Any]],
    control_checks: list[dict[str, Any]],
    intervention_checks: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if len(controls) != len(interventions):
        raise ValueError("Control and intervention workflow counts differ")
    expected_calls = sum(len(row["steps"]) for row in controls)
    if len(control_checks) != expected_calls or len(intervention_checks) != expected_calls:
        raise ValueError("Generation/verification count does not match routed steps")
    transitions = {name: Counter() for name in METRICS}
    pairs: list[dict[str, Any]] = []
    check_index = 0
    for control, intervention in zip(controls, interventions, strict=True):
        if control["sample_id"] != intervention["sample_id"]:
            raise ValueError("Control and intervention workflow IDs differ")
        if control["benchmark_mode"] != intervention["benchmark_mode"]:
            raise ValueError("Benchmark mode drifted between conditions")
        control_steps = control["steps"]
        intervention_steps = intervention["steps"]
        if len(control_steps) != len(intervention_steps):
            raise ValueError("Control and intervention step counts differ")
        step_pairs = []
        for left, right in zip(control_steps, intervention_steps, strict=True):
            if (left["step_id"], left["step_index"]) != (right["step_id"], right["step_index"]):
                raise ValueError("Control and intervention step identities differ")
            left_check = control_checks[check_index]
            right_check = intervention_checks[check_index]
            check_index += 1
            if left_check["enabled"] or left_check["restored_exactly"] is not None or right_check["restored_exactly"] is not True:
                raise RuntimeError("Attention parameter restoration failed")
            for metric in ("tool_selection", "argument_match", "step_outcome"):
                transitions[metric][_transition(left.get(METRICS[metric]), right.get(METRICS[metric]))] += 1
            step_pairs.append({
                "step_id": left["step_id"],
                "step_index": left["step_index"],
                "routed_query_changed": left["routed_query"] != right["routed_query"],
                "control": {key: left.get(key) for key in ("selected_tool", "selected_args", "parse_status", "tool_selection_correct", "argument_match_correct", "execution_success", "final_outcome_correct")},
                "intervention": {key: right.get(key) for key in ("selected_tool", "selected_args", "parse_status", "tool_selection_correct", "argument_match_correct", "execution_success", "final_outcome_correct")},
                "control_verification": left_check,
                "intervention_verification": right_check,
            })
        for metric in ("all_steps_correct", "final_step_outcome", "final_program_execution"):
            transitions[metric][_transition(control.get(METRICS[metric]), intervention.get(METRICS[metric]))] += 1
        pairs.append({
            "sample_id": control["sample_id"],
            "domain": control["domain"],
            "source": control["source"],
            "benchmark_mode": control["benchmark_mode"],
            "workflow_execution_mode": control["workflow_execution_mode"],
            "control": {key: control.get(key) for key in ("all_tools_correct", "all_arguments_correct", "all_steps_correct", "final_step_outcome_correct", "final_program_execution_correct")},
            "intervention": {key: intervention.get(key) for key in ("all_tools_correct", "all_arguments_correct", "all_steps_correct", "final_step_outcome_correct", "final_program_execution_correct")},
            "steps": step_pairs,
        })
    return pairs, {
        "paired_workflow_count": len(pairs),
        "paired_step_count": check_index,
        "metric_transitions": {name: dict(counts) for name, counts in transitions.items()},
        "changed_routed_query_steps": sum(step["routed_query_changed"] for pair in pairs for step in pair["steps"]),
        "no_op_interventions": sum(check["perturbation"]["status"] == "no_op" for check in intervention_checks),
    }


async def run_evaluation_async(
    *,
    dataset_group: str,
    checkpoint_dir: Path,
    output_dir: Path,
    config: MultistepConfig,
    execution_mode: str = "paired",
    control_reference: Path | None = None,
    generator_loader: Callable[[str], Any] = router.load_generator,
) -> Path:
    if execution_mode not in {"paired", "reference", "intervention_only"}:
        raise ValueError("execution_mode must be paired, reference, or intervention_only")
    if execution_mode == "intervention_only" and control_reference is None:
        raise ValueError("intervention_only execution requires a canonical control reference")
    if execution_mode != "intervention_only" and control_reference is not None:
        raise ValueError("control_reference applies only to intervention_only execution")
    benchmark, raw_rows, dataset = _selected_dataset(dataset_group, config.workflow_limit)
    checkpoint = checkpoint_dir.expanduser().resolve(strict=True)
    if not checkpoint.is_dir():
        raise ValueError(f"Checkpoint is not a directory: {checkpoint}")
    identity = checkpoint_identity(checkpoint)
    if generator_loader is router.load_generator and identity["status"] != "complete":
        raise ValueError("GPT-OSS checkpoint/config/Harmony asset provenance is incomplete")
    generator = generator_loader(str(checkpoint))
    verify_identity_inputs(identity)
    if generator_loader is router.load_generator:
        native_generator_identity(generator, identity)
    if config.layer not in available_attention_layers(generator.model):
        raise ValueError(f"Unavailable attention layer: {config.layer}")
    if config.target not in available_attention_targets(generator.model, config.layer):
        raise ValueError(f"Unavailable attention target {config.target!r} at layer {config.layer}")

    selected_ids = [sample.id for sample in dataset]
    parent_plan = (
        _load_parent_plan(config.plan_manifest, dataset_group=dataset_group, benchmark=benchmark,
                          sample_ids=selected_ids, config=config)
        if config.plan_manifest is not None else None
    )

    target = output_dir.expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.mkdir()
    in_progress = target / "RUN_IN_PROGRESS"
    stage = "setup"
    run_id = f"multistep-{uuid.uuid4()}"
    started_at = _utc_now()
    checks: dict[str, list[dict[str, Any]]] = {"control": [], "intervention": []}
    try:
        in_progress.write_text(json.dumps({"run_id": run_id, "started_at": started_at}) + "\n", encoding="utf-8")
        config_record = {
            "kind": RUN_KIND,
            "execution_mode": execution_mode,
            "dataset_group": dataset_group,
            "benchmark_path": str(benchmark),
            "benchmark_sha256": hashlib.sha256(benchmark.read_bytes()).hexdigest(),
            "selected_workflow_ids": [sample.id for sample in dataset],
            "selected_workflow_count": len(dataset),
            "selected_step_count": sum(len(sample.expected_steps) for sample in dataset),
            "raw_benchmark_mode_counts": dict(Counter(row.get("benchmark_mode", "<absent>") for row in raw_rows)),
            "effective_benchmark_mode_counts": dict(Counter(sample.benchmark_mode for sample in dataset)),
            "raw_workflow_execution_mode_counts": dict(Counter(row.get("workflow_execution_mode", "<absent>") for row in raw_rows)),
            "effective_declared_workflow_execution_mode_counts": dict(Counter(sample.workflow_execution_mode for sample in dataset)),
            "evaluator_defaults": {"benchmark_mode": DEFAULT_BENCHMARK_MODE, "workflow_execution_mode": DEFAULT_WORKFLOW_EXECUTION_MODE},
            "evaluation_protocol": MULTISTEP_EVALUATION_PROTOCOL,
            "effective_rollout_execution_mode": PREDICTED_ROLLOUT_EXECUTION_MODE,
            "condition_isolation": "fresh_evaluator_mcp_session_per_condition",
            "intervention_scope": "selected_attention_parameters_during_each_generation",
            "config": {**asdict(config), "plan_manifest": str(config.plan_manifest) if config.plan_manifest else None},
            "parent_plan": parent_plan,
            "checkpoint_identity": identity,
            "model_binding": "native_loader" if generator_loader is router.load_generator else "injected_loader_unverified",
            "runtime": runtime_identity(generator.model),
            "run_id": run_id,
        }
        (target / "run_config.json").write_bytes(_canonical_json_bytes(config_record))
        manifest = {
            "manifest_kind": RUN_MANIFEST_KIND,
            "run_id": run_id,
            "workflow_type": "multi_step",
            "execution_mode": execution_mode,
            "dataset": {
                "group": dataset_group, "path": str(benchmark),
                "sha256": config_record["benchmark_sha256"], "selected_workflow_ids": selected_ids,
            },
            "model": {"checkpoint_identity": identity, "binding": config_record["model_binding"]},
            "intervention": {
                "layer": config.layer, "target": config.target, "method": config.method,
                "strength": config.strength, "strength_mode": config.strength_mode, "seed": config.seed,
                "scope": config_record["intervention_scope"],
            },
            "evaluation": {
                "protocol": MULTISTEP_EVALUATION_PROTOCOL,
                "condition_isolation": config_record["condition_isolation"],
                "reasoning_mode": "reasoning", "reasoning_effort": "low",
            },
            "runtime": config_record["runtime"],
            "parent_plan": parent_plan,
            "started_at": started_at,
            "job_details": _job_details(),
            "lifecycle_status_file": "run_status.json",
        }
        manifest_bytes = _canonical_json_bytes(manifest)
        (target / "run_manifest.json").write_bytes(manifest_bytes)
        manifest_reference = {"run_id": run_id, "filename": "run_manifest.json", "sha256": _sha256_bytes(manifest_bytes)}
        (target / "run_status.json").write_bytes(_canonical_json_bytes({
            "run_id": run_id, "status": "running", "started_at": started_at,
            "started_stage": stage,
        }))
        expected_catalog: str | None = None

        def route_for(condition: str):
            def route_query(_router, query, live_tools, schemas, descriptions, reasoning_mode, reasoning_effort):
                nonlocal expected_catalog
                if _router is not router or reasoning_mode != "reasoning" or reasoning_effort != "low":
                    raise ValueError("Unexpected multi-step router or reasoning condition")
                catalog = json.dumps([live_tools, schemas, descriptions], sort_keys=True)
                if expected_catalog is None:
                    expected_catalog = catalog
                elif expected_catalog != catalog:
                    raise ValueError("Live tool catalog changed between paired conditions")
                native_tools = router.build_native_tools(live_tools, schemas, descriptions)
                intervention = None
                try:
                    with temporary_attention_intervention(
                        generator.model, config.layer, target=config.target,
                        method=config.method, strength=config.strength, seed=config.seed,
                        strength_mode=config.strength_mode, enabled=condition == "intervention",
                    ) as active_intervention:
                        intervention = active_intervention
                        prediction = router.generate_prediction(generator, query, live_tools, native_tools, schemas, "low")
                finally:
                    if intervention is not None:
                        verification = {
                            "enabled": condition == "intervention",
                            "restored_exactly": intervention.restored_exactly,
                            "changed_attention_parameter_names": list(intervention.changed_parameter_names),
                            "perturbation": intervention.perturbation,
                        }
                        checks[condition].append(verification)
                        if condition == "intervention" and not intervention.restored_exactly:
                            raise RuntimeError("Attention weights did not restore exactly")
                return (
                    prediction.selected_tool, prediction.selected_args,
                    prediction.raw_output, prediction.parse_status,
                    prediction.attempted_tool, prediction.diagnostic,
                )
            return route_query

        if execution_mode in {"paired", "reference"}:
            stage = "control"
            await _evaluate_multistep_with_server(
                dataset, benchmark, SERVER_PATH, True, "gpt-oss-local",
                reasoning_mode="reasoning", reasoning_effort="low",
                output_dir=target / "control", route_query=route_for("control"),
            )
        if execution_mode in {"paired", "intervention_only"}:
            stage = "intervention"
            await _evaluate_multistep_with_server(
                dataset, benchmark, SERVER_PATH, True, "gpt-oss-local",
                reasoning_mode="reasoning", reasoning_effort="low",
                output_dir=target / "intervention", route_query=route_for("intervention"),
            )
        primary_summary = _read_json(target / ("control" if execution_mode == "reference" else "intervention" if execution_mode == "intervention_only" else "control") / "summary.json")
        manifest["tool_registry"] = {field: primary_summary[field] for field in (
            "tool_pool", "tool_count", "tool_registry_fingerprint", "tool_registry_fingerprint_version")}
        manifest["generation"] = primary_summary.get("generation")
        compatibility = {
            "workflow_type": "multi_step", "experiment_kind": RUN_KIND,
            "checkpoint_fingerprint": identity["fingerprint"], "dataset_group": dataset_group,
            "benchmark_sha256": config_record["benchmark_sha256"], "selected_workflow_ids": selected_ids,
            "evaluation": manifest["evaluation"], "runtime": config_record["runtime"],
            "tool_registry": manifest["tool_registry"], "generation": manifest["generation"],
        }
        if execution_mode == "reference":
            stage = "completion"
            reference = write_reference(
                target / "canonical_control", workflow_type="multi_step", compatibility=compatibility,
                controls=[{"sample_id": row["sample_id"], "control": row} for row in _read_workflows(target / "control" / "samples.jsonl")],
                metadata={"run_id": run_id, "started_at": started_at, "completed_at": _utc_now(),
                          "job_details": _job_details()},
            )
            (target / "summary.json").write_text(json.dumps({"kind": REFERENCE_KIND, "execution_mode": "reference",
                "reference": reference, "control": primary_summary}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            pairs, diagnostics = [], {"paired_workflow_count": len(dataset), "paired_step_count": sum(len(item.expected_steps) for item in dataset)}
        else:
            if execution_mode == "paired":
                control_summary = _read_json(target / "control" / "summary.json")
                intervention_summary = _read_json(target / "intervention" / "summary.json")
                for field in ("tool_pool", "tool_count", "tool_registry_fingerprint", "tool_registry_fingerprint_version", "evaluation_protocol", "benchmark_sha256"):
                    if control_summary[field] != intervention_summary[field]:
                        raise ValueError(f"Paired condition metadata differs: {field}")
                controls = _read_workflows(target / "control" / "samples.jsonl")
                control_checks = checks["control"]
                reference = None
            else:
                assert control_reference is not None
                # This validates lifecycle, hashes, exact workflow coverage, and all
                # immutable protocol fields before any comparison is written.
                reference_manifest = _read_json(control_reference / REFERENCE_MANIFEST)
                reference, indexed_controls = load_reference(
                    control_reference, workflow_type="multi_step",
                    compatibility=reference_manifest.get("compatibility"), expected_ids=selected_ids,
                )
                if reference_manifest.get("compatibility") != compatibility:
                    raise ValueError("Canonical multi-step control is incompatible with this intervention")
                controls = [indexed_controls[sample_id]["control"] for sample_id in selected_ids]
                control_checks = [{"enabled": False, "changed_attention_parameter_names": [], "restored_exactly": None,
                                   "perturbation": {"status": "disabled", "changed_elements": 0}}
                                  for row in controls for _ in row["steps"]]
                control_summary = {"reference": reference}
                intervention_summary = primary_summary
                config_record["control_reference"] = reference
                manifest["control_reference"] = reference
            manifest["manifest_kind"] = RUN_MANIFEST_KIND if execution_mode == "paired" else INTERVENTION_ONLY_MANIFEST_KIND
            manifest_bytes = _canonical_json_bytes(manifest)
            (target / "run_manifest.json").write_bytes(manifest_bytes)
            manifest_reference = {"run_id": run_id, "filename": "run_manifest.json", "sha256": _sha256_bytes(manifest_bytes)}
            pairs, diagnostics = _paired_results(
                controls, _read_workflows(target / "intervention" / "samples.jsonl"), control_checks, checks["intervention"],
            )
            if diagnostics["paired_workflow_count"] != len(dataset):
                raise ValueError("Paired workflow count does not match selected dataset")
            for pair in pairs:
                pair.update({"kind": RUN_KIND, "run_manifest": manifest_reference, "execution_mode": execution_mode,
                             "layer_index": config.layer, "seed": config.seed, "target": config.target,
                             "method": config.method, "strength": float(config.strength),
                             "strength_mode": config.strength_mode, "dataset_group": dataset_group})
                if reference is not None:
                    pair["control_reference"] = reference
                    pair["reference_control_id"] = pair["sample_id"]
                    pair.pop("control")
                    for step in pair["steps"]:
                        step.pop("control")
                        step.pop("control_verification")
            stage = "completion"
            verify_identity_inputs(identity)
            with (target / "paired_records.jsonl").open("x", encoding="utf-8") as handle:
                for pair in pairs:
                    handle.write(json.dumps(pair, sort_keys=True) + "\n")
            (target / "summary.json").write_text(json.dumps({"kind": RUN_KIND, "execution_mode": execution_mode,
                "dataset_group": dataset_group, "run_manifest": manifest_reference,
                "benchmark_mode_counts": config_record["effective_benchmark_mode_counts"],
                "control": control_summary, "intervention": intervention_summary, **diagnostics}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (target / "run_config.json").write_bytes(_canonical_json_bytes(config_record))
        in_progress.unlink()
        (target / "run_status.json").write_bytes(_canonical_json_bytes({
            "run_id": run_id, "status": "completed", "completed_stage": stage,
            "started_at": started_at, "completed_at": _utc_now(),
            "paired_workflow_count": len(pairs), "paired_generation_count": len(checks["intervention"]),
        }))
        (target / "RUN_COMPLETE").write_text("\n", encoding="utf-8")
    except BaseException as error:
        (target / "RUN_FAILED.json").write_text(json.dumps({
            "kind": RUN_KIND, "failure_stage": stage,
            "run_id": run_id, "error_type": type(error).__name__, "error": str(error),
            "control_generation_checks": checks.get("control", []),
            "intervention_generation_checks": checks.get("intervention", []),
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (target / "run_status.json").write_bytes(_canonical_json_bytes({
            "run_id": run_id, "status": "failed", "failure_stage": stage,
            "started_at": started_at, "failed_at": _utc_now(),
            "completed_generation_checks": len(checks.get("intervention", [])),
        }))
        raise
    return target


def run_evaluation(**kwargs: Any) -> Path:
    return asyncio.run(run_evaluation_async(**kwargs))


def _parse_layer_request(value: str) -> tuple[int, ...] | str:
    if value in {"all", "representative"}:
        return value
    try:
        layers = tuple(int(item) for item in value.split(","))
    except ValueError as error:
        raise ValueError("--layers must be comma-separated integer IDs, all, or representative") from error
    if not layers or any(layer < 0 for layer in layers):
        raise ValueError("--layers must contain non-negative layer IDs")
    return layers


def _parse_seeds(value: str) -> tuple[int, ...]:
    try:
        seeds = tuple(int(item) for item in value.split(","))
    except ValueError as error:
        raise ValueError("--seeds must be comma-separated integers") from error
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("--seeds must contain one or more distinct integers")
    return seeds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-group", required=True, choices=sorted(DATASET_GROUPS))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--layer", type=int)
    parser.add_argument("--layers", default="representative", help="Plan-only layer IDs, all, or representative")
    parser.add_argument("--target", choices=sorted(SUPPORTED_ATTENTION_TARGETS), default="qkv")
    parser.add_argument("--method", choices=("noise", "replace"), default="noise")
    parser.add_argument("--strength", type=float, default=0.01)
    parser.add_argument("--strength-mode", choices=sorted(SUPPORTED_STRENGTH_MODES), default="absolute")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--seeds", default="1234", help="Plan-only comma-separated seeds")
    parser.add_argument("--workflow-limit", type=int)
    parser.add_argument("--plan-manifest", type=Path, help="Frozen multi-step screen plan to validate this layer/seed chunk")
    parser.add_argument("--plan-only", action="store_true", help="Discover safe layers and write a frozen plan without generation")
    parser.add_argument("--plan-output", type=Path, help="New plan JSON written by --plan-only")
    parser.add_argument("--execution-mode", choices=("paired", "reference", "intervention-only"), default="paired")
    parser.add_argument("--control-reference", type=Path)
    args = parser.parse_args()
    if args.plan_only:
        if args.plan_output is None:
            parser.error("--plan-only requires --plan-output")
        try:
            payload = plan_evaluation(
                dataset_group=args.dataset_group, checkpoint_dir=args.checkpoint,
                layer_request=_parse_layer_request(args.layers), seeds=_parse_seeds(args.seeds),
                target=args.target, method=args.method, strength=args.strength,
                strength_mode=args.strength_mode, workflow_limit=args.workflow_limit,
            )
        except ValueError as error:
            parser.error(str(error))
        _write_plan(args.plan_output, payload)
        print(args.plan_output.expanduser().resolve())
        return
    if args.output_dir is None or args.layer is None:
        parser.error("--output-dir and --layer are required unless --plan-only is used")
    if args.execution_mode == "intervention-only" and args.control_reference is None:
        parser.error("--control-reference is required with --execution-mode intervention-only")
    if args.execution_mode != "intervention-only" and args.control_reference is not None:
        parser.error("--control-reference applies only to intervention-only execution")
    config = MultistepConfig(
        args.layer, args.target, args.method, args.strength, args.seed, args.workflow_limit,
        args.strength_mode, args.plan_manifest,
    )
    destination = run_evaluation(
        dataset_group=args.dataset_group, checkpoint_dir=args.checkpoint,
        output_dir=args.output_dir, config=config,
        execution_mode=args.execution_mode.replace("-", "_"), control_reference=args.control_reference,
    )
    print(destination)


if __name__ == "__main__":
    main()
