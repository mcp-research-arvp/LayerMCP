"""Paired GPT-OSS attention interventions over canonical multi-step rollouts."""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable

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
    available_attention_layers,
    available_attention_targets,
    temporary_attention_intervention,
)


RUN_KIND = "phase2_gpt_oss_multistep_attention_intervention_v1"
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

    def __post_init__(self) -> None:
        if self.target not in SUPPORTED_ATTENTION_TARGETS:
            raise ValueError(f"Unsupported attention target: {self.target}")
        if self.method not in {"noise", "replace"}:
            raise ValueError(f"Unsupported intervention method: {self.method}")
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
                "control": {key: left.get(key) for key in ("selected_tool", "selected_args", "tool_selection_correct", "argument_match_correct", "execution_success", "final_outcome_correct")},
                "intervention": {key: right.get(key) for key in ("selected_tool", "selected_args", "tool_selection_correct", "argument_match_correct", "execution_success", "final_outcome_correct")},
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
    generator_loader: Callable[[str], Any] = router.load_generator,
) -> Path:
    if dataset_group not in DATASET_GROUPS:
        raise ValueError(f"Unknown active multi-step dataset group: {dataset_group}")
    benchmark = DATASET_GROUPS[dataset_group].resolve(strict=True)
    dataset_counts(benchmark)
    raw_rows = json.loads(benchmark.read_text(encoding="utf-8"))
    dataset = load_benchmark(benchmark)
    if config.workflow_limit is not None:
        dataset = dataset[:config.workflow_limit]
        raw_rows = raw_rows[:config.workflow_limit]
    if not dataset or not all(sample.expected_steps for sample in dataset):
        raise ValueError("Selected dataset has no runnable multi-step workflows")
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

    target = output_dir.expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.mkdir()
    in_progress = target / "RUN_IN_PROGRESS"
    stage = "setup"
    try:
        in_progress.write_text("\n", encoding="utf-8")
        config_record = {
            "kind": RUN_KIND,
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
            "config": asdict(config),
            "checkpoint_identity": identity,
            "model_binding": "native_loader" if generator_loader is router.load_generator else "injected_loader_unverified",
            "runtime": runtime_identity(generator.model),
        }
        (target / "run_config.json").write_text(json.dumps(config_record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        expected_catalog: str | None = None
        checks: dict[str, list[dict[str, Any]]] = {"control": [], "intervention": []}

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
                with temporary_attention_intervention(
                    generator.model, config.layer, target=config.target,
                    method=config.method, strength=config.strength, seed=config.seed,
                    enabled=condition == "intervention",
                ) as intervention:
                    prediction = router.generate_prediction(generator, query, live_tools, native_tools, schemas, "low")
                verification = {
                    "enabled": condition == "intervention",
                    "restored_exactly": intervention.restored_exactly,
                    "changed_attention_parameter_names": list(intervention.changed_parameter_names),
                    "perturbation": intervention.perturbation,
                }
                if condition == "intervention" and not intervention.restored_exactly:
                    raise RuntimeError("Attention weights did not restore exactly")
                checks[condition].append(verification)
                return (
                    prediction.selected_tool, prediction.selected_args,
                    prediction.raw_output, prediction.parse_status,
                    prediction.attempted_tool, prediction.diagnostic,
                )
            return route_query

        stage = "control"
        await _evaluate_multistep_with_server(
            dataset, benchmark, SERVER_PATH, True, "gpt-oss-local",
            reasoning_mode="reasoning", reasoning_effort="low",
            output_dir=target / "control", route_query=route_for("control"),
        )
        stage = "intervention"
        await _evaluate_multistep_with_server(
            dataset, benchmark, SERVER_PATH, True, "gpt-oss-local",
            reasoning_mode="reasoning", reasoning_effort="low",
            output_dir=target / "intervention", route_query=route_for("intervention"),
        )
        control_summary = _read_json(target / "control" / "summary.json")
        intervention_summary = _read_json(target / "intervention" / "summary.json")
        if control_summary["evaluation_protocol"] != MULTISTEP_EVALUATION_PROTOCOL:
            raise ValueError("Control did not use the expected guided rollout protocol")
        for field in ("tool_pool", "tool_count", "tool_registry_fingerprint", "tool_registry_fingerprint_version", "evaluation_protocol", "benchmark_sha256"):
            if control_summary[field] != intervention_summary[field]:
                raise ValueError(f"Paired condition metadata differs: {field}")
        pairs, diagnostics = _paired_results(
            _read_workflows(target / "control" / "samples.jsonl"),
            _read_workflows(target / "intervention" / "samples.jsonl"),
            checks["control"], checks["intervention"],
        )
        if diagnostics["paired_workflow_count"] != len(dataset):
            raise ValueError("Paired workflow count does not match selected dataset")
        if [pair["sample_id"] for pair in pairs] != [sample.id for sample in dataset]:
            raise ValueError("Paired workflow IDs do not match selected dataset")
        stage = "completion"
        verify_identity_inputs(identity)
        with (target / "paired_records.jsonl").open("x", encoding="utf-8") as handle:
            for pair in pairs:
                handle.write(json.dumps(pair, sort_keys=True) + "\n")
        (target / "summary.json").write_text(json.dumps({
            "kind": RUN_KIND,
            "dataset_group": dataset_group,
            "benchmark_mode_counts": config_record["effective_benchmark_mode_counts"],
            "control": control_summary,
            "intervention": intervention_summary,
            **diagnostics,
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        in_progress.unlink()
        (target / "RUN_COMPLETE").write_text("\n", encoding="utf-8")
    except BaseException as error:
        (target / "RUN_FAILED.json").write_text(json.dumps({
            "kind": RUN_KIND, "failure_stage": stage,
            "error_type": type(error).__name__, "error": str(error),
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        raise
    return target


def run_evaluation(**kwargs: Any) -> Path:
    return asyncio.run(run_evaluation_async(**kwargs))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-group", required=True, choices=sorted(DATASET_GROUPS))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--target", choices=sorted(SUPPORTED_ATTENTION_TARGETS), default="qkv")
    parser.add_argument("--method", choices=("noise", "replace"), default="noise")
    parser.add_argument("--strength", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--workflow-limit", type=int)
    args = parser.parse_args()
    config = MultistepConfig(args.layer, args.target, args.method, args.strength, args.seed, args.workflow_limit)
    destination = run_evaluation(
        dataset_group=args.dataset_group, checkpoint_dir=args.checkpoint,
        output_dir=args.output_dir, config=config,
    )
    print(destination)


if __name__ == "__main__":
    main()
