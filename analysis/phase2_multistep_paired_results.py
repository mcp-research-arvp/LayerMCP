"""Combine completed planned GPT-OSS multi-step paired-evaluation chunks.

This module reads saved artifacts only.  It never loads a checkpoint, starts an
MCP server, reruns scoring, or treats repeated unchanged controls as independent
workflows.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
from itertools import product
import json
from pathlib import Path
from typing import Any, Iterable
from research.phase2.control_reference import REFERENCE_MANIFEST, REFERENCE_KIND, load_reference


RUN_KIND = "phase2_gpt_oss_multistep_attention_intervention_v2"
MANIFEST_KIND = "phase2_multistep_paired_run_manifest_v1"
PLAN_KIND = "phase2_multistep_layer_screen_plan_v1"
SUMMARY_KIND = "phase2_gpt_oss_multistep_paired_results_summary_v1"


def _require(value: bool, message: str) -> None:
    if not value:
        raise ValueError(message)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"{path}: expected JSON object")
    return value


def _read_records(path: Path) -> list[dict[str, Any]]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    _require(bool(records) and all(isinstance(record, dict) for record in records), f"{path}: invalid paired records")
    return records


def _plan_grid(path: Path | None) -> tuple[set[tuple[int, int, str]] | None, dict[str, Any] | None]:
    if path is None:
        return None, None
    content = path.expanduser().resolve(strict=True).read_bytes()
    payload = json.loads(content)
    _require(payload.get("kind") == PLAN_KIND, f"{path}: invalid multi-step plan kind")
    plan = payload.get("screen_plan", {})
    layers, seeds, samples = plan.get("selected_layers"), plan.get("seeds"), plan.get("sample_ids")
    _require(isinstance(layers, list) and isinstance(seeds, list) and isinstance(samples, list), f"{path}: invalid screen plan")
    _require(all(type(value) is int for value in layers + seeds) and all(isinstance(value, str) and value for value in samples), f"{path}: invalid screen plan IDs")
    return set(product(layers, seeds, samples)), {
        "path": str(path.expanduser().resolve()),
        "sha256": "sha256:" + hashlib.sha256(content).hexdigest(),
        "screen_plan": plan,
    }


def _control_identity(pair: dict[str, Any]) -> dict[str, Any]:
    return {
        "workflow": pair.get("control"),
        "steps": [step.get("control") for step in pair.get("steps", [])],
    }


def _check_step(pair_key: tuple[int, int, str], step: dict[str, Any]) -> None:
    control = step.get("control_verification", {})
    intervention = step.get("intervention_verification", {})
    _require(control.get("enabled") is False and control.get("changed_attention_parameter_names") == [],
             f"{pair_key}: changed control generation")
    _require(intervention.get("enabled") is True and intervention.get("restored_exactly") is True,
             f"{pair_key}: intervention generation did not restore exactly")
    perturbation = intervention.get("perturbation", {})
    _require(perturbation.get("status") in {"changed", "no_op"}, f"{pair_key}: missing perturbation measurement")


def _counts(values: Iterable[Any]) -> dict[str, int]:
    result = {"correct": 0, "wrong": 0, "unscored": 0}
    for value in values:
        result["correct" if value is True else "wrong" if value is False else "unscored"] += 1
    return result


def combine_completed_runs(run_dirs: Iterable[Path | str], *, expected_plan: Path | None = None) -> dict[str, Any]:
    expected, expected_reference = _plan_grid(expected_plan)
    paths = [Path(path).expanduser().resolve() for path in run_dirs]
    _require(bool(paths), "at least one completed run directory is required")
    compatibility: dict[str, Any] | None = None
    controls: dict[str, dict[str, Any]] = {}
    seen: set[tuple[int, int, str]] = set()
    by_layer_seed: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    inputs = []

    for path in paths:
        _require((path / "RUN_COMPLETE").is_file(), f"{path}: missing RUN_COMPLETE")
        _require(not (path / "RUN_IN_PROGRESS").exists() and not (path / "RUN_FAILED.json").exists(), f"{path}: conflicting lifecycle markers")
        config = _read_json(path / "run_config.json")
        manifest_path = path / "run_manifest.json"
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes)
        status = _read_json(path / "run_status.json")
        _require(config.get("kind") == RUN_KIND and manifest.get("manifest_kind") == MANIFEST_KIND, f"{path}: unsupported multi-step protocol")
        _require(status.get("status") == "completed" and status.get("run_id") == config.get("run_id") == manifest.get("run_id"), f"{path}: incomplete lifecycle")
        _require(isinstance(manifest.get("job_details", {}).get("actual"), dict) and
                 isinstance(manifest.get("job_details", {}).get("requested"), dict),
                 f"{path}: missing scheduler provenance")
        _require(isinstance(manifest.get("runtime"), dict) and isinstance(manifest.get("tool_registry"), dict),
                 f"{path}: missing runtime or tool-registry provenance")
        _require(isinstance(manifest.get("started_at"), str) and
                 isinstance(status.get("started_at"), str) and isinstance(status.get("completed_at"), str),
                 f"{path}: missing lifecycle timestamps")
        run_config = config.get("config", {})
        _require(isinstance(run_config, dict), f"{path}: missing intervention configuration")
        layer, seed = run_config.get("layer"), run_config.get("seed")
        _require(type(layer) is int and type(seed) is int, f"{path}: invalid layer or seed")
        dataset = manifest.get("dataset", {})
        intervention = manifest.get("intervention", {})
        _require(dataset.get("selected_workflow_ids") == config.get("selected_workflow_ids"), f"{path}: manifest workflow selection mismatch")
        current = {
            "workflow_type": manifest.get("workflow_type"), "dataset_group": dataset.get("group"),
            "benchmark_sha256": dataset.get("sha256"), "selected_workflow_ids": dataset.get("selected_workflow_ids"),
            "checkpoint_fingerprint": manifest.get("model", {}).get("checkpoint_identity", {}).get("fingerprint"),
            "evaluation": manifest.get("evaluation"),
            "target": intervention.get("target"), "method": intervention.get("method"),
            "strength": intervention.get("strength"), "strength_mode": intervention.get("strength_mode"),
        }
        _require(all(value is not None for key, value in current.items() if key != "checkpoint_fingerprint"),
                 f"{path}: incomplete manifest provenance")
        if compatibility is None:
            compatibility = current
        else:
            _require(current == compatibility, f"{path}: incompatible model, dataset, scoring/protocol, or intervention settings")
        reference = {"run_id": manifest["run_id"], "filename": "run_manifest.json", "sha256": "sha256:" + hashlib.sha256(manifest_bytes).hexdigest()}
        records = _read_records(path / "paired_records.jsonl")
        local = set()
        for pair in records:
            sample_id = pair.get("sample_id")
            key = (layer, seed, sample_id)
            _require(pair.get("kind") == RUN_KIND and pair.get("run_manifest") == reference, f"{path}: unlinked paired record")
            _require(pair.get("layer_index") == layer and pair.get("seed") == seed, f"{key}: record layer/seed mismatch")
            _require(pair.get("dataset_group") == dataset.get("group"), f"{key}: dataset group mismatch")
            _require(isinstance(sample_id, str) and sample_id in dataset.get("selected_workflow_ids", []), f"{key}: unknown workflow ID")
            _require(key not in seen, f"duplicate layer–seed–workflow pair: {key}")
            for step in pair.get("steps", []):
                _check_step(key, step)
            identity = _control_identity(pair)
            if sample_id in controls:
                _require(controls[sample_id] == identity, f"{key}: repeated unchanged controls disagree")
            else:
                controls[sample_id] = identity
            seen.add(key)
            local.add(key)
            by_layer_seed[(layer, seed)].append(pair)
        expected_local = {(layer, seed, sample) for sample in dataset["selected_workflow_ids"]}
        _require(local == expected_local, f"{path}: incomplete layer–seed workflow chunk")
        inputs.append({"directory": str(path), "pair_count": len(records), "run_manifest_sha256": reference["sha256"]})

    if expected is not None:
        _require(seen == expected, "completed chunks do not match the frozen expected layer–seed–workflow grid")

    rows = []
    for (layer, seed), pairs in sorted(by_layer_seed.items()):
        steps = [step for pair in pairs for step in pair["steps"]]
        row = {
            "layer_index": layer, "seed": seed,
            "workflow_count": len(pairs), "step_count": len(steps),
            "step_metrics": {}, "workflow_metrics": {},
        }
        for condition in ("control", "intervention"):
            row["step_metrics"][condition] = {
                "tool_selection": _counts(step[condition].get("tool_selection_correct") for step in steps),
                "argument_match": _counts(step[condition].get("argument_match_correct") for step in steps),
                "execution_success": _counts(step[condition].get("execution_success") for step in steps),
                "final_outcome": _counts(step[condition].get("final_outcome_correct") for step in steps),
                "invalid_outputs": sum(step[condition].get("parse_status") != "ok" for step in steps),
                "no_call_outputs": sum(step[condition].get("selected_tool") is None for step in steps),
            }
            row["workflow_metrics"][condition] = {
                "all_steps_correct": _counts(pair[condition].get("all_steps_correct") for pair in pairs),
                "final_step_outcome": _counts(pair[condition].get("final_step_outcome_correct") for pair in pairs),
                "final_program_execution": _counts(pair[condition].get("final_program_execution_correct") for pair in pairs),
            }
        row["step_metrics"]["tool_selection_repairs"] = sum(
            step["control"].get("tool_selection_correct") is False and step["intervention"].get("tool_selection_correct") is True
            for step in steps
        )
        row["step_metrics"]["tool_selection_damage"] = sum(
            step["control"].get("tool_selection_correct") is True and step["intervention"].get("tool_selection_correct") is False
            for step in steps
        )
        row["workflow_metrics"]["all_steps_repairs"] = sum(
            pair["control"].get("all_steps_correct") is False and pair["intervention"].get("all_steps_correct") is True
            for pair in pairs
        )
        row["workflow_metrics"]["all_steps_damage"] = sum(
            pair["control"].get("all_steps_correct") is True and pair["intervention"].get("all_steps_correct") is False
            for pair in pairs
        )
        rows.append(row)
    return {
        "kind": SUMMARY_KIND, "compatibility": compatibility, "expected_plan": expected_reference,
        "input_runs": inputs, "unique_workflow_count": len(controls), "paired_observation_count": len(seen),
        "rows": rows,
    }


def save_summary(summary: dict[str, Any], output: Path) -> None:
    output = output.expanduser().resolve()
    for run in summary["input_runs"]:
        _require(not output.is_relative_to(Path(run["directory"])), "output must be outside input run directories")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def combine_linked_reference_runs(
    run_dirs: Iterable[Path | str], *, control_reference: Path, expected_plan: Path | None = None,
) -> dict[str, Any]:
    """Combine intervention-only multi-step chunks against one canonical rollout."""
    reference_dir = control_reference.expanduser().resolve(strict=True)
    reference_manifest = _read_json(reference_dir / REFERENCE_MANIFEST)
    _require(reference_manifest.get("kind") == REFERENCE_KIND and reference_manifest.get("workflow_type") == "multi_step",
             f"{reference_dir}: unsupported canonical multi-step reference")
    compatibility = reference_manifest.get("compatibility")
    _require(isinstance(compatibility, dict), f"{reference_dir}: missing reference compatibility")
    reference, indexed = load_reference(reference_dir, workflow_type="multi_step", compatibility=compatibility,
                                        expected_ids=reference_manifest.get("selected_ids", []))
    expected, expected_reference = _plan_grid(expected_plan)
    paths = [Path(path).expanduser().resolve() for path in run_dirs]
    _require(bool(paths), "at least one intervention-only run directory is required")
    seen: set[tuple[int, int, str]] = set()
    by_layer_seed: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    inputs = []
    for path in paths:
        _require((path / "RUN_COMPLETE").is_file() and not (path / "RUN_FAILED.json").exists(), f"{path}: incomplete intervention run")
        config = _read_json(path / "run_config.json")
        manifest_bytes = (path / "run_manifest.json").read_bytes()
        manifest = json.loads(manifest_bytes)
        status = _read_json(path / "run_status.json")
        _require(config.get("execution_mode") == "intervention_only" and
                 manifest.get("manifest_kind") == "phase2_multistep_intervention_only_run_manifest_v1" and
                 status.get("status") == "completed", f"{path}: not a completed intervention-only run")
        _require(manifest.get("control_reference") == reference, f"{path}: linked to another control reference")
        actual = {"workflow_type": "multi_step", "experiment_kind": config.get("kind"),
                  "checkpoint_fingerprint": config.get("checkpoint_identity", {}).get("fingerprint"),
                  "dataset_group": config.get("dataset_group"), "benchmark_sha256": config.get("benchmark_sha256"),
                  "selected_workflow_ids": config.get("selected_workflow_ids"),
                  "evaluation": manifest.get("evaluation"), "runtime": config.get("runtime"),
                  "tool_registry": manifest.get("tool_registry"), "generation": manifest.get("generation")}
        _require(actual == compatibility, f"{path}: incompatible model/dataset/generation/scoring protocol")
        run_config = config.get("config", {})
        layer, seed = run_config.get("layer"), run_config.get("seed")
        _require(type(layer) is int and type(seed) is int, f"{path}: invalid layer/seed")
        records = _read_records(path / "paired_records.jsonl")
        local = set()
        for pair in records:
            sample_id = pair.get("sample_id")
            key = (layer, seed, sample_id)
            _require(pair.get("control_reference") == reference and sample_id in indexed, f"{path}: unlinked workflow record")
            _require(key not in seen, f"duplicate layer–seed–workflow pair: {key}")
            control = indexed[sample_id]["control"]
            intervention = pair.get("intervention")
            _require(isinstance(intervention, dict), f"{key}: missing intervention workflow")
            materialized_steps = []
            control_steps = control.get("steps", [])
            for left, right in zip(control_steps, pair.get("steps", []), strict=True):
                _require((left.get("step_id"), left.get("step_index")) == (right.get("step_id"), right.get("step_index")),
                         f"{key}: changed step identity")
                _check_step(key, {"control_verification": {"enabled": False, "changed_attention_parameter_names": []},
                                  "intervention_verification": right.get("intervention_verification", {})})
                materialized_steps.append({**right, "control": {name: left.get(name) for name in (
                    "selected_tool", "selected_args", "parse_status", "tool_selection_correct", "argument_match_correct", "execution_success", "final_outcome_correct")}})
            materialized = {**pair, "control": {name: control.get(name) for name in (
                "all_tools_correct", "all_arguments_correct", "all_steps_correct", "final_step_outcome_correct", "final_program_execution_correct")},
                "steps": materialized_steps}
            seen.add(key); local.add(key); by_layer_seed[(layer, seed)].append(materialized)
        _require(local == {(layer, seed, sample) for sample in indexed}, f"{path}: incomplete layer–seed workflow chunk")
        inputs.append({"directory": str(path), "intervention_record_count": len(records),
                       "run_manifest_sha256": "sha256:" + hashlib.sha256(manifest_bytes).hexdigest()})
    if expected is not None:
        _require(seen == expected, "completed chunks do not match the frozen expected layer–seed–workflow grid")
    rows = []
    for (layer, seed), pairs in sorted(by_layer_seed.items()):
        steps = [step for pair in pairs for step in pair["steps"]]
        row = {"layer_index": layer, "seed": seed, "workflow_count": len(pairs), "step_count": len(steps),
               "step_metrics": {}, "workflow_metrics": {}}
        for condition in ("control", "intervention"):
            row["step_metrics"][condition] = {
                "tool_selection": _counts(step[condition].get("tool_selection_correct") for step in steps),
                "argument_match": _counts(step[condition].get("argument_match_correct") for step in steps),
                "execution_success": _counts(step[condition].get("execution_success") for step in steps),
                "final_outcome": _counts(step[condition].get("final_outcome_correct") for step in steps),
                "invalid_outputs": sum(step[condition].get("parse_status") != "ok" for step in steps),
                "no_call_outputs": sum(step[condition].get("selected_tool") is None for step in steps),
            }
            row["workflow_metrics"][condition] = {
                "all_steps_correct": _counts(pair[condition].get("all_steps_correct") for pair in pairs),
                "final_step_outcome": _counts(pair[condition].get("final_step_outcome_correct") for pair in pairs),
                "final_program_execution": _counts(pair[condition].get("final_program_execution_correct") for pair in pairs),
            }
        row["step_metrics"]["tool_selection_repairs"] = sum(not step["control"].get("tool_selection_correct") and step["intervention"].get("tool_selection_correct") for step in steps)
        row["step_metrics"]["tool_selection_damage"] = sum(step["control"].get("tool_selection_correct") and not step["intervention"].get("tool_selection_correct") for step in steps)
        row["workflow_metrics"]["all_steps_repairs"] = sum(not pair["control"].get("all_steps_correct") and pair["intervention"].get("all_steps_correct") for pair in pairs)
        row["workflow_metrics"]["all_steps_damage"] = sum(pair["control"].get("all_steps_correct") and not pair["intervention"].get("all_steps_correct") for pair in pairs)
        rows.append(row)
    return {"kind": SUMMARY_KIND, "execution_mode": "intervention_only", "control_reference": reference,
            "compatibility": compatibility, "expected_plan": expected_reference, "input_runs": inputs,
            "unique_workflow_count": len(indexed), "unique_control_count": len(indexed),
            "paired_observation_count": len(seen), "rows": rows}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-plan", type=Path)
    parser.add_argument("--control-reference", type=Path)
    args = parser.parse_args(argv)
    try:
        summary = (combine_linked_reference_runs(args.run_dir, control_reference=args.control_reference,
                                                 expected_plan=args.expected_plan)
                   if args.control_reference is not None else
                   combine_completed_runs(args.run_dir, expected_plan=args.expected_plan))
        save_summary(summary, args.output)
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.exit(2, f"Cannot combine multi-step paired results: {error}\n")
    print(f"Saved {len(summary['rows'])} layer/seed rows; {summary['unique_workflow_count']} unique workflows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
