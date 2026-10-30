"""Read completed GPT-OSS paired records; never load models or rescore outputs."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
from itertools import product
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable

from analysis.benchmark_inventory import infer_benchmark_class
from research.phase2.control_reference import REFERENCE_MANIFEST, REFERENCE_KIND, load_reference


RUN_KIND = "phase2_gpt_oss_attention_intervention_eval_v1"
HARDENED_RUN_KIND = "phase2_gpt_oss_attention_intervention_eval_v2"
CURRENT_RUN_KIND = "phase2_gpt_oss_attention_intervention_eval_v3"
SUPPORTED_RUN_KINDS = (RUN_KIND, HARDENED_RUN_KIND, CURRENT_RUN_KIND)
SUMMARY_KIND = "phase2_gpt_oss_paired_results_summary_v1"
METRICS = (
    "tool_selection_correct", "argument_match_correct",
    "execution_success", "final_outcome_correct",
)
PROVENANCE_KEYS = (
    "model", "expected_model_name", "reasoning_mode", "reasoning_method",
    "reasoning_effort", "prompt_template_id", "generation",
    "effective_generation_limit", "effective_generation_limit_unit",
    "evaluation_protocol", "benchmark_mode", "tool_pool", "tool_count",
    "tool_registry_fingerprint", "tool_registry_fingerprint_version",
    "final_outcome_matchers",
)
SAMPLE_KEYS = (
    "query", "prompt_context", "expected_tool", "expected_args", "expected_answer",
    "benchmark_path", "source", "domain", "task_type", "benchmark_mode",
    "final_outcome_matcher", "model_name", "reasoning_mode", "reasoning_method",
    "reasoning_effort", "prompt_template", "evaluation_protocol", "tool_pool",
    "tool_count", "tool_registry_fingerprint", "tool_registry_fingerprint_version",
)
CONTROL_KEYS = (
    "selected_tool", "selected_args", "raw_model_output", "parse_status",
    "invalid_output", "no_call_outcome", *METRICS,
    "tool_result_value", "final_outcome_status",
)


def _require(test: bool, message: str) -> None:
    if not test:
        raise ValueError(message)


def _select(record: dict[str, Any], keys: Iterable[str]) -> dict[str, Any]:
    return {key: record.get(key) for key in keys}


def _ids(value: Any, label: str) -> list[int]:
    _require(isinstance(value, list) and bool(value), f"{label}: expected nonempty integer list")
    _require(all(type(item) is int and item >= 0 for item in value), f"{label}: invalid integer")
    _require(len(set(value)) == len(value), f"{label}: duplicate values")
    return value


def _condition(value: Any, sample_id: str) -> dict[str, Any]:
    _require(isinstance(value, dict), f"{sample_id}: missing full evaluation condition")
    _require(value.get("sample_id") == sample_id, f"{sample_id}: condition sample ID mismatch")
    for key in (*METRICS[:-1], "invalid_output", "no_call_outcome"):
        _require(type(value.get(key)) is bool, f"{sample_id}: missing/invalid saved boolean {key}")
    _require("final_outcome_correct" in value and
             (value["final_outcome_correct"] is None or type(value["final_outcome_correct"]) is bool),
             f"{sample_id}: missing/invalid saved final outcome")
    _require(isinstance(value.get("parse_status"), str), f"{sample_id}: missing parse status")
    _require("selected_tool" in value, f"{sample_id}: missing selected tool")
    _require(value["selected_tool"] is None or isinstance(value["selected_tool"], str),
             f"{sample_id}: invalid selected tool")
    _require(value["invalid_output"] == (value["parse_status"] != "ok"),
             f"{sample_id}: inconsistent saved parse flags")
    _require(value["no_call_outcome"] == (value["selected_tool"] is None),
             f"{sample_id}: inconsistent saved no-call flag")
    for alias, canonical in (("tool_choice_correct", "tool_selection_correct"),
                             ("chosen_tool", "selected_tool")):
        _require(alias not in value or value[alias] == value[canonical],
                 f"{sample_id}: inconsistent saved {alias}")
    _require(not value["tool_selection_correct"] or
             (not value["invalid_output"] and not value["no_call_outcome"]),
             f"{sample_id}: invalid/no-call output marked tool-correct")
    _require(bool(value.get("tool_registry_fingerprint")), f"{sample_id}: missing registry fingerprint")
    for key in ("query", "prompt_context", "expected_tool", "expected_args", "expected_answer",
                "benchmark_path", "benchmark_mode", "selected_args", "raw_model_output"):
        _require(key in value, f"{sample_id}: missing saved {key}")
    return value


def _bucket(condition: dict[str, Any]) -> str:
    # Mutually exclusive categories; no_call_outputs also reports the overlap.
    if condition["invalid_output"]:
        return "invalid_outputs"
    if condition["no_call_outcome"]:
        return "valid_no_call_outputs"
    return "correct_tool_calls" if condition["tool_selection_correct"] else "valid_wrong_tool_calls"


def _hardened_provenance(manifest: dict[str, Any], path: Path) -> dict[str, Any]:
    """Validate saved v2 identities offline; never reopen checkpoint weights."""
    identity = manifest.get("checkpoint_identity", {})
    _require(isinstance(identity, dict) and identity.get("status") == "complete",
             f"{path}: incomplete hardened checkpoint identity")
    _require(identity.get("checkpoint_directory") == manifest.get("checkpoint_path"),
             f"{path}: hardened identity points at a different checkpoint")
    files = identity.get("manifest")
    _require(isinstance(files, dict) and set(files) == {"weights", "config", "tokenizer"},
             f"{path}: missing hardened file manifest")
    for category, entries in files.items():
        _require(isinstance(entries, list) and bool(entries), f"{path}: missing {category} identity")
        names = []
        for entry in entries:
            _require(isinstance(entry, dict) and isinstance(entry.get("name"), str), f"{path}: invalid {category} manifest")
            names.append(entry["name"])
            _require(type(entry.get("bytes")) is int and entry["bytes"] >= 0 and
                     isinstance(entry.get("sha256"), str) and
                     re.fullmatch(r"sha256:[0-9a-f]{64}", entry["sha256"]) is not None,
                     f"{path}: invalid {category} file hash/size")
        _require(len(set(names)) == len(names), f"{path}: duplicate manifest files")
    fingerprint = "sha256:" + hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    _require(identity.get("fingerprint") == fingerprint and manifest.get("checkpoint_fingerprint") == fingerprint,
             f"{path}: hardened fingerprint disagrees with saved file manifest")
    _require(manifest.get("experiment_kind") in (HARDENED_RUN_KIND, CURRENT_RUN_KIND) and
             manifest.get("condition_isolation") == "fresh_mcp_server_per_condition" and
             manifest.get("strength_mode") in ("absolute", "tensor_rms"), f"{path}: invalid hardened protocol labels")
    _require(manifest.get("model_binding") == "native_loader", f"{path}: unverified hardened model binding")
    input_kind = manifest.get("input_kind", "saved_source_run")
    _require(input_kind in ("saved_source_run", "benchmark_files"), f"{path}: unknown input kind")
    if input_kind == "saved_source_run":
        _require(manifest.get("source_run_id") == Path(manifest.get("source_run_directory", "")).name,
                 f"{path}: inconsistent source run ID")
    else:
        hashes = manifest.get("benchmark_sha256")
        selected = manifest.get("selected_sample_manifest")
        _require(isinstance(hashes, dict) and hashes and all(isinstance(k, str) and re.fullmatch(r"[0-9a-f]{64}", v or "") for k, v in hashes.items()),
                 f"{path}: missing direct benchmark hashes")
        _require(isinstance(selected, list) and selected, f"{path}: missing direct selected-sample manifest")
    runtime = manifest.get("runtime")
    _require(isinstance(runtime, dict) and isinstance(runtime.get("git_commit"), str) and
             re.fullmatch(r"[0-9a-f]{40}", runtime["git_commit"]) is not None,
             f"{path}: missing current runtime commit")
    for field in ("tracked_worktree_dirty", "python", "torch", "cuda_runtime", "packages",
                  "devices", "dtypes", "gpu_names", "gpu_driver", "deterministic_algorithms",
                  "cudnn_deterministic", "cudnn_benchmark", "matmul_allow_tf32", "generation", "runtime_code_fingerprint"):
        _require(field in runtime, f"{path}: missing runtime setting {field}")
    _require(isinstance(runtime["runtime_code_fingerprint"], str) and
             re.fullmatch(r"sha256:[0-9a-f]{64}", runtime["runtime_code_fingerprint"]),
             f"{path}: invalid runtime code fingerprint")
    keys = ("experiment_kind", "input_kind", "checkpoint_fingerprint", "model_binding",
            "condition_isolation", "strength_mode", "runtime")
    result = {key: manifest.get(key) for key in keys}
    if input_kind == "saved_source_run":
        result.update({key: manifest.get(key) for key in ("source_run_id", "source_checkpoint_equivalence")})
    else:
        result["benchmark_sha256"] = manifest["benchmark_sha256"]
        result["selected_sample_manifest"] = manifest["selected_sample_manifest"]
    return result


def _hardened_pair(record: dict[str, Any], provenance: dict[str, Any], key: tuple,
                   manifest_reference: dict[str, Any] | None = None) -> None:
    record_keys = tuple(key for key in provenance if key not in ("benchmark_sha256", "selected_sample_manifest"))
    _require(_select(record, record_keys) == _select(provenance, record_keys), f"{key}: hardened record provenance differs from manifest")
    if manifest_reference is not None:
        _require(record.get("run_manifest") == manifest_reference, f"{key}: paired record is not linked to this run manifest")
    for name in ("control", "intervention"):
        condition = record.get(name, {})
        _require(isinstance(condition, dict) and condition.get("condition") == name and condition.get("experiment_kind") in (HARDENED_RUN_KIND, CURRENT_RUN_KIND),
                 f"{key}: missing hardened condition labels")
    control_verification = record.get("control_verification", {})
    verification = record.get("intervention_verification", {})
    _require(isinstance(control_verification, dict) and isinstance(verification, dict), f"{key}: missing verification")
    control = control_verification.get("perturbation", {})
    _require(isinstance(control, dict) and control.get("status") == "disabled" and control.get("changed_elements") == 0,
             f"{key}: missing disabled-control diagnostics")
    diagnostic = verification.get("perturbation", {})
    _require(isinstance(diagnostic, dict), f"{key}: missing actual perturbation diagnostics")
    status, changed = diagnostic.get("status"), diagnostic.get("changed_elements")
    _require(status in ("changed", "no_op") and type(changed) is int and changed >= 0 and
             (status == "changed") == (changed > 0) and diagnostic.get("strength_mode") in ("absolute", "tensor_rms"),
             f"{key}: inconsistent actual/no-op perturbation diagnostics")


def _expected_grid(plan_path: Path | None) -> tuple[set[tuple[int, int, str]] | None, dict[str, Any] | None]:
    if plan_path is None:
        return None, None
    content = plan_path.expanduser().resolve(strict=True).read_bytes()
    payload = json.loads(content)
    plan = payload.get("screen_plan", {})
    layers, seeds, samples = plan.get("selected_layers"), plan.get("seeds"), plan.get("sample_ids")
    _require(isinstance(layers, list) and isinstance(seeds, list) and isinstance(samples, list),
             f"{plan_path}: invalid frozen screen plan")
    _require(all(type(layer) is int for layer in layers) and all(type(seed) is int for seed in seeds) and
             all(isinstance(sample, str) and sample for sample in samples), f"{plan_path}: invalid expected grid")
    return set(product(layers, seeds, samples)), {
        "path": str(plan_path.expanduser().resolve()),
        "sha256": "sha256:" + hashlib.sha256(content).hexdigest(),
        "screen_plan": plan,
    }


def combine_completed_runs(run_dirs: Iterable[Path | str], *, expected_plan: Path | None = None) -> dict[str, Any]:
    """Validate completed shards and return a compact, provenance-preserving summary.

    Layers/seeds may differ; every shard must contain the same complete sample panel.
    Repeated controls must agree in output, parsed call and saved scores (not latency).
    """
    paths = [Path(path).expanduser().resolve() for path in run_dirs]
    _require(bool(paths), "At least one completed run directory is required")
    compatibility = None
    controls: dict[str, dict[str, Any]] = {}
    identities: dict[str, dict[str, Any]] = {}
    seen: set[tuple[int, int, str]] = set()
    groups: dict[tuple[int, int, str, str], list[dict[str, Any]]] = defaultdict(list)
    inputs = []
    expected_keys, expected_plan_reference = _expected_grid(expected_plan)
    for path in paths:
        _require((path / "RUN_COMPLETE").is_file(), f"{path}: missing RUN_COMPLETE")
        _require(not (path / "RUN_IN_PROGRESS").exists() and not (path / "RUN_FAILED.json").exists(),
                 f"{path}: conflicting completion/failure markers")
        config_bytes = (path / "run_config.json").read_bytes()
        record_bytes = (path / "paired_records.jsonl").read_bytes()
        manifest = json.loads(config_bytes)
        _require(isinstance(manifest, dict), f"{path}: run configuration must be an object")
        run_kind = manifest.get("kind")
        _require(run_kind in SUPPORTED_RUN_KINDS, f"{path}: unsupported run kind")
        _require(manifest.get("call_predicted_tools") is True, f"{path}: not a full tool-execution run")
        config = manifest.get("resolved_config", manifest.get("config", {}))
        _require(isinstance(config, dict), f"{path}: missing concrete configuration")
        layers = _ids(config.get("layers"), "layers")
        seeds = _ids(config.get("seeds"), "seeds")
        strength = config.get("strength")
        _require(type(strength) in (int, float) and math.isfinite(strength) and strength >= 0,
                 f"{path}: invalid strength")
        _require(config.get("method") in ("noise", "replace"), f"{path}: unsupported method")
        _require(config.get("target") in ("attention_all", "qkv", "out"), f"{path}: unsupported GPT-OSS target")
        records = [json.loads(line) for line in record_bytes.splitlines() if line.strip()]
        _require(bool(records), f"{path}: empty completed run")
        _require(all(isinstance(record, dict) and isinstance(record.get("sample_id"), str)
                     and bool(record["sample_id"]) for record in records), f"{path}: invalid records/sample IDs")
        samples = sorted({record["sample_id"] for record in records})
        declared_samples = config.get("sample_ids")
        if declared_samples is not None:
            _require(sorted(declared_samples) == samples, f"{path}: sample panel differs from configuration")
        _require(manifest.get("example_count") == len(samples), f"{path}: example_count mismatch")
        metadata = manifest.get("source_run_metadata", {})
        _require(isinstance(metadata, dict), f"{path}: invalid source metadata")
        current = {"experiment_kind": run_kind, "sample_ids": samples,
                   **_select(config, ("target", "method", "strength", "strength_mode"))}
        input_kind = manifest.get("input_kind", "saved_source_run")
        if input_kind == "saved_source_run":
            current["source_run_metadata"] = _select(metadata, PROVENANCE_KEYS)
        elif input_kind == "benchmark_files":
            current["benchmark_sha256"] = manifest.get("benchmark_sha256")
            current["selected_sample_manifest"] = manifest.get("selected_sample_manifest")
        else:
            current["source_run_metadata"] = _select(metadata, PROVENANCE_KEYS)
        for key in ("checkpoint_path",):
            value = manifest.get(key)
            _require(isinstance(value, str) and Path(value).is_absolute(), f"{path}: missing absolute {key}")
            current[key] = str(Path(value).resolve())
        if input_kind == "saved_source_run":
            value = manifest.get("source_run_directory")
            _require(isinstance(value, str) and Path(value).is_absolute(), f"{path}: missing absolute source_run_directory")
            current["source_run_directory"] = str(Path(value).resolve())
        if input_kind == "saved_source_run" or run_kind == RUN_KIND:
            for field in ("tool_registry_fingerprint", "tool_registry_fingerprint_version", "tool_pool", "tool_count"):
                _require(bool(metadata.get(field)), f"{path}: missing source registry metadata {field}")
        hardened = _hardened_provenance(manifest, path) if run_kind in (HARDENED_RUN_KIND, CURRENT_RUN_KIND) else None
        manifest_reference = None
        run_registry_metadata: dict[str, Any] | None = None
        if run_kind == CURRENT_RUN_KIND:
            manifest_path = path / "run_manifest.json"
            _require(manifest_path.is_file(), f"{path}: missing run_manifest.json")
            manifest_bytes = manifest_path.read_bytes()
            full_manifest = json.loads(manifest_bytes)
            _require(full_manifest.get("manifest_kind") == "phase2_paired_run_manifest_v1", f"{path}: invalid run manifest kind")
            _require(full_manifest.get("run_id") == manifest.get("run_id"), f"{path}: run manifest/config run ID mismatch")
            _require(full_manifest.get("tool_registry", {}).get("metadata"), f"{path}: missing run-level registry identity")
            status_path = path / "run_status.json"
            _require(status_path.is_file(), f"{path}: missing run lifecycle status")
            status = json.loads(status_path.read_text())
            _require(status.get("run_id") == manifest.get("run_id") and status.get("status") == "completed",
                     f"{path}: run lifecycle is not completed")
            run_registry_metadata = full_manifest["tool_registry"]["metadata"]
            manifest_reference = {"run_id": manifest["run_id"], "filename": "run_manifest.json",
                                  "sha256": "sha256:" + hashlib.sha256(manifest_bytes).hexdigest()}
            current["parent_plan"] = manifest.get("parent_plan")
        if hardened is not None:
            current["hardened_provenance"] = hardened
        if compatibility is None:
            compatibility = current
        else:
            _require(current == compatibility, f"{path}: incompatible checkpoint/source-run, samples, settings or provenance")
        local_keys = set()
        for record in records:
            sample_id = record["sample_id"]
            _require(record.get("kind") == run_kind, f"{path}: unsupported record kind")
            _require(all(record.get(key) == config[key] for key in ("target", "method", "strength")),
                     f"{sample_id}: record settings differ from run configuration")
            layer, seed = record.get("layer_index"), record.get("seed")
            _require(type(layer) is int and type(seed) is int and layer in layers and seed in seeds,
                     f"{sample_id}: undeclared layer/seed")
            key = (layer, seed, sample_id)
            if hardened is not None:
                _hardened_pair(record, hardened, key, manifest_reference)
            _require(key not in seen, f"Duplicate layer–seed–sample pair: {key}")
            seen.add(key)
            local_keys.add(key)
            verification = record.get("intervention_verification", {})
            _require(isinstance(verification, dict), f"{key}: missing exact weight restoration")
            _require(verification.get("enabled") is True and verification.get("restored_exactly") is True,
                     f"{key}: missing exact weight restoration")
            control_check = record.get("control_verification", {})
            _require(isinstance(control_check, dict), f"{key}: missing unchanged-control verification")
            # Disabled controls legitimately have restored_exactly=null: no weights changed.
            _require(control_check.get("enabled") is False and
                     control_check.get("changed_attention_parameter_names") == [],
                     f"{key}: control was not unchanged")
            _require(record.get("registry_exact_match") is True, f"{key}: registry was not an exact match")
            control = _condition(record.get("control"), sample_id)
            intervention = _condition(record.get("intervention"), sample_id)
            identity = _select(control, SAMPLE_KEYS)
            _require(identity == _select(intervention, SAMPLE_KEYS), f"{key}: sample/provenance differs between conditions")
            expected_registry = metadata if run_registry_metadata is None else run_registry_metadata
            for field in ("tool_pool", "tool_count", "tool_registry_fingerprint", "tool_registry_fingerprint_version",
                          "reasoning_mode", "reasoning_method", "reasoning_effort", "evaluation_protocol"):
                if field in expected_registry:
                    _require(control.get(field) == expected_registry.get(field), f"{key}: condition/source {field} mismatch")
            if sample_id in controls:
                _require(identity == identities[sample_id], f"{key}: changed sample definition")
                _require(_select(control, CONTROL_KEYS) == _select(controls[sample_id], CONTROL_KEYS),
                         f"{key}: repeated unchanged controls disagree; review control drift separately")
            else:
                controls[sample_id], identities[sample_id] = control, identity
            classification = infer_benchmark_class(Path(control.get("benchmark_path", "unknown")), [control])
            groups[(layer, seed, classification, control.get("benchmark_mode", "unknown"))].append(record)
        _require(local_keys == set(product(layers, seeds, samples)), f"{path}: incomplete layer–seed–sample grid")
        inputs.append({"directory": str(path), "pair_count": len(records),
                       "run_config_sha256": hashlib.sha256(config_bytes).hexdigest(),
                       "paired_records_sha256": hashlib.sha256(record_bytes).hexdigest()})

    if expected_keys is not None:
        _require(seen == expected_keys,
                 "Completed chunks do not match the frozen expected layer–seed–sample grid")

    rows = []
    for (layer, seed, classification, mode), records in sorted(groups.items()):
        row = {"layer_index": layer, "seed": seed, "benchmark_classification": classification,
               "benchmark_mode": mode, "sample_count": len(records),
               "tool_choice_changes": 0, "tool_choice_repairs": 0, "tool_choice_damage": 0,
               "correct_correct": 0, "correct_wrong": 0, "wrong_correct": 0, "wrong_wrong": 0,
               "no_op_interventions": 0,
               "final_outcome_paired_scored": 0, "final_outcome_repairs": 0, "final_outcome_damage": 0}
        for condition in ("control", "intervention"):
            for category in ("invalid_outputs", "valid_no_call_outputs", "valid_wrong_tool_calls", "correct_tool_calls",
                             "no_call_outputs"):
                row[f"{condition}_{category}"] = 0
            for metric in METRICS:
                row[f"{condition}_{metric}"] = 0
            row[f"{condition}_final_outcome_scored"] = 0
        for record in records:
            control, intervention = record["control"], record["intervention"]
            transition = f"{'correct' if control['tool_selection_correct'] else 'wrong'}_{'correct' if intervention['tool_selection_correct'] else 'wrong'}"
            row[transition] += 1
            row["no_op_interventions"] += record.get("intervention_verification", {}).get("perturbation", {}).get("status") == "no_op"
            row["tool_choice_changes"] += control["selected_tool"] != intervention["selected_tool"]
            row["tool_choice_repairs"] += not control["tool_selection_correct"] and intervention["tool_selection_correct"]
            row["tool_choice_damage"] += control["tool_selection_correct"] and not intervention["tool_selection_correct"]
            if control["final_outcome_correct"] is not None and intervention["final_outcome_correct"] is not None:
                row["final_outcome_paired_scored"] += 1
                row["final_outcome_repairs"] += not control["final_outcome_correct"] and intervention["final_outcome_correct"]
                row["final_outcome_damage"] += control["final_outcome_correct"] and not intervention["final_outcome_correct"]
            for condition in ("control", "intervention"):
                value = record[condition]
                row[f"{condition}_{_bucket(value)}"] += 1
                row[f"{condition}_no_call_outputs"] += value["no_call_outcome"]
                for metric in METRICS:
                    row[f"{condition}_{metric}"] += value[metric] is True
                row[f"{condition}_final_outcome_scored"] += value["final_outcome_correct"] is not None
        successes = row["control_tool_selection_correct"]
        failures = row["sample_count"] - successes
        row["tool_choice_repair_rate"] = row["tool_choice_repairs"] / failures if failures else None
        row["tool_choice_damage_rate"] = row["tool_choice_damage"] / successes if successes else None
        rows.append(row)
    return {"kind": SUMMARY_KIND, "compatibility": compatibility, "input_runs": inputs,
            "provenance_status": "linked_run_manifest" if compatibility["experiment_kind"] == CURRENT_RUN_KIND else "hardened_file_and_runtime_identity" if compatibility["experiment_kind"] == HARDENED_RUN_KIND else "legacy_path_based_unverified_identity",
            "expected_plan": expected_plan_reference,
            "unique_sample_count": len(controls), "paired_observation_count": len(seen),
            "unique_controls": [{"sample_id": sample_id, **_select(controls[sample_id], METRICS)}
                                for sample_id in sorted(controls)],
            "rows": rows}


def combine_linked_reference_runs(
    run_dirs: Iterable[Path | str], *, control_reference: Path, expected_plan: Path | None = None,
) -> dict[str, Any]:
    """Combine intervention-only chunks with one independently stored control.

    This intentionally materializes controls only in memory for comparison. The
    output counts the reference controls once, not once for each layer/seed.
    """
    reference_dir = control_reference.expanduser().resolve(strict=True)
    reference_manifest = json.loads((reference_dir / REFERENCE_MANIFEST).read_text(encoding="utf-8"))
    _require(reference_manifest.get("kind") == REFERENCE_KIND and reference_manifest.get("workflow_type") == "single_step",
             f"{reference_dir}: unsupported canonical control reference")
    reference_compatibility = reference_manifest.get("compatibility")
    _require(isinstance(reference_compatibility, dict), f"{reference_dir}: missing reference compatibility")
    reference, controls = load_reference(reference_dir, workflow_type="single_step", compatibility=reference_compatibility,
                                         expected_ids=reference_manifest.get("selected_ids", []))
    paths = [Path(path).expanduser().resolve() for path in run_dirs]
    _require(bool(paths), "At least one intervention-only run directory is required")
    expected_keys, expected_plan_reference = _expected_grid(expected_plan)
    seen: set[tuple[int, int, str]] = set()
    groups: dict[tuple[int, int, str, str], list[dict[str, Any]]] = defaultdict(list)
    inputs = []
    for path in paths:
        _require((path / "RUN_COMPLETE").is_file() and not (path / "RUN_FAILED.json").exists(), f"{path}: incomplete intervention run")
        config = json.loads((path / "run_config.json").read_text(encoding="utf-8"))
        manifest_bytes = (path / "run_manifest.json").read_bytes()
        manifest = json.loads(manifest_bytes)
        status = json.loads((path / "run_status.json").read_text(encoding="utf-8"))
        _require(config.get("execution_mode") == "intervention_only" and
                 manifest.get("manifest_kind") == "phase2_intervention_only_run_manifest_v1" and
                 status.get("status") == "completed", f"{path}: not a completed intervention-only run")
        _require(manifest.get("control_reference") == reference, f"{path}: references a different canonical control")
        input_keys = ("source_run_directory", "source_run_metadata", "selected_sample_ids") if config.get("input_kind") == "saved_source_run" else (
            "benchmark_paths", "benchmark_sha256", "selected_sample_manifest", "selected_sample_ids")
        current_compatibility = {
            "workflow_type": "single_step", "experiment_kind": config.get("experiment_kind"),
            "checkpoint_fingerprint": config.get("checkpoint_fingerprint"), "model_binding": config.get("model_binding"),
            "runtime": config.get("runtime"), "inputs": {key: config.get(key) for key in input_keys},
            "generation": {"reasoning_mode": "reasoning", "reasoning_effort": "low"},
            "evaluation": {"call_predicted_tools": config.get("call_predicted_tools", True),
                           "condition_isolation": config.get("condition_isolation")},
            "tool_registry": manifest.get("tool_registry"),
        }
        _require(current_compatibility == reference_compatibility, f"{path}: incompatible reference protocol")
        resolved = config.get("resolved_config", {})
        layers, seeds = _ids(resolved.get("layers"), "layers"), _ids(resolved.get("seeds"), "seeds")
        records = [json.loads(line) for line in (path / "paired_records.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        local = set()
        for record in records:
            sample_id, layer, seed = record.get("sample_id"), record.get("layer_index"), record.get("seed")
            key = (layer, seed, sample_id)
            _require(record.get("execution_mode") == "intervention_only" and record.get("control_reference") == reference,
                     f"{path}: unlinked intervention record")
            _require(sample_id in controls and layer in layers and seed in seeds, f"{path}: invalid intervention coverage")
            _require(key not in seen, f"Duplicate layer–seed–sample pair: {key}")
            intervention = _condition(record.get("intervention"), sample_id)
            control = _condition(controls[sample_id].get("control"), sample_id)
            verification = record.get("intervention_verification", {})
            _require(verification.get("enabled") is True and verification.get("restored_exactly") is True,
                     f"{key}: intervention did not restore exactly")
            _require(record.get("registry_exact_match") is True, f"{key}: registry was not exact")
            materialized = {**record, "control": control}
            seen.add(key); local.add(key)
            groups[(layer, seed, infer_benchmark_class(Path(control.get("benchmark_path", "unknown")), [control]), control.get("benchmark_mode", "unknown"))].append(materialized)
        _require(local == set(product(layers, seeds, controls)), f"{path}: incomplete layer–seed–sample grid")
        inputs.append({"directory": str(path), "intervention_record_count": len(records),
                       "run_manifest_sha256": "sha256:" + hashlib.sha256(manifest_bytes).hexdigest()})
    if expected_keys is not None:
        _require(seen == expected_keys, "Completed chunks do not match the frozen expected layer–seed–sample grid")
    rows = []
    for (layer, seed, classification, mode), records in sorted(groups.items()):
        row = {"layer_index": layer, "seed": seed, "benchmark_classification": classification, "benchmark_mode": mode,
               "sample_count": len(records), "tool_choice_changes": 0, "tool_choice_repairs": 0, "tool_choice_damage": 0,
               "correct_correct": 0, "correct_wrong": 0, "wrong_correct": 0, "wrong_wrong": 0,
               "no_op_interventions": 0, "final_outcome_paired_scored": 0,
               "final_outcome_repairs": 0, "final_outcome_damage": 0}
        for condition in ("control", "intervention"):
            for metric in METRICS:
                row[f"{condition}_{metric}"] = 0
            for category in ("invalid_outputs", "valid_no_call_outputs", "valid_wrong_tool_calls", "correct_tool_calls", "no_call_outputs"):
                row[f"{condition}_{category}"] = 0
            row[f"{condition}_final_outcome_scored"] = 0
        for record in records:
            control, intervention = record["control"], record["intervention"]
            transition = f"{'correct' if control['tool_selection_correct'] else 'wrong'}_{'correct' if intervention['tool_selection_correct'] else 'wrong'}"
            row[transition] += 1
            row["tool_choice_changes"] += control["selected_tool"] != intervention["selected_tool"]
            row["tool_choice_repairs"] += not control["tool_selection_correct"] and intervention["tool_selection_correct"]
            row["tool_choice_damage"] += control["tool_selection_correct"] and not intervention["tool_selection_correct"]
            row["no_op_interventions"] += record["intervention_verification"]["perturbation"]["status"] == "no_op"
            if control["final_outcome_correct"] is not None and intervention["final_outcome_correct"] is not None:
                row["final_outcome_paired_scored"] += 1
                row["final_outcome_repairs"] += not control["final_outcome_correct"] and intervention["final_outcome_correct"]
                row["final_outcome_damage"] += control["final_outcome_correct"] and not intervention["final_outcome_correct"]
            for condition, value in (("control", control), ("intervention", intervention)):
                row[f"{condition}_{_bucket(value)}"] += 1
                row[f"{condition}_no_call_outputs"] += value["no_call_outcome"]
                for metric in METRICS:
                    row[f"{condition}_{metric}"] += value[metric] is True
                row[f"{condition}_final_outcome_scored"] += value["final_outcome_correct"] is not None
        rows.append(row)
    return {"kind": SUMMARY_KIND, "execution_mode": "intervention_only", "control_reference": reference,
            "compatibility": reference_compatibility, "input_runs": inputs, "expected_plan": expected_plan_reference,
            "unique_sample_count": len(controls), "unique_control_count": len(controls),
            "paired_observation_count": len(seen), "rows": rows}


def save_summary(summary: dict[str, Any], output: Path) -> None:
    """Write only a new report outside all input folders; never overwrite artifacts."""
    output = output.expanduser().resolve()
    for run in summary["input_runs"]:
        _require(not output.is_relative_to(Path(run["directory"])), "Output must be outside input run directories")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", action="append", required=True, type=Path,
                        help="Completed paired output folder; repeat for each shard")
    parser.add_argument("--output", required=True, type=Path, help="New JSON report outside input folders")
    parser.add_argument("--expected-plan", type=Path,
                        help="Frozen plan; reject missing or extra layer/seed/sample coverage")
    parser.add_argument("--control-reference", type=Path,
                        help="Canonical-control directory required for intervention-only chunks")
    args = parser.parse_args(argv)
    try:
        summary = (combine_linked_reference_runs(args.run_dir, control_reference=args.control_reference,
                                                 expected_plan=args.expected_plan)
                   if args.control_reference is not None else
                   combine_completed_runs(args.run_dir, expected_plan=args.expected_plan))
        save_summary(summary, args.output)
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.exit(2, f"Cannot combine paired results: {error}\n")
    print(f"Saved {len(summary['rows'])} layer/seed/benchmark rows; "
          f"{summary['unique_sample_count']} unique examples, "
          f"{summary['paired_observation_count']} paired observations to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
