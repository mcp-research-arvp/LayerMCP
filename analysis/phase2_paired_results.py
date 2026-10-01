"""Read completed GPT-OSS paired records; never load models or rescore outputs."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
from itertools import product
import json
import math
from pathlib import Path
from typing import Any, Iterable

from analysis.benchmark_inventory import infer_benchmark_class


RUN_KIND = "phase2_gpt_oss_attention_intervention_eval_v1"
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


def combine_completed_runs(run_dirs: Iterable[Path | str]) -> dict[str, Any]:
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
    for path in paths:
        _require((path / "RUN_COMPLETE").is_file(), f"{path}: missing RUN_COMPLETE")
        _require(not (path / "RUN_IN_PROGRESS").exists() and not (path / "RUN_FAILED.json").exists(),
                 f"{path}: conflicting completion/failure markers")
        config_bytes = (path / "run_config.json").read_bytes()
        record_bytes = (path / "paired_records.jsonl").read_bytes()
        manifest = json.loads(config_bytes)
        _require(isinstance(manifest, dict), f"{path}: run configuration must be an object")
        _require(manifest.get("kind") == RUN_KIND, f"{path}: unsupported run kind")
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
        current = {"sample_ids": samples, **_select(config, ("target", "method", "strength")),
                   "source_run_metadata": _select(metadata, PROVENANCE_KEYS)}
        for key in ("checkpoint_path", "source_run_directory"):
            value = manifest.get(key)
            _require(isinstance(value, str) and Path(value).is_absolute(), f"{path}: missing absolute {key}")
            current[key] = str(Path(value).resolve())
        for field in ("tool_registry_fingerprint", "tool_registry_fingerprint_version", "tool_pool", "tool_count"):
            _require(bool(metadata.get(field)), f"{path}: missing source registry metadata {field}")
        if compatibility is None:
            compatibility = current
        else:
            _require(current == compatibility, f"{path}: incompatible checkpoint/source-run, samples, settings or provenance")
        local_keys = set()
        for record in records:
            sample_id = record["sample_id"]
            _require(record.get("kind") == RUN_KIND, f"{path}: unsupported record kind")
            _require(all(record.get(key) == config[key] for key in ("target", "method", "strength")),
                     f"{sample_id}: record settings differ from run configuration")
            layer, seed = record.get("layer_index"), record.get("seed")
            _require(type(layer) is int and type(seed) is int and layer in layers and seed in seeds,
                     f"{sample_id}: undeclared layer/seed")
            key = (layer, seed, sample_id)
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
            for field in ("tool_pool", "tool_count", "tool_registry_fingerprint", "tool_registry_fingerprint_version",
                          "reasoning_mode", "reasoning_method", "reasoning_effort", "evaluation_protocol"):
                _require(control.get(field) == metadata.get(field), f"{key}: condition/source {field} mismatch")
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

    rows = []
    for (layer, seed, classification, mode), records in sorted(groups.items()):
        row = {"layer_index": layer, "seed": seed, "benchmark_classification": classification,
               "benchmark_mode": mode, "sample_count": len(records),
               "tool_choice_changes": 0, "tool_choice_repairs": 0, "tool_choice_damage": 0,
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
            "unique_sample_count": len(controls), "paired_observation_count": len(seen),
            "unique_controls": [{"sample_id": sample_id, **_select(controls[sample_id], METRICS)}
                                for sample_id in sorted(controls)],
            "rows": rows}


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
    args = parser.parse_args(argv)
    try:
        summary = combine_completed_runs(args.run_dir)
        save_summary(summary, args.output)
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.exit(2, f"Cannot combine paired results: {error}\n")
    print(f"Saved {len(summary['rows'])} layer/seed/benchmark rows; "
          f"{summary['unique_sample_count']} unique examples, "
          f"{summary['paired_observation_count']} paired observations to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
