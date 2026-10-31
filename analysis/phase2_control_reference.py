"""Inspect or register immutable Phase 2 canonical controls from completed runs.

Registration writes a new directory and never changes the source artifact.
Only current hardened paired artifacts are accepted; older baselines are
reported as unverified because their missing identity fields cannot be inferred.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from research.phase2.control_reference import inspect_candidate, write_reference


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def register(source: Path, destination: Path, workflow: str) -> Path:
    source = source.expanduser().resolve(strict=True)
    if not (source / "RUN_COMPLETE").is_file():
        raise ValueError(f"{source}: missing RUN_COMPLETE")
    config = _read(source / "run_config.json")
    manifest = _read(source / "run_manifest.json")
    status = _read(source / "run_status.json")
    if status.get("status") != "completed":
        raise ValueError(f"{source}: run is not completed")
    if workflow == "single":
        if config.get("kind") != "phase2_gpt_oss_attention_intervention_eval_v3" or manifest.get("manifest_kind") != "phase2_paired_run_manifest_v1":
            raise ValueError("single-step source is not a current hardened paired artifact; provenance is unverified")
        selected = config.get("screen_plan", {}).get("sample_ids")
        input_keys = ("source_run_directory", "source_run_metadata", "selected_sample_ids") if config.get("input_kind") == "saved_source_run" else (
            "benchmark_paths", "benchmark_sha256", "selected_sample_manifest", "selected_sample_ids")
        inputs = {key: config.get(key) for key in input_keys}
        inputs["selected_sample_ids"] = selected
        compatibility = {
            "workflow_type": "single_step", "experiment_kind": config["kind"],
            "checkpoint_fingerprint": config.get("checkpoint_fingerprint"), "model_binding": config.get("model_binding"),
            "runtime": config.get("runtime"), "inputs": inputs,
            "generation": {"reasoning_mode": "reasoning", "reasoning_effort": "low"},
            "evaluation": {"call_predicted_tools": True, "condition_isolation": config.get("condition_isolation")},
            "tool_registry": manifest.get("tool_registry"),
        }
        rows = [json.loads(line) for line in (source / "paired_records.jsonl").read_text().splitlines() if line.strip()]
        controls = []
        for row in rows:
            if row.get("sample_id") not in selected or "control" not in row:
                raise ValueError("source has incomplete control records")
            controls.append({"sample_id": row["sample_id"], "benchmark_path": row.get("benchmark_path"),
                             "control": row["control"], "control_verification": row.get("control_verification"),
                             "registry_exact_match": row.get("registry_exact_match"),
                             "saved_control_comparison": row.get("saved_control_comparison")})
        unique = {row["sample_id"]: row for row in controls}
        if list(unique) != selected or len(unique) != len(controls):
            raise ValueError("source repeats or misses controls; register one complete canonical control shard")
        controls = [unique[item] for item in selected]
        workflow_type = "single_step"
    elif workflow == "multi":
        if config.get("kind") != "phase2_gpt_oss_multistep_attention_intervention_v2" or manifest.get("manifest_kind") != "phase2_multistep_paired_run_manifest_v1":
            raise ValueError("multi-step source is not a current hardened paired artifact; provenance is unverified")
        selected = config.get("selected_workflow_ids")
        compatibility = {
            "workflow_type": "multi_step", "experiment_kind": config["kind"],
            "checkpoint_fingerprint": config.get("checkpoint_identity", {}).get("fingerprint"),
            "dataset_group": config.get("dataset_group"), "benchmark_sha256": config.get("benchmark_sha256"),
            "selected_workflow_ids": selected, "evaluation": manifest.get("evaluation"),
            "runtime": config.get("runtime"), "tool_registry": manifest.get("tool_registry"),
            "generation": manifest.get("generation"),
        }
        controls = [{"sample_id": row["sample_id"], "control": row}
                    for row in [json.loads(line) for line in (source / "control" / "samples.jsonl").read_text().splitlines() if line.strip()]]
        if [row["sample_id"] for row in controls] != selected:
            raise ValueError("source multi-step control coverage differs from its selected workflows")
        workflow_type = "multi_step"
    else:
        raise ValueError("workflow must be single or multi")
    write_reference(destination.expanduser().resolve(), workflow_type=workflow_type, compatibility=compatibility,
                    controls=controls, metadata={"registered_from": str(source), "run_id": config.get("run_id"),
                                                   "started_at": status.get("started_at"), "completed_at": status.get("completed_at"),
                                                   "source_run_manifest": manifest})
    return destination.expanduser().resolve()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    inspect = sub.add_parser("inspect")
    inspect.add_argument("--candidate", required=True, type=Path)
    register_parser = sub.add_parser("register")
    register_parser.add_argument("--workflow", choices=("single", "multi"), required=True)
    register_parser.add_argument("--source-run", required=True, type=Path)
    register_parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "inspect":
        print(json.dumps(inspect_candidate(args.candidate), indent=2, sort_keys=True))
    else:
        print(register(args.source_run, args.output_dir, args.workflow))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
