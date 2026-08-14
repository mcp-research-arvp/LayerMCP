"""Recover a summary-only failed multi-step run from complete saved workflows."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any

from analysis.multi_step_run import (
    dataset_counts,
    validate_and_index_multistep,
    validate_complete_multistep_run,
)
from analysis.run_artifacts import _load_json_object, _load_jsonl_objects
from evaluation.evaluate import (
    MULTISTEP_EVALUATION_PROTOCOL,
    PREDICTED_ROLLOUT_EXECUTION_MODE,
    _build_multistep_metrics,
)


RECOVERY_METHOD = "summary_only_from_complete_saved_workflows_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _single_dataset_directory(source_run: Path) -> Path:
    candidates = sorted(source_run.glob("domains/*/*"))
    candidates = [path for path in candidates if path.is_dir()]
    if len(candidates) != 1:
        raise ValueError(
            f"Expected exactly one source dataset directory, found {len(candidates)}"
        )
    return candidates[0]


def _uniform(records: list[dict[str, Any]], field: str) -> Any:
    values = {json.dumps(record.get(field), sort_keys=True) for record in records}
    if len(values) != 1:
        raise ValueError(f"Saved workflows have inconsistent {field}")
    return records[0].get(field)


def _git_head(repository: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def recover_multistep_run(
    *, source_run: Path, output_run: Path, benchmark: Path, repository: Path
) -> Path:
    source_run = source_run.resolve()
    output_run = output_run.resolve()
    benchmark = benchmark.resolve()
    repository = repository.resolve()
    if output_run.exists():
        raise FileExistsError(f"Recovery destination already exists: {output_run}")
    if (source_run / "RUN_COMPLETE").exists():
        raise ValueError("Source run is already complete and does not need recovery")
    if (source_run / "artifact_index.jsonl").exists():
        raise ValueError("Source run already has an artifact index")

    source_dataset = _single_dataset_directory(source_run)
    source_samples = source_dataset / "samples.jsonl"
    source_log = source_dataset / "evaluation.log"
    for path in (source_run / "run_metadata.json", source_samples, source_log):
        if not path.is_file():
            raise FileNotFoundError(f"Required recovery source is missing: {path}")

    metadata = _load_json_object(source_run / "run_metadata.json")
    records = _load_jsonl_objects(source_samples)
    expected_workflows, expected_steps = dataset_counts(benchmark)
    if len(records) != expected_workflows:
        raise ValueError("Saved workflow count does not match the benchmark")
    observed_ids: dict[str, list[str]] = {}
    for record in records:
        workflow_id = str(record.get("sample_id"))
        if workflow_id in observed_ids:
            raise ValueError(f"Duplicate saved workflow: {workflow_id}")
        steps = record.get("steps")
        if not isinstance(steps, list) or not steps:
            raise ValueError(f"Saved workflow has no steps: {workflow_id}")
        step_ids = [str(step.get("step_id")) for step in steps]
        if len(step_ids) != len(set(step_ids)):
            raise ValueError(f"Duplicate saved step in workflow: {workflow_id}")
        observed_ids[workflow_id] = step_ids
    expected_rows = json.loads(benchmark.read_text(encoding="utf-8"))
    expected_ids = {
        str(row["id"]): [str(step["id"]) for step in row["expected_steps"]]
        for row in expected_rows
    }
    if observed_ids != expected_ids or sum(map(len, observed_ids.values())) != expected_steps:
        raise ValueError("Saved workflow and step membership is incomplete")

    required_record_fields = (
        "model_name", "prompt_template", "evaluation_protocol", "reasoning_mode",
        "reasoning_method", "effective_generation_limit",
        "effective_generation_limit_unit", "tool_pool", "tool_count",
        "tool_registry_fingerprint", "tool_registry_fingerprint_version",
        "benchmark_mode", "workflow_execution_mode",
    )
    values = {field: _uniform(records, field) for field in required_record_fields}
    if values["evaluation_protocol"] != MULTISTEP_EVALUATION_PROTOCOL:
        raise ValueError("Saved workflows use an unsupported evaluation protocol")
    if values["workflow_execution_mode"] != PREDICTED_ROLLOUT_EXECUTION_MODE:
        raise ValueError("Saved workflows are not predicted-sequence rollout records")
    if values["model_name"] != metadata.get("expected_model_name"):
        raise ValueError("Saved model does not match source run metadata")
    for field, metadata_field in (
        ("prompt_template", "prompt_template_id"),
        ("reasoning_mode", "reasoning_mode"),
        ("reasoning_method", "reasoning_method"),
        ("effective_generation_limit", "effective_generation_limit"),
        ("effective_generation_limit_unit", "effective_generation_limit_unit"),
        ("tool_pool", "tool_pool"),
        ("tool_count", "tool_count"),
        ("tool_registry_fingerprint", "tool_registry_fingerprint"),
        ("tool_registry_fingerprint_version", "tool_registry_fingerprint_version"),
    ):
        if values[field] != metadata.get(metadata_field):
            raise ValueError(f"Saved {field} does not match source run metadata")

    output_dataset = output_run / "domains" / source_dataset.parent.name / source_dataset.name
    output_dataset.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(source_samples, output_dataset / "samples.jsonl")
    shutil.copyfile(source_log, output_dataset / "source_evaluation.log")
    (output_dataset / "evaluation.log").write_text(
        "Reporting-only recovery from complete saved workflow records.\n"
        f"Source run: {source_run}\nRecovery method: {RECOVERY_METHOD}\n",
        encoding="utf-8",
    )

    step_records = [step for record in records for step in record["steps"]]
    metrics = _build_multistep_metrics(records, step_records)
    first = records[0]
    summary = {
        "timestamp": source_run.name.split("_", 1)[0],
        "benchmark_path": str(benchmark),
        "model_name": values["model_name"],
        "router_id": first.get("router_id"),
        "router_backend": first.get("router_backend"),
        "architecture_source": "preserved_saved_workflow_records",
        "weight_source": "preserved_saved_workflow_records",
        "prompt_template": values["prompt_template"],
        "reasoning_mode": values["reasoning_mode"],
        "reasoning_method": values["reasoning_method"],
        "effective_generation_limit": values["effective_generation_limit"],
        "effective_generation_limit_unit": values["effective_generation_limit_unit"],
        "evaluation_protocol": values["evaluation_protocol"],
        "evaluation_protocol_description": first.get("evaluation_protocol_description"),
        "workflow_execution_modes": [values["workflow_execution_mode"]],
        "declared_workflow_execution_modes": sorted(
            {str(record.get("declared_workflow_execution_mode")) for record in records}
        ),
        "tool_pool": values["tool_pool"],
        "tool_count": values["tool_count"],
        "tool_names": first.get("tool_names", []),
        "tool_registry_fingerprint": values["tool_registry_fingerprint"],
        "tool_registry_fingerprint_version": values["tool_registry_fingerprint_version"],
        **metrics,
        "average_step_latency_seconds": (
            sum(float(step.get("latency_seconds", 0.0)) for step in step_records)
            / len(step_records)
        ),
        "executed_tool_calls": sum(step.get("called_tool") is not None for step in step_records),
        "errors_count": sum(
            step.get("called_tool") is not None and not step.get("execution_success", False)
            for step in step_records
        ),
        "recovered_from_complete_saved_inference": True,
        "recovery_method": RECOVERY_METHOD,
        "source_run_path": str(source_run),
        "source_job_id": str(metadata.get("slurm_job_id")),
    }
    (output_dataset / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    recovery_commit = _git_head(repository)
    source_hashes = {
        str(path.relative_to(source_run)): _sha256(path)
        for path in sorted(source_run.rglob("*"))
        if path.is_file()
    }
    manifest = {
        "recovered_from_complete_saved_inference": True,
        "recovery_method": RECOVERY_METHOD,
        "recovery_commit": recovery_commit,
        "source_run_path": str(source_run),
        "source_job_id": str(metadata.get("slurm_job_id")),
        "source_git_commit": metadata.get("git_commit"),
        "source_artifact_sha256": source_hashes,
        "recovered_samples_sha256": _sha256(output_dataset / "samples.jsonl"),
        "benchmark_path": str(benchmark),
        "benchmark_sha256": _sha256(benchmark),
    }
    (output_run / "recovery_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    recovered_metadata = {
        **metadata,
        "benchmark_paths": [str(benchmark)],
        "source_counts": {
            str(benchmark): {"workflows": expected_workflows, "routed_steps": expected_steps}
        },
        "benchmark_mode_distributions": {
            str(benchmark): {str(values["benchmark_mode"]): expected_workflows}
        },
        "recovered_from_complete_saved_inference": True,
        "recovery_method": RECOVERY_METHOD,
        "recovery_commit": recovery_commit,
        "source_run_path": str(source_run),
        "source_job_id": str(metadata.get("slurm_job_id")),
        "source_artifact_sha256": source_hashes,
    }
    (output_run / "run_metadata.json").write_text(
        json.dumps(recovered_metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    validate_and_index_multistep(
        dataset_directory=output_dataset,
        index_path=output_run / "artifact_index.jsonl",
        source_benchmark=benchmark,
        evaluated_benchmark=benchmark,
        expected_model=values["model_name"],
        expected_prompt_template=values["prompt_template"],
        expected_registry_fingerprint=values["tool_registry_fingerprint"],
        expected_registry_fingerprint_version=values["tool_registry_fingerprint_version"],
        expected_tool_count=values["tool_count"],
        expected_tool_pool=values["tool_pool"],
        expected_reasoning_mode=values["reasoning_mode"],
        expected_reasoning_method=values["reasoning_method"],
        expected_generation_limit=values["effective_generation_limit"],
    )
    validate_complete_multistep_run(
        output_run / "artifact_index.jsonl",
        [benchmark],
        output_run / "run_metadata.json",
    )
    (output_run / "RUN_COMPLETE").write_text(
        "recovered from complete saved inference\n", encoding="utf-8"
    )
    return output_run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output-run", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    args = parser.parse_args()
    recovered = recover_multistep_run(
        source_run=args.source_run,
        output_run=args.output_run,
        benchmark=args.benchmark,
        repository=args.repository,
    )
    print(recovered)


if __name__ == "__main__":
    main()
