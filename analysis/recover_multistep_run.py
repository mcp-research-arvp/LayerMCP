"""Recover a summary-only failed multi-step run from complete saved workflows."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
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


def _paths_overlap(first: Path, second: Path) -> bool:
    """Return whether either resolved path contains the other."""
    return first == second or first in second.parents or second in first.parents


def _benchmark_relative_path(path: Path) -> Path | None:
    """Return the stable benchmark-relative suffix of a benchmark path."""
    parts = path.parts
    try:
        index = parts.index("benchmark")
    except ValueError:
        return None
    return Path(*parts[index:])


def _benchmark_reference_matches(recorded: Any, benchmark: Path) -> bool:
    """Compare a recorded benchmark path across repository worktrees safely."""
    if not isinstance(recorded, str) or not recorded:
        return False
    recorded_path = Path(recorded).expanduser().resolve()
    if recorded_path == benchmark:
        return True
    if (
        _benchmark_relative_path(recorded_path)
        != _benchmark_relative_path(benchmark)
        or not recorded_path.is_file()
    ):
        return False
    return _sha256(recorded_path) == _sha256(benchmark)


def _metadata_entry_for_benchmark(
    value: Any, *, field: str, benchmark: Path
) -> Any:
    """Return the one metadata value belonging to the supplied benchmark."""
    if not isinstance(value, dict) or len(value) != 1:
        raise ValueError(f"Source run metadata {field} must contain exactly one benchmark")
    recorded_path, recorded_value = next(iter(value.items()))
    if not _benchmark_reference_matches(recorded_path, benchmark):
        raise ValueError(
            f"Source run metadata {field} does not match supplied benchmark"
        )
    return recorded_value


def _validate_source_benchmark_metadata(
    metadata: dict[str, Any], *, benchmark: Path, workflow_count: int,
    step_count: int, benchmark_modes: dict[str, int],
) -> None:
    """Validate launcher-recorded benchmark provenance before recovery."""
    benchmark_paths = metadata.get("benchmark_paths")
    if not isinstance(benchmark_paths, list) or len(benchmark_paths) != 1:
        raise ValueError(
            "Source run metadata benchmark_paths must contain exactly one benchmark"
        )
    if not _benchmark_reference_matches(benchmark_paths[0], benchmark):
        raise ValueError(
            "Source run metadata benchmark_paths does not match supplied benchmark"
        )

    source_counts = _metadata_entry_for_benchmark(
        metadata.get("source_counts"), field="source_counts", benchmark=benchmark
    )
    expected_counts = {"workflows": workflow_count, "routed_steps": step_count}
    if source_counts != expected_counts:
        raise ValueError(
            f"Source run metadata source_counts mismatch: "
            f"{source_counts!r} != {expected_counts!r}"
        )

    recorded_modes = _metadata_entry_for_benchmark(
        metadata.get("benchmark_mode_distributions"),
        field="benchmark_mode_distributions",
        benchmark=benchmark,
    )
    if recorded_modes != benchmark_modes:
        raise ValueError(
            "Source run metadata benchmark_mode_distributions mismatch: "
            f"{recorded_modes!r} != {benchmark_modes!r}"
        )


def recover_multistep_run(
    *, source_run: Path, output_run: Path, benchmark: Path, repository: Path
) -> Path:
    source_run = source_run.resolve(strict=True)
    output_run = output_run.resolve()
    benchmark = benchmark.resolve(strict=True)
    repository = repository.resolve(strict=True)
    if _paths_overlap(source_run, output_run):
        raise ValueError(
            "Recovery source and destination trees must not overlap: "
            f"{source_run} and {output_run}"
        )
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
        if not _benchmark_reference_matches(record.get("benchmark_path"), benchmark):
            raise ValueError(
                f"Saved workflow benchmark does not match supplied benchmark: "
                f"{workflow_id}"
            )
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

    step_records = [step for record in records for step in record["steps"]]
    benchmark_modes = dict(sorted(Counter(
        str(record["benchmark_mode"]) for record in records
    ).items()))
    _validate_source_benchmark_metadata(
        metadata,
        benchmark=benchmark,
        workflow_count=expected_workflows,
        step_count=expected_steps,
        benchmark_modes=benchmark_modes,
    )

    # Complete every source-only operation before exposing recovery artifacts.
    source_hashes = {
        str(path.relative_to(source_run)): _sha256(path)
        for path in sorted(source_run.rglob("*"))
        if path.is_file()
    }
    benchmark_hash = _sha256(benchmark)
    recovery_commit = _git_head(repository)

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
    recovered_metadata = {
        **metadata,
        "source_run_metadata": metadata,
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

    output_run.parent.mkdir(parents=True, exist_ok=True)
    temporary_run = Path(tempfile.mkdtemp(
        prefix=f".{output_run.name}.recovery-", dir=output_run.parent
    ))
    try:
        output_dataset = (
            temporary_run / "domains" / source_dataset.parent.name / source_dataset.name
        )
        output_dataset.mkdir(parents=True, exist_ok=False)
        shutil.copyfile(source_samples, output_dataset / "samples.jsonl")
        shutil.copyfile(source_log, output_dataset / "source_evaluation.log")
        (output_dataset / "evaluation.log").write_text(
            "Reporting-only recovery from complete saved workflow records.\n"
            f"Source run: {source_run}\nRecovery method: {RECOVERY_METHOD}\n",
            encoding="utf-8",
        )
        (output_dataset / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

        manifest = {
            "recovered_from_complete_saved_inference": True,
            "recovery_method": RECOVERY_METHOD,
            "recovery_commit": recovery_commit,
            "source_run_path": str(source_run),
            "source_job_id": str(metadata.get("slurm_job_id")),
            "source_git_commit": metadata.get("git_commit"),
            "source_run_metadata": metadata,
            "source_artifact_sha256": source_hashes,
            "recovered_samples_sha256": _sha256(output_dataset / "samples.jsonl"),
            "benchmark_path": str(benchmark),
            "benchmark_sha256": benchmark_hash,
        }
        (temporary_run / "recovery_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (temporary_run / "run_metadata.json").write_text(
            json.dumps(recovered_metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        validate_and_index_multistep(
            dataset_directory=output_dataset,
            index_path=temporary_run / "artifact_index.jsonl",
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
            temporary_run / "artifact_index.jsonl",
            [benchmark],
            temporary_run / "run_metadata.json",
        )
        (temporary_run / "RUN_COMPLETE").write_text(
            "recovered from complete saved inference\n", encoding="utf-8"
        )
        if output_run.exists():
            raise FileExistsError(
                f"Recovery destination already exists: {output_run}"
            )
        os.rename(temporary_run, output_run)
    except BaseException:
        if temporary_run.exists():
            shutil.rmtree(temporary_run)
        raise
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
