"""Rescore complete multi-step runs into immutable, versioned derived artifacts."""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
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
    MODEL_FINAL_RESPONSE_CONTRACT,
    MULTISTEP_EVALUATION_PROTOCOL,
    PREDICTED_ROLLOUT_EXECUTION_MODE,
    WORKFLOW_FINAL_SCORING_VERSION,
    _build_multistep_metrics,
    _score_workflow_final_answer,
    _score_workflow_final_program_execution,
    _score_workflow_final_tool_result,
)


RESCORE_METHOD = "saved_multistep_workflow_scoring_v1"
_SCORING_PREFIXES = (
    "workflow_final_answer_",
    "workflow_final_program_",
    "workflow_final_tool_result_",
)
_SCORING_TARGET_FIELDS = {
    "expected_final_program_result",
    "expected_final_tool_result",
}
_BENCHMARK_SCORING_FIELDS = {
    "workflow_final_answer_contract",
    "workflow_final_answer_expected",
    "workflow_final_program_contract",
    "expected_final_program_result",
    "workflow_final_tool_result_contract",
    "expected_final_tool_result",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _single_dataset_directory(run: Path) -> Path:
    candidates = [path for path in sorted(run.glob("domains/*/*")) if path.is_dir()]
    if len(candidates) != 1:
        raise ValueError(f"Expected exactly one dataset directory, found {len(candidates)}")
    return candidates[0]


def _benchmark_relative_path(value: str) -> Path:
    parts = Path(value).parts
    try:
        index = parts.index("benchmark")
    except ValueError as exc:
        raise ValueError("Saved benchmark path has no benchmark-relative identity") from exc
    return Path(*parts[index:])


def _git_blob(repository: Path, commit: str, relative: Path) -> bytes:
    result = subprocess.run(
        ["git", "show", f"{commit}:{relative.as_posix()}"],
        cwd=repository,
        check=True,
        capture_output=True,
    )
    return result.stdout


def _uniform(records: list[dict[str, Any]], field: str) -> Any:
    values = {_canonical(record.get(field)) for record in records}
    if len(values) != 1:
        raise ValueError(f"Saved workflows have inconsistent {field}")
    return records[0].get(field)


def _without_scoring(record: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in record.items()
        if key != "workflow_final_scoring_version"
        and key not in _SCORING_TARGET_FIELDS
        and not any(key.startswith(prefix) for prefix in _SCORING_PREFIXES)
    }


def _without_benchmark_scoring(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value for key, value in row.items()
        if key not in _BENCHMARK_SCORING_FIELDS
    }


def _score_record(
    original: dict[str, Any], benchmark_row: dict[str, Any]
) -> dict[str, Any]:
    record = deepcopy(original)
    for key in list(record):
        if key == "workflow_final_scoring_version" or key in _SCORING_TARGET_FIELDS or any(
            key.startswith(prefix) for prefix in _SCORING_PREFIXES
        ):
            del record[key]

    final_value = record.get("final_step_result_value")
    executed = bool(record.get("steps")) and all(
        step.get("execution_success") is not None for step in record["steps"]
    )
    answer = _score_workflow_final_answer(
        expected_final_answer=benchmark_row.get("expected_final_answer")
    )
    program = _score_workflow_final_program_execution(
        expected_final_program_result=benchmark_row.get(
            "expected_final_program_result"
        ),
        workflow_final_program_contract=benchmark_row.get(
            "workflow_final_program_contract"
        ),
        final_tool_result_value=final_value,
        call_predicted_tools=executed,
    )
    tool_result = _score_workflow_final_tool_result(
        expected_final_tool_result=benchmark_row.get("expected_final_tool_result"),
        workflow_final_tool_result_contract=benchmark_row.get(
            "workflow_final_tool_result_contract"
        ),
        final_tool_result_value=final_value,
        call_predicted_tools=executed,
    )
    record.update(
        {
            "workflow_final_answer_contract": MODEL_FINAL_RESPONSE_CONTRACT,
            "workflow_final_answer_correct": answer.correct,
            "workflow_final_answer_status": answer.status,
            "workflow_final_answer_matcher": answer.matcher,
            "workflow_final_answer_diagnostic": answer.diagnostic,
            "expected_final_program_result": benchmark_row.get(
                "expected_final_program_result"
            ),
            "workflow_final_program_contract": benchmark_row.get(
                "workflow_final_program_contract"
            ),
            "workflow_final_program_execution_correct": program.correct,
            "workflow_final_program_execution_status": program.status,
            "workflow_final_program_execution_matcher": program.matcher,
            "workflow_final_program_execution_diagnostic": program.diagnostic,
            "expected_final_tool_result": benchmark_row.get(
                "expected_final_tool_result"
            ),
            "workflow_final_tool_result_contract": benchmark_row.get(
                "workflow_final_tool_result_contract"
            ),
            "workflow_final_tool_result_correct": tool_result.correct,
            "workflow_final_tool_result_status": tool_result.status,
            "workflow_final_tool_result_matcher": tool_result.matcher,
            "workflow_final_tool_result_diagnostic": tool_result.diagnostic,
            "workflow_final_scoring_version": WORKFLOW_FINAL_SCORING_VERSION,
        }
    )
    return record


def rescore_multistep_run(
    *, source_run: Path, output_run: Path, benchmark: Path, repository: Path
) -> Path:
    source_run = source_run.resolve(strict=True)
    output_run = output_run.resolve()
    benchmark = benchmark.resolve(strict=True)
    repository = repository.resolve(strict=True)
    if output_run.exists():
        raise FileExistsError(f"Rescore destination already exists: {output_run}")
    if source_run == output_run or source_run in output_run.parents or output_run in source_run.parents:
        raise ValueError("Source and destination trees must not overlap")
    for required in ("RUN_COMPLETE", "artifact_index.jsonl", "run_metadata.json"):
        if not (source_run / required).is_file():
            raise FileNotFoundError(f"Complete source run is missing {required}")

    source_dataset = _single_dataset_directory(source_run)
    source_samples = source_dataset / "samples.jsonl"
    source_summary = source_dataset / "summary.json"
    source_log = source_dataset / "evaluation.log"
    for required in (source_samples, source_summary, source_log):
        if not required.is_file():
            raise FileNotFoundError(f"Complete source run is missing {required}")

    metadata = _load_json_object(source_run / "run_metadata.json")
    records = _load_jsonl_objects(source_samples)
    benchmark_rows = json.loads(benchmark.read_text(encoding="utf-8"))
    if not isinstance(benchmark_rows, list) or not benchmark_rows:
        raise ValueError("Scoring benchmark must be a non-empty JSON list")
    benchmark_by_id = {str(row["id"]): row for row in benchmark_rows}
    workflow_count, step_count = dataset_counts(benchmark)
    if len(records) != workflow_count or len(benchmark_by_id) != workflow_count:
        raise ValueError("Saved workflow count does not match scoring benchmark")

    recorded_benchmark = _uniform(records, "benchmark_path")
    if not isinstance(recorded_benchmark, str) or not recorded_benchmark:
        raise ValueError("Saved workflows are missing benchmark identity")
    relative = _benchmark_relative_path(recorded_benchmark)
    if relative != _benchmark_relative_path(str(benchmark)):
        raise ValueError("Saved and scoring benchmarks have different identities")
    source_commit = str(metadata.get("git_commit") or "")
    if not source_commit:
        raise ValueError("Source metadata is missing its Git commit")
    source_benchmark_bytes = _git_blob(repository, source_commit, relative)
    source_benchmark_rows = json.loads(source_benchmark_bytes)
    if not isinstance(source_benchmark_rows, list):
        raise ValueError("Historical benchmark must be a JSON list")
    if _canonical([
        _without_benchmark_scoring(row) for row in source_benchmark_rows
    ]) != _canonical([
        _without_benchmark_scoring(row) for row in benchmark_rows
    ]):
        raise ValueError(
            "Historical and scoring benchmarks differ outside versioned "
            "workflow-scoring metadata"
        )
    source_ids = {
        str(row["id"]): [str(step["id"]) for step in row["expected_steps"]]
        for row in source_benchmark_rows
    }
    scoring_ids = {
        str(row["id"]): [str(step["id"]) for step in row["expected_steps"]]
        for row in benchmark_rows
    }
    observed_ids = {
        str(row.get("sample_id")): [str(step.get("step_id")) for step in row.get("steps", [])]
        for row in records
    }
    if observed_ids != source_ids or observed_ids != scoring_ids:
        raise ValueError("Saved workflow membership is incomplete or changed")
    for record in records:
        benchmark_row = benchmark_by_id[str(record["sample_id"])]
        if record.get("expected_final_answer") != benchmark_row.get(
            "expected_final_answer"
        ):
            raise ValueError("User-facing final-answer gold changed since inference")
    if sum(map(len, observed_ids.values())) != step_count:
        raise ValueError("Saved step count does not match scoring benchmark")

    required_uniform = (
        "model_name", "prompt_template", "reasoning_mode", "reasoning_method",
        "effective_generation_limit", "effective_generation_limit_unit",
        "tool_pool", "tool_count", "tool_registry_fingerprint",
        "tool_registry_fingerprint_version", "evaluation_protocol",
        "workflow_execution_mode",
    )
    values = {field: _uniform(records, field) for field in required_uniform}
    if values["model_name"] != metadata.get("expected_model_name"):
        raise ValueError("Saved model does not match run metadata")
    metadata_pairs = {
        "prompt_template": "prompt_template_id",
        "reasoning_mode": "reasoning_mode",
        "reasoning_method": "reasoning_method",
        "effective_generation_limit": "effective_generation_limit",
        "effective_generation_limit_unit": "effective_generation_limit_unit",
        "tool_pool": "tool_pool",
        "tool_count": "tool_count",
        "tool_registry_fingerprint": "tool_registry_fingerprint",
        "tool_registry_fingerprint_version": "tool_registry_fingerprint_version",
    }
    for record_field, metadata_field in metadata_pairs.items():
        if values[record_field] != metadata.get(metadata_field):
            raise ValueError(f"Saved {record_field} does not match run metadata")
    if values["evaluation_protocol"] != MULTISTEP_EVALUATION_PROTOCOL:
        raise ValueError("Unsupported source evaluation protocol")
    if values["workflow_execution_mode"] != PREDICTED_ROLLOUT_EXECUTION_MODE:
        raise ValueError("Source is not a predicted-sequence workflow run")

    rescored = [
        _score_record(row, benchmark_by_id[str(row["sample_id"])]) for row in records
    ]
    for before, after in zip(records, rescored, strict=True):
        if _canonical(_without_scoring(before)) != _canonical(_without_scoring(after)):
            raise AssertionError("Rescoring changed a non-scoring workflow field")

    source_hashes = {
        str(path.relative_to(source_run)): _sha256(path)
        for path in sorted(source_run.rglob("*")) if path.is_file()
    }
    step_records = [step for record in rescored for step in record["steps"]]
    metrics = _build_multistep_metrics(rescored, step_records)
    old_summary = _load_json_object(source_summary)
    summary = {
        **old_summary,
        **metrics,
        "benchmark_path": recorded_benchmark,
        "workflow_final_scoring_version": WORKFLOW_FINAL_SCORING_VERSION,
        "rescored_from_complete_saved_run": True,
        "rescore_method": RESCORE_METHOD,
        "source_run_path": str(source_run),
        "source_run_identity": source_run.name,
        "source_git_commit": source_commit,
        "source_benchmark_sha256": _sha256_bytes(source_benchmark_bytes),
        "scoring_benchmark_sha256": _sha256(benchmark),
    }
    rescored_metadata = {
        **metadata,
        "benchmark_paths": [str(benchmark)],
        "source_counts": {
            str(benchmark): {
                "workflows": workflow_count,
                "routed_steps": step_count,
            }
        },
        "workflow_final_scoring_version": WORKFLOW_FINAL_SCORING_VERSION,
        "workflow_level_metrics": [
            "workflow_final_answer_accuracy",
            "workflow_final_program_execution_accuracy",
            "workflow_final_tool_result_accuracy",
        ],
        "workflow_metric_summaries": {
            str(benchmark): {
                key: value
                for key, value in summary.items()
                if key == "workflow_final_scoring_version"
                or key.startswith("workflow_final_answer_")
                or key.startswith("workflow_final_program_execution_")
                or key.startswith("workflow_final_tool_result_")
            }
        },
        "rescored_from_complete_saved_run": True,
        "rescore_method": RESCORE_METHOD,
        "rescore_git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repository, text=True
        ).strip(),
        "source_run_path": str(source_run),
        "source_run_identity": source_run.name,
        "source_artifact_sha256": source_hashes,
        "source_benchmark_sha256": _sha256_bytes(source_benchmark_bytes),
        "scoring_benchmark_path": str(benchmark),
        "scoring_benchmark_sha256": _sha256(benchmark),
    }

    output_run.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_run.name}.rescore-", dir=output_run.parent))
    try:
        output_dataset = temporary / "domains" / source_dataset.parent.name / source_dataset.name
        output_dataset.mkdir(parents=True)
        with (output_dataset / "samples.jsonl").open("x", encoding="utf-8") as handle:
            for row in rescored:
                handle.write(json.dumps(row, ensure_ascii=True) + "\n")
        (output_dataset / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        shutil.copyfile(source_log, output_dataset / "source_evaluation.log")
        (output_dataset / "evaluation.log").write_text(
            f"Scoring-only rescore of {source_run}\nMethod: {RESCORE_METHOD}\n",
            encoding="utf-8",
        )
        manifest = {
            "rescore_method": RESCORE_METHOD,
            "workflow_final_scoring_version": WORKFLOW_FINAL_SCORING_VERSION,
            "source_run_path": str(source_run),
            "source_run_identity": source_run.name,
            "source_artifact_sha256": source_hashes,
            "source_samples_sha256": _sha256(source_samples),
            "rescored_samples_sha256": _sha256(output_dataset / "samples.jsonl"),
            "source_benchmark_path": recorded_benchmark,
            "source_benchmark_sha256": _sha256_bytes(source_benchmark_bytes),
            "scoring_benchmark_path": str(benchmark),
            "scoring_benchmark_sha256": _sha256(benchmark),
            "preserved_non_scoring_records_sha256": _sha256_bytes(
                "\n".join(_canonical(_without_scoring(row)) for row in records).encode()
            ),
        }
        (temporary / "rescore_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (temporary / "run_metadata.json").write_text(
            json.dumps(rescored_metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        validate_and_index_multistep(
            dataset_directory=output_dataset,
            index_path=temporary / "artifact_index.jsonl",
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
            temporary / "artifact_index.jsonl", [benchmark], temporary / "run_metadata.json"
        )
        (temporary / "RUN_COMPLETE").write_text("rescored and validated\n", encoding="utf-8")
        if output_run.exists():
            raise FileExistsError(f"Rescore destination already exists: {output_run}")
        os.rename(temporary, output_run)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return output_run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output-run", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    args = parser.parse_args()
    print(rescore_multistep_run(
        source_run=args.source_run,
        output_run=args.output_run,
        benchmark=args.benchmark,
        repository=args.repository,
    ))


if __name__ == "__main__":
    main()
