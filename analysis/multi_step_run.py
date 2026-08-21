"""Prepare and validate collision-safe multi-step evaluation runs."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

from analysis.run_artifacts import (
    _canonical_benchmark,
    _existing_index_records,
    _index_path_value,
    _load_json_object,
    _load_jsonl_objects,
    _resolve_index_path,
    validate_run_metadata,
)
from evaluation.evaluate import (
    DEFAULT_BENCHMARK_MODE,
    DEFAULT_WORKFLOW_EXECUTION_MODE,
    MULTISTEP_EVALUATION_PROTOCOL,
    OUTCOME_METRIC_NAMES,
    PREDICTED_ROLLOUT_EXECUTION_MODE,
    _multistep_query,
    load_benchmark,
)


DATASET_GROUPS = {
    "coding_sweagent": Path("benchmark/coding/coding_sweagent_multistep.json"),
    "coding_nebius_replay": Path(
        "benchmark/coding/coding_nebius_sweagent_replay_multistep.json"
    ),
    "enterprise_tau2": Path("benchmark/enterprise/enterprise_public_workflows.json"),
    "finance_convfinqa": Path("benchmark/finance/finance_convfinqa_multistep.json"),
    "finance_finqa": Path("benchmark/finance/finance_finqa_test_multistep.json"),
    "finance_finretrieval_replay": Path(
        "benchmark/finance/finance_finretrieval_replay_multistep.json"
    ),
    "math_controlled": Path("benchmark/math/math_multistep_controlled.json"),
}
EMPTY_PLACEHOLDER = Path(
    "benchmark/coding/coding_nebius_swerebench_openhands_replay_multistep.json"
)

FINAL_METRIC_SUFFIXES = (
    "accuracy",
    "gold",
    "scored",
    "correct",
    "mismatch",
    "extraction_error",
    "unavailable",
    "status_counts",
    "contracts",
    "matchers",
)
CORE_SUMMARY_METRIC_FIELDS = (
    "tool_selection_accuracy",
    "argument_accuracy",
    "step_outcome_accuracy",
    "step_outcome_scored",
    "step_outcome_status_counts",
    "step_outcome_matchers",
    "all_tools_correct_accuracy",
    "all_arguments_correct_accuracy",
    "all_steps_correct_accuracy",
    "all_steps_correct_scored",
)
FINAL_STEP_METRIC_FIELDS = tuple(
    f"final_step_outcome_{suffix}" for suffix in FINAL_METRIC_SUFFIXES
)
FINAL_PROGRAM_METRIC_FIELDS = tuple(
    f"final_program_execution_{suffix}" for suffix in FINAL_METRIC_SUFFIXES
)
WORKFLOW_OUTCOME_FIELDS = (
    "all_tools_correct",
    "all_arguments_correct",
    "all_steps_correct",
    "expected_final_step_outcome",
    "final_step_outcome_contract",
    "final_step_outcome_correct",
    "final_step_outcome_status",
    "final_step_outcome_matcher",
)
STEP_OUTCOME_FIELDS = (
    "tool_selection_correct",
    "argument_match_correct",
    "final_outcome_correct",
    "final_outcome_status",
    "final_outcome_matcher",
)


def _require_fields(value: dict[str, Any], fields: tuple[str, ...], label: str) -> None:
    missing = [field for field in fields if field not in value]
    if missing:
        raise ValueError(f"{label} is missing required fields: {', '.join(missing)}")


def _require_optional_complete_bundle(
    value: dict[str, Any], fields: tuple[str, ...], label: str
) -> None:
    if any(field in value for field in fields):
        _require_fields(value, fields, label)


def resolve_dataset_groups(group: str, run_kind: str) -> list[tuple[str, Path]]:
    if run_kind not in {"short_test", "full"}:
        raise ValueError(f"Unsupported RUN_KIND: {run_kind}")
    if group == "all":
        if run_kind != "short_test":
            raise ValueError("DATASET_GROUP=all is allowed only for RUN_KIND=short_test")
        return list(DATASET_GROUPS.items())
    if group not in DATASET_GROUPS:
        raise ValueError(f"Unsupported DATASET_GROUP: {group}")
    return [(group, DATASET_GROUPS[group])]


def dataset_counts(path: Path) -> tuple[int, int]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Runnable multi-step dataset must be a nonempty list: {path}")
    workflow_ids: set[str] = set()
    step_ids: set[tuple[str, str]] = set()
    steps = 0
    for row in rows:
        workflow_id = str(row["id"])
        if workflow_id in workflow_ids:
            raise ValueError(f"Duplicate workflow ID in {path}: {workflow_id}")
        workflow_ids.add(workflow_id)
        expected_steps = row.get("expected_steps")
        if not isinstance(expected_steps, list) or not expected_steps:
            raise ValueError(f"Workflow has no expected_steps: {workflow_id}")
        for step in expected_steps:
            key = workflow_id, str(step["id"])
            if key in step_ids:
                raise ValueError(f"Duplicate step ID in {path}: {key}")
            step_ids.add(key)
            steps += 1
    return len(rows), steps


def select_longest_workflows(
    path: Path,
    prompt_length: Callable[[str], int],
    count: int = 1,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if count <= 0:
        raise ValueError("Short-test workflow count must be positive")
    raw_rows = json.loads(path.read_text(encoding="utf-8"))
    samples = load_benchmark(path)
    candidates: list[tuple[int, str, int]] = []
    for sample_index, sample in enumerate(samples):
        history: list[dict[str, Any]] = []
        maximum = 0
        for step in sample.expected_steps:
            maximum = max(maximum, prompt_length(_multistep_query(sample, step, history)))
            # Selection only: approximate the maximum rendered history size with
            # reference-shaped values. These values are never written into the
            # evaluated subset or supplied to the model during guided rollout.
            history.append(
                {
                    "step_id": step.id,
                    "query": step.query,
                    "selected_tool": step.expected_tool,
                    "selected_args": step.expected_args,
                    "execution_success": True,
                    "tool_result_value": step.expected_answer,
                    "tool_error": None,
                }
            )
        candidates.append((maximum, sample.id, sample_index))
    ranked = sorted(candidates, key=lambda item: (item[0], item[1]), reverse=True)
    selected_candidates = ranked[: min(count, len(ranked))]
    selected = [raw_rows[item[2]] for item in selected_candidates]
    source_bytes = path.read_bytes()
    metadata = {
        "original_benchmark_path": str(path.resolve()),
        "original_benchmark_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "selected_workflow_ids": [item[1] for item in selected_candidates],
        "selected_workflow_steps": [len(row["expected_steps"]) for row in selected],
        "selected_workflow_count": len(selected),
        "selection_rule": (
            "workflows with the largest exactly rendered step prompt for the "
            "selected model; ties broken by reverse lexicographic workflow ID"
        ),
        "largest_prompt_tokens": [item[0] for item in selected_candidates],
        "headline_eligible": False,
        "retention": "removable_after_corresponding_full_run_completes",
    }
    return selected, metadata


def select_longest_workflow(
    path: Path,
    prompt_length: Callable[[str], int],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Backward-compatible one-workflow selection helper."""
    selected, metadata = select_longest_workflows(path, prompt_length, 1)
    metadata["selected_workflow_id"] = metadata["selected_workflow_ids"][0]
    return selected[0], metadata


def write_short_test_subset(
    path: Path,
    subset_path: Path,
    provenance_path: Path,
    prompt_length: Callable[[str], int],
    count: int = 1,
) -> dict[str, Any]:
    selected, metadata = select_longest_workflows(path, prompt_length, count)
    payload = json.dumps(selected, ensure_ascii=True, indent=2) + "\n"
    subset_path.parent.mkdir(parents=True, exist_ok=True)
    with subset_path.open("x", encoding="utf-8") as handle:
        handle.write(payload)
    metadata["short_test_subset_path"] = str(subset_path.resolve())
    metadata["short_test_subset_sha256"] = hashlib.sha256(payload.encode()).hexdigest()
    with provenance_path.open("x", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return metadata


def write_model_short_test_subset(
    *, source: Path, subset: Path, provenance: Path, model: str,
    checkpoint: Path, registry_snapshot: Path, reasoning_mode: str,
) -> dict[str, Any]:
    """Render the production router prompt with a tokenizer, never model weights."""
    from models.routers.structured_tool_call import build_native_tools, build_tool_call_prompt

    snapshot = _load_json_object(registry_snapshot)
    names = snapshot["tool_names"]
    schemas = snapshot["tool_schemas"]
    descriptions = snapshot["tool_descriptions"]
    if model == "gemma-4-local":
        from models.architectures.gemma4_pytorch.inference import get_tokenizer
    elif model == "qwen-3.6-local":
        from models.architectures.qwen36_pytorch.inference import get_tokenizer
    elif model == "phi-4-local":
        from models.architectures.phi4_pytorch.inference import get_tokenizer
    elif model == "llama-3.1-8b-local":
        from models.architectures.llama31_8b_pytorch.inference import get_tokenizer
    else:
        raise ValueError(f"Unsupported short-test model: {model}")
    tokenizer = get_tokenizer(str(checkpoint))

    def prompt_length(query: str) -> int:
        if model == "llama-3.1-8b-local":
            content = (
                "You are an MCP client in a tool-routing benchmark. Call exactly "
                "one of the tools supplied by the chat template. Do not answer the "
                "request directly and do not explain the call.\n\nUser query:\n" + query
            )
            rendered = tokenizer.apply_chat_template(
                [{"role": "user", "content": content}],
                tools=build_native_tools(names, schemas, descriptions),
                tokenize=False,
                add_generation_prompt=True,
            )
        elif model == "phi-4-local":
            content = build_tool_call_prompt(query, names, schemas, descriptions)
            rendered = tokenizer.apply_chat_template(
                [{"role": "user", "content": content}],
                tokenize=False,
                add_generation_prompt=True,
            )
        elif model == "gemma-4-local":
            content = build_tool_call_prompt(query, names, schemas, descriptions)
            rendered = tokenizer.apply_chat_template(
                [{"role": "user", "content": content}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=reasoning_mode == "reasoning",
            )
        elif model == "qwen-3.6-local":
            content = (
                "This is a tool-routing benchmark. You must call exactly one of "
                "the provided functions and must not answer the request directly, "
                "even if you can solve it without a tool.\n\nUser request:\n" + query
            )
            rendered = tokenizer.apply_chat_template(
                [{"role": "user", "content": content}],
                tools=build_native_tools(names, schemas, descriptions),
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=reasoning_mode == "reasoning",
            )
        else:
            raise ValueError(f"Unsupported short-test model: {model}")
        return len(tokenizer.encode(rendered, add_special_tokens=False))

    count = 3 if model == "gemma-4-local" else 1
    return write_short_test_subset(
        source, subset, provenance, prompt_length, count=count
    )


def validate_and_index_multistep(
    *, dataset_directory: Path, index_path: Path, source_benchmark: Path,
    evaluated_benchmark: Path, expected_model: str, expected_prompt_template: str,
    expected_registry_fingerprint: str, expected_registry_fingerprint_version: str,
    expected_tool_count: int, expected_tool_pool: str,
    expected_reasoning_mode: str, expected_reasoning_method: str,
    expected_generation_limit: int,
    short_test_provenance: Path | None = None,
) -> dict[str, Any]:
    dataset_directory = dataset_directory.resolve()
    index_path = index_path.resolve()
    run_directory = index_path.parent
    try:
        relative = dataset_directory.relative_to(run_directory)
    except ValueError as exc:
        raise ValueError("Dataset directory must be inside the run directory") from exc
    if not relative.parts or relative.parts[0] != "domains":
        raise ValueError("Dataset directory must be under the run domains directory")
    paths = [dataset_directory / name for name in ("samples.jsonl", "summary.json", "evaluation.log")]
    resolved = [path.resolve() for path in paths]
    if len(set(resolved)) != 3 or any(path.parent != dataset_directory for path in resolved):
        raise ValueError("Dataset artifact paths must be distinct and confined")
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Required dataset artifact is missing: {path}")
    samples = _load_jsonl_objects(paths[0])
    summary = _load_json_object(paths[1])
    source_count, source_steps = dataset_counts(evaluated_benchmark)
    if len(samples) != source_count or summary.get("total_workflows") != source_count:
        raise ValueError("Workflow count does not match evaluated benchmark and summary")
    if summary.get("total_steps") != source_steps:
        raise ValueError("Routed-step count does not match evaluated benchmark")
    expected_path = _canonical_benchmark(evaluated_benchmark)
    required_summary = {
        "model_name": expected_model,
        "prompt_template": expected_prompt_template,
        "evaluation_protocol": MULTISTEP_EVALUATION_PROTOCOL,
        "reasoning_mode": expected_reasoning_mode,
        "reasoning_method": expected_reasoning_method,
        "effective_generation_limit": expected_generation_limit,
        "effective_generation_limit_unit": "tokens",
        "tool_pool": expected_tool_pool,
        "tool_count": expected_tool_count,
        "tool_registry_fingerprint": expected_registry_fingerprint,
        "tool_registry_fingerprint_version": expected_registry_fingerprint_version,
        "outcome_metric_names": list(OUTCOME_METRIC_NAMES),
    }
    for field, value in required_summary.items():
        if summary.get(field) != value:
            raise ValueError(f"Unexpected summary {field}: {summary.get(field)!r} != {value!r}")
    _require_fields(
        summary,
        CORE_SUMMARY_METRIC_FIELDS + FINAL_STEP_METRIC_FIELDS,
        "Multi-step summary",
    )
    _require_optional_complete_bundle(
        summary, FINAL_PROGRAM_METRIC_FIELDS, "Final program metrics"
    )
    if _canonical_benchmark(summary.get("benchmark_path", "")) != expected_path:
        raise ValueError("Unexpected summary benchmark path")
    evaluated_hash = hashlib.sha256(evaluated_benchmark.read_bytes()).hexdigest()
    if summary.get("benchmark_sha256") != evaluated_hash:
        raise ValueError("Summary benchmark SHA-256 does not match evaluated benchmark")
    expected_rows = json.loads(evaluated_benchmark.read_text(encoding="utf-8"))
    expected_ids = {str(row["id"]): [str(step["id"]) for step in row["expected_steps"]] for row in expected_rows}
    observed_ids: dict[str, list[str]] = {}
    benchmark_modes: set[str] = set()
    execution_modes: set[str] = set()
    matchers: set[str] = set()
    for record in samples:
        workflow_id = str(record.get("sample_id"))
        if workflow_id in observed_ids:
            raise ValueError(f"Duplicate evaluated workflow: {workflow_id}")
        if record.get("model_name") != expected_model:
            raise ValueError(f"Mixed or unexpected model in workflow {workflow_id}")
        if _canonical_benchmark(record.get("benchmark_path", "")) != expected_path:
            raise ValueError(f"Unexpected benchmark in workflow {workflow_id}")
        if record.get("benchmark_sha256") != evaluated_hash:
            raise ValueError(
                f"Benchmark SHA-256 mismatch in workflow {workflow_id}"
            )
        if record.get("evaluation_protocol") != MULTISTEP_EVALUATION_PROTOCOL:
            raise ValueError(f"Unexpected evaluation protocol in workflow {workflow_id}")
        for field, value in required_summary.items():
            if record.get(field) != value:
                raise ValueError(
                    f"Metadata mismatch for {field} in workflow {workflow_id}"
                )
        _require_fields(record, WORKFLOW_OUTCOME_FIELDS, f"Workflow {workflow_id}")
        _require_optional_complete_bundle(
            record,
            (
                "expected_final_program_result",
                "final_program_execution_contract",
                "final_program_execution_correct",
                "final_program_execution_status",
                "final_program_execution_matcher",
            ),
            f"Workflow {workflow_id} final program outcome",
        )
        steps = record.get("steps")
        if not isinstance(steps, list):
            raise ValueError(f"Missing routed steps in workflow {workflow_id}")
        step_ids = [str(step.get("step_id")) for step in steps]
        if len(step_ids) != len(set(step_ids)):
            raise ValueError(f"Duplicate routed step in workflow {workflow_id}")
        for step in steps:
            _require_fields(
                step,
                STEP_OUTCOME_FIELDS,
                f"Workflow {workflow_id} step {step.get('step_id')}",
            )
        observed_ids[workflow_id] = step_ids
        benchmark_modes.add(str(record.get("benchmark_mode", DEFAULT_BENCHMARK_MODE)))
        execution_modes.add(str(record.get("workflow_execution_mode", DEFAULT_WORKFLOW_EXECUTION_MODE)))
        matchers.update(str(step["final_outcome_matcher"]) for step in steps if step.get("final_outcome_matcher"))
    if observed_ids != expected_ids:
        raise ValueError("Evaluated workflow/step IDs do not exactly match the benchmark")
    summary_modes = set(summary.get("benchmark_mode_counts", {}))
    if summary_modes != benchmark_modes:
        raise ValueError("Summary benchmark-mode distribution does not match workflows")
    if set(summary.get("workflow_execution_modes", [])) != execution_modes:
        raise ValueError("Summary workflow-execution modes do not match workflows")
    if execution_modes != {PREDICTED_ROLLOUT_EXECUTION_MODE}:
        raise ValueError(
            "PR35 multi-step artifacts must use predicted_sequence execution"
        )
    indexed = {
        "samples_path": _index_path_value(paths[0], run_directory),
        "summary_path": _index_path_value(paths[1], run_directory),
        "evaluation_log_path": _index_path_value(paths[2], run_directory),
    }
    existing = _existing_index_records(index_path)
    used = {_resolve_index_path(record[field], run_directory) for record in existing for field in indexed}
    if any(_resolve_index_path(value, run_directory) in used for value in indexed.values()):
        raise ValueError("Artifact path is already indexed")
    record = {
        "benchmark_path": str(source_benchmark.resolve()),
        "evaluated_benchmark_path": str(evaluated_benchmark.resolve()),
        "benchmark_sha256": hashlib.sha256(
            source_benchmark.read_bytes()
        ).hexdigest(),
        "evaluated_benchmark_sha256": hashlib.sha256(
            evaluated_benchmark.read_bytes()
        ).hexdigest(),
        "workflow_count": source_count,
        "expected_step_count": source_steps,
        "benchmark_modes": sorted(benchmark_modes),
        "workflow_execution_modes": sorted(execution_modes),
        "evaluation_protocol": MULTISTEP_EVALUATION_PROTOCOL,
        "final_outcome_matchers": sorted(matchers),
        "outcome_metric_names": summary["outcome_metric_names"],
        **{
            field: summary[field]
            for field in (
                CORE_SUMMARY_METRIC_FIELDS
                + FINAL_STEP_METRIC_FIELDS
                + FINAL_PROGRAM_METRIC_FIELDS
            )
            if field in summary
        },
        "model_name": expected_model,
        "prompt_template": expected_prompt_template,
        "reasoning_mode": summary["reasoning_mode"],
        "reasoning_method": summary["reasoning_method"],
        "effective_generation_limit": summary["effective_generation_limit"],
        "effective_generation_limit_unit": summary[
            "effective_generation_limit_unit"
        ],
        **{field: summary[field] for field in (
            "tool_pool", "tool_count", "tool_registry_fingerprint",
            "tool_registry_fingerprint_version")},
        **indexed,
    }
    if short_test_provenance is not None:
        record["short_test_provenance_path"] = _index_path_value(
            short_test_provenance, run_directory
        )
    with index_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
    return record


def validate_complete_multistep_run(
    index_path: Path,
    expected_benchmarks: list[Path],
    run_metadata_path: Path | None = None,
) -> None:
    index_path = index_path.resolve()
    run_directory = index_path.parent
    records = _existing_index_records(index_path)
    expected = Counter(str(path.resolve()) for path in expected_benchmarks)
    observed = Counter(str(Path(record.get("benchmark_path", "")).resolve()) for record in records)
    if expected != observed or any(value != 1 for value in observed.values()):
        raise ValueError(f"Run index is incomplete: expected {expected}, observed {observed}")
    artifact_paths: list[Path] = []
    for record in records:
        if record.get("outcome_metric_names") != list(OUTCOME_METRIC_NAMES):
            raise ValueError("Run index declares unexpected outcome metrics")
        _require_fields(
            record,
            CORE_SUMMARY_METRIC_FIELDS + FINAL_STEP_METRIC_FIELDS,
            "Run index entry",
        )
        _require_optional_complete_bundle(
            record, FINAL_PROGRAM_METRIC_FIELDS, "Run index final program metrics"
        )
        source_benchmark = Path(str(record.get("benchmark_path", ""))).resolve()
        if not source_benchmark.is_file():
            raise FileNotFoundError(
                f"Indexed benchmark does not exist: {source_benchmark}"
            )
        if record.get("benchmark_sha256") != hashlib.sha256(
            source_benchmark.read_bytes()
        ).hexdigest():
            raise ValueError("Indexed benchmark SHA-256 does not match its file")
        evaluated_benchmark = Path(
            str(record.get("evaluated_benchmark_path", ""))
        ).resolve()
        if not evaluated_benchmark.is_file():
            raise FileNotFoundError(
                f"Indexed evaluated benchmark does not exist: {evaluated_benchmark}"
            )
        if record.get("evaluated_benchmark_sha256") != hashlib.sha256(
            evaluated_benchmark.read_bytes()
        ).hexdigest():
            raise ValueError(
                "Indexed evaluated benchmark SHA-256 does not match its file"
            )
        for field in ("samples_path", "summary_path", "evaluation_log_path"):
            if not record.get(field):
                raise ValueError(f"Run index entry is missing {field}")
            artifact = _resolve_index_path(str(record[field]), run_directory)
            if not artifact.is_file():
                raise FileNotFoundError(f"Indexed artifact does not exist: {artifact}")
            artifact_paths.append(artifact)
        provenance = record.get("short_test_provenance_path")
        if provenance and not _resolve_index_path(str(provenance), run_directory).is_file():
            raise FileNotFoundError("Indexed short-test provenance does not exist")
    if len(artifact_paths) != len(set(artifact_paths)):
        raise ValueError("Run artifact index reuses an artifact path")
    modes = {mode for record in records for mode in record.get("benchmark_modes", [])}
    if len(modes) > 1:
        # Mixed modes are permitted only as distinct index entries; never pooled.
        if any(len(record.get("benchmark_modes", [])) != 1 for record in records):
            raise ValueError("A dataset index entry pools multiple benchmark modes")
    if run_metadata_path is not None:
        validate_run_metadata(index_path=index_path, metadata_path=run_metadata_path)
        metadata = _load_json_object(run_metadata_path)
        if not isinstance(metadata.get("git_commit"), str) or not metadata[
            "git_commit"
        ].strip():
            raise ValueError("Run metadata is missing its producing Git commit")
        benchmark_hashes = metadata.get("benchmark_sha256")
        if not isinstance(benchmark_hashes, dict):
            raise ValueError("Run metadata is missing benchmark SHA-256 values")
        expected_hashes = {
            str(Path(record["benchmark_path"]).resolve()): record["benchmark_sha256"]
            for record in records
        }
        normalized_hashes = {
            str(Path(path).resolve()): value
            for path, value in benchmark_hashes.items()
        }
        if normalized_hashes != expected_hashes:
            raise ValueError("Run metadata benchmark SHA-256 values do not match")
        if metadata.get("evaluation_protocol") != MULTISTEP_EVALUATION_PROTOCOL:
            raise ValueError("Run metadata uses an obsolete multi-step protocol")
        if metadata.get("workflow_execution_mode") != PREDICTED_ROLLOUT_EXECUTION_MODE:
            raise ValueError("Run metadata does not declare predicted_sequence")
        if metadata.get("outcome_metric_names") != list(OUTCOME_METRIC_NAMES):
            raise ValueError("Run metadata declares unexpected outcome metrics")
        metric_summaries = metadata.get("outcome_metric_summaries")
        if not isinstance(metric_summaries, dict):
            raise ValueError("Run metadata is missing outcome metric summaries")
        for record in records:
            expected_metric_summary = {
                key: value
                for key, value in record.items()
                if key in {
                    "outcome_metric_names",
                    "tool_selection_accuracy",
                    "argument_accuracy",
                    "step_outcome_accuracy",
                    "step_outcome_scored",
                    "step_outcome_status_counts",
                    "step_outcome_matchers",
                    "all_tools_correct_accuracy",
                    "all_arguments_correct_accuracy",
                    "all_steps_correct_accuracy",
                    "all_steps_correct_scored",
                }
                or key.startswith("final_step_outcome_")
                or key.startswith("final_program_execution_")
            }
            if metric_summaries.get(record["benchmark_path"]) != expected_metric_summary:
                raise ValueError("Run metadata outcome metrics do not match its index")
        run_kind = metadata.get("run_kind")
        if run_kind not in {"short_test", "full"}:
            raise ValueError(f"Unsupported run kind in metadata: {run_kind!r}")
        if metadata.get("headline_eligible") is not (run_kind == "full"):
            raise ValueError("Run headline eligibility does not match run kind")
        source_counts = metadata.get("source_counts")
        if not isinstance(source_counts, dict):
            raise ValueError("Run metadata is missing source_counts")
        for record in records:
            source = str(Path(record["benchmark_path"]).resolve())
            counts = source_counts.get(source)
            if not isinstance(counts, dict):
                raise ValueError(f"Run metadata is missing counts for {source}")
            if run_kind == "full" and (
                record.get("workflow_count") != counts.get("workflows")
                or record.get("expected_step_count") != counts.get("routed_steps")
            ):
                raise ValueError(f"Full-run counts do not match source counts for {source}")
        if run_kind == "short_test" and not metadata.get("short_test_selection"):
            raise ValueError("Short-test metadata is missing exact selection provenance")
        if run_kind == "short_test":
            declared_selections = list(metadata["short_test_selection"].values())
            for record in records:
                provenance_value = record.get("short_test_provenance_path")
                if not provenance_value:
                    raise ValueError("Short-test index is missing selection provenance")
                provenance = _load_json_object(
                    _resolve_index_path(str(provenance_value), run_directory)
                )
                if provenance not in declared_selections:
                    raise ValueError(
                        "Indexed short-test selection does not match run metadata"
                    )
                samples = _load_jsonl_objects(
                    _resolve_index_path(str(record["samples_path"]), run_directory)
                )
                observed_ids = [str(sample.get("sample_id")) for sample in samples]
                if observed_ids != provenance.get("selected_workflow_ids"):
                    raise ValueError(
                        "Short-test samples do not match the recorded selection"
                    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counts", type=Path)
    parser.add_argument("--make-short-test", action="store_true")
    parser.add_argument("--subset", type=Path)
    parser.add_argument("--provenance", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--registry-snapshot", type=Path)
    parser.add_argument("--reasoning-mode")
    parser.add_argument("--validate-index", action="store_true")
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--index", type=Path)
    parser.add_argument("--source-benchmark", type=Path)
    parser.add_argument("--evaluated-benchmark", type=Path)
    parser.add_argument("--model")
    parser.add_argument("--prompt-template")
    parser.add_argument("--registry-fingerprint")
    parser.add_argument("--registry-fingerprint-version")
    parser.add_argument("--tool-count", type=int)
    parser.add_argument("--tool-pool")
    parser.add_argument("--reasoning-method")
    parser.add_argument("--generation-limit", type=int)
    parser.add_argument("--short-test-provenance", type=Path)
    parser.add_argument("--run-metadata", type=Path)
    args = parser.parse_args()
    if args.counts:
        workflows, steps = dataset_counts(args.counts)
        print(f"{workflows} {steps}")
        return
    if args.make_short_test:
        write_model_short_test_subset(
            source=args.source_benchmark,
            subset=args.subset,
            provenance=args.provenance,
            model=args.model,
            checkpoint=args.checkpoint,
            registry_snapshot=args.registry_snapshot,
            reasoning_mode=args.reasoning_mode,
        )
        return
    if args.validate_index:
        validate_complete_multistep_run(
            args.index,
            [Path(line) for line in args.source_benchmark.read_text().splitlines() if line],
            args.run_metadata,
        )
        return
    validate_and_index_multistep(
        dataset_directory=args.dataset_dir, index_path=args.index,
        source_benchmark=args.source_benchmark,
        evaluated_benchmark=args.evaluated_benchmark,
        expected_model=args.model, expected_prompt_template=args.prompt_template,
        expected_registry_fingerprint=args.registry_fingerprint,
        expected_registry_fingerprint_version=args.registry_fingerprint_version,
        expected_tool_count=args.tool_count, expected_tool_pool=args.tool_pool,
        expected_reasoning_mode=args.reasoning_mode,
        expected_reasoning_method=args.reasoning_method,
        expected_generation_limit=args.generation_limit,
        short_test_provenance=args.short_test_provenance,
    )


if __name__ == "__main__":
    main()
