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
)
from evaluation.evaluate import (
    DEFAULT_BENCHMARK_MODE,
    DEFAULT_WORKFLOW_EXECUTION_MODE,
    MULTISTEP_EVALUATION_PROTOCOL,
    _gold_history_item,
    _multistep_query,
    load_benchmark,
)


DATASET_GROUPS = {
    "coding_sweagent": Path("benchmark/coding/coding_sweagent_multistep.json"),
    "coding_nebius_replay": Path(
        "benchmark/coding/coding_nebius_sweagent_replay_multistep.json"
    ),
    "enterprise_tau2": Path("benchmark/enterprise/enterprise_public_workflows.json"),
    "convfinqa": Path("benchmark/finance/finance_convfinqa_multistep.json"),
    "finqa": Path("benchmark/finance/finance_finqa_test_multistep.json"),
    "finretrieval_replay": Path(
        "benchmark/finance/finance_finretrieval_replay_multistep.json"
    ),
    "mathematics": Path("benchmark/math/math_multistep_controlled.json"),
}
EMPTY_PLACEHOLDER = Path(
    "benchmark/coding/coding_nebius_swerebench_openhands_replay_multistep.json"
)


def resolve_dataset_groups(group: str, run_kind: str) -> list[tuple[str, Path]]:
    if run_kind not in {"preflight", "full"}:
        raise ValueError(f"Unsupported RUN_KIND: {run_kind}")
    if group == "all":
        if run_kind != "preflight":
            raise ValueError("DATASET_GROUP=all is allowed only for RUN_KIND=preflight")
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


def select_longest_workflow(
    path: Path,
    prompt_length: Callable[[str], int],
) -> tuple[dict[str, Any], dict[str, Any]]:
    raw_rows = json.loads(path.read_text(encoding="utf-8"))
    samples = load_benchmark(path)
    candidates: list[tuple[int, str, int]] = []
    for sample_index, sample in enumerate(samples):
        history: list[dict[str, Any]] = []
        maximum = 0
        for step in sample.expected_steps:
            maximum = max(maximum, prompt_length(_multistep_query(sample, step, history)))
            history.append(_gold_history_item(step))
        candidates.append((maximum, sample.id, sample_index))
    maximum, workflow_id, sample_index = max(candidates, key=lambda item: (item[0], item[1]))
    selected = raw_rows[sample_index]
    source_bytes = path.read_bytes()
    metadata = {
        "original_benchmark_path": str(path.resolve()),
        "original_benchmark_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "selected_workflow_id": workflow_id,
        "selected_workflow_steps": len(selected["expected_steps"]),
        "selection_rule": (
            "workflow containing the largest exactly rendered step prompt for the "
            "selected model; ties broken by lexicographically greatest workflow ID"
        ),
        "largest_prompt_tokens": maximum,
        "headline_eligible": False,
        "retention": "removable_after_corresponding_full_run_completes",
    }
    return selected, metadata


def write_preflight_subset(
    path: Path,
    subset_path: Path,
    provenance_path: Path,
    prompt_length: Callable[[str], int],
) -> dict[str, Any]:
    selected, metadata = select_longest_workflow(path, prompt_length)
    payload = json.dumps([selected], ensure_ascii=True, indent=2) + "\n"
    subset_path.parent.mkdir(parents=True, exist_ok=True)
    with subset_path.open("x", encoding="utf-8") as handle:
        handle.write(payload)
    metadata["preflight_subset_path"] = str(subset_path.resolve())
    metadata["preflight_subset_sha256"] = hashlib.sha256(payload.encode()).hexdigest()
    with provenance_path.open("x", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return metadata


def write_model_preflight_subset(
    *, source: Path, subset: Path, provenance: Path, model: str,
    checkpoint: Path, registry_snapshot: Path,
) -> dict[str, Any]:
    """Render the production router prompt with a tokenizer, never model weights."""
    from transformers import AutoTokenizer
    from models.routers.structured_tool_call import build_native_tools, build_tool_call_prompt

    snapshot = _load_json_object(registry_snapshot)
    names = snapshot["tool_names"]
    schemas = snapshot["tool_schemas"]
    descriptions = snapshot["tool_descriptions"]
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)

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
        else:
            raise ValueError(f"Unsupported preflight model: {model}")
        return len(tokenizer.encode(rendered, add_special_tokens=False))

    return write_preflight_subset(source, subset, provenance, prompt_length)


def validate_and_index_multistep(
    *, dataset_directory: Path, index_path: Path, source_benchmark: Path,
    evaluated_benchmark: Path, expected_model: str, expected_prompt_template: str,
    expected_registry_fingerprint: str, expected_registry_fingerprint_version: str,
    expected_tool_count: int, expected_tool_pool: str,
    preflight_provenance: Path | None = None,
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
        "tool_pool": expected_tool_pool,
        "tool_count": expected_tool_count,
        "tool_registry_fingerprint": expected_registry_fingerprint,
        "tool_registry_fingerprint_version": expected_registry_fingerprint_version,
    }
    for field, value in required_summary.items():
        if summary.get(field) != value:
            raise ValueError(f"Unexpected summary {field}: {summary.get(field)!r} != {value!r}")
    if _canonical_benchmark(summary.get("benchmark_path", "")) != expected_path:
        raise ValueError("Unexpected summary benchmark path")
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
        if record.get("evaluation_protocol") != MULTISTEP_EVALUATION_PROTOCOL:
            raise ValueError(f"Unexpected evaluation protocol in workflow {workflow_id}")
        steps = record.get("steps")
        if not isinstance(steps, list):
            raise ValueError(f"Missing routed steps in workflow {workflow_id}")
        step_ids = [str(step.get("step_id")) for step in steps]
        if len(step_ids) != len(set(step_ids)):
            raise ValueError(f"Duplicate routed step in workflow {workflow_id}")
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
        "workflow_count": source_count,
        "expected_step_count": source_steps,
        "benchmark_modes": sorted(benchmark_modes),
        "workflow_execution_modes": sorted(execution_modes),
        "evaluation_protocol": MULTISTEP_EVALUATION_PROTOCOL,
        "final_outcome_matchers": sorted(matchers),
        "model_name": expected_model,
        "prompt_template": expected_prompt_template,
        **{field: summary[field] for field in (
            "tool_pool", "tool_count", "tool_registry_fingerprint",
            "tool_registry_fingerprint_version")},
        **indexed,
    }
    if preflight_provenance is not None:
        record["preflight_provenance_path"] = _index_path_value(preflight_provenance, run_directory)
    with index_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
    return record


def validate_complete_multistep_run(index_path: Path, expected_benchmarks: list[Path]) -> None:
    index_path = index_path.resolve()
    run_directory = index_path.parent
    records = _existing_index_records(index_path)
    expected = Counter(str(path.resolve()) for path in expected_benchmarks)
    observed = Counter(str(Path(record.get("benchmark_path", "")).resolve()) for record in records)
    if expected != observed or any(value != 1 for value in observed.values()):
        raise ValueError(f"Run index is incomplete: expected {expected}, observed {observed}")
    artifact_paths: list[Path] = []
    for record in records:
        for field in ("samples_path", "summary_path", "evaluation_log_path"):
            if not record.get(field):
                raise ValueError(f"Run index entry is missing {field}")
            artifact = _resolve_index_path(str(record[field]), run_directory)
            if not artifact.is_file():
                raise FileNotFoundError(f"Indexed artifact does not exist: {artifact}")
            artifact_paths.append(artifact)
        provenance = record.get("preflight_provenance_path")
        if provenance and not _resolve_index_path(str(provenance), run_directory).is_file():
            raise FileNotFoundError("Indexed preflight provenance does not exist")
    if len(artifact_paths) != len(set(artifact_paths)):
        raise ValueError("Run artifact index reuses an artifact path")
    modes = {mode for record in records for mode in record.get("benchmark_modes", [])}
    if len(modes) > 1:
        # Mixed modes are permitted only as distinct index entries; never pooled.
        if any(len(record.get("benchmark_modes", [])) != 1 for record in records):
            raise ValueError("A dataset index entry pools multiple benchmark modes")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counts", type=Path)
    parser.add_argument("--make-preflight", action="store_true")
    parser.add_argument("--subset", type=Path)
    parser.add_argument("--provenance", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--registry-snapshot", type=Path)
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
    parser.add_argument("--preflight-provenance", type=Path)
    args = parser.parse_args()
    if args.counts:
        workflows, steps = dataset_counts(args.counts)
        print(f"{workflows} {steps}")
        return
    if args.make_preflight:
        write_model_preflight_subset(
            source=args.source_benchmark,
            subset=args.subset,
            provenance=args.provenance,
            model=args.model,
            checkpoint=args.checkpoint,
            registry_snapshot=args.registry_snapshot,
        )
        return
    if args.validate_index:
        validate_complete_multistep_run(args.index, [Path(line) for line in args.source_benchmark.read_text().splitlines() if line])
        return
    validate_and_index_multistep(
        dataset_directory=args.dataset_dir, index_path=args.index,
        source_benchmark=args.source_benchmark,
        evaluated_benchmark=args.evaluated_benchmark,
        expected_model=args.model, expected_prompt_template=args.prompt_template,
        expected_registry_fingerprint=args.registry_fingerprint,
        expected_registry_fingerprint_version=args.registry_fingerprint_version,
        expected_tool_count=args.tool_count, expected_tool_pool=args.tool_pool,
        preflight_provenance=args.preflight_provenance,
    )


if __name__ == "__main__":
    main()
