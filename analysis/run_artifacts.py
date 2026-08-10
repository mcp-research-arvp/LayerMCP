"""Validate and index one completed evaluator dataset artifact directory."""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
import json
from pathlib import Path
import re
from typing import Any


SAFE_PATH_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
REGISTRY_FINGERPRINT = re.compile(r"^sha256:[0-9a-f]{64}$")


def safe_path_component(value: str, *, label: str) -> str:
    if (
        not value
        or value in {".", ".."}
        or ".." in value
        or SAFE_PATH_COMPONENT.fullmatch(value) is None
    ):
        raise ValueError(f"Unsafe {label} path component: {value!r}")
    return value


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _load_jsonl_objects(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected a JSON object at {path}:{line_number}")
            records.append(value)
    return records


def _canonical_benchmark(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _index_path_value(path: Path, run_directory: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(run_directory.resolve()))
    except ValueError:
        return str(resolved)


def _resolve_index_path(value: str, run_directory: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = run_directory / path
    return path.resolve()


def _existing_index_records(index_path: Path) -> list[dict[str, Any]]:
    if not index_path.exists():
        return []
    return _load_jsonl_objects(index_path)


def _required_registry_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    required = (
        "tool_pool",
        "tool_count",
        "tool_registry_fingerprint",
        "tool_registry_fingerprint_version",
    )
    missing = [field for field in required if metadata.get(field) in (None, "")]
    if missing:
        raise ValueError("Live registry metadata is missing: " + ", ".join(missing))
    if not isinstance(metadata["tool_pool"], str) or not metadata["tool_pool"].strip():
        raise ValueError("Live registry tool_pool must be a nonempty string")
    if (
        isinstance(metadata["tool_count"], bool)
        or not isinstance(metadata["tool_count"], int)
        or metadata["tool_count"] <= 0
    ):
        raise ValueError("Live registry tool_count must be a positive integer")
    if REGISTRY_FINGERPRINT.fullmatch(metadata["tool_registry_fingerprint"]) is None:
        raise ValueError("Live registry fingerprint must be a sha256 digest")
    if (
        not isinstance(metadata["tool_registry_fingerprint_version"], str)
        or not metadata["tool_registry_fingerprint_version"].strip()
    ):
        raise ValueError("Live registry fingerprint version must be a nonempty string")
    return {field: metadata[field] for field in required}


async def capture_live_registry_metadata(server_path: Path) -> dict[str, Any]:
    """Build metadata through the evaluator's canonical live-registry path."""
    return _required_registry_metadata(
        await capture_live_registry_snapshot(server_path)
    )


async def capture_live_registry_snapshot(server_path: Path) -> dict[str, Any]:
    """Capture canonical registry metadata plus the exact model-visible catalog."""
    from evaluation.evaluate import _run_server_session, _tool_pool_metadata, _tool_schema

    async with _run_server_session(server_path) as session:
        listed_tools = await session.list_tools()
        registered = listed_tools.tools
        names = [tool.name for tool in registered]
        schemas = {tool.name: _tool_schema(tool) for tool in registered}
        descriptions = {
            tool.name: str(getattr(tool, "description", "") or "") for tool in registered
        }
    metadata = _required_registry_metadata(
        _tool_pool_metadata(names, schemas, descriptions)
    )
    return {
        **metadata,
        "tool_names": names,
        "tool_schemas": schemas,
        "tool_descriptions": descriptions,
    }


def validate_and_index_dataset(
    *,
    dataset_directory: Path,
    index_path: Path,
    expected_benchmark: Path,
    expected_model: str,
    expected_prompt_template: str,
    expected_registry_fingerprint: str,
    expected_registry_fingerprint_version: str,
    expected_tool_count: int,
    expected_tool_pool: str,
) -> dict[str, Any]:
    dataset_directory = dataset_directory.resolve()
    index_path = index_path.resolve()
    run_directory = index_path.parent
    samples_path = dataset_directory / "samples.jsonl"
    summary_path = dataset_directory / "summary.json"
    log_path = dataset_directory / "evaluation.log"
    try:
        relative_dataset_directory = dataset_directory.relative_to(run_directory)
    except ValueError as exc:
        raise ValueError(
            f"Dataset directory must be inside run directory {run_directory}: "
            f"{dataset_directory}"
        ) from exc
    if not relative_dataset_directory.parts or relative_dataset_directory.parts[0] != "domains":
        raise ValueError(
            f"Dataset directory must be under {run_directory / 'domains'}: "
            f"{dataset_directory}"
        )
    resolved_artifacts = tuple(
        path.resolve() for path in (samples_path, summary_path, log_path)
    )
    if len(set(resolved_artifacts)) != len(resolved_artifacts):
        raise ValueError(f"Dataset artifact paths must be distinct: {dataset_directory}")
    for path, resolved in zip(
        (samples_path, summary_path, log_path),
        resolved_artifacts,
    ):
        if resolved.parent != dataset_directory:
            raise ValueError(
                f"Dataset artifact escapes its intended directory: {path} -> {resolved}"
            )
    for path in (samples_path, summary_path, log_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required dataset artifact is missing: {path}")

    summary = _load_json_object(summary_path)
    samples = _load_jsonl_objects(samples_path)
    expected_benchmark_path = _canonical_benchmark(expected_benchmark)

    if int(summary.get("total_samples", -1)) != len(samples):
        raise ValueError(
            f"Row-count mismatch for {samples_path}: summary reports "
            f"{summary.get('total_samples')}, JSONL contains {len(samples)}"
        )
    if str(summary.get("model_name")) != expected_model:
        raise ValueError(
            f"Unexpected summary model in {summary_path}: "
            f"{summary.get('model_name')!r} != {expected_model!r}"
        )
    if _canonical_benchmark(str(summary.get("benchmark_path", ""))) != expected_benchmark_path:
        raise ValueError(
            f"Unexpected benchmark in {summary_path}: "
            f"{summary.get('benchmark_path')!r} != {str(expected_benchmark)!r}"
        )
    if str(summary.get("prompt_template")) != expected_prompt_template:
        raise ValueError(
            f"Unexpected prompt template in {summary_path}: "
            f"{summary.get('prompt_template')!r} != {expected_prompt_template!r}"
        )
    if summary.get("tool_registry_fingerprint") != expected_registry_fingerprint:
        raise ValueError(
            f"Unexpected registry fingerprint in {summary_path}: "
            f"{summary.get('tool_registry_fingerprint')!r} != "
            f"{expected_registry_fingerprint!r}"
        )
    expected_registry = {
        "tool_pool": expected_tool_pool,
        "tool_count": expected_tool_count,
        "tool_registry_fingerprint": expected_registry_fingerprint,
        "tool_registry_fingerprint_version": expected_registry_fingerprint_version,
    }
    for field, expected_value in expected_registry.items():
        if summary.get(field) != expected_value:
            raise ValueError(
                f"Unexpected registry metadata {field!r} in {summary_path}: "
                f"{summary.get(field)!r} != {expected_value!r}"
            )

    if summary.get("reasoning_mode") not in {"direct", "reasoning"}:
        raise ValueError(f"Invalid or missing reasoning_mode in {summary_path}")
    if not isinstance(summary.get("reasoning_method"), str) or not summary[
        "reasoning_method"
    ].strip():
        raise ValueError(f"Invalid or missing reasoning_method in {summary_path}")
    generation_limit = summary.get("effective_generation_limit")
    if (
        isinstance(generation_limit, bool)
        or not isinstance(generation_limit, int)
        or generation_limit <= 0
    ):
        raise ValueError(
            f"Invalid or missing effective_generation_limit in {summary_path}"
        )
    if summary.get("effective_generation_limit_unit") != "tokens":
        raise ValueError(
            f"Invalid or missing effective_generation_limit_unit in {summary_path}"
        )

    shared_fields = (
        "model_name",
        "prompt_template",
        "evaluation_protocol",
        "reasoning_mode",
        "reasoning_method",
        "effective_generation_limit",
        "effective_generation_limit_unit",
        "tool_pool",
        "tool_count",
        "tool_registry_fingerprint",
        "tool_registry_fingerprint_version",
    )
    matchers: set[str] = set()
    benchmark_modes: set[str] = set()
    for line_number, sample in enumerate(samples, start=1):
        if sample.get("model_name") != expected_model:
            raise ValueError(
                f"Mixed or unexpected model at {samples_path}:{line_number}: "
                f"{sample.get('model_name')!r} != {expected_model!r}"
            )
        if _canonical_benchmark(str(sample.get("benchmark_path", ""))) != expected_benchmark_path:
            raise ValueError(
                f"Unexpected benchmark at {samples_path}:{line_number}: "
                f"{sample.get('benchmark_path')!r} != {str(expected_benchmark)!r}"
            )
        for field in shared_fields:
            if sample.get(field) != summary.get(field):
                raise ValueError(
                    f"Metadata mismatch for {field!r} at "
                    f"{samples_path}:{line_number}"
                )
        matcher = sample.get("final_outcome_matcher")
        if matcher:
            matchers.add(str(matcher))
        mode = sample.get("benchmark_mode")
        if mode:
            benchmark_modes.add(str(mode))

    indexed_paths = {
        "samples_path": _index_path_value(samples_path, run_directory),
        "summary_path": _index_path_value(summary_path, run_directory),
        "evaluation_log_path": _index_path_value(log_path, run_directory),
    }
    existing = _existing_index_records(index_path)
    used_paths = {
        _resolve_index_path(str(record[field]), run_directory)
        for record in existing
        for field in ("samples_path", "summary_path", "evaluation_log_path")
        if record.get(field)
    }
    for field, value in indexed_paths.items():
        resolved = _resolve_index_path(value, run_directory)
        if resolved in used_paths:
            raise ValueError(f"Artifact path is already indexed ({field}): {resolved}")

    record = {
        "benchmark_path": str(expected_benchmark),
        "benchmark_modes": sorted(benchmark_modes),
        "evaluation_protocol": summary["evaluation_protocol"],
        "final_outcome_matchers": sorted(matchers),
        "model_name": expected_model,
        "prompt_template": expected_prompt_template,
        "reasoning_mode": summary["reasoning_mode"],
        "reasoning_method": summary["reasoning_method"],
        "effective_generation_limit": summary["effective_generation_limit"],
        "effective_generation_limit_unit": summary[
            "effective_generation_limit_unit"
        ],
        "tool_count": summary["tool_count"],
        "tool_pool": summary["tool_pool"],
        "tool_registry_fingerprint": summary["tool_registry_fingerprint"],
        "tool_registry_fingerprint_version": summary[
            "tool_registry_fingerprint_version"
        ],
        **indexed_paths,
    }
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with index_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
    return record


def validate_complete_run(
    *,
    index_path: Path,
    expected_benchmarks: list[Path],
) -> None:
    index_path = index_path.resolve()
    run_directory = index_path.parent
    records = _existing_index_records(index_path)
    expected = [_canonical_benchmark(path) for path in expected_benchmarks]
    observed = [
        _canonical_benchmark(str(record.get("benchmark_path", "")))
        for record in records
    ]
    expected_counts = Counter(expected)
    observed_counts = Counter(observed)
    if observed_counts != expected_counts or any(count != 1 for count in observed_counts.values()):
        raise ValueError(
            "Run artifact index is incomplete or contains duplicate/unexpected "
            f"benchmarks: expected {dict(expected_counts)}, observed "
            f"{dict(observed_counts)}"
        )

    used_paths: list[Path] = []
    for line_number, record in enumerate(records, start=1):
        for field in ("samples_path", "summary_path", "evaluation_log_path"):
            if not record.get(field):
                raise ValueError(f"{index_path}:{line_number} is missing {field!r}")
            path = _resolve_index_path(str(record[field]), run_directory)
            if not path.is_file():
                raise FileNotFoundError(
                    f"Indexed artifact does not exist ({field}): {path}"
                )
            used_paths.append(path)
    if len(used_paths) != len(set(used_paths)):
        raise ValueError("Run artifact index reuses an artifact path")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-run", action="store_true")
    parser.add_argument("--safe-component", nargs=2, metavar=("LABEL", "VALUE"))
    parser.add_argument("--capture-live-registry", action="store_true")
    parser.add_argument("--include-catalog", action="store_true")
    parser.add_argument("--registry-metadata-out", type=Path)
    parser.add_argument("--server", type=Path)
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--index", type=Path)
    parser.add_argument("--benchmark", type=Path)
    parser.add_argument("--model")
    parser.add_argument("--prompt-template")
    parser.add_argument("--registry-fingerprint")
    parser.add_argument("--registry-fingerprint-version")
    parser.add_argument("--tool-count", type=int)
    parser.add_argument("--tool-pool")
    parser.add_argument("--resolved-datasets", type=Path)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.safe_component is not None:
        label, value = args.safe_component
        print(safe_path_component(value, label=label))
        return
    if args.capture_live_registry:
        if args.server is None or args.registry_metadata_out is None:
            raise SystemExit(
                "--capture-live-registry requires --server and "
                "--registry-metadata-out"
            )
        capture = (
            capture_live_registry_snapshot
            if args.include_catalog
            else capture_live_registry_metadata
        )
        metadata = asyncio.run(capture(args.server))
        with args.registry_metadata_out.open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True)
            handle.write("\n")
        return
    if args.validate_run:
        if args.index is None or args.resolved_datasets is None:
            raise SystemExit("--validate-run requires --index and --resolved-datasets")
        expected_benchmarks = [
            Path(line)
            for line in args.resolved_datasets.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        validate_complete_run(
            index_path=args.index,
            expected_benchmarks=expected_benchmarks,
        )
        return
    required = {
        "--index": args.index,
        "--dataset-dir": args.dataset_dir,
        "--benchmark": args.benchmark,
        "--model": args.model,
        "--prompt-template": args.prompt_template,
        "--registry-fingerprint": args.registry_fingerprint,
        "--registry-fingerprint-version": args.registry_fingerprint_version,
        "--tool-count": args.tool_count,
        "--tool-pool": args.tool_pool,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise SystemExit("Missing required arguments: " + ", ".join(missing))
    validate_and_index_dataset(
        dataset_directory=args.dataset_dir,
        index_path=args.index,
        expected_benchmark=args.benchmark,
        expected_model=args.model,
        expected_prompt_template=args.prompt_template,
        expected_registry_fingerprint=args.registry_fingerprint,
        expected_registry_fingerprint_version=args.registry_fingerprint_version,
        expected_tool_count=args.tool_count,
        expected_tool_pool=args.tool_pool,
    )


if __name__ == "__main__":
    main()
