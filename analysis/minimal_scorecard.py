"""Generate a minimal, reporting-only scorecard from completed run artifacts."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Iterable, Sequence

from analysis.benchmark_inventory import infer_benchmark_class
from evaluation.evaluate import DEFAULT_BENCHMARK_MODE


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATTERN = re.compile(r"^Results:\s*(.+?)\s*$", re.MULTILINE)
SUMMARY_PATTERN = re.compile(r"^Summary:\s*(.+?)\s*$", re.MULTILINE)
SCHEMA_VALID_UNAVAILABLE = "—"

DOMAIN_LABELS = {
    "coding": "Coding",
    "enterprise": "Enterprise",
    "enterprise_automation": "Enterprise",
    "finance": "Finance",
    "mathematics": "Mathematics",
}


@dataclass(frozen=True)
class RunRecord:
    model: str
    benchmark_mode: str
    mode_source: str
    benchmark: str
    benchmark_family: str
    domain: str
    sample: dict[str, Any]


@dataclass(frozen=True)
class LoadedRuns:
    records: tuple[RunRecord, ...]
    fingerprints: tuple[str, ...]
    fingerprint_versions: tuple[str, ...]
    tool_counts: tuple[int, ...]
    tool_pools: tuple[str, ...]


def _resolve_artifact(raw_path: str, *, log_path: Path) -> Path:
    candidate = Path(raw_path).expanduser()
    if candidate.exists():
        return candidate.resolve()

    alternatives = (
        log_path.parent / candidate.name,
        PROJECT_ROOT / "results" / candidate.name,
    )
    for alternative in alternatives:
        if alternative.exists():
            return alternative.resolve()
    raise FileNotFoundError(
        f"Artifact referenced by {log_path} does not exist: {raw_path}"
    )


def _artifact_from_log(
    log_path: Path,
    *,
    pattern: re.Pattern[str],
    label: str,
) -> Path:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    matches = pattern.findall(text)
    if not matches:
        raise ValueError(f"{log_path} does not contain a {label}: line")
    if len(set(matches)) != 1:
        raise ValueError(f"{log_path} contains inconsistent {label}: paths")
    return _resolve_artifact(matches[-1], log_path=log_path)


def _artifacts_from_index(run_directory: Path) -> list[tuple[Path, Path]]:
    index_path = run_directory / "artifact_index.jsonl"
    if not index_path.exists():
        return []
    if not (run_directory / "RUN_COMPLETE").is_file():
        raise ValueError(
            f"Indexed run is incomplete (missing RUN_COMPLETE): {run_directory}"
        )
    pairs: list[tuple[Path, Path]] = []
    for line_number, record in enumerate(_load_jsonl(index_path), start=1):
        try:
            summary_raw = str(record["summary_path"])
            samples_raw = str(record["samples_path"])
        except KeyError as exc:
            raise ValueError(
                f"{index_path}:{line_number} is missing {exc.args[0]!r}"
            ) from exc

        def resolve(raw: str) -> Path:
            path = Path(raw).expanduser()
            if not path.is_absolute():
                path = run_directory / path
            if not path.is_file():
                raise FileNotFoundError(
                    f"Artifact referenced by {index_path} does not exist: {path}"
                )
            return path.resolve()

        pairs.append((resolve(summary_raw), resolve(samples_raw)))
    return pairs


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            records.append(value)
    return records


def _benchmark_rows(benchmark_path: str, samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    path = Path(benchmark_path)
    candidates = [path]
    if not path.is_absolute():
        candidates.append(PROJECT_ROOT / path)
    for candidate in candidates:
        if not candidate.exists():
            continue
        payload = json.loads(candidate.read_text(encoding="utf-8"))
        if isinstance(payload, list) and all(isinstance(row, dict) for row in payload):
            return payload
    return samples


def _benchmark_family(benchmark_path: str, samples: list[dict[str, Any]]) -> str:
    return infer_benchmark_class(
        Path(benchmark_path),
        _benchmark_rows(benchmark_path, samples),
    )


def _validate_single_step(
    summary: dict[str, Any],
    samples: list[dict[str, Any]],
    *,
    summary_path: Path,
) -> None:
    protocol = summary.get("evaluation_protocol")
    if protocol not in {None, "single_step_tool_routing_v1"}:
        raise ValueError(
            f"Minimal scorecard accepts single-step runs only; "
            f"{summary_path} uses {protocol!r}"
        )
    if int(summary.get("total_samples", len(samples))) != len(samples):
        raise ValueError(
            f"Summary/sample count mismatch for {summary_path}: "
            f"{summary.get('total_samples')} != {len(samples)}"
        )
    for index, sample in enumerate(samples):
        sample_protocol = sample.get("evaluation_protocol")
        if sample_protocol not in {None, "single_step_tool_routing_v1"}:
            raise ValueError(
                f"Sample {index} associated with {summary_path} is not single-step"
            )


def _require_registry_metadata(
    summary: dict[str, Any], *, summary_path: Path
) -> tuple[str, str, int, str]:
    required_fields = (
        "tool_registry_fingerprint",
        "tool_registry_fingerprint_version",
        "tool_count",
        "tool_pool",
    )
    for field in required_fields:
        value = summary.get(field)
        if (
            value is None
            or (isinstance(value, str) and not value.strip())
            or value == 0
        ):
            raise ValueError(
                f"{summary_path} is missing required registry field {field!r}; "
                "registry compatibility cannot be verified."
            )
    return (
        str(summary["tool_registry_fingerprint"]),
        str(summary["tool_registry_fingerprint_version"]),
        int(summary["tool_count"]),
        str(summary["tool_pool"]),
    )


def load_runs(run_directories: Sequence[Path]) -> LoadedRuns:
    if not run_directories:
        raise ValueError("At least one run directory is required.")

    records: list[RunRecord] = []
    fingerprints: set[str] = set()
    fingerprint_versions: set[str] = set()
    tool_counts: set[int] = set()
    tool_pools: set[str] = set()
    seen_summaries: set[Path] = set()
    seen_samples: set[Path] = set()

    for run_directory in sorted(Path(path).resolve() for path in run_directories):
        if not run_directory.is_dir():
            raise FileNotFoundError(f"Run directory does not exist: {run_directory}")
        indexed_artifacts = _artifacts_from_index(run_directory)
        if indexed_artifacts:
            artifacts = indexed_artifacts
        else:
            logs = sorted(run_directory.glob("*.log"))
            if not logs:
                raise ValueError(
                    f"Run directory contains no artifact index or .log files: "
                    f"{run_directory}"
                )
            artifacts = [
                (
                    _artifact_from_log(
                        log_path,
                        pattern=SUMMARY_PATTERN,
                        label="Summary",
                    ),
                    _artifact_from_log(
                        log_path,
                        pattern=RESULTS_PATTERN,
                        label="Results",
                    ),
                )
                for log_path in logs
            ]

        for summary_path, samples_path in artifacts:
            if summary_path in seen_summaries or samples_path in seen_samples:
                raise ValueError(
                    f"Duplicate result artifact in run directory: {run_directory}"
                )
            seen_summaries.add(summary_path)
            seen_samples.add(samples_path)

            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if not isinstance(summary, dict):
                raise ValueError(f"Summary is not a JSON object: {summary_path}")
            samples = _load_jsonl(samples_path)
            _validate_single_step(summary, samples, summary_path=summary_path)

            model = str(summary.get("model_name") or run_directory.name)
            benchmark_path = str(summary.get("benchmark_path") or summary_path.parent.name)
            family = _benchmark_family(benchmark_path, samples)

            fingerprint, fingerprint_version, tool_count, tool_pool = (
                _require_registry_metadata(summary, summary_path=summary_path)
            )
            fingerprints.add(fingerprint)
            fingerprint_versions.add(fingerprint_version)
            tool_counts.add(tool_count)
            tool_pools.add(tool_pool)

            for sample in samples:
                sample_model = str(sample.get("model_name") or model)
                if sample_model != model:
                    raise ValueError(
                        f"Model mismatch between {summary_path} and {samples_path}: "
                        f"{model!r} != {sample_model!r}"
                    )
                domain = str(sample.get("domain") or "unknown")
                if "benchmark_mode" not in sample:
                    benchmark_mode = DEFAULT_BENCHMARK_MODE
                    mode_source = "defaulted (missing)"
                elif sample["benchmark_mode"] is None:
                    benchmark_mode = DEFAULT_BENCHMARK_MODE
                    mode_source = "defaulted (null)"
                else:
                    benchmark_mode = str(sample["benchmark_mode"])
                    mode_source = "explicit"
                records.append(
                    RunRecord(
                        model=model,
                        benchmark_mode=benchmark_mode,
                        mode_source=mode_source,
                        benchmark=Path(benchmark_path).stem,
                        benchmark_family=family,
                        domain=domain,
                        sample=sample,
                    )
                )

    if len(fingerprints) > 1:
        raise ValueError(
            "Refusing to aggregate incompatible tool registry fingerprints: "
            + ", ".join(sorted(fingerprints))
        )
    if len(fingerprint_versions) > 1:
        raise ValueError(
            "Refusing to aggregate incompatible registry fingerprint versions: "
            + ", ".join(sorted(fingerprint_versions))
        )
    if len(tool_counts) > 1:
        raise ValueError(
            "Refusing to aggregate runs with different tool counts: "
            + ", ".join(map(str, sorted(tool_counts)))
        )
    if len(tool_pools) > 1:
        raise ValueError(
            "Refusing to aggregate runs with different tool pools: "
            + ", ".join(sorted(tool_pools))
        )
    if not records:
        raise ValueError("No sample records were loaded.")

    return LoadedRuns(
        records=tuple(records),
        fingerprints=tuple(sorted(fingerprints)),
        fingerprint_versions=tuple(sorted(fingerprint_versions)),
        tool_counts=tuple(sorted(tool_counts)),
        tool_pools=tuple(sorted(tool_pools)),
    )


def _rate(numerator: int, denominator: int) -> str:
    return (
        f"{100 * numerator / denominator:.1f}%"
        if denominator
        else SCHEMA_VALID_UNAVAILABLE
    )


def _coverage(scored: int, total: int) -> str:
    percentage = 100 * scored / total if total else 0.0
    return f"{scored}/{total} ({percentage:.1f}%)"


def _metrics(records: Iterable[RunRecord]) -> dict[str, Any]:
    rows = list(records)
    total = len(rows)
    final_scored = sum(
        record.sample.get("final_outcome_correct") is not None for record in rows
    )
    return {
        "n": total,
        "final_scored": final_scored,
        "final_coverage": _coverage(final_scored, total),
        "tsa": _rate(
            sum(record.sample.get("tool_selection_correct") is True for record in rows),
            total,
        ),
        "sgoa": _rate(
            sum(record.sample.get("final_outcome_correct") is True for record in rows),
            final_scored,
        ),
        "exact_args": _rate(
            sum(record.sample.get("argument_match_correct") is True for record in rows),
            total,
        ),
        "execution": _rate(
            sum(record.sample.get("execution_success") is True for record in rows),
            total,
        ),
        "no_call": _rate(
            sum(record.sample.get("failure_category") == "no_tool_call" for record in rows),
            total,
        ),
        "wrong_tool": sum(
            record.sample.get("failure_category") == "wrong_tool" for record in rows
        ),
        "wrong_args": sum(
            record.sample.get("failure_category") == "wrong_args" for record in rows
        ),
        "args_false_final_true": sum(
            record.sample.get("argument_match_correct") is False
            and record.sample.get("final_outcome_correct") is True
            for record in rows
        ),
        "execution_true_final_false": sum(
            record.sample.get("execution_success") is True
            and record.sample.get("final_outcome_correct") is False
            for record in rows
        ),
    }


def _group_records(
    records: Sequence[RunRecord],
    fields: Sequence[str],
) -> list[tuple[tuple[str, ...], list[RunRecord]]]:
    grouped: dict[tuple[str, ...], list[RunRecord]] = defaultdict(list)
    for record in records:
        grouped[tuple(str(getattr(record, field)) for field in fields)].append(record)
    return sorted(grouped.items())


def _domain_label(domain: str) -> str:
    return DOMAIN_LABELS.get(domain, domain.replace("_", " ").title())


def _table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in rows)
    return "\n".join(lines)


def render_markdown(loaded: LoadedRuns) -> str:
    records = loaded.records
    models = _group_records(records, ("model", "benchmark_mode", "mode_source"))
    domains = _group_records(
        records,
        ("model", "benchmark_mode", "mode_source", "domain"),
    )
    families = _group_records(
        records,
        ("model", "benchmark_mode", "mode_source", "domain", "benchmark_family"),
    )
    records_by_model = _group_records(records, ("model",))

    registry_parts = []
    if loaded.tool_pools:
        registry_parts.append(f"pool `{loaded.tool_pools[0]}`")
    if loaded.tool_counts:
        registry_parts.append(f"{loaded.tool_counts[0]} tools")
    if loaded.fingerprints:
        registry_parts.append(f"fingerprint `{loaded.fingerprints[0]}`")
    registry_note = ", ".join(registry_parts) or "registry metadata unavailable"
    overall_rows = []
    diagnostics_rows = []
    matcher_rows = []
    for (model, benchmark_mode, mode_source), group in models:
        metrics = _metrics(group)
        overall_rows.append(
            (
                model,
                benchmark_mode,
                mode_source,
                metrics["n"],
                metrics["sgoa"],
                metrics["final_coverage"],
                metrics["tsa"],
                SCHEMA_VALID_UNAVAILABLE,
            )
        )
        diagnostics_rows.append(
            (
                model,
                benchmark_mode,
                mode_source,
                metrics["exact_args"],
                metrics["execution"],
                metrics["no_call"],
                metrics["wrong_tool"],
                metrics["wrong_args"],
                metrics["args_false_final_true"],
                metrics["execution_true_final_false"],
            )
        )

    domain_rows = []
    for (model, benchmark_mode, mode_source, domain), group in domains:
        metrics = _metrics(group)
        domain_rows.append(
            (
                model,
                benchmark_mode,
                mode_source,
                _domain_label(domain),
                metrics["n"],
                metrics["sgoa"],
                metrics["final_coverage"],
                metrics["tsa"],
                SCHEMA_VALID_UNAVAILABLE,
            )
        )

    family_rows = []
    for (model, benchmark_mode, mode_source, domain, family), group in families:
        metrics = _metrics(group)
        family_rows.append(
            (
                model,
                benchmark_mode,
                mode_source,
                _domain_label(domain),
                family,
                metrics["n"],
                metrics["sgoa"],
                metrics["final_coverage"],
                metrics["tsa"],
                SCHEMA_VALID_UNAVAILABLE,
            )
        )

    for (model,), group in records_by_model:
        matchers = sorted(
            {
                str(record.sample["final_outcome_matcher"])
                for record in group
                if record.sample.get("final_outcome_matcher")
            }
        )
        modes = sorted(
            {f"{record.benchmark_mode} ({record.mode_source})" for record in group}
        )
        matcher_rows.append(
            (
                model,
                ", ".join(f"`{mode}`" for mode in modes),
                ", ".join(f"`{matcher}`" for matcher in matchers)
                if matchers
                else "matcher metadata unavailable",
            )
        )

    return "\n".join(
        [
            "# Preliminary Minimal Scorecard",
            "",
            "## Executive note",
            "",
            "These are preliminary audited baselines. Final Outcome Accuracy is the "
            "primary reported success metric; it compares executed predicted MCP "
            "output with the benchmark's declared structured expected answer and is "
            "not natural-language answer accuracy. Tool Selection Accuracy is the "
            "second headline metric. Valid Arguments / Schema-Valid Tool Call (SVCA) "
            "is unavailable until full raw JSON Schema validation is implemented. "
            "Exact Reference Argument Match is a secondary diagnostic against one "
            "reference call; it is not SVCA or a primary correctness score.",
            "",
            f"Registry compatibility check: {registry_note}.",
            "",
            "## Scoring provenance",
            "",
            _table(
                (
                    "Model",
                    "Observed benchmark modes",
                    "Observed final-outcome matchers",
                ),
                matcher_rows,
            ),
            "",
            "## Overall model scorecard",
            "",
            _table(
                (
                    "Model",
                    "Benchmark mode",
                    "Mode source",
                    "N",
                    "Final Outcome Accuracy",
                    "Final Outcome Coverage",
                    "Tool Selection Accuracy",
                    "Valid Arguments / SVCA",
                ),
                overall_rows,
            ),
            "",
            "## Model × domain scorecard",
            "",
            _table(
                (
                    "Model",
                    "Benchmark mode",
                    "Mode source",
                    "Domain",
                    "N",
                    "Final Outcome Accuracy",
                    "Final Outcome Coverage",
                    "Tool Selection Accuracy",
                    "Valid Arguments / SVCA",
                ),
                domain_rows,
            ),
            "",
            "## Model × domain × benchmark-family scorecard",
            "",
            _table(
                (
                    "Model",
                    "Benchmark mode",
                    "Mode source",
                    "Domain",
                    "Benchmark family",
                    "N",
                    "Final Outcome Accuracy",
                    "Final Outcome Coverage",
                    "Tool Selection Accuracy",
                    "Valid Arguments / SVCA",
                ),
                family_rows,
            ),
            "",
            "## Diagnostics",
            "",
            _table(
                (
                    "Model",
                    "Benchmark mode",
                    "Mode source",
                    "Exact Reference Argument Match",
                    "Runtime execution",
                    "No/unknown call",
                    "Wrong tool",
                    "Wrong args",
                    "Args false / final true",
                    "Execution true / final false",
                ),
                diagnostics_rows,
            ),
            "",
            "## Caveats",
            "",
            "- Results use the full MCP registry setting shown above.",
            "- Benchmark modes and their explicit/defaulted provenance are reported "
            "in separate metric rows; grounded and offline/replay results are never "
            "silently pooled.",
            "- Benchmark families must not be silently pooled; use the family table for "
            "research comparisons and treat the overall/domain tables as run summaries.",
            "- Final-outcome scoring uses the matcher names reported above. PR #29 "
            "Finance table rows may use `finance_query_table_rows_v1`, while other "
            "rows may use `recursive_json_subset_v1`; this valid mixture is reported "
            "rather than rejected.",
            "- Llama Coding includes outcomes from calls whose scalar strings were "
            "coerced by MCP/Pydantic at runtime; SVCA is intentionally unavailable.",
            "- Enterprise tau2 single-step rows are adapted standalone action routing, "
            "not autonomous tau2 task success.",
            "- Exact Reference Argument Match, runtime execution, no-call behavior, "
            "coercion, aliases, and canonicalization remain diagnostics rather than "
            "headline correctness metrics.",
            "",
        ]
    )


def build_scorecard(run_directories: Sequence[Path]) -> str:
    return render_markdown(load_runs(run_directories))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a reporting-only final-outcome/tool-selection scorecard with "
            "schema-valid calls marked unavailable."
        ),
    )
    parser.add_argument("run_directories", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    markdown = build_scorecard(args.run_directories)
    if args.output is None:
        print(markdown, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(markdown, encoding="utf-8")
        print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
