"""Inventory active LayerMCP benchmark datasets without loading fixtures."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any, Iterable

from evaluation.evaluate import DEFAULT_BENCHMARK_MODE, DEFAULT_WORKFLOW_EXECUTION_MODE


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_DOMAINS = ("math", "enterprise", "coding", "finance")


def discover_benchmark_files(project_root: Path = PROJECT_ROOT) -> list[Path]:
    benchmark_root = project_root / "benchmark"
    paths: list[Path] = []
    for domain in BENCHMARK_DOMAINS:
        domain_root = benchmark_root / domain
        if not domain_root.exists():
            continue
        for path in domain_root.rglob("*.json"):
            if path.name == "mathqa_operation_mapping.json":
                continue
            relative_parts = path.relative_to(benchmark_root).parts
            if any(
                part in {"archive", "fixtures", "__pycache__"}
                or "pycache" in part.lower()
                for part in relative_parts
            ):
                continue
            paths.append(path)
    return sorted(paths)


def infer_benchmark_class(path: Path, rows: list[dict[str, Any]]) -> str:
    # Checkout directory names are not benchmark evidence. In particular, a
    # temporary parent directory containing "smoke" must not change a file's
    # classification.
    values = [path.name.lower()]
    for row in rows:
        for key in (
            "source",
            "source_dataset",
            "provenance_type",
            "benchmark_mode",
            "task_type",
            "notes",
        ):
            value = row.get(key)
            if value is not None:
                values.append(str(value).lower())
    evidence = " ".join(values)

    if "offline_trace_replay" in evidence or "replay" in path.name.lower():
        return "replay/offline"
    if "smoke" in path.name.lower() or "smoke" in evidence:
        return "smoke"
    if "controlled" in path.name.lower() or "controlled_synthetic" in evidence:
        return "controlled"
    if "diagnostic" in evidence or "adapted" in path.name.lower() or "public_adapted" in evidence:
        return "diagnostic/adapted"
    is_workflow = any(
        row.get("task_type") == "multi_step_tool_routing"
        or bool(row.get("expected_steps"))
        for row in rows
    )
    if is_workflow and any(
        token in evidence for token in ("public", "source-derived", "derived")
    ):
        return "public/source-derived workflow"
    if is_workflow:
        return "workflow"
    if any(token in evidence for token in ("public", "source-derived", "derived")):
        return "public/source-derived"
    return "unknown"


def _value_counts(rows: Iterable[dict[str, Any]], key: str) -> list[str]:
    return sorted({str(row[key]) for row in rows if row.get(key) is not None})


def _answer_counts(items: Iterable[dict[str, Any]]) -> dict[str, int]:
    counts = {"populated": 0, "null": 0, "missing": 0}
    for item in items:
        if "expected_answer" not in item:
            counts["missing"] += 1
        elif item["expected_answer"] is None:
            counts["null"] += 1
        else:
            counts["populated"] += 1
    return counts


def _is_multistep(row: dict[str, Any]) -> bool:
    return row.get("task_type") == "multi_step_tool_routing"


def summarize_file(
    path: Path,
    *,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not all(isinstance(row, dict) for row in payload):
        raise ValueError(f"Benchmark must be a JSON list of objects: {path}")
    rows: list[dict[str, Any]] = payload
    steps = [
        step
        for row in rows
        for step in row.get("expected_steps", [])
        if isinstance(step, dict)
    ]
    multistep_rows = [row for row in rows if _is_multistep(row)]
    single_step_rows = [
        row for row in rows if row.get("task_type") == "single_tool_routing"
    ]

    expected_tools = Counter(
        str(row["expected_tool"])
        for row in single_step_rows
        if row.get("expected_tool") is not None
    )
    expected_tools.update(
        str(step["expected_tool"])
        for step in steps
        if step.get("expected_tool") is not None
    )

    domains = _value_counts(rows, "domain")
    task_types = _value_counts(rows, "task_type")
    warnings: list[str] = []
    missing_domains = sum(not row.get("domain") for row in rows)
    missing_task_types = sum(not row.get("task_type") for row in rows)
    empty_multistep = sum(
        _is_multistep(row) and not row.get("expected_steps") for row in rows
    )
    if missing_domains:
        warnings.append(f"{missing_domains} row(s) have missing domain")
    if len(domains) > 1:
        warnings.append(f"inconsistent domain values: {domains}")
    if missing_task_types:
        warnings.append(f"{missing_task_types} row(s) have missing task_type")
    if len(task_types) > 1:
        warnings.append(f"inconsistent task_type values: {task_types}")
    if empty_multistep:
        warnings.append(
            f"{empty_multistep} multi-step row(s) have zero expected_steps"
        )
    if (
        any(token in path.name.lower() for token in ("fixture", "replay"))
        and (len(rows) >= 100 or path.stat().st_size >= 5_000_000)
    ):
        warnings.append(
            "suspicious large fixture/replay-like file is active "
            f"({len(rows)} rows, {path.stat().st_size} bytes)"
        )

    benchmark_modes = sorted(
        {
            str(row.get("benchmark_mode") or DEFAULT_BENCHMARK_MODE)
            for row in rows
        }
    )
    workflow_execution_modes = sorted(
        {
            str(
                row.get("workflow_execution_mode")
                or DEFAULT_WORKFLOW_EXECUTION_MODE
            )
            for row in rows
        }
    )
    relative_path = path.relative_to(project_root).as_posix()
    return {
        "path": relative_path,
        "domain_values": domains,
        "task_type_values": task_types,
        "row_count": len(rows),
        "workflow_count": len(multistep_rows),
        "expected_step_count": len(steps),
        "single_step_count": len(single_step_rows),
        "multi_step_workflow_count": len(multistep_rows),
        "source_values": _value_counts(rows, "source"),
        "source_dataset_values": _value_counts(rows, "source_dataset"),
        "benchmark_mode_values": benchmark_modes,
        "workflow_execution_mode_values": workflow_execution_modes,
        "source_action_role_values": sorted(
            set(_value_counts(rows, "source_action_role"))
            | set(_value_counts(steps, "source_action_role"))
        ),
        "top_level_expected_answer": _answer_counts(rows),
        "step_level_expected_answer": _answer_counts(steps),
        "top_level_prompt_context_non_empty_count": sum(
            bool(str(row.get("prompt_context") or "").strip()) for row in rows
        ),
        "step_level_prompt_context_non_empty_count": sum(
            bool(str(step.get("prompt_context") or "").strip()) for step in steps
        ),
        "depends_on_length_distribution": dict(
            sorted(
                Counter(
                    len(step.get("depends_on", []))
                    if isinstance(step.get("depends_on", []), list)
                    else 0
                    for step in steps
                ).items()
            )
        ),
        "expected_tool_distribution": dict(sorted(expected_tools.items())),
        "inferred_benchmark_class": infer_benchmark_class(path, rows),
        "warnings": warnings,
    }


def _empty_metrics() -> dict[str, int]:
    return {
        "file_count": 0,
        "row_count": 0,
        "workflow_count": 0,
        "expected_step_count": 0,
        "single_step_count": 0,
        "multi_step_workflow_count": 0,
    }


def _add_row(target: dict[str, int], row: dict[str, Any]) -> None:
    steps = [
        step for step in row.get("expected_steps", []) if isinstance(step, dict)
    ]
    is_multistep = _is_multistep(row)
    target["row_count"] += 1
    target["workflow_count"] += int(is_multistep)
    target["expected_step_count"] += len(steps)
    target["single_step_count"] += int(
        row.get("task_type") == "single_tool_routing"
    )
    target["multi_step_workflow_count"] += int(is_multistep)


def build_inventory(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    discovered_paths = discover_benchmark_files(project_root)
    all_summaries = [
        summarize_file(path, project_root=project_root) for path in discovered_paths
    ]
    summaries = [summary for summary in all_summaries if summary["row_count"] > 0]
    placeholders = [
        {
            **summary,
            "warnings": [
                *summary["warnings"],
                "empty placeholder excluded from runnable active totals",
            ],
        }
        for summary in all_summaries
        if summary["row_count"] == 0
    ]
    by_domain: dict[str, dict[str, int]] = defaultdict(_empty_metrics)
    by_task_type: dict[str, dict[str, int]] = defaultdict(_empty_metrics)
    by_class: dict[str, dict[str, int]] = defaultdict(_empty_metrics)
    domain_step_kind: dict[str, Counter[str]] = defaultdict(Counter)
    domain_files: dict[str, set[str]] = defaultdict(set)
    task_type_files: dict[str, set[str]] = defaultdict(set)
    class_files: dict[str, set[str]] = defaultdict(set)
    directory_domain_names = {
        "math": "mathematics",
        "enterprise": "enterprise_automation",
        "coding": "coding",
        "finance": "finance",
    }

    summaries_by_path = {summary["path"]: summary for summary in summaries}
    for path in discovered_paths:
        relative_path = path.relative_to(project_root).as_posix()
        if relative_path not in summaries_by_path:
            continue
        path_parts = Path(relative_path).parts
        directory_domain = (
            directory_domain_names.get(path_parts[1], "<missing>")
            if len(path_parts) > 1
            else "<missing>"
        )
        rows = json.loads(path.read_text(encoding="utf-8"))
        for row in rows:
            domain = str(row.get("domain") or directory_domain)
            task_type = str(row.get("task_type") or "<missing>")
            benchmark_class = infer_benchmark_class(Path(relative_path), [row])
            _add_row(by_domain[domain], row)
            _add_row(by_task_type[task_type], row)
            _add_row(by_class[benchmark_class], row)
            domain_files[domain].add(relative_path)
            task_type_files[task_type].add(relative_path)
            class_files[benchmark_class].add(relative_path)
            domain_step_kind[domain]["single_step"] += int(
                task_type == "single_tool_routing"
            )
            domain_step_kind[domain]["multi_step"] += int(
                task_type == "multi_step_tool_routing"
            )

    for buckets, files in (
        (by_domain, domain_files),
        (by_task_type, task_type_files),
        (by_class, class_files),
    ):
        for value, metrics in buckets.items():
            metrics["file_count"] = len(files[value])

    return {
        "files": summaries,
        "placeholders": placeholders,
        "aggregates": {
            "by_domain": dict(sorted(by_domain.items())),
            "by_task_type": dict(sorted(by_task_type.items())),
            "by_inferred_benchmark_class": dict(sorted(by_class.items())),
            "by_domain_x_single_step_multi_step": {
                domain: dict(sorted(counts.items()))
                for domain, counts in sorted(domain_step_kind.items())
            },
            "total_active_files": len(summaries),
            "total_placeholder_files": len(placeholders),
            "total_active_rows": sum(item["row_count"] for item in summaries),
            "total_workflows": sum(item["workflow_count"] for item in summaries),
            "total_expected_steps": sum(
                item["expected_step_count"] for item in summaries
            ),
        },
    }


def render_markdown(inventory: dict[str, Any]) -> str:
    aggregates = inventory["aggregates"]
    lines = [
        "# Active Benchmark Inventory",
        "",
        f"- Active files: {aggregates['total_active_files']}",
        f"- Empty placeholders: {aggregates['total_placeholder_files']}",
        f"- Active rows: {aggregates['total_active_rows']}",
        f"- Multi-step workflows: {aggregates['total_workflows']}",
        f"- Expected steps: {aggregates['total_expected_steps']}",
        "",
        "| Path | Class | Rows | Single | Workflows | Steps | Warnings |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for item in inventory["files"]:
        warnings = "; ".join(item["warnings"])
        lines.append(
            f"| `{item['path']}` | {item['inferred_benchmark_class']} | "
            f"{item['row_count']} | {item['single_step_count']} | "
            f"{item['workflow_count']} | {item['expected_step_count']} | "
            f"{warnings} |"
        )
    if inventory["placeholders"]:
        lines.extend(["", "## Excluded Empty Placeholders", ""])
        for item in inventory["placeholders"]:
            lines.append(f"- `{item['path']}`")
    for heading, key in (
        ("By Domain", "by_domain"),
        ("By Task Type", "by_task_type"),
        ("By Inferred Benchmark Class", "by_inferred_benchmark_class"),
    ):
        lines.extend(
            [
                "",
                f"## {heading}",
                "",
                "| Value | Files | Rows | Single | Workflows | Steps |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for value, counts in aggregates[key].items():
            lines.append(
                f"| {value} | {counts['file_count']} | {counts['row_count']} | "
                f"{counts['single_step_count']} | {counts['workflow_count']} | "
                f"{counts['expected_step_count']} |"
            )
    lines.extend(
        [
            "",
            "## By Domain and Step Kind",
            "",
            "| Domain | Single-step rows | Multi-step workflows |",
            "| --- | ---: | ---: |",
        ]
    )
    for domain, counts in aggregates[
        "by_domain_x_single_step_multi_step"
    ].items():
        lines.append(
            f"| {domain} | {counts.get('single_step', 0)} | "
            f"{counts.get('multi_step', 0)} |"
        )
    return "\n".join(lines) + "\n"


def _write(path: str | None, content: str) -> None:
    if path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(content, encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Print JSON to stdout.")
    parser.add_argument("--markdown-out", help="Write Markdown inventory to PATH.")
    parser.add_argument("--json-out", help="Write JSON inventory to PATH.")
    args = parser.parse_args(argv)

    inventory = build_inventory()
    json_text = json.dumps(inventory, ensure_ascii=True, indent=2) + "\n"
    markdown_text = render_markdown(inventory)
    _write(args.markdown_out, markdown_text)
    _write(args.json_out, json_text)
    print(json_text if args.json else markdown_text, end="")


if __name__ == "__main__":
    main()
