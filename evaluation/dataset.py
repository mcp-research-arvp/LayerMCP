from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

NO_TOOL_NAME = "hallucinated_tool"

_TOOL_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_DIFFICULTIES = {"easy", "medium", "hard"}
_REQUIRED_FIELDS = {
    "id",
    "domain",
    "query",
    "tools",
    "expected_tool",
    "expected_arguments",
    "expected_result",
    "difficulty",
}


class BenchmarkValidationError(ValueError):
    """Raised when a benchmark row does not match the expected schema."""


@dataclass(frozen=True)
class BenchmarkSample:
    id: str
    domain: str
    query: str
    tools: tuple[str, ...]
    expected_tool: str
    expected_arguments: dict[str, Any]
    expected_result: Any
    difficulty: str

    @property
    def expects_tool_call(self) -> bool:
        return self.expected_tool != NO_TOOL_NAME


def load_benchmark(path: Path) -> list[BenchmarkSample]:
    if not path.exists():
        raise FileNotFoundError(
            f"Benchmark dataset not found: {path}. "
            "Create benchmark/tool_routing.jsonl or update --dataset."
        )

    records = _read_records(path)
    samples = [_validate_record(record, index) for index, record in records]
    _validate_unique_ids(samples)
    return samples


def _read_records(path: Path) -> list[tuple[int, Any]]:
    if path.suffix == ".jsonl":
        return _read_jsonl(path)
    if path.suffix == ".json":
        return _read_json_list(path)

    raise ValueError(f"Unsupported benchmark file extension: {path.suffix}")


def _read_jsonl(path: Path) -> list[tuple[int, Any]]:
    records: list[tuple[int, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue

            try:
                records.append((line_number, json.loads(stripped)))
            except json.JSONDecodeError as exc:
                raise BenchmarkValidationError(
                    f"Invalid JSON on line {line_number}: {exc.msg}"
                ) from exc

    if not records:
        raise BenchmarkValidationError("Benchmark JSONL file must not be empty.")

    return records


def _read_json_list(path: Path) -> list[tuple[int, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        dataset = json.load(handle)

    if not isinstance(dataset, list):
        raise BenchmarkValidationError("Benchmark JSON file must contain a list.")

    return list(enumerate(dataset, start=1))


def _validate_record(record: Any, index: int) -> BenchmarkSample:
    if not isinstance(record, dict):
        raise BenchmarkValidationError(f"Row {index} must be a JSON object.")

    missing = sorted(_REQUIRED_FIELDS - set(record))
    if missing:
        raise BenchmarkValidationError(
            f"Row {index} is missing required field(s): {', '.join(missing)}"
        )

    sample_id = _required_string(record, "id", index)
    domain = _required_string(record, "domain", index)
    query = _required_string(record, "query", index)
    expected_tool = _required_string(record, "expected_tool", index)
    difficulty = _required_string(record, "difficulty", index)

    if difficulty not in _DIFFICULTIES:
        raise BenchmarkValidationError(
            f"Row {index} difficulty must be one of: "
            f"{', '.join(sorted(_DIFFICULTIES))}"
        )

    tools = _validate_tools(record["tools"], index)
    expected_arguments = _validate_expected_arguments(
        record["expected_arguments"],
        index,
    )

    if expected_tool != NO_TOOL_NAME and expected_tool not in tools:
        raise BenchmarkValidationError(
            f"Row {index} expected_tool must be listed in tools or be "
            f"{NO_TOOL_NAME!r}."
        )

    if expected_tool == NO_TOOL_NAME and expected_arguments:
        raise BenchmarkValidationError(
            f"Row {index} expected_arguments must be empty when no tool is expected."
        )

    return BenchmarkSample(
        id=sample_id,
        domain=domain,
        query=query,
        tools=tools,
        expected_tool=expected_tool,
        expected_arguments=expected_arguments,
        expected_result=record["expected_result"],
        difficulty=difficulty,
    )


def _required_string(record: dict[str, Any], field: str, index: int) -> str:
    value = record[field]
    if not isinstance(value, str) or not value.strip():
        raise BenchmarkValidationError(
            f"Row {index} field {field!r} must be a non-empty string."
        )
    return value.strip()


def _validate_tools(value: Any, index: int) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise BenchmarkValidationError(
            f"Row {index} field 'tools' must be a non-empty list."
        )

    tools: list[str] = []
    for tool in value:
        if not isinstance(tool, str) or not _TOOL_NAME_PATTERN.fullmatch(tool):
            raise BenchmarkValidationError(
                f"Row {index} contains invalid tool name: {tool!r}."
            )
        tools.append(tool)

    duplicates = sorted({tool for tool in tools if tools.count(tool) > 1})
    if duplicates:
        raise BenchmarkValidationError(
            f"Row {index} contains duplicate tool(s): {', '.join(duplicates)}"
        )

    return tuple(tools)


def _validate_expected_arguments(value: Any, index: int) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise BenchmarkValidationError(
            f"Row {index} field 'expected_arguments' must be an object."
        )

    for key in value:
        if not isinstance(key, str) or not key:
            raise BenchmarkValidationError(
                f"Row {index} expected argument keys must be non-empty strings."
            )

    return dict(value)


def _validate_unique_ids(samples: list[BenchmarkSample]) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for sample in samples:
        if sample.id in seen:
            duplicates.add(sample.id)
        seen.add(sample.id)

    if duplicates:
        raise BenchmarkValidationError(
            f"Benchmark sample id(s) must be unique: {', '.join(sorted(duplicates))}"
        )
