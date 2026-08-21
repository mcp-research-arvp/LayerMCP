"""Build the pinned FinQA test-split expansion used by LayerMCP.

The builder deliberately does not download data. Pass the official FinQA
``dataset/test.json`` file from the pinned revision with ``--source-test``.
It intentionally excludes 15 long-context source rows and writes:

* 642 one-operation, single-tool samples;
* 490 multi-operation, multi-call samples; and
* a compact 1,753-row operation/result fixture.

Each multi-step call is a mechanical adaptation of one exact gold program
operation. Operation zero is replayed through ``finance_query_table`` so the
workflow begins with fixture-backed evidence. Later arithmetic operations use
``calculator`` with ``#N`` references replaced by their prior numeric results;
later ``greater`` or ``table_*`` operations remain fixture-backed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import operator
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.finance.grounding import (  # noqa: E402
    calculator_prompt_context,
    table_query_prompt_context,
)

SOURCE_DATASET = "FinQA"
SOURCE_REPOSITORY = "czyssrs/FinQA"
SOURCE_REPOSITORY_URL = "https://github.com/czyssrs/FinQA"
SOURCE_REVISION = "0f16e2867befa6840783e58be38c9efb9229d742"
SOURCE_FILE_SHA256 = "831dbfb2e785dbc227f895ce3f24046433467aec67b09db2bd6ac7692a8a30dc"
SOURCE_URL = (
    f"https://github.com/czyssrs/FinQA/blob/{SOURCE_REVISION}/dataset/test.json"
)
SOURCE_LICENSE = "MIT"
SOURCE_COPYRIGHT = "Copyright (c) 2021 Zhiyu Chen"
SOURCE_LICENSE_URL = f"https://github.com/czyssrs/FinQA/blob/{SOURCE_REVISION}/LICENSE"
SOURCE_PAPER_URL = "https://aclanthology.org/2021.emnlp-main.300/"

SOURCE_EXAMPLE_COUNT = 1_147
SINGLE_OPERATION_COUNT = 642
MULTI_OPERATION_COUNT = 490
OPERATION_ROW_COUNT = 1_753
MAX_COMPACT_FIXTURE_ROWS = 10_000

FIXTURE_DATASET_ID = "finqa-public-test-program-results-v1"
SINGLE_OUTPUT_NAME = "finance_finqa_test_single.json"
MULTISTEP_OUTPUT_NAME = "finance_finqa_test_multistep.json"
FIXTURE_OUTPUT_NAME = "fixtures/finqa_test_program_results_cells.json"
CANONICAL_FIXTURE_FILE = (
    "benchmark/finance/fixtures/finqa_test_program_results_cells.json"
)
FIXTURE_COLUMNS = [
    {"name": "source_split", "type": "TEXT"},
    {"name": "source_row_index", "type": "INTEGER"},
    {"name": "source_id", "type": "TEXT"},
    {"name": "operation_index", "type": "INTEGER"},
    {"name": "operation", "type": "TEXT"},
    {"name": "operator", "type": "TEXT"},
    {"name": "arg1", "type": "TEXT"},
    {"name": "arg2", "type": "TEXT"},
    {"name": "source_result", "type": "TEXT"},
    {"name": "numeric_result", "type": "REAL"},
    {"name": "text_result", "type": "TEXT"},
    {"name": "is_final", "type": "INTEGER"},
]
FIXTURE_GROUNDING_TABLE = {
    "dataset_id": FIXTURE_DATASET_ID,
    "columns": FIXTURE_COLUMNS,
    "rows": [],
}

INTENTIONALLY_EXCLUDED_LONG_CONTEXT_FINQA_SOURCE_INDICES = frozenset({
    0,
    1,
    8,
    9,
    10,
    32,
    42,
    44,
    45,
    78,
    92,
    105,
    172,
    264,
    347,
})
FINQA_EXCLUSION_POLICY = (
    "These 15 FinQA test rows are intentionally outside the active benchmark "
    "because their table-grounding contexts exceed the bounded inference "
    "budget used for reproducible local evaluation."
)

_OPERATION_PATTERN = re.compile(r"^([a-z_]+)\((.*)\)$")
_REFERENCE_PATTERN = re.compile(r"^#(\d+)$")
_NUMERIC_LITERAL_PATTERN = re.compile(r"^-?(?:\d+(?:\.\d*)?|\.\d+)%?$")
_SUPPORTED_OPERATORS = {
    "add",
    "divide",
    "exp",
    "greater",
    "multiply",
    "subtract",
    "table_average",
    "table_max",
    "table_min",
    "table_sum",
}
_CALCULATOR_SYMBOLS = {
    "add": "+",
    "divide": "/",
    "exp": "**",
    "multiply": "*",
    "subtract": "-",
}
_ARITHMETIC_FUNCTIONS: dict[str, Callable[[int | float, int | float], int | float]] = {
    "add": operator.add,
    "divide": operator.truediv,
    "exp": operator.pow,
    "multiply": operator.mul,
    "subtract": operator.sub,
}


@dataclass(frozen=True)
class GoldOperation:
    """One parsed and deterministically evaluated FinQA gold operation."""

    index: int
    canonical: str
    operator: str
    arg1: str
    arg2: str
    source_result: str
    runtime_result: int | float | str
    dependencies: tuple[int, ...]


@dataclass(frozen=True)
class BuildResult:
    """Paths and counts produced by :func:`build_expansion`."""

    single_path: Path
    multistep_path: Path
    fixture_path: Path
    single_count: int
    multistep_count: int
    operation_count: int


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"JSON source does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc


def _source_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except FileNotFoundError as exc:
        raise ValueError(f"FinQA source file does not exist: {path}") from exc
    return digest.hexdigest()


def _split_top_level_operations(program: str) -> list[str]:
    if not isinstance(program, str) or not program.strip():
        raise ValueError("FinQA gold program must be a non-empty string.")

    operations: list[str] = []
    depth = 0
    start = 0
    for index, character in enumerate(program):
        if character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
            if depth < 0:
                raise ValueError(f"Unbalanced FinQA program: {program!r}")
        elif character == "," and depth == 0:
            operation = program[start:index].strip()
            if not operation:
                raise ValueError(f"Empty operation in FinQA program: {program!r}")
            operations.append(operation)
            start = index + 1

    if depth != 0:
        raise ValueError(f"Unbalanced FinQA program: {program!r}")
    final_operation = program[start:].strip()
    if not final_operation:
        raise ValueError(f"Empty final operation in FinQA program: {program!r}")
    operations.append(final_operation)
    return operations


def _parse_operation(canonical: str) -> tuple[str, str, str]:
    match = _OPERATION_PATTERN.fullmatch(canonical)
    if match is None:
        raise ValueError(f"Unsupported FinQA operation syntax: {canonical!r}")
    operation_name, raw_arguments = match.groups()
    if operation_name not in _SUPPORTED_OPERATORS:
        raise ValueError(f"Unsupported FinQA operator: {operation_name!r}")

    arguments = [argument.strip() for argument in raw_arguments.split(",", 1)]
    if len(arguments) != 2 or not all(arguments):
        raise ValueError(
            f"FinQA operation must contain exactly two arguments: {canonical!r}"
        )
    return operation_name, arguments[0], arguments[1]


def _literal_value(token: str) -> int | float:
    if token == "const_m1":
        return -1
    if token.startswith("const_"):
        constant = token.removeprefix("const_")
        if not constant.isdigit():
            raise ValueError(f"Unsupported FinQA constant: {token!r}")
        return int(constant)

    if not _NUMERIC_LITERAL_PATTERN.fullmatch(token):
        raise ValueError(f"Unsupported FinQA numeric argument: {token!r}")
    if token.endswith("%"):
        return float(token[:-1]) / 100
    if "." in token:
        return float(token)
    return int(token)


def _resolve_numeric_argument(
    token: str,
    prior_results: Sequence[int | float | str],
) -> int | float:
    reference = _REFERENCE_PATTERN.fullmatch(token)
    if reference is None:
        return _literal_value(token)

    result_index = int(reference.group(1))
    if result_index >= len(prior_results):
        raise ValueError(
            f"FinQA reference {token!r} does not identify a prior operation."
        )
    result = prior_results[result_index]
    if isinstance(result, bool) or not isinstance(result, (int, float)):
        raise ValueError(
            f"FinQA reference {token!r} does not contain a numeric result."
        )
    return result


def _parse_source_numeric_result(value: Any) -> int | float:
    text = str(value).strip().replace(",", "")
    if text.endswith("%"):
        return float(text[:-1]) / 100
    if not _NUMERIC_LITERAL_PATTERN.fullmatch(text):
        raise ValueError(f"Unsupported FinQA source result: {value!r}")
    if "." in text:
        return float(text)
    return int(text)


def _final_result_matches(actual: int | float | str, expected: Any) -> bool:
    if isinstance(expected, str):
        return isinstance(actual, str) and actual == expected
    if isinstance(expected, bool) or not isinstance(expected, (int, float)):
        return False
    if isinstance(actual, bool) or not isinstance(actual, (int, float)):
        return False
    return math.isclose(
        round(float(actual), 5),
        float(expected),
        rel_tol=0.0,
        abs_tol=1e-5,
    )


def _gold_operations(qa: dict[str, Any]) -> list[GoldOperation]:
    program = qa.get("program")
    canonical_operations = _split_top_level_operations(program)
    source_steps = qa.get("steps")
    if not isinstance(source_steps, list) or len(source_steps) != len(
        canonical_operations
    ):
        raise ValueError(
            "FinQA qa.steps must align one-to-one with the gold program "
            f"operations: {program!r}."
        )

    prior_results: list[int | float | str] = []
    operations: list[GoldOperation] = []
    for operation_index, (canonical, source_step) in enumerate(
        zip(canonical_operations, source_steps, strict=True)
    ):
        if not isinstance(source_step, dict) or "res" not in source_step:
            raise ValueError(
                f"FinQA step {operation_index} has no published result: {program!r}."
            )
        operation_name, arg1, arg2 = _parse_operation(canonical)
        dependencies = tuple(
            sorted(
                {
                    int(reference.group(1))
                    for token in (arg1, arg2)
                    if (reference := _REFERENCE_PATTERN.fullmatch(token)) is not None
                }
            )
        )
        if any(dependency >= operation_index for dependency in dependencies):
            raise ValueError(
                f"FinQA operation {canonical!r} has a future/self reference."
            )

        is_final = operation_index == len(canonical_operations) - 1
        if operation_name.startswith("table_"):
            if is_final:
                execution_answer = qa.get("exe_ans")
                if isinstance(execution_answer, str):
                    runtime_result: int | float | str = execution_answer
                elif isinstance(execution_answer, (int, float)) and not isinstance(
                    execution_answer, bool
                ):
                    runtime_result = execution_answer
                else:
                    raise ValueError(
                        f"Unsupported FinQA execution answer: {execution_answer!r}"
                    )
            else:
                runtime_result = _parse_source_numeric_result(source_step["res"])
        else:
            left = _resolve_numeric_argument(arg1, prior_results)
            right = _resolve_numeric_argument(arg2, prior_results)
            if operation_name == "greater":
                runtime_result = "yes" if left > right else "no"
            else:
                try:
                    runtime_result = _ARITHMETIC_FUNCTIONS[operation_name](left, right)
                except ZeroDivisionError as exc:
                    raise ValueError(
                        f"Division by zero in FinQA operation: {canonical!r}"
                    ) from exc
                if isinstance(runtime_result, complex) or not math.isfinite(
                    float(runtime_result)
                ):
                    raise ValueError(
                        f"Non-finite result in FinQA operation: {canonical!r}"
                    )

        prior_results.append(runtime_result)
        operations.append(
            GoldOperation(
                index=operation_index,
                canonical=canonical,
                operator=operation_name,
                arg1=arg1,
                arg2=arg2,
                source_result=str(source_step["res"]),
                runtime_result=runtime_result,
                dependencies=dependencies,
            )
        )

    if not _final_result_matches(prior_results[-1], qa.get("exe_ans")):
        raise ValueError(
            "Mechanically evaluated FinQA program does not match its pinned "
            f"execution answer: {program!r}; calculated={prior_results[-1]!r}, "
            f"expected={qa.get('exe_ans')!r}."
        )
    return operations


def _format_calculator_number(value: int | float | str) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Calculator dependency is not numeric: {value!r}")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"Calculator dependency is not finite: {value!r}")
    return repr(value)


def _calculator_expression(
    operation: GoldOperation,
    prior_results: Sequence[int | float | str],
) -> str:
    if operation.operator not in _CALCULATOR_SYMBOLS:
        raise ValueError(
            f"Operation cannot be mapped to calculator: {operation.canonical!r}"
        )

    def resolve_for_expression(token: str) -> str:
        reference = _REFERENCE_PATTERN.fullmatch(token)
        if reference is not None:
            reference_index = int(reference.group(1))
            if reference_index >= len(prior_results):
                raise ValueError(
                    f"FinQA reference {token!r} does not identify a prior result."
                )
            return _format_calculator_number(prior_results[reference_index])
        return _format_calculator_number(_literal_value(token))

    left = resolve_for_expression(operation.arg1)
    right = resolve_for_expression(operation.arg2)
    return f"({left}) {_CALCULATOR_SYMBOLS[operation.operator]} ({right})"


def _query_table_call(
    source_row_index: int,
    operation: GoldOperation,
) -> tuple[dict[str, Any], dict[str, Any]]:
    result_column = (
        "text_result" if isinstance(operation.runtime_result, str) else "numeric_result"
    )
    sql = (
        f"SELECT {result_column} AS result FROM data "
        f"WHERE source_row_index = {source_row_index} "
        f"AND operation_index = {operation.index}"
    )
    result: str | float
    if isinstance(operation.runtime_result, str):
        result = operation.runtime_result
    else:
        # The declared SQLite column is REAL, so normalize retrieved numbers to
        # floats even when the source operation happens to yield an integer.
        result = float(operation.runtime_result)
    return (
        {
            "dataset_id": FIXTURE_DATASET_ID,
            "sql": sql,
        },
        {
            "dataset_id": FIXTURE_DATASET_ID,
            "columns": ["result"],
            "rows": [[result]],
            "row_count": 1,
            "truncated": False,
        },
    )


def _calculator_call(
    operation: GoldOperation,
    prior_results: Sequence[int | float | str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    expression = _calculator_expression(operation, prior_results)
    return (
        {"expression": expression},
        {
            "expression": expression,
            "result": operation.runtime_result,
        },
    )


def _source_metadata(
    source_row_index: int,
    source_record: dict[str, Any],
    qa: dict[str, Any],
) -> dict[str, Any]:
    return {
        "source_dataset": SOURCE_DATASET,
        "source_repository": SOURCE_REPOSITORY,
        "source_revision": SOURCE_REVISION,
        "source_split": "test",
        "source_row_index": source_row_index,
        "source_id": source_record["id"],
        "source_filename": source_record.get("filename"),
        "source_program": qa["program"],
        "source_exe_answer": qa["exe_ans"],
        "source_answer": qa.get("answer"),
        "source_license": SOURCE_LICENSE,
        "source_copyright": SOURCE_COPYRIGHT,
        "source_url": SOURCE_URL,
        "source_file_sha256": SOURCE_FILE_SHA256,
        "source_license_url": SOURCE_LICENSE_URL,
        "source_paper_url": SOURCE_PAPER_URL,
        "query_origin": "official_research_dataset_question",
        "fixture_dataset_id": FIXTURE_DATASET_ID,
        "fixture_file": CANONICAL_FIXTURE_FILE,
    }


def _single_sample(
    source_row_index: int,
    source_record: dict[str, Any],
    qa: dict[str, Any],
    operation: GoldOperation,
) -> dict[str, Any]:
    expected_args, expected_answer = _query_table_call(source_row_index, operation)
    return {
        "id": f"finance_public_finqa_test_single_{source_row_index:04d}",
        "domain": "finance",
        "task_type": "single_tool_routing",
        "difficulty": "easy",
        "source": "public_finance_paper_derived",
        "query": qa["question"],
        "prompt_context": table_query_prompt_context(
            expected_args,
            FIXTURE_GROUNDING_TABLE,
        ),
        "expected_tool": "finance_query_table",
        "expected_args": expected_args,
        "expected_answer": expected_answer,
        "perturbation_type": "gold_program",
        "notes": (
            "Exact FinQA test question with its one-operation gold program "
            "replayed from the compact pinned result fixture."
        ),
        **_source_metadata(source_row_index, source_record, qa),
        "source_operation_count": 1,
        "source_operation": operation.canonical,
        "source_operation_result": operation.source_result,
        "provenance_type": "research_paper_dataset_adaptation",
        "adaptation_notes": (
            "The question is unchanged. The gold operation/result is stored "
            "as one fixture row and retrieved with finance_query_table."
        ),
    }


def _multistep_sample(
    source_row_index: int,
    source_record: dict[str, Any],
    qa: dict[str, Any],
    operations: Sequence[GoldOperation],
) -> dict[str, Any]:
    expected_steps: list[dict[str, Any]] = []
    prior_results: list[int | float | str] = []
    for operation in operations:
        use_query_table = (
            operation.index == 0
            or operation.operator == "greater"
            or operation.operator.startswith("table_")
        )
        if use_query_table:
            expected_tool = "finance_query_table"
            expected_args, expected_answer = _query_table_call(
                source_row_index, operation
            )
            prompt_context = table_query_prompt_context(
                expected_args,
                FIXTURE_GROUNDING_TABLE,
            )
        else:
            expected_tool = "calculator"
            expected_args, expected_answer = _calculator_call(operation, prior_results)
            prompt_context = calculator_prompt_context(expected_args)

        expected_steps.append(
            {
                "id": f"operation_{operation.index + 1:03d}",
                # This is the exact canonical operation published by FinQA,
                # not a generated follow-up question.
                "query": operation.canonical,
                "prompt_context": prompt_context,
                "expected_tool": expected_tool,
                "expected_args": expected_args,
                "expected_answer": expected_answer,
                "depends_on": [
                    f"operation_{dependency + 1:03d}"
                    for dependency in operation.dependencies
                ],
                "source_program": operation.canonical,
                "source_operator": operation.operator,
                "source_result": operation.source_result,
            }
        )
        prior_results.append(operation.runtime_result)

    return {
        "id": f"finance_public_finqa_test_multistep_{source_row_index:04d}",
        "domain": "finance",
        "task_type": "multi_step_tool_routing",
        "difficulty": "medium" if len(operations) == 2 else "hard",
        "source": "public_finance_paper_derived",
        "query": qa["question"],
        "expected_steps": expected_steps,
        "expected_final_answer": qa.get("answer"),
        "workflow_final_answer_contract": "finqa_execution_v1",
        "workflow_final_answer_expected": qa["exe_ans"],
        "perturbation_type": "gold_program_sequence",
        "notes": (
            "Exact FinQA test question with one ordered tool call per exact "
            "gold program operation."
        ),
        **_source_metadata(source_row_index, source_record, qa),
        "source_operation_count": len(operations),
        "source_execution_answers": [
            operation.source_result for operation in operations
        ],
        "tool_sequence_origin": "mechanical_adaptation_of_gold_program",
        "provenance_type": "research_paper_dataset_multistep_adaptation",
        "adaptation_notes": (
            "The top-level question and canonical program operations are "
            "unchanged. Operation zero and comparison/table operations query "
            "the compact fixture; later arithmetic operations call calculator "
            "after substituting prior #N results."
        ),
    }


def _fixture_row(
    source_row_index: int,
    source_record: dict[str, Any],
    operation: GoldOperation,
    operation_count: int,
) -> list[Any]:
    if isinstance(operation.runtime_result, str):
        numeric_result = None
        text_result = operation.runtime_result
    else:
        numeric_result = float(operation.runtime_result)
        text_result = None
    return [
        "test",
        source_row_index,
        source_record["id"],
        operation.index,
        operation.canonical,
        operation.operator,
        operation.arg1,
        operation.arg2,
        operation.source_result,
        numeric_result,
        text_result,
        int(operation.index == operation_count - 1),
    ]


def _fixture(
    rows: list[list[Any]],
    excluded_source_indices: set[int],
) -> dict[str, Any]:
    return {
        "dataset_id": FIXTURE_DATASET_ID,
        "description": (
            "One compact result row per gold operation for the 1,132 official "
            "FinQA test examples not already represented by LayerMCP."
        ),
        "columns": [dict(column) for column in FIXTURE_COLUMNS],
        "rows": rows,
        "provenance": {
            "source_dataset": SOURCE_DATASET,
            "source_repository": SOURCE_REPOSITORY_URL,
            "source_revision": SOURCE_REVISION,
            "source_url": SOURCE_URL,
            "source_file_sha256": SOURCE_FILE_SHA256,
            "source_split": "test",
            "source_license": SOURCE_LICENSE,
            "source_copyright": SOURCE_COPYRIGHT,
            "source_license_url": SOURCE_LICENSE_URL,
            "source_paper_url": SOURCE_PAPER_URL,
            "local_license_file": "benchmark/finance/fixtures/FINQA_LICENSE.txt",
            "excluded_source_row_indices": sorted(excluded_source_indices),
            "source_example_count": SOURCE_EXAMPLE_COUNT,
            "included_example_count": (SINGLE_OPERATION_COUNT + MULTI_OPERATION_COUNT),
            "single_operation_count": SINGLE_OPERATION_COUNT,
            "multi_operation_count": MULTI_OPERATION_COUNT,
            "operation_row_count": OPERATION_ROW_COUNT,
            "adaptation": (
                "Each included gold operation is retained verbatim with its "
                "arguments and published step result. numeric_result/text_result "
                "store the deterministic execution value used by the offline "
                "tool replay. No source table or filing text is copied."
            ),
        },
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def build_expansion(source_test: Path, output_root: Path) -> BuildResult:
    """Build and write the three deterministic FinQA expansion artifacts."""

    actual_source_hash = _source_sha256(source_test)
    if actual_source_hash != SOURCE_FILE_SHA256:
        raise ValueError(
            "FinQA source hash does not match the pinned official test split: "
            f"expected {SOURCE_FILE_SHA256}, got {actual_source_hash}."
        )

    source_records = _load_json(source_test)
    if not isinstance(source_records, list) or len(source_records) != (
        SOURCE_EXAMPLE_COUNT
    ):
        actual_count = len(source_records) if isinstance(source_records, list) else None
        raise ValueError(
            "Pinned FinQA test split must contain exactly "
            f"{SOURCE_EXAMPLE_COUNT} examples; found {actual_count!r}."
        )

    excluded_source_indices = INTENTIONALLY_EXCLUDED_LONG_CONTEXT_FINQA_SOURCE_INDICES
    single_samples: list[dict[str, Any]] = []
    multistep_samples: list[dict[str, Any]] = []
    fixture_rows: list[list[Any]] = []

    for source_row_index, source_record in enumerate(source_records):
        if source_row_index in excluded_source_indices:
            continue
        if not isinstance(source_record, dict) or not isinstance(
            source_record.get("qa"), dict
        ):
            raise ValueError(f"FinQA source row {source_row_index} has no qa object.")
        if not isinstance(source_record.get("id"), str):
            raise ValueError(
                f"FinQA source row {source_row_index} has no stable source ID."
            )

        qa = source_record["qa"]
        if not isinstance(qa.get("question"), str):
            raise ValueError(
                f"FinQA source row {source_row_index} has no question string."
            )
        operations = _gold_operations(qa)
        fixture_rows.extend(
            _fixture_row(
                source_row_index,
                source_record,
                operation,
                len(operations),
            )
            for operation in operations
        )
        if len(operations) == 1:
            single_samples.append(
                _single_sample(
                    source_row_index,
                    source_record,
                    qa,
                    operations[0],
                )
            )
        else:
            multistep_samples.append(
                _multistep_sample(
                    source_row_index,
                    source_record,
                    qa,
                    operations,
                )
            )

    if len(single_samples) != SINGLE_OPERATION_COUNT:
        raise ValueError(
            f"Expected {SINGLE_OPERATION_COUNT} one-operation rows, "
            f"generated {len(single_samples)}."
        )
    if len(multistep_samples) != MULTI_OPERATION_COUNT:
        raise ValueError(
            f"Expected {MULTI_OPERATION_COUNT} multi-operation rows, "
            f"generated {len(multistep_samples)}."
        )
    if len(fixture_rows) != OPERATION_ROW_COUNT:
        raise ValueError(
            f"Expected {OPERATION_ROW_COUNT} operation fixture rows, "
            f"generated {len(fixture_rows)}."
        )
    if len(fixture_rows) >= MAX_COMPACT_FIXTURE_ROWS:
        raise ValueError(
            f"Generated FinQA fixture is no longer compact: {len(fixture_rows)} rows."
        )

    source_142 = next(
        (sample for sample in multistep_samples if sample["source_row_index"] == 142),
        None,
    )
    if source_142 is None or [
        step["expected_tool"] for step in source_142["expected_steps"]
    ] != ["finance_query_table", "finance_query_table"]:
        raise ValueError(
            "Pinned FinQA source row 142 must keep its later greater operation "
            "on finance_query_table."
        )

    single_path = output_root / SINGLE_OUTPUT_NAME
    multistep_path = output_root / MULTISTEP_OUTPUT_NAME
    fixture_path = output_root / FIXTURE_OUTPUT_NAME
    _write_json(single_path, single_samples)
    _write_json(multistep_path, multistep_samples)
    _write_json(
        fixture_path,
        _fixture(fixture_rows, excluded_source_indices),
    )
    return BuildResult(
        single_path=single_path,
        multistep_path=multistep_path,
        fixture_path=fixture_path,
        single_count=len(single_samples),
        multistep_count=len(multistep_samples),
        operation_count=len(fixture_rows),
    )


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build the deterministic LayerMCP expansion from the pinned "
            "official FinQA test.json file."
        )
    )
    parser.add_argument(
        "--source-test",
        required=True,
        type=Path,
        help="Path to FinQA dataset/test.json at the pinned source revision.",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        type=Path,
        help=(
            "Destination finance benchmark directory. The builder writes two "
            "benchmark JSON files and one fixtures/*.json file below it."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _argument_parser()
    arguments = parser.parse_args(argv)
    try:
        result = build_expansion(arguments.source_test, arguments.output_root)
    except ValueError as exc:
        parser.error(str(exc))

    print(
        "Built FinQA expansion: "
        f"{result.single_count} single-tool samples, "
        f"{result.multistep_count} multi-step samples, "
        f"{result.operation_count} operation fixture rows."
    )
    print(result.single_path)
    print(result.multistep_path)
    print(result.fixture_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
