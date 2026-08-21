"""Build LayerMCP's pinned ten-conversation ConvFinQA benchmark.

The builder does not download data. Pass ``data.zip`` from the exact pinned
ConvFinQA revision. It verifies the archive before reading ``data/dev.json``,
selects the ten fixed source rows below, normalizes the twenty cited evidence
cells, and mechanically translates all 35 cumulative turn programs into the
existing LayerMCP table-query and calculator contracts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.finance.grounding import (  # noqa: E402
    calculator_prompt_context,
    table_query_prompt_context,
)
from mcp_server.tool_impls import calculator  # noqa: E402


SOURCE_DATASET = "ConvFinQA"
SOURCE_REPOSITORY = "czyssrs/ConvFinQA"
SOURCE_REPOSITORY_URL = "https://github.com/czyssrs/ConvFinQA"
SOURCE_REVISION = "cf3eed2d5984960bf06bb8145bcea5e80b0222a6"
SOURCE_ARCHIVE = "data.zip"
SOURCE_ARCHIVE_SHA256 = (
    "d764271fae60d81b62e6d58dfc481807ebc8cfbcd633811241723c4a2101072a"
)
SOURCE_MEMBER = "data/dev.json"
SOURCE_MEMBER_SHA256 = (
    "9665dd60f106e5cd41d37537edac97bbd26eb7629ad63ca2390b19d4619ba51b"
)
SOURCE_SPLIT = "dev"
SOURCE_EXAMPLE_COUNT = 421
SOURCE_LICENSE = "MIT"

SELECTED_SOURCE_ROW_INDICES = (0, 2, 3, 6, 7, 8, 9, 10, 11, 13)
SELECTED_CONVERSATION_IDS = (
    "Single_MRO/2007/page_134.pdf-1",
    "Single_IPG/2009/page_85.pdf-3",
    "Single_UNP/2008/page_77.pdf-2",
    "Single_UNP/2014/page_35.pdf-3",
    "Single_DVN/2007/page_58.pdf-3",
    "Single_AON/2009/page_46.pdf-3",
    "Single_CDNS/2018/page_31.pdf-1",
    "Single_BLK/2014/page_119.pdf-1",
    "Single_ETR/2008/page_355.pdf-4",
    "Single_MO/2012/page_44.pdf-4",
)
WORKFLOW_COUNT = 10
STEP_COUNT = 35
EVIDENCE_ROW_COUNT = 20

FIXTURE_DATASET_ID = "convfinqa-dev-v1"
BENCHMARK_OUTPUT_NAME = "finance_convfinqa_multistep.json"
FIXTURE_OUTPUT_NAME = "fixtures/convfinqa_dev_cells.json"
SUBSET_OUTPUT_NAME = "fixtures/convfinqa_dev_source_subset.json"
MANIFEST_OUTPUT_NAME = "fixtures/convfinqa_build_manifest.json"

# Canonical JSON hash of the selected source-owned fields. This is independent
# of pretty-printing and detects changed questions, programs, answers, evidence,
# row indices, IDs, filenames, or turn selection.
EXPECTED_SOURCE_SUBSET_SHA256 = (
    "7cbacbe3958c747799ef697178489e309d51a444b9efa8ed30bcc27b29493c71"
)
# Canonical JSON hash of all 35 source-program/tool/argument/dependency mappings.
EXPECTED_TRANSLATION_SHA256 = (
    "4eacf7682b40c50bd6d411d71b123d17c082fe5bba762648c77a7d12779a75aa"
)
EXPECTED_BENCHMARK_SHA256 = (
    "f2f830f657e5684b3496fc15b951aebf22ba795519a8598d494f1128b3766ecc"
)
EXPECTED_FIXTURE_SHA256 = (
    "e15dbcc953395495657c074f937426090c469a9d4912d299ba0276d6b2e505f1"
)
EXPECTED_SUBSET_FILE_SHA256 = (
    "e256bb075346df471d62aa390fd66045c1f24fba6d10b5feaa899e7e88aa44e1"
)
ORIGIN_MAIN_MODEL_FACING_SHA256 = (
    "0ebaa05699c5dda75119dc38ffea1e14132bf5a413b8238c667a22bf7456c7f4"
)

FIXTURE_COLUMNS = [
    {"name": "source_split", "type": "TEXT"},
    {"name": "source_row_index", "type": "INTEGER"},
    {"name": "conversation_id", "type": "TEXT"},
    {"name": "filename", "type": "TEXT"},
    {"name": "metric", "type": "TEXT"},
    {"name": "period", "type": "TEXT"},
    {"name": "numeric_value", "type": "REAL"},
    {"name": "evidence_key", "type": "TEXT"},
    {"name": "evidence_text", "type": "TEXT"},
]


@dataclass(frozen=True)
class EvidenceSpec:
    source_row_index: int
    metric: str
    period: str
    numeric_value: float
    evidence_key: str


EVIDENCE_SPECS = (
    EvidenceSpec(0, "weighted average exercise price per share", "2005", 25.14, "table_1"),
    EvidenceSpec(0, "weighted average exercise price per share", "2007", 60.94, "table_1"),
    EvidenceSpec(2, "discretionary company contributions", "2009", 3.8, "text_15"),
    EvidenceSpec(2, "total expensed amounts for savings plans", "2009", 35.1, "text_14"),
    EvidenceSpec(3, "equipment rents payable", "2007", 103.0, "table_6"),
    EvidenceSpec(3, "equipment rents payable", "2008", 93.0, "table_6"),
    EvidenceSpec(6, "cash provided by operating activities", "2012", 6161.0, "table_1"),
    EvidenceSpec(6, "cash provided by operating activities", "2013", 6823.0, "table_1"),
    EvidenceSpec(7, "total oil and gas", "canada mmboe", 60.0, "table_3"),
    EvidenceSpec(7, "total oil and gas", "total mmboe", 243.0, "table_5"),
    EvidenceSpec(8, "risk and insurance brokerage services segment revenue", "2008", 6197.0, "table_1"),
    EvidenceSpec(8, "risk and insurance brokerage services segment revenue", "2009", 6305.0, "table_1"),
    EvidenceSpec(9, "s&p 500", "2015", 110.28, "table_3"),
    EvidenceSpec(9, "s&p 500", "2016", 129.05, "table_3"),
    EvidenceSpec(10, "total long-term borrowings", "carrying value", 4938.0, "table_7"),
    EvidenceSpec(10, "total long-term borrowings", "fair value", 5309.0, "table_7"),
    EvidenceSpec(11, "gas customers", "2007", 86000.0, "text_5"),
    EvidenceSpec(11, "gas customers", "2008", 93000.0, "text_5"),
    EvidenceSpec(13, "total smokeless products shipment volume", "2010", 724.4, "table_5"),
    EvidenceSpec(13, "total smokeless products shipment volume", "2011", 734.6, "table_5"),
)

_OPERATION_PATTERN = re.compile(r"^([a-z_]+)\((.*)\)$")
_REFERENCE_PATTERN = re.compile(r"^#(\d+)$")
_NUMBER_PATTERN = re.compile(r"^-?(?:\d+(?:\.\d*)?|\.\d+)$")
_SYMBOLS = {"add": "+", "divide": "/", "multiply": "*", "subtract": "-"}


@dataclass(frozen=True)
class Operation:
    operator: str
    arg1: str
    arg2: str


@dataclass(frozen=True)
class BuildResult:
    benchmark_path: Path
    fixture_path: Path
    subset_path: Path
    manifest_path: Path
    workflow_count: int
    step_count: int


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except FileNotFoundError as exc:
        raise ValueError(f"ConvFinQA source archive does not exist: {path}") from exc
    return digest.hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _pretty_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def _write_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)


def _split_operations(program: str) -> list[str]:
    if not isinstance(program, str) or not program.strip():
        raise ValueError("ConvFinQA turn program must be a non-empty string.")
    if _NUMBER_PATTERN.fullmatch(program.strip()):
        return []
    operations: list[str] = []
    depth = 0
    start = 0
    for index, character in enumerate(program):
        if character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
            if depth < 0:
                raise ValueError(f"Unbalanced ConvFinQA program: {program!r}")
        elif character == "," and depth == 0:
            operations.append(program[start:index].strip())
            start = index + 1
    if depth != 0:
        raise ValueError(f"Unbalanced ConvFinQA program: {program!r}")
    operations.append(program[start:].strip())
    if not all(operations):
        raise ValueError(f"Empty operation in ConvFinQA program: {program!r}")
    return operations


def _parse_operation(operation: str) -> Operation:
    match = _OPERATION_PATTERN.fullmatch(operation)
    if match is None:
        raise ValueError(f"Unsupported ConvFinQA operation syntax: {operation!r}")
    operator, arguments = match.groups()
    if operator not in _SYMBOLS:
        raise ValueError(f"Unsupported ConvFinQA operator: {operator!r}")
    parts = [part.strip() for part in arguments.split(",")]
    if len(parts) != 2 or not all(parts):
        raise ValueError(f"Operation must have two arguments: {operation!r}")
    return Operation(operator, parts[0], parts[1])


def _parsed_program(program: str) -> list[Operation]:
    return [_parse_operation(value) for value in _split_operations(program)]


def _literal(token: str) -> int | float:
    if token.startswith("const_"):
        token = token.removeprefix("const_")
    if not _NUMBER_PATTERN.fullmatch(token):
        raise ValueError(f"Expected numeric ConvFinQA program token, got {token!r}.")
    return float(token) if "." in token else int(token)


def _render_token(token: str, operations: Sequence[Operation]) -> str:
    reference = _REFERENCE_PATTERN.fullmatch(token)
    if reference is None:
        value = _literal(token)
        return str(value)
    index = int(reference.group(1))
    if index >= len(operations):
        raise ValueError(f"Invalid ConvFinQA operation reference: {token!r}")
    return f"({_render_operation(operations[index], operations)})"


def _render_operation(operation: Operation, operations: Sequence[Operation]) -> str:
    return (
        f"{_render_token(operation.arg1, operations)} "
        f"{_SYMBOLS[operation.operator]} "
        f"{_render_token(operation.arg2, operations)}"
    )


def _calculator_expression(program: str) -> str:
    operations = _parsed_program(program)
    if not operations:
        raise ValueError("A literal program cannot be translated to calculator.")
    return _render_operation(operations[-1], operations)


def _extract_source_subset(source_records: Any) -> list[dict[str, Any]]:
    if not isinstance(source_records, list) or len(source_records) != SOURCE_EXAMPLE_COUNT:
        actual = len(source_records) if isinstance(source_records, list) else None
        raise ValueError(
            f"Pinned ConvFinQA dev split must contain {SOURCE_EXAMPLE_COUNT} rows; "
            f"found {actual!r}."
        )
    subset: list[dict[str, Any]] = []
    for source_row_index in SELECTED_SOURCE_ROW_INDICES:
        row = source_records[source_row_index]
        if not isinstance(row, dict):
            raise ValueError(f"ConvFinQA source row {source_row_index} is not an object.")
        qa = row.get("qa")
        annotation = row.get("annotation")
        if not isinstance(qa, dict) or not isinstance(annotation, dict):
            raise ValueError(f"ConvFinQA source row {source_row_index} is incomplete.")
        try:
            subset.append(
                {
                    "source_row_index": source_row_index,
                    "source_conversation_id": row["id"],
                    "source_filename": row["filename"],
                    "question": qa["question"],
                    "display_answer": qa["answer"],
                    "gold_evidence": qa["gold_inds"],
                    "source_program": qa["program"],
                    "source_execution_answer": qa["exe_ans"],
                    "original_program": annotation["original_program"],
                    "dialogue_break": annotation["dialogue_break"],
                    "turn_program": annotation["turn_program"],
                    "execution_answers": annotation["exe_ans_list"],
                }
            )
        except KeyError as exc:
            raise ValueError(
                f"ConvFinQA source row {source_row_index} lacks {exc.args[0]!r}."
            ) from exc
    return subset


def _validate_subset(subset: Any) -> list[dict[str, Any]]:
    if not isinstance(subset, list) or len(subset) != WORKFLOW_COUNT:
        raise ValueError(f"ConvFinQA subset must contain {WORKFLOW_COUNT} rows.")
    indices = tuple(row.get("source_row_index") for row in subset if isinstance(row, dict))
    ids = tuple(row.get("source_conversation_id") for row in subset if isinstance(row, dict))
    if indices != SELECTED_SOURCE_ROW_INDICES:
        raise ValueError(f"ConvFinQA inclusion list changed: {indices!r}.")
    if ids != SELECTED_CONVERSATION_IDS:
        raise ValueError(f"ConvFinQA conversation IDs changed: {ids!r}.")
    actual_hash = _sha256_bytes(_canonical_bytes(subset))
    if actual_hash != EXPECTED_SOURCE_SUBSET_SHA256:
        raise ValueError(
            "ConvFinQA selected source fields changed: expected canonical subset "
            f"hash {EXPECTED_SOURCE_SUBSET_SHA256}, got {actual_hash}."
        )
    for row in subset:
        turn_count = len(row["dialogue_break"])
        if not (
            turn_count
            == len(row["turn_program"])
            == len(row["execution_answers"])
        ):
            raise ValueError(
                f"ConvFinQA row {row['source_row_index']} has inconsistent turns."
            )
        normalized_original_program = re.sub(
            r"\bA(\d+)\b", r"#\1", row["original_program"]
        )
        if normalized_original_program != row["source_program"]:
            raise ValueError(
                f"ConvFinQA row {row['source_row_index']} source programs disagree."
            )
        if row["source_execution_answer"] != row["execution_answers"][-1]:
            raise ValueError(
                f"ConvFinQA row {row['source_row_index']} execution answers disagree."
            )
    if sum(len(row["dialogue_break"]) for row in subset) != STEP_COUNT:
        raise ValueError(f"ConvFinQA subset must contain {STEP_COUNT} turns.")
    return subset


def _read_pinned_archive(source_archive: Path) -> list[dict[str, Any]]:
    actual_hash = _sha256_file(source_archive)
    if actual_hash != SOURCE_ARCHIVE_SHA256:
        raise ValueError(
            "ConvFinQA archive hash mismatch: expected "
            f"{SOURCE_ARCHIVE_SHA256}, got {actual_hash}."
        )
    try:
        with zipfile.ZipFile(source_archive) as archive:
            source_bytes = archive.read(SOURCE_MEMBER)
    except (zipfile.BadZipFile, KeyError) as exc:
        raise ValueError(
            f"Pinned ConvFinQA archive must contain {SOURCE_MEMBER!r}."
        ) from exc
    member_hash = _sha256_bytes(source_bytes)
    if member_hash != SOURCE_MEMBER_SHA256:
        raise ValueError(
            "ConvFinQA dev.json hash mismatch: expected "
            f"{SOURCE_MEMBER_SHA256}, got {member_hash}."
        )
    try:
        records = json.loads(source_bytes)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid ConvFinQA {SOURCE_MEMBER}: {exc}") from exc
    return _validate_subset(_extract_source_subset(records))


def _fixture_rows(subset: Sequence[dict[str, Any]]) -> list[list[Any]]:
    by_index = {row["source_row_index"]: row for row in subset}
    rows: list[list[Any]] = []
    for spec in EVIDENCE_SPECS:
        source = by_index[spec.source_row_index]
        evidence = source["gold_evidence"]
        if spec.evidence_key not in evidence:
            raise ValueError(
                f"Source row {spec.source_row_index} lacks evidence {spec.evidence_key!r}."
            )
        rows.append(
            [
                SOURCE_SPLIT,
                spec.source_row_index,
                source["source_conversation_id"],
                source["source_filename"],
                spec.metric,
                spec.period,
                spec.numeric_value,
                spec.evidence_key,
                evidence[spec.evidence_key],
            ]
        )
    if len(rows) != EVIDENCE_ROW_COUNT:
        raise ValueError(f"ConvFinQA fixture must contain {EVIDENCE_ROW_COUNT} rows.")
    return rows


def _fixture(rows: list[list[Any]]) -> dict[str, Any]:
    return {
        "dataset_id": FIXTURE_DATASET_ID,
        "description": (
            "Normalized gold evidence cells for ten exact multi-turn conversations "
            "from the official ConvFinQA development split."
        ),
        "columns": [dict(column) for column in FIXTURE_COLUMNS],
        "rows": rows,
        "provenance": {
            "source_dataset": SOURCE_DATASET,
            "source_split": SOURCE_SPLIT,
            "source_repository": SOURCE_REPOSITORY_URL,
            "source_revision": SOURCE_REVISION,
            "source_archive": SOURCE_ARCHIVE,
            "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
            "source_license": SOURCE_LICENSE,
            "adaptation": (
                "Only the gold evidence cells needed by the selected exact "
                "conversations were normalized into the allowlisted SQLite fixture."
            ),
        },
    }


def _evidence_for_value(
    source_row_index: int,
    value: int | float,
) -> EvidenceSpec:
    matches = [
        spec
        for spec in EVIDENCE_SPECS
        if spec.source_row_index == source_row_index
        and spec.numeric_value == float(value)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one normalized evidence cell for row {source_row_index} "
            f"and value {value!r}; found {len(matches)}."
        )
    return matches[0]


def _quote(value: str) -> str:
    return value.replace("'", "''")


def _query_sql(source_row_index: int, program: str) -> str:
    operations = _parsed_program(program)
    if not operations:
        spec = _evidence_for_value(source_row_index, _literal(program))
        return (
            "SELECT MAX(numeric_value) AS result FROM data WHERE "
            f"source_row_index = {source_row_index} AND metric = '{_quote(spec.metric)}' "
            f"AND period = '{_quote(spec.period)}'"
        )
    operation = operations[-1]
    if _REFERENCE_PATTERN.fullmatch(operation.arg1) or _REFERENCE_PATTERN.fullmatch(
        operation.arg2
    ):
        raise ValueError("First-turn fixture query cannot use prior operation references.")
    left = _evidence_for_value(source_row_index, _literal(operation.arg1))
    right = _evidence_for_value(source_row_index, _literal(operation.arg2))
    symbol = _SYMBOLS[operation.operator]
    if left.metric == right.metric and left.period != right.period:
        return (
            f"SELECT MAX(CASE WHEN period = '{_quote(left.period)}' THEN numeric_value END) "
            f"{symbol} MAX(CASE WHEN period = '{_quote(right.period)}' THEN numeric_value END) "
            f"AS result FROM data WHERE source_row_index = {source_row_index} "
            f"AND metric = '{_quote(left.metric)}'"
        )
    if left.period == right.period and left.metric != right.metric:
        return (
            f"SELECT MAX(CASE WHEN metric = '{_quote(left.metric)}' THEN numeric_value END) "
            f"{symbol} MAX(CASE WHEN metric = '{_quote(right.metric)}' THEN numeric_value END) "
            f"AS result FROM data WHERE source_row_index = {source_row_index} "
            f"AND period = '{_quote(left.period)}'"
        )
    raise ValueError(
        f"Cannot translate source row {source_row_index} program {program!r} to SQL."
    )


def _query_answer(sql: str, fixture_rows: Sequence[Sequence[Any]]) -> dict[str, Any]:
    connection = sqlite3.connect(":memory:")
    try:
        definitions = ", ".join(
            f"{column['name']} {column['type']}" for column in FIXTURE_COLUMNS
        )
        connection.execute(f"CREATE TABLE data ({definitions})")
        placeholders = ", ".join("?" for _ in FIXTURE_COLUMNS)
        connection.executemany(f"INSERT INTO data VALUES ({placeholders})", fixture_rows)
        cursor = connection.execute(sql)
        rows = [list(row) for row in cursor.fetchall()]
        columns = [description[0] for description in cursor.description or []]
    finally:
        connection.close()
    return {"columns": columns, "rows": rows, "row_count": len(rows), "truncated": False}


def _operation_dependency_turns(
    program: str,
    prior_programs: Sequence[str],
) -> list[str]:
    operations = _parsed_program(program)
    if not operations:
        return []
    root = operations[-1]
    dependencies: set[int] = set()
    for token in (root.arg1, root.arg2):
        reference = _REFERENCE_PATTERN.fullmatch(token)
        if reference is not None:
            operation_index = int(reference.group(1))
            prefix = ", ".join(_split_operations(program)[: operation_index + 1])
            matches = [
                index for index, prior in enumerate(prior_programs) if prior == prefix
            ]
        else:
            literal = _literal(token)
            matches = [
                index
                for index, prior in enumerate(prior_programs)
                if not _parsed_program(prior) and _literal(prior) == literal
            ]
        if matches:
            dependencies.add(matches[-1])
    return [f"turn_{index + 1:03d}" for index in sorted(dependencies)]


def _expected_step(
    source_row_index: int,
    turn_index: int,
    query: str,
    program: str,
    prior_programs: Sequence[str],
    fixture: dict[str, Any],
) -> dict[str, Any]:
    use_query_table = turn_index == 0 or not _parsed_program(program)
    if use_query_table:
        sql = _query_sql(source_row_index, program)
        expected_args = {"dataset_id": FIXTURE_DATASET_ID, "sql": sql}
        expected_answer = _query_answer(sql, fixture["rows"])
        prompt_context = table_query_prompt_context(expected_args, fixture)
        expected_tool = "finance_query_table"
        dependencies: list[str] = []
    else:
        expression = _calculator_expression(program)
        expected_args = {"expression": expression}
        expected_answer = {"result": calculator(expression)["result"]}
        prompt_context = calculator_prompt_context(expected_args)
        expected_tool = "calculator"
        dependencies = _operation_dependency_turns(program, prior_programs)
    return {
        "id": f"turn_{turn_index + 1:03d}",
        "query": query,
        "prompt_context": prompt_context,
        "expected_tool": expected_tool,
        "expected_args": expected_args,
        "expected_answer": expected_answer,
        "depends_on": dependencies,
        "source_program": program,
    }


def _workflow(source: dict[str, Any], fixture: dict[str, Any]) -> dict[str, Any]:
    source_row_index = source["source_row_index"]
    programs = source["turn_program"]
    steps = [
        _expected_step(
            source_row_index,
            turn_index,
            query,
            programs[turn_index],
            programs[:turn_index],
            fixture,
        )
        for turn_index, query in enumerate(source["dialogue_break"])
    ]
    turn_count = len(steps)
    turn_label = {0: "five", 2: "two", 3: "five"}.get(
        source_row_index, str(turn_count)
    )
    return {
        "id": f"finance_multistep_convfinqa_dev_{source_row_index:03d}",
        "domain": "finance",
        "task_type": "multi_step_tool_routing",
        "difficulty": "hard",
        "source": "public_finance_conversation_derived",
        "query": source["question"],
        "expected_steps": steps,
        "expected_final_answer": source["display_answer"],
        "final_program_execution_contract": "convfinqa_program_execution",
        "expected_final_program_result": source["execution_answers"][-1],
        "source_dataset": SOURCE_DATASET,
        "source_split": SOURCE_SPLIT,
        "source_row_index": source_row_index,
        "source_conversation_id": source["source_conversation_id"],
        "source_filename": source["source_filename"],
        "source_original_program": source["original_program"],
        "source_execution_answers": source["execution_answers"],
        "query_origin": "exact_public_dataset_dialogue",
        "tool_sequence_origin": "mechanical_adaptation_of_gold_turn_programs",
        "fixture_dataset_id": FIXTURE_DATASET_ID,
        "source_revision": SOURCE_REVISION,
        "source_license": SOURCE_LICENSE,
        "provenance_type": "public_dataset_multistep_adaptation",
        "perturbation_type": "paper_conversation",
        "notes": (
            f"Exact {turn_label}-turn ConvFinQA conversation with a mechanically "
            "adapted retrieval-and-calculation tool sequence."
        ),
    }


def _translation_hash(workflows: Sequence[dict[str, Any]]) -> str:
    translation = [
        {
            "source_row_index": workflow["source_row_index"],
            "turns": [
                {
                    "source_program": step["source_program"],
                    "expected_tool": step["expected_tool"],
                    "expected_args": step["expected_args"],
                    "depends_on": step["depends_on"],
                }
                for step in workflow["expected_steps"]
            ],
        }
        for workflow in workflows
    ]
    return _sha256_bytes(_canonical_bytes(translation))


def _model_facing_hash(workflows: Sequence[dict[str, Any]]) -> str:
    projection = [
        {
            key: workflow[key]
            for key in ("id", "query", "expected_steps", "expected_final_answer")
        }
        for workflow in workflows
    ]
    return _sha256_bytes(_canonical_bytes(projection))


def _manifest(
    benchmark_bytes: bytes,
    fixture_bytes: bytes,
    subset_bytes: bytes,
) -> dict[str, Any]:
    return {
        "builder": "benchmark.finance.build_convfinqa_multistep",
        "builder_version": "convfinqa_source_to_artifact_v1",
        "source": {
            "repository": SOURCE_REPOSITORY,
            "repository_url": SOURCE_REPOSITORY_URL,
            "revision": SOURCE_REVISION,
            "archive": SOURCE_ARCHIVE,
            "archive_sha256": SOURCE_ARCHIVE_SHA256,
            "member": SOURCE_MEMBER,
            "member_sha256": SOURCE_MEMBER_SHA256,
            "split": SOURCE_SPLIT,
            "license": SOURCE_LICENSE,
        },
        "selection": {
            "source_row_indices": list(SELECTED_SOURCE_ROW_INDICES),
            "conversation_ids": list(SELECTED_CONVERSATION_IDS),
            "workflow_count": WORKFLOW_COUNT,
            "step_count": STEP_COUNT,
            "evidence_row_count": EVIDENCE_ROW_COUNT,
            "source_subset_canonical_sha256": EXPECTED_SOURCE_SUBSET_SHA256,
            "translation_canonical_sha256": EXPECTED_TRANSLATION_SHA256,
            "origin_main_model_facing_sha256": ORIGIN_MAIN_MODEL_FACING_SHA256,
        },
        "generated_artifacts": {
            BENCHMARK_OUTPUT_NAME: _sha256_bytes(benchmark_bytes),
            FIXTURE_OUTPUT_NAME: _sha256_bytes(fixture_bytes),
            SUBSET_OUTPUT_NAME: _sha256_bytes(subset_bytes),
        },
    }


def build_from_subset(subset: Any, output_root: Path) -> BuildResult:
    """Build all artifacts from an already extracted, source-owned subset."""

    selected = _validate_subset(subset)
    fixture = _fixture(_fixture_rows(selected))
    workflows = [_workflow(source, fixture) for source in selected]
    if len(workflows) != WORKFLOW_COUNT:
        raise ValueError(f"Expected {WORKFLOW_COUNT} ConvFinQA workflows.")
    step_count = sum(len(workflow["expected_steps"]) for workflow in workflows)
    if step_count != STEP_COUNT:
        raise ValueError(f"Expected {STEP_COUNT} ConvFinQA steps, got {step_count}.")
    translation_hash = _translation_hash(workflows)
    if translation_hash != EXPECTED_TRANSLATION_SHA256:
        raise ValueError(
            "ConvFinQA program-to-tool translation changed: expected "
            f"{EXPECTED_TRANSLATION_SHA256}, got {translation_hash}."
        )
    model_facing_hash = _model_facing_hash(workflows)
    if model_facing_hash != ORIGIN_MAIN_MODEL_FACING_SHA256:
        raise ValueError(
            "ConvFinQA model-facing content differs from origin/main: expected "
            f"{ORIGIN_MAIN_MODEL_FACING_SHA256}, got {model_facing_hash}."
        )
    benchmark_bytes = _pretty_bytes(workflows)
    fixture_bytes = _pretty_bytes(fixture)
    subset_bytes = _pretty_bytes(selected)
    expected_hashes = {
        BENCHMARK_OUTPUT_NAME: EXPECTED_BENCHMARK_SHA256,
        FIXTURE_OUTPUT_NAME: EXPECTED_FIXTURE_SHA256,
        SUBSET_OUTPUT_NAME: EXPECTED_SUBSET_FILE_SHA256,
    }
    generated_bytes = {
        BENCHMARK_OUTPUT_NAME: benchmark_bytes,
        FIXTURE_OUTPUT_NAME: fixture_bytes,
        SUBSET_OUTPUT_NAME: subset_bytes,
    }
    for name, expected_hash in expected_hashes.items():
        actual_hash = _sha256_bytes(generated_bytes[name])
        if actual_hash != expected_hash:
            raise ValueError(
                f"ConvFinQA serialization or generated content changed for {name}: "
                f"expected {expected_hash}, got {actual_hash}."
            )
    manifest = _manifest(benchmark_bytes, fixture_bytes, subset_bytes)

    benchmark_path = output_root / BENCHMARK_OUTPUT_NAME
    fixture_path = output_root / FIXTURE_OUTPUT_NAME
    subset_path = output_root / SUBSET_OUTPUT_NAME
    manifest_path = output_root / MANIFEST_OUTPUT_NAME
    _write_bytes(benchmark_path, benchmark_bytes)
    _write_bytes(fixture_path, fixture_bytes)
    _write_bytes(subset_path, subset_bytes)
    _write_bytes(manifest_path, _pretty_bytes(manifest))
    return BuildResult(
        benchmark_path,
        fixture_path,
        subset_path,
        manifest_path,
        len(workflows),
        step_count,
    )


def build_convfinqa(source_archive: Path, output_root: Path) -> BuildResult:
    """Verify the pinned archive and deterministically build all artifacts."""

    return build_from_subset(_read_pinned_archive(source_archive), output_root)


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-archive",
        required=True,
        type=Path,
        help="Path to data.zip from the pinned ConvFinQA revision.",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        type=Path,
        help="Destination benchmark/finance directory.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _argument_parser()
    arguments = parser.parse_args(argv)
    try:
        result = build_convfinqa(arguments.source_archive, arguments.output_root)
    except ValueError as exc:
        parser.error(str(exc))
    print(
        f"Wrote {result.workflow_count} workflows and {result.step_count} steps "
        f"to {result.benchmark_path}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
