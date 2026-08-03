"""Build the executable CodeSearchNet single-step coding benchmark.

The generated rows preserve the exact published CodeSearchNet query strings and
pair each query with one deterministic relevance-3 annotation occurrence.  The
first 15 historical LayerMCP selections retain their existing coordinates; the
remaining queries prefer the earliest Python relevance-3 annotation and then
the earliest relevance-3 annotation in any language.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FIXTURE_OUTPUT = (
    PROJECT_ROOT
    / "benchmark"
    / "coding"
    / "fixtures"
    / "codesearchnet_public_annotations.json"
)
DEFAULT_BENCHMARK_OUTPUT = (
    PROJECT_ROOT
    / "benchmark"
    / "coding"
    / "coding_codesearchnet_public_derived.json"
)
LOCAL_LICENSE_PATH = (
    PROJECT_ROOT
    / "benchmark"
    / "coding"
    / "fixtures"
    / "CODESEARCHNET_LICENSE.txt"
)

SOURCE_REVISION = "bb121a53a559e99a6849409355ee5c83803f2e87"
VERIFIED_REPOSITORY_TIP = "106e827405c968597da938f6b373d30183918869"
QUERY_SHA256 = "037509c717c2e164721f0fd3ea45cb05f36669551af643f53930a92b76b146cf"
ANNOTATION_SHA256 = "0340af32b551ceadb74fec147f97642b7fedf3ff039e38fb86baff49ee899846"
LICENSE_SHA256 = "5ba1fd8a344040f2698ed3234aeb8f4b3e85211aa54a37048021f3eb0043be22"

QUERY_URL = (
    "https://github.com/github/CodeSearchNet/blob/"
    f"{SOURCE_REVISION}/resources/queries.csv"
)
ANNOTATION_URL = (
    "https://github.com/github/CodeSearchNet/blob/"
    f"{SOURCE_REVISION}/resources/annotationStore.csv"
)
LICENSE_URL = (
    "https://github.com/github/CodeSearchNet/blob/"
    f"{SOURCE_REVISION}/LICENSE"
)
PAPER_URL = "https://arxiv.org/abs/1909.09436"

REPOSITORY_ID = "codesearchnet-public-v1"
FIXTURE_VERSION = "coding_codesearchnet_fixture_v2"
SELECTION_VERSION = "codesearchnet_relevance3_query_coverage_v2"
ANNOTATION_PATH = "resources/annotationStore_selected.jsonl"
QUERIES_PATH = "resources/queries_selected.txt"

# Preserve the first 15 benchmark rows and their exact published coordinates.
LEGACY_COORDINATES = (
    (20, 1635),
    (13, 1666),
    (28, 1680),
    (29, 1721),
    (43, 1801),
    (12, 1832),
    (74, 1929),
    (25, 1952),
    (19, 2028),
    (39, 2071),
    (57, 2144),
    (79, 2259),
    (24, 2264),
    (8, 2344),
    (11, 2398),
)

EXPECTED_QUERY_COUNT = 99
EXPECTED_ANNOTATION_COUNT = 4006
EXPECTED_SELECTED_COUNT = 97
EXPECTED_EXCLUDED_QUERIES = (
    "set file attrib hidden",
    "concatenate several file remove header lines",
)


def _choose_annotation_candidate(
    candidates: list[tuple[int, dict[str, str]]],
) -> tuple[int, dict[str, str]]:
    """Prefer the earliest Python row, then the earliest row in source order."""

    if not candidates:
        raise ValueError("At least one annotation candidate is required.")
    return min(
        candidates,
        key=lambda item: (
            item[1]["Language"] != "Python",
            item[0],
        ),
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_csv(path: Path, expected_fields: tuple[str, ...]) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != expected_fields:
            raise RuntimeError(
                f"Unexpected columns in {path}: {reader.fieldnames!r}"
            )
        return [dict(row) for row in reader]


def _validate_source_file(path: Path, expected_hash: str) -> None:
    actual_hash = _sha256(path)
    if actual_hash != expected_hash:
        raise RuntimeError(
            f"Pinned source hash mismatch for {path}: "
            f"{actual_hash} != {expected_hash}"
        )


def _select_records(
    queries: list[str],
    annotations: list[dict[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pair_relevance_counts: dict[tuple[str, str, str], Counter[str]] = {}
    for row in annotations:
        key = (row["Language"], row["Query"], row["GitHubUrl"])
        pair_relevance_counts.setdefault(key, Counter())[row["Relevance"]] += 1
    selected: list[dict[str, Any]] = []
    selected_queries: set[str] = set()

    def add(query_index: int, annotation_index: int) -> None:
        query = queries[query_index]
        annotation = annotations[annotation_index]
        if annotation["Query"] != query:
            raise RuntimeError(
                "Pinned CodeSearchNet query/annotation coordinate mismatch: "
                f"query[{query_index}]={query!r}, "
                f"annotation[{annotation_index}]={annotation['Query']!r}"
            )
        if annotation["Relevance"] != "3":
            raise RuntimeError(
                f"Selected annotation {annotation_index} is not relevance 3."
            )
        if query in selected_queries:
            raise RuntimeError(f"Duplicate selected query: {query!r}")

        annotation_key = (
            annotation["Language"],
            annotation["Query"],
            annotation["GitHubUrl"],
        )
        relevance_counts = dict(
            sorted(
                pair_relevance_counts[annotation_key].items(),
                key=lambda item: int(item[0]),
            )
        )
        selected_queries.add(query)
        selected.append(
            {
                "source_query_index_zero_based": query_index,
                "source_annotation_index_zero_based": annotation_index,
                "source_annotation_pair_multiplicity": sum(
                    relevance_counts.values()
                ),
                "source_annotation_pair_relevance_counts": relevance_counts,
                "language": annotation["Language"],
                "query": query,
                "github_url": annotation["GitHubUrl"],
                "relevance": 3,
                "notes": annotation["Notes"],
                "source_annotation_record": annotation,
            }
        )

    for query_index, annotation_index in LEGACY_COORDINATES:
        add(query_index, annotation_index)

    annotations_by_query: dict[str, list[tuple[int, dict[str, str]]]] = {}
    for annotation_index, annotation in enumerate(annotations):
        if annotation["Relevance"] == "3":
            annotations_by_query.setdefault(annotation["Query"], []).append(
                (annotation_index, annotation)
            )

    excluded: list[dict[str, Any]] = []
    for query_index, query in enumerate(queries):
        if query in selected_queries:
            continue
        candidates = annotations_by_query.get(query, [])
        if not candidates:
            excluded.append(
                {
                    "source_query_index_zero_based": query_index,
                    "query": query,
                    "reason": "no_relevance_3_annotation",
                }
            )
            continue
        annotation_index, _ = _choose_annotation_candidate(candidates)
        add(query_index, annotation_index)

    if len(selected) != EXPECTED_SELECTED_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_SELECTED_COUNT} selected records, got {len(selected)}."
        )
    if tuple(item["query"] for item in excluded) != EXPECTED_EXCLUDED_QUERIES:
        raise RuntimeError(f"Unexpected excluded queries: {excluded!r}")

    return selected, excluded


def _jsonl_lines(records: list[dict[str, Any]]) -> list[str]:
    lines = [
        json.dumps(record, ensure_ascii=False, separators=(",", ":"))
        for record in records
    ]
    for index, record in enumerate(records):
        query_bytes = record["query"].encode("utf-8")
        matching_lines = [
            line_index
            for line_index, line in enumerate(lines)
            if query_bytes in line.encode("utf-8")
        ]
        if matching_lines != [index]:
            raise RuntimeError(
                f"Query {record['query']!r} is not line-unique: {matching_lines}."
            )
        if len(lines[index].encode("utf-8")) > 2_000:
            raise RuntimeError(
                f"Selected annotation line {index + 1} exceeds search excerpt bound."
            )
    return lines


def _fixture_payload(
    records: list[dict[str, Any]],
    excluded: list[dict[str, Any]],
    jsonl_lines: list[str],
) -> dict[str, Any]:
    readme = (
        "# CodeSearchNet Public Routing Fixture\n\n"
        f"This deterministic fixture contains {len(records)} selected "
        "CodeSearchNet published queries and one relevance-3 annotation "
        "occurrence for each query.\n\n"
        "No target repository source code is copied. Published GitHub URLs are "
        "retained only as annotation evidence and provenance. This fixture "
        "supports bounded, case-sensitive fixed-string lookup; it does not "
        "reproduce semantic code retrieval, ranking, or NDCG.\n\n"
        f"Source revision: {SOURCE_REVISION}\n"
        "License: MIT; the pinned notice is preserved in "
        "benchmark/coding/fixtures/CODESEARCHNET_LICENSE.txt.\n"
    )
    selection = (
        f"Preserve the {len(LEGACY_COORDINATES)} historical LayerMCP "
        "query/annotation coordinates. For every remaining published query, "
        "select the earliest Python relevance-3 annotation when available, "
        "otherwise the earliest relevance-3 annotation in source order. "
        f"This retains {len(records)} of {EXPECTED_QUERY_COUNT} queries; queries "
        "without a relevance-3 annotation are excluded."
    )
    rating_summary = {
        "selected_pairs": len(records),
        "selected_pairs_with_multiple_annotations": sum(
            record["source_annotation_pair_multiplicity"] > 1
            for record in records
        ),
        "only_relevance_3_pairs": sum(
            set(record["source_annotation_pair_relevance_counts"]) == {"3"}
            for record in records
        ),
        "mixed_relevance_pairs": sum(
            len(record["source_annotation_pair_relevance_counts"]) > 1
            for record in records
        ),
        "pairs_where_relevance_3_is_a_minority": sum(
            record["source_annotation_pair_relevance_counts"]["3"]
            < record["source_annotation_pair_multiplicity"]
            - record["source_annotation_pair_relevance_counts"]["3"]
            for record in records
        ),
    }
    return {
        "repo_id": REPOSITORY_ID,
        "fixture_version": FIXTURE_VERSION,
        "description": (
            f"Declarative repository fixture containing {len(records)} "
            "CodeSearchNet published query/relevance-3 annotation pairs. This "
            "is a bounded lexical routing fixture, not a reproduction of "
            "semantic retrieval or NDCG."
        ),
        "files": {
            "README.md": readme,
            QUERIES_PATH: "".join(f"{record['query']}\n" for record in records),
            ANNOTATION_PATH: "\n".join(jsonl_lines) + "\n",
        },
        "record_count": len(records),
        "records": records,
        "provenance": {
            "source_dataset": "CodeSearchNet Challenge human evaluation",
            "source_repository": "github/CodeSearchNet",
            "source_revision": SOURCE_REVISION,
            "verified_repository_tip": VERIFIED_REPOSITORY_TIP,
            "source_query_file": "resources/queries.csv",
            "source_query_url": QUERY_URL,
            "source_query_sha256": QUERY_SHA256,
            "source_annotation_file": "resources/annotationStore.csv",
            "source_annotation_url": ANNOTATION_URL,
            "source_annotation_sha256": ANNOTATION_SHA256,
            "source_license": "MIT",
            "source_license_url": LICENSE_URL,
            "source_license_sha256": LICENSE_SHA256,
            "local_license_file": (
                "benchmark/coding/fixtures/CODESEARCHNET_LICENSE.txt"
            ),
            "attribution_file": (
                "benchmark/coding/fixtures/CODESEARCHNET_ATTRIBUTION.md"
            ),
            "source_paper_url": PAPER_URL,
            "query_origin": "codesearchnet_published_query",
            "provenance_type": "research_dataset_adaptation",
            "selection_version": SELECTION_VERSION,
            "selection": selection,
            "selected_pair_rating_summary": rating_summary,
            "excluded_queries": excluded,
            "adaptation": (
                "Only selected MIT-licensed CodeSearchNet query and annotation "
                "records are normalized. No target source code is copied. The "
                "fixture evaluates lexical routing and annotation lookup, not "
                "semantic retrieval or NDCG."
            ),
        },
    }


def _all_byte_columns(line: str, pattern: str) -> list[int]:
    line_bytes = line.encode("utf-8")
    pattern_bytes = pattern.encode("utf-8")
    columns: list[int] = []
    start = 0
    while True:
        offset = line_bytes.find(pattern_bytes, start)
        if offset < 0:
            break
        columns.append(offset + 1)
        start = offset + len(pattern_bytes)
    return columns


def _expected_answer(
    record: dict[str, Any],
    line_number: int,
    line: str,
) -> dict[str, Any]:
    columns = _all_byte_columns(line, record["query"])
    if not columns:
        raise RuntimeError(f"Selected query is absent from fixture line {line_number}.")
    return {
        "repo_id": REPOSITORY_ID,
        "pattern": record["query"],
        "path_glob": ANNOTATION_PATH,
        "case_sensitive": True,
        "matches": [
            {
                "path": ANNOTATION_PATH,
                "line": line_number,
                "column": columns[0],
                "columns": columns,
                "columns_truncated": False,
                "text": line,
                "text_truncated": False,
            }
        ],
        "count": 1,
        "truncated": False,
        "engine": "ripgrep-fixed-string",
        "source": "coding-fixture",
    }


def _benchmark_rows(
    records: list[dict[str, Any]],
    jsonl_lines: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, (record, line) in enumerate(zip(records, jsonl_lines), start=1):
        query = record["query"]
        annotation = record["source_annotation_record"]
        rows.append(
            {
                "id": f"coding_public_codesearchnet_search_text_{index:03d}",
                "domain": "coding",
                "task_type": "single_tool_routing",
                "benchmark_mode": "grounded_tool_execution",
                "difficulty": "medium",
                "source": "public_coding_research_derived",
                "query": (
                    f"In repository {REPOSITORY_ID}, search only "
                    f"{ANNOTATION_PATH} for the exact text "
                    f"{json.dumps(query, ensure_ascii=False)}. Match case exactly "
                    "and return at most one result."
                ),
                "expected_tool": "code_search_text",
                "expected_args": {
                    "repo_id": REPOSITORY_ID,
                    "pattern": query,
                    "path_glob": ANNOTATION_PATH,
                    "case_sensitive": True,
                    "max_results": 1,
                },
                "expected_answer": _expected_answer(record, index, line),
                "perturbation_type": "none",
                "notes": (
                    "Generated self-contained search instruction wrapping the "
                    "exact CodeSearchNet query stored in original_query."
                ),
                "source_dataset": "CodeSearchNet Challenge human evaluation",
                "source_repository": "github/CodeSearchNet",
                "source_revision": SOURCE_REVISION,
                "verified_repository_tip": VERIFIED_REPOSITORY_TIP,
                "original_query": query,
                "query_wrapper_id": "codesearchnet_annotation_lookup_v1",
                "selection_version": SELECTION_VERSION,
                "source_query_index_zero_based": record[
                    "source_query_index_zero_based"
                ],
                "source_annotation_index_zero_based": record[
                    "source_annotation_index_zero_based"
                ],
                "source_annotation_pair_multiplicity": record[
                    "source_annotation_pair_multiplicity"
                ],
                "source_annotation_pair_relevance_counts": record[
                    "source_annotation_pair_relevance_counts"
                ],
                "source_language": record["language"],
                "source_relevance": 3,
                "source_github_url": record["github_url"],
                "source_annotation_record": annotation,
                "source_query_url": QUERY_URL,
                "source_annotation_url": ANNOTATION_URL,
                "source_query_sha256": QUERY_SHA256,
                "source_annotation_sha256": ANNOTATION_SHA256,
                "source_license": "MIT",
                "source_license_url": LICENSE_URL,
                "source_license_sha256": LICENSE_SHA256,
                "source_paper_url": PAPER_URL,
                "query_origin": "generated_wrapper_around_codesearchnet_query",
                "original_query_origin": "codesearchnet_published_query",
                "provenance_type": "research_dataset_adaptation",
                "fixture_id": REPOSITORY_ID,
                "fixture_version": FIXTURE_VERSION,
                "fixture_file": (
                    "benchmark/coding/fixtures/"
                    "codesearchnet_public_annotations.json"
                ),
                "license_file": (
                    "benchmark/coding/fixtures/CODESEARCHNET_LICENSE.txt"
                ),
                "attribution_file": (
                    "benchmark/coding/fixtures/CODESEARCHNET_ATTRIBUTION.md"
                ),
                "adaptation_notes": (
                    "original_query is the exact CodeSearchNet published query. "
                    "query adds only the repository, file, case-sensitivity, and "
                    "result-bound instructions needed to make the expected tool "
                    "arguments inferable. The fixture normalizes one selected "
                    "MIT-licensed relevance-3 annotation occurrence and copies "
                    "no target repository source code. This row evaluates lexical "
                    "tool routing and annotation lookup, not semantic retrieval, "
                    "ranking, or NDCG reproduction."
                ),
            }
        )
    return rows


def _serialized(value: Any) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False) + "\n"


def _write_or_check(path: Path, content: str, *, check: bool) -> None:
    if check:
        if not path.exists() or path.read_text(encoding="utf-8") != content:
            raise RuntimeError(f"Generated artifact is stale: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries-csv", type=Path, required=True)
    parser.add_argument("--annotations-csv", type=Path, required=True)
    parser.add_argument(
        "--fixture-output",
        type=Path,
        default=DEFAULT_FIXTURE_OUTPUT,
    )
    parser.add_argument(
        "--benchmark-output",
        type=Path,
        default=DEFAULT_BENCHMARK_OUTPUT,
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify that the existing outputs match a fresh generation.",
    )
    args = parser.parse_args()

    _validate_source_file(args.queries_csv, QUERY_SHA256)
    _validate_source_file(args.annotations_csv, ANNOTATION_SHA256)
    _validate_source_file(LOCAL_LICENSE_PATH, LICENSE_SHA256)
    query_rows = _load_csv(args.queries_csv, ("query",))
    annotations = _load_csv(
        args.annotations_csv,
        ("Language", "Query", "GitHubUrl", "Relevance", "Notes"),
    )
    queries = [row["query"] for row in query_rows]
    if len(queries) != EXPECTED_QUERY_COUNT or len(set(queries)) != len(queries):
        raise RuntimeError(
            f"Expected {EXPECTED_QUERY_COUNT} unique queries, got {len(queries)}."
        )
    if len(annotations) != EXPECTED_ANNOTATION_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_ANNOTATION_COUNT} annotations, "
            f"got {len(annotations)}."
        )

    records, excluded = _select_records(queries, annotations)
    jsonl_lines = _jsonl_lines(records)
    fixture = _fixture_payload(records, excluded, jsonl_lines)
    rows = _benchmark_rows(records, jsonl_lines)

    _write_or_check(
        args.fixture_output,
        _serialized(fixture),
        check=args.check,
    )
    _write_or_check(
        args.benchmark_output,
        _serialized(rows),
        check=args.check,
    )
    action = "verified" if args.check else "wrote"
    print(f"{action} {args.fixture_output}")
    print(f"{action} {args.benchmark_output}")
    print(f"selected queries: {len(records)}")
    print(f"excluded queries: {len(excluded)}")
    print(
        "language distribution:",
        dict(Counter(record["language"] for record in records)),
    )


if __name__ == "__main__":
    main()
