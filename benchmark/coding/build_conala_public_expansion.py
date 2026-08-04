"""Build the executable CoNaLa single-step coding benchmark.

The generated rows preserve 133 distinct curated CoNaLa test intents from a
pinned NeuLab release. Source code snippets are not copied into the fixture;
stable source coordinates and cryptographic hashes retain provenance.
"""

from __future__ import annotations

import argparse
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
    / "conala_public_test.json"
)
DEFAULT_BENCHMARK_OUTPUT = (
    PROJECT_ROOT
    / "benchmark"
    / "coding"
    / "coding_conala_public_derived.json"
)

SOURCE_REVISION = "fbc749f1c537e5c3834e93b15784302e331debe2"
SOURCE_SHA256 = "3a7e5eea6deeccb5e7c9557534af860854fd2f0ae870752b42c296ed30e53cb7"
DATASET_CARD_SHA256 = (
    "326072b41743fff642a4639ade350308a47942ff67608f3dfe447014453f3e74"
)
LOADER_SHA256 = "1f106c699e97915f6d02b4f3a169a33f6de8e7306cc30b10cc1354d4b86f0f2d"

SOURCE_FILE = "data/conala-paired-test.json"
SOURCE_URL = (
    "https://huggingface.co/datasets/neulab/conala/resolve/"
    f"{SOURCE_REVISION}/{SOURCE_FILE}"
)
DATASET_REPOSITORY_URL = (
    "https://huggingface.co/datasets/neulab/conala/tree/" f"{SOURCE_REVISION}"
)
DATASET_CARD_URL = (
    "https://huggingface.co/datasets/neulab/conala/blob/"
    f"{SOURCE_REVISION}/README.md"
)
LOADER_URL = (
    "https://huggingface.co/datasets/neulab/conala/blob/"
    f"{SOURCE_REVISION}/conala.py"
)
PROJECT_URL = "https://conala-corpus.github.io/"
PAPER_URL = "https://arxiv.org/abs/1805.08949"
PAPER_DOI_URL = "https://doi.org/10.1145/3196398.3196408"

REPOSITORY_ID = "conala-public-test-v1"
FIXTURE_VERSION = "coding_conala_fixture_v1"
SELECTION_VERSION = "conala_curated_test_line_unique_133_v1"
SOURCE_RECORD_COUNT = 500
SELECTED_RECORD_COUNT = 133
EXPECTED_LAST_SELECTED_SOURCE_INDEX = 145
MAX_TOOL_PATTERN_LENGTH = 500
MANIFEST_PATH = "resources/conala_curated_test_selected.jsonl"
QUERIES_PATH = "resources/conala_curated_test_queries_selected.txt"

EXPECTED_SOURCE_FIELDS = {
    "question_id",
    "intent",
    "rewritten_intent",
    "snippet",
}


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _sha256(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _validate_source_file(path: Path, expected_hash: str) -> None:
    actual_hash = _sha256(path)
    if actual_hash != expected_hash:
        raise RuntimeError(
            f"Pinned source hash mismatch for {path}: "
            f"{actual_hash} != {expected_hash}"
        )


def _canonical_record_sha256(record: dict[str, Any]) -> str:
    content = json.dumps(
        record,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return _sha256_bytes(content)


def _load_source(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise RuntimeError(f"Unable to read pinned CoNaLa source {path}.") from exc
    if len(lines) != SOURCE_RECORD_COUNT:
        raise RuntimeError(
            f"Expected {SOURCE_RECORD_COUNT} CoNaLa source lines, got {len(lines)}."
        )

    for index, line in enumerate(lines):
        try:
            raw_record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Malformed JSON at CoNaLa source line {index + 1}."
            ) from exc
        if (
            not isinstance(raw_record, dict)
            or set(raw_record) != EXPECTED_SOURCE_FIELDS
        ):
            actual_fields = (
                sorted(raw_record)
                if isinstance(raw_record, dict)
                else type(raw_record)
            )
            raise RuntimeError(
                f"Unexpected CoNaLa record at source index {index}: "
                f"{actual_fields!r}"
            )
        record = dict(raw_record)
        if not isinstance(record["question_id"], int) or record["question_id"] <= 0:
            raise RuntimeError(f"Invalid question_id at CoNaLa source index {index}.")
        for field in ("intent", "snippet"):
            if not isinstance(record[field], str):
                raise RuntimeError(
                    f"Invalid {field} at CoNaLa source index {index}."
                )
        if record["rewritten_intent"] is not None and not isinstance(
            record["rewritten_intent"], str
        ):
            raise RuntimeError(
                f"Invalid rewritten_intent at CoNaLa source index {index}."
            )
        records.append(record)
    return records


def _is_selectable_query(query: object, selected_queries: list[str]) -> bool:
    if (
        not isinstance(query, str)
        or not query
        or len(query) > MAX_TOOL_PATTERN_LENGTH
        or "\x00" in query
        or "\n" in query
        or "\r" in query
    ):
        return False
    return not any(
        query in existing or existing in query for existing in selected_queries
    )


def _select_records(source_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    selected_queries: list[str] = []
    for source_index, source_record in enumerate(source_records):
        query = source_record["rewritten_intent"]
        if not _is_selectable_query(query, selected_queries):
            continue
        assert isinstance(query, str)

        question_id = source_record["question_id"]
        selected_queries.append(query)
        selected.append(
            {
                "source_record_index_zero_based": source_index,
                "question_id": question_id,
                "query": query,
                "stackoverflow_url": (
                    f"https://stackoverflow.com/questions/{question_id}"
                ),
                "source_record_canonical_sha256": _canonical_record_sha256(
                    source_record
                ),
                "source_snippet_sha256": _sha256_bytes(
                    source_record["snippet"].encode("utf-8")
                ),
            }
        )
        if len(selected) == SELECTED_RECORD_COUNT:
            break

    if len(selected) != SELECTED_RECORD_COUNT:
        raise RuntimeError(
            f"Expected {SELECTED_RECORD_COUNT} selected CoNaLa records, "
            f"got {len(selected)}."
        )
    if selected[-1]["source_record_index_zero_based"] != (
        EXPECTED_LAST_SELECTED_SOURCE_INDEX
    ):
        raise RuntimeError(
            "Pinned CoNaLa selection ended at an unexpected source index."
        )
    if len(set(selected_queries)) != SELECTED_RECORD_COUNT:
        raise RuntimeError("Selected CoNaLa intents are not unique.")
    for index, query in enumerate(selected_queries):
        matching_lines = [
            line_index
            for line_index, candidate in enumerate(selected_queries)
            if query in candidate
        ]
        if matching_lines != [index]:
            raise RuntimeError(
                f"Selected CoNaLa query {query!r} is not line-unique: "
                f"{matching_lines}."
            )
    return selected


def _fixture_payload(records: list[dict[str, Any]]) -> dict[str, Any]:
    selection = (
        "Scan the pinned curated test split in source order. Keep non-empty, "
        "single-line rewritten_intent values only when neither the candidate "
        "nor an already selected query is a substring of the other; stop after "
        f"{SELECTED_RECORD_COUNT} records. This makes every fixed-string lookup "
        "line-unique while supplying the exact shortfall required to bring "
        "LayerMCP's active Coding single-step benchmark to 300 rows."
    )
    readme = (
        "# CoNaLa Public Routing Fixture\n\n"
        f"This deterministic fixture contains {len(records)} exact "
        "crowd-rewritten intents from the curated CoNaLa test split. It "
        "supports bounded, case-sensitive fixed-string lookup and does not "
        "reproduce CoNaLa code generation or BLEU evaluation.\n\n"
        "Original Stack Overflow titles and paired Python snippets are not "
        "copied. Stable source coordinates and SHA-256 values preserve their "
        "relationship to each selected rewritten intent.\n\n"
        f"Source revision: {SOURCE_REVISION}\n"
        "Pinned dataset-card license metadata: MIT (`license` includes `mit`); "
        "no standalone dataset license notice is present.\n"
    )
    manifest_lines = [
        json.dumps(record, ensure_ascii=False, separators=(",", ":"))
        for record in records
    ]
    return {
        "repo_id": REPOSITORY_ID,
        "fixture_version": FIXTURE_VERSION,
        "description": (
            f"Declarative repository fixture containing {len(records)} exact "
            "CoNaLa curated-test rewritten intents. This is a bounded lexical "
            "routing fixture, not a reproduction of code generation or BLEU."
        ),
        "files": {
            "README.md": readme,
            QUERIES_PATH: "".join(f"{record['query']}\n" for record in records),
            MANIFEST_PATH: "\n".join(manifest_lines) + "\n",
        },
        "record_count": len(records),
        "records": records,
        "provenance": {
            "source_dataset": "CoNaLa",
            "source_configuration": "curated",
            "source_split": "test",
            "source_dataset_version": "1.1.0",
            "source_repository": "neulab/conala",
            "source_revision": SOURCE_REVISION,
            "source_file": SOURCE_FILE,
            "source_file_url": SOURCE_URL,
            "source_file_sha256": SOURCE_SHA256,
            "source_file_record_count": SOURCE_RECORD_COUNT,
            "source_dataset_repository_url": DATASET_REPOSITORY_URL,
            "source_dataset_card_url": DATASET_CARD_URL,
            "source_dataset_card_sha256": DATASET_CARD_SHA256,
            "source_loader_url": LOADER_URL,
            "source_loader_sha256": LOADER_SHA256,
            "source_project_url": PROJECT_URL,
            "source_paper_url": PAPER_URL,
            "source_paper_doi_url": PAPER_DOI_URL,
            "query_origin": "conala_crowd_rewritten_intent",
            "provenance_type": "research_dataset_adaptation",
            "selection_version": SELECTION_VERSION,
            "selection": selection,
            "selected_record_count": len(records),
            "selected_question_count": len(
                {record["question_id"] for record in records}
            ),
            "selected_unique_snippet_count": len(
                {record["source_snippet_sha256"] for record in records}
            ),
            "source_license": "MIT",
            "source_license_scope": "dataset",
            "source_license_evidence_type": "dataset_card_metadata",
            "source_license_evidence": (
                "Pinned dataset-card YAML front matter "
                "(`license` includes `mit`)"
            ),
            "source_license_evidence_sha256": DATASET_CARD_SHA256,
            "source_license_url": DATASET_CARD_URL,
            "attribution_file": (
                "benchmark/coding/fixtures/CONALA_ATTRIBUTION.md"
            ),
            "adaptation": (
                "Exact curated rewritten intents and bounded provenance "
                "metadata are normalized for fixed-string lookup. Original "
                "Stack Overflow titles and paired Python snippets are not "
                "redistributed. This evaluates lexical tool routing, not "
                "code generation or BLEU."
            ),
        },
    }


def _expected_answer(record: dict[str, Any], line_number: int) -> dict[str, Any]:
    return {
        "repo_id": REPOSITORY_ID,
        "pattern": record["query"],
        "path_glob": QUERIES_PATH,
        "case_sensitive": True,
        "matches": [
            {
                "path": QUERIES_PATH,
                "line": line_number,
                "column": 1,
                "columns": [1],
                "columns_truncated": False,
                "text": record["query"],
                "text_truncated": False,
            }
        ],
        "count": 1,
        "truncated": False,
        "engine": "ripgrep-fixed-string",
        "source": "coding-fixture",
    }


def _benchmark_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        query = record["query"]
        rows.append(
            {
                "id": f"coding_public_conala_search_text_{index:03d}",
                "domain": "coding",
                "task_type": "single_tool_routing",
                "benchmark_mode": "grounded_tool_execution",
                "difficulty": "medium",
                "source": "public_coding_research_derived",
                "query": (
                    f"In repository {REPOSITORY_ID}, search only {QUERIES_PATH} "
                    f"for the exact curated intent "
                    f"{json.dumps(query, ensure_ascii=False)}. Match case exactly "
                    "and return at most one result."
                ),
                "expected_tool": "code_search_text",
                "expected_args": {
                    "repo_id": REPOSITORY_ID,
                    "pattern": query,
                    "path_glob": QUERIES_PATH,
                    "case_sensitive": True,
                    "max_results": 1,
                },
                "expected_answer": _expected_answer(record, index),
                "perturbation_type": "none",
                "notes": (
                    "Generated self-contained search instruction wrapping the "
                    "exact CoNaLa rewritten intent stored in original_query."
                ),
                "source_dataset": "CoNaLa",
                "source_configuration": "curated",
                "source_split": "test",
                "source_dataset_version": "1.1.0",
                "source_repository": "neulab/conala",
                "source_revision": SOURCE_REVISION,
                "source_file": SOURCE_FILE,
                "source_file_url": SOURCE_URL,
                "source_file_sha256": SOURCE_SHA256,
                "source_file_record_count": SOURCE_RECORD_COUNT,
                "source_record_index_zero_based": record[
                    "source_record_index_zero_based"
                ],
                "source_question_id": record["question_id"],
                "source_stackoverflow_url": record["stackoverflow_url"],
                "source_record_canonical_sha256": record[
                    "source_record_canonical_sha256"
                ],
                "source_snippet_sha256": record["source_snippet_sha256"],
                "source_original_intent_redistributed": False,
                "source_rewritten_intent_redistributed": True,
                "source_snippet_redistributed": False,
                "original_query": query,
                "query_wrapper_id": "conala_curated_intent_lookup_v1",
                "selection_version": SELECTION_VERSION,
                "source_project_url": PROJECT_URL,
                "source_paper_url": PAPER_URL,
                "source_paper_doi_url": PAPER_DOI_URL,
                "source_license": "MIT",
                "source_license_scope": "dataset",
                "source_license_evidence_type": "dataset_card_metadata",
                "source_license_evidence": (
                    "Pinned dataset-card YAML front matter "
                    "(`license` includes `mit`)"
                ),
                "source_license_evidence_sha256": DATASET_CARD_SHA256,
                "source_license_url": DATASET_CARD_URL,
                "query_origin": "generated_wrapper_around_conala_intent",
                "original_query_origin": "conala_crowd_rewritten_intent",
                "provenance_type": "research_dataset_adaptation",
                "fixture_id": REPOSITORY_ID,
                "fixture_version": FIXTURE_VERSION,
                "fixture_file": (
                    "benchmark/coding/fixtures/conala_public_test.json"
                ),
                "attribution_file": (
                    "benchmark/coding/fixtures/CONALA_ATTRIBUTION.md"
                ),
                "adaptation_notes": (
                    "original_query is the exact crowd-rewritten CoNaLa intent. "
                    "query adds only the repository, file, case-sensitivity, "
                    "and result-bound instructions needed to make tool arguments "
                    "inferable. Original Stack Overflow titles and paired code "
                    "are not copied. This row evaluates lexical tool routing "
                    "rather than code generation or BLEU reproduction."
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
    parser.add_argument("--source-jsonl", type=Path, required=True)
    parser.add_argument("--dataset-card", type=Path, required=True)
    parser.add_argument("--loader", type=Path, required=True)
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

    _validate_source_file(args.source_jsonl, SOURCE_SHA256)
    _validate_source_file(args.dataset_card, DATASET_CARD_SHA256)
    _validate_source_file(args.loader, LOADER_SHA256)
    source_records = _load_source(args.source_jsonl)
    records = _select_records(source_records)
    fixture = _fixture_payload(records)
    rows = _benchmark_rows(records)

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
    print(f"source records: {len(source_records)}")


if __name__ == "__main__":
    main()
