from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

from benchmark.finance import build_convfinqa_multistep as builder


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FINANCE_ROOT = PROJECT_ROOT / "benchmark" / "finance"
BENCHMARK_PATH = FINANCE_ROOT / builder.BENCHMARK_OUTPUT_NAME
FIXTURE_PATH = FINANCE_ROOT / builder.FIXTURE_OUTPUT_NAME
SUBSET_PATH = FINANCE_ROOT / builder.SUBSET_OUTPUT_NAME
MANIFEST_PATH = FINANCE_ROOT / builder.MANIFEST_OUTPUT_NAME


def _load(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ConvFinQABuilderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.subset = _load(SUBSET_PATH)
        cls.benchmark = _load(BENCHMARK_PATH)
        cls.fixture = _load(FIXTURE_PATH)
        cls.manifest = _load(MANIFEST_PATH)

    def test_exact_membership_and_source_fields(self) -> None:
        self.assertEqual(
            tuple(row["source_row_index"] for row in self.subset),
            builder.SELECTED_SOURCE_ROW_INDICES,
        )
        self.assertEqual(
            tuple(row["source_conversation_id"] for row in self.subset),
            builder.SELECTED_CONVERSATION_IDS,
        )
        self.assertEqual(len(self.benchmark), builder.WORKFLOW_COUNT)
        self.assertEqual(
            sum(len(row["expected_steps"]) for row in self.benchmark),
            builder.STEP_COUNT,
        )
        for source, generated in zip(self.subset, self.benchmark, strict=True):
            self.assertEqual(generated["query"], source["question"])
            self.assertEqual(
                [step["query"] for step in generated["expected_steps"]],
                source["dialogue_break"],
            )
            self.assertEqual(
                [step["source_program"] for step in generated["expected_steps"]],
                source["turn_program"],
            )
            self.assertEqual(
                generated["source_execution_answers"],
                source["execution_answers"],
            )
            self.assertEqual(
                generated["expected_final_program_result"],
                source["execution_answers"][-1],
            )
            self.assertEqual(
                generated["expected_final_answer"], source["display_answer"]
            )
            self.assertEqual(
                generated["source_conversation_id"],
                source["source_conversation_id"],
            )
            self.assertEqual(generated["source_filename"], source["source_filename"])
            self.assertEqual(
                generated["source_original_program"], source["original_program"]
            )
            self.assertEqual(
                generated["expected_steps"][-1]["source_program"],
                source["source_program"],
            )
            self.assertEqual(
                generated["expected_final_program_result"],
                source["source_execution_answer"],
            )

    def test_fixture_is_exactly_the_selected_normalized_evidence(self) -> None:
        self.assertEqual(len(self.fixture["rows"]), builder.EVIDENCE_ROW_COUNT)
        sources = {row["source_row_index"]: row for row in self.subset}
        for spec, row in zip(builder.EVIDENCE_SPECS, self.fixture["rows"], strict=True):
            source = sources[spec.source_row_index]
            self.assertEqual(
                row,
                [
                    "dev",
                    spec.source_row_index,
                    source["source_conversation_id"],
                    source["source_filename"],
                    spec.metric,
                    spec.period,
                    spec.numeric_value,
                    spec.evidence_key,
                    source["gold_evidence"][spec.evidence_key],
                ],
            )

    def test_program_to_tool_translation_is_pinned_and_mechanical(self) -> None:
        self.assertEqual(
            builder._translation_hash(self.benchmark),
            builder.EXPECTED_TRANSLATION_SHA256,
        )
        for workflow in self.benchmark:
            for turn_index, step in enumerate(workflow["expected_steps"]):
                parsed = builder._parsed_program(step["source_program"])
                expected_tool = (
                    "finance_query_table"
                    if turn_index == 0 or not parsed
                    else "calculator"
                )
                self.assertEqual(step["expected_tool"], expected_tool)
                if expected_tool == "calculator":
                    self.assertEqual(
                        step["expected_args"]["expression"],
                        builder._calculator_expression(step["source_program"]),
                    )
                else:
                    self.assertEqual(
                        step["expected_args"]["sql"],
                        builder._query_sql(
                            workflow["source_row_index"], step["source_program"]
                        ),
                    )

    def test_display_answer_and_execution_target_are_separate(self) -> None:
        for row in self.benchmark:
            self.assertEqual(
                row["final_program_execution_contract"],
                "convfinqa_program_execution",
            )
            self.assertIn("expected_final_answer", row)
            self.assertIn("expected_final_program_result", row)
        row_1877 = next(row for row in self.benchmark if row["source_row_index"] == 9)
        self.assertEqual(row_1877["expected_final_answer"], "1877")
        self.assertEqual(row_1877["expected_final_program_result"], 18770.0)
        self.assertEqual(row_1877["source_execution_answers"][-1], 18770.0)

    def test_model_facing_projection_matches_origin_main_hash(self) -> None:
        self.assertEqual(
            builder._model_facing_hash(self.benchmark),
            builder.ORIGIN_MAIN_MODEL_FACING_SHA256,
        )

    def test_repeated_build_is_byte_identical_to_committed_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            first_result = builder.build_from_subset(copy.deepcopy(self.subset), Path(first))
            second_result = builder.build_from_subset(copy.deepcopy(self.subset), Path(second))
            for attribute, committed in (
                ("benchmark_path", BENCHMARK_PATH),
                ("fixture_path", FIXTURE_PATH),
                ("subset_path", SUBSET_PATH),
                ("manifest_path", MANIFEST_PATH),
            ):
                first_path = getattr(first_result, attribute)
                second_path = getattr(second_result, attribute)
                self.assertEqual(first_path.read_bytes(), second_path.read_bytes())
                self.assertEqual(first_path.read_bytes(), committed.read_bytes())

    def test_manifest_records_all_source_and_generated_hashes(self) -> None:
        source = self.manifest["source"]
        self.assertEqual(source["repository"], builder.SOURCE_REPOSITORY)
        self.assertEqual(source["revision"], builder.SOURCE_REVISION)
        self.assertEqual(source["archive_sha256"], builder.SOURCE_ARCHIVE_SHA256)
        self.assertEqual(source["member"], builder.SOURCE_MEMBER)
        self.assertEqual(source["member_sha256"], builder.SOURCE_MEMBER_SHA256)
        self.assertEqual(source["license"], builder.SOURCE_LICENSE)
        generated = self.manifest["generated_artifacts"]
        self.assertEqual(generated[builder.BENCHMARK_OUTPUT_NAME], _sha256(BENCHMARK_PATH))
        self.assertEqual(generated[builder.FIXTURE_OUTPUT_NAME], _sha256(FIXTURE_PATH))
        self.assertEqual(generated[builder.SUBSET_OUTPUT_NAME], _sha256(SUBSET_PATH))
        self.assertEqual(
            self.manifest["selection"]["source_subset_canonical_sha256"],
            builder.EXPECTED_SOURCE_SUBSET_SHA256,
        )

    def test_archive_and_member_checksum_validation(self) -> None:
        records: list[object] = [{} for _ in range(builder.SOURCE_EXAMPLE_COUNT)]
        for source in self.subset:
            records[source["source_row_index"]] = {
                "id": source["source_conversation_id"],
                "filename": source["source_filename"],
                "qa": {
                    "question": source["question"],
                    "answer": source["display_answer"],
                    "gold_inds": source["gold_evidence"],
                    "program": source["source_program"],
                    "exe_ans": source["source_execution_answer"],
                },
                "annotation": {
                    "original_program": source["original_program"],
                    "dialogue_break": source["dialogue_break"],
                    "turn_program": source["turn_program"],
                    "exe_ans_list": source["execution_answers"],
                },
            }
        member_bytes = json.dumps(records, ensure_ascii=False).encode("utf-8")
        with tempfile.TemporaryDirectory() as temporary:
            archive_path = Path(temporary) / "data.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr(builder.SOURCE_MEMBER, member_bytes)
            archive_hash = _sha256(archive_path)
            member_hash = hashlib.sha256(member_bytes).hexdigest()
            with mock.patch.object(builder, "SOURCE_ARCHIVE_SHA256", archive_hash), mock.patch.object(
                builder, "SOURCE_MEMBER_SHA256", member_hash
            ):
                self.assertEqual(builder._read_pinned_archive(archive_path), self.subset)
                archive_path.write_bytes(archive_path.read_bytes() + b"altered")
                with self.assertRaisesRegex(ValueError, "archive hash mismatch"):
                    builder._read_pinned_archive(archive_path)

    def test_altered_source_rows_programs_hashes_and_translations_fail(self) -> None:
        altered_question = copy.deepcopy(self.subset)
        altered_question[0]["question"] += " altered"
        with self.assertRaisesRegex(ValueError, "selected source fields changed"):
            builder._validate_subset(altered_question)

        altered_program = copy.deepcopy(self.subset)
        altered_program[0]["turn_program"][0] = "60.95"
        with self.assertRaisesRegex(ValueError, "selected source fields changed"):
            builder._validate_subset(altered_program)

        with mock.patch.object(builder, "EXPECTED_SOURCE_SUBSET_SHA256", "0" * 64):
            with self.assertRaisesRegex(ValueError, "canonical subset hash"):
                builder._validate_subset(copy.deepcopy(self.subset))

        altered_symbols = dict(builder._SYMBOLS)
        altered_symbols["subtract"] = "+"
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(
            builder, "_SYMBOLS", altered_symbols
        ):
            with self.assertRaisesRegex(ValueError, "translation changed"):
                builder.build_from_subset(copy.deepcopy(self.subset), Path(temporary))

    def test_altered_evidence_and_serialization_fail_before_writing(self) -> None:
        first = builder.EVIDENCE_SPECS[0]
        altered_specs = (
            builder.EvidenceSpec(
                first.source_row_index,
                first.metric,
                first.period,
                first.numeric_value + 0.01,
                first.evidence_key,
            ),
            *builder.EVIDENCE_SPECS[1:],
        )
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(
            builder, "EVIDENCE_SPECS", altered_specs
        ):
            with self.assertRaises(ValueError):
                builder.build_from_subset(copy.deepcopy(self.subset), Path(temporary))

        original_pretty = builder._pretty_bytes

        def altered_pretty(value: object) -> bytes:
            return original_pretty(value) + b"\n"

        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(
            builder, "_pretty_bytes", altered_pretty
        ):
            root = Path(temporary)
            with self.assertRaisesRegex(ValueError, "serialization or generated content"):
                builder.build_from_subset(copy.deepcopy(self.subset), root)
            self.assertFalse((root / builder.BENCHMARK_OUTPUT_NAME).exists())


if __name__ == "__main__":
    unittest.main()
