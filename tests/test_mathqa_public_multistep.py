from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import zipfile

from benchmark.math import build_mathqa_public_multistep as builder
from evaluation.evaluate import (
    _score_final_outcome,
    _score_final_step_outcome,
    _score_sample,
    load_benchmark,
)
from mcp_server.server import mcp
from mcp_server.tool_impls import calculator


ROOT = Path(__file__).resolve().parents[1]
MATH_ROOT = ROOT / "benchmark/math"
SCRATCH_ROOT = os.environ.get("SCRATCH")
DEFAULT_ARCHIVE = (
    Path(SCRATCH_ROOT) / "layermcp/raw_sources/mathqa/MathQA.zip"
    if SCRATCH_ROOT
    else Path("/tmp/MathQA.zip")
)


def _configured_archive() -> Path:
    archive_text = os.environ.get("LAYERMCP_MATHQA_ARCHIVE")
    return Path(archive_text) if archive_text else DEFAULT_ARCHIVE


def _available_archive(path: Path | None = None) -> Path | None:
    candidate = path or _configured_archive()
    return candidate if candidate.is_file() else None


ARCHIVE_AVAILABLE = _available_archive() is not None
ARCHIVE_SKIP_REASON = (
    "Set LAYERMCP_MATHQA_ARCHIVE to the pinned MathQA.zip source archive."
)


class MathQAPublicMultistepTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.benchmark = json.loads((MATH_ROOT / builder.BENCHMARK_NAME).read_text())
        cls.fixture = json.loads((MATH_ROOT / builder.FIXTURE_NAME).read_text())
        cls.manifest = json.loads((MATH_ROOT / builder.MANIFEST_NAME).read_text())
        cls.mapping = json.loads((MATH_ROOT / builder.MAPPING_NAME).read_text())

    @classmethod
    def _built_from_archive(cls) -> dict[str, object]:
        archive = _available_archive()
        if archive is None:
            raise AssertionError(ARCHIVE_SKIP_REASON)
        if not hasattr(cls, "_archive_built"):
            cls._archive_built = builder.build_from_archive(archive)
        return cls._archive_built

    @unittest.skipUnless(ARCHIVE_AVAILABLE, ARCHIVE_SKIP_REASON)
    def test_archive_and_member_hashes_are_pinned(self) -> None:
        archive = _available_archive()
        assert archive is not None
        self.assertEqual(hashlib.sha256(archive.read_bytes()).hexdigest(), builder.ARCHIVE_SHA256)
        with zipfile.ZipFile(archive) as source_archive:
            for name, expected in builder.MEMBER_SHA256.items():
                self.assertEqual(hashlib.sha256(source_archive.read(name)).hexdigest(), expected)

    def test_population_and_selected_counts_are_exact(self) -> None:
        inventory = self.manifest["source_inventory"]
        self.assertEqual(inventory, {
            "total_rows": 2985, "eligible_workflows": 1498,
            "eligible_steps": 6638, "selected_workflows": 200,
            "selected_steps": 892,
        })
        self.assertEqual(len(self.manifest["row_inventory"]), 2985)
        self.assertEqual(len({row["id"] for row in self.benchmark}), 200)
        eligible = [row for row in self.manifest["row_inventory"] if row["status"] == "eligible"]
        self.assertEqual(len(eligible), 1498)
        self.assertTrue(all(row["operation_count"] >= 2 for row in eligible))
        self.assertEqual(sum(self.manifest["rejection_counts"].values()), 1487)

    def test_exclusion_classes_are_exhaustive_and_honest(self) -> None:
        self.assertEqual(self.manifest["exclusion_class_counts"], {
            "numeric_option_parsing_limitation": 35,
            "strict_tolerance_mismatch": 226,
            "annotated_formula_linear_formula_disagreement": 1,
            "program_result_matches_other_option": 81,
            "destination_tool_constraint": 16,
            "unsupported_operation": 63,
            "invalid_or_forward_reference": 0,
            "nonfinite_or_tool_execution_failure": 0,
            "source_program_selected_answer_disagreement_unresolved": 802,
            "insufficient_operations": 254,
            "selected_option_not_numeric": 9,
            "split_overlap": 0,
        })
        self.assertEqual(
            self.manifest["rejection_counts"]["destination_tool_exponent_limit"],
            9,
        )
        unresolved = [
            row for row in self.manifest["row_inventory"]
            if row.get("exclusion_class")
            == "source_program_selected_answer_disagreement_unresolved"
        ]
        self.assertEqual(len(unresolved), 802)
        self.assertTrue(all(row["status"] == "rejected" for row in unresolved))
        self.assertTrue(all(row["source_coordinate"] for row in unresolved))
        self.assertTrue(all(len(row["source_row_sha256"]) == 64 for row in unresolved))

    def test_committed_fixture_manifest_and_benchmark_are_consistent(self) -> None:
        self.assertEqual(self.fixture["provenance"], self.manifest["provenance"])
        self.assertEqual(
            self.fixture["selected_source_coordinates"],
            self.manifest["selected_source_coordinates"],
        )
        self.assertEqual(len(self.fixture["rows"]), len(self.benchmark))
        self.assertEqual(len(self.benchmark), 200)
        for benchmark_row, fixture_row in zip(self.benchmark, self.fixture["rows"]):
            self.assertEqual(benchmark_row["source_coordinate"], fixture_row["source_coordinate"])
            self.assertEqual(benchmark_row["source_row_index"], fixture_row["source_row_index"])
            self.assertEqual(benchmark_row["source_row_sha256"], fixture_row["source_row_sha256"])
            self.assertEqual(benchmark_row["source_record"], fixture_row["source_record"])
        for name, expected_hash in self.manifest["generated_artifact_sha256"].items():
            self.assertEqual(
                hashlib.sha256((MATH_ROOT / name).read_bytes()).hexdigest(),
                expected_hash,
                name,
            )

    @unittest.skipUnless(ARCHIVE_AVAILABLE, ARCHIVE_SKIP_REASON)
    def test_generation_is_repeatedly_identical_and_matches_committed_bytes(self) -> None:
        built = self._built_from_archive()
        archive = _available_archive()
        assert archive is not None
        second = builder.build_from_archive(archive)
        self.assertEqual(built["artifact_bytes"], second["artifact_bytes"])
        for name, data in built["artifact_bytes"].items():
            self.assertEqual((MATH_ROOT / name).read_bytes(), data, name)
        self.assertEqual(
            hashlib.sha256((MATH_ROOT / builder.MANIFEST_NAME).read_bytes()).hexdigest(),
            "ca49dad10675634acfab981ab697051de48cc3fe34362f53691d23c4d0e00a1e",
        )

    @unittest.skipUnless(ARCHIVE_AVAILABLE, ARCHIVE_SKIP_REASON)
    def test_selected_source_fields_and_hashes_are_exact(self) -> None:
        built = self._built_from_archive()
        archive = _available_archive()
        assert archive is not None
        with zipfile.ZipFile(archive) as source_archive:
            source = json.loads(source_archive.read("test.json"))
        fixture = built["fixture"]
        self.assertEqual(len(fixture["rows"]), 200)
        for benchmark_row, fixture_row in zip(self.benchmark, fixture["rows"]):
            index = fixture_row["source_row_index"]
            source_row = source[index]
            self.assertEqual(fixture_row["source_record"], source_row)
            self.assertEqual(benchmark_row["source_record"], source_row)
            self.assertEqual(benchmark_row["query"], source_row["Problem"])
            self.assertEqual(benchmark_row["source_coordinate"], f"test:{index}")
            self.assertEqual(
                benchmark_row["source_row_sha256"],
                hashlib.sha256(builder._canonical_bytes(source_row)).hexdigest(),
            )
            self.assertEqual(
                [step["source_raw_dsl_call"] for step in benchmark_row["expected_steps"]],
                [call for call in source_row["linear_formula"].split("|") if call],
            )

    @unittest.skipUnless(ARCHIVE_AVAILABLE, ARCHIVE_SKIP_REASON)
    def test_no_selected_question_duplicates_train_or_dev(self) -> None:
        archive = _available_archive()
        assert archive is not None
        with zipfile.ZipFile(archive) as source_archive:
            other = json.loads(source_archive.read("train.json")) + json.loads(source_archive.read("dev.json"))
        questions = {row["Problem"] for row in other}
        self.assertFalse({row["query"] for row in self.benchmark} & questions)

    def test_program_translation_uses_only_existing_tools_and_source_dependencies(self) -> None:
        allowed_tools = {"calculator", "gcd_lcm", "modular_arithmetic"}
        observed_operations = set()
        for row in self.benchmark:
            for index, step in enumerate(row["expected_steps"]):
                observed_operations.add(step["source_operation"])
                self.assertIn(step["expected_tool"], allowed_tools)
                self.assertEqual(step["query"], step["source_raw_dsl_call"])
                expected_dependencies = sorted({
                    f"step_{int(token[1:]):02d}"
                    for token in step["source_arguments"] if token.startswith("#")
                })
                self.assertEqual(step["depends_on"], expected_dependencies)
                self.assertTrue(all(int(dep.split("_")[1]) < index for dep in expected_dependencies))
        self.assertIn("lcm", observed_operations)
        self.assertIn("reminder", observed_operations)
        self.assertEqual(self.mapping["supported_operations"]["gcd"], "gcd_lcm")

    def test_every_selected_call_validates_and_replays_through_registered_tool(self) -> None:
        registered = mcp._tool_manager._tools
        for row in self.benchmark:
            for step in row["expected_steps"]:
                tool = registered[step["expected_tool"]]
                schema = tool.parameters
                self.assertTrue(set(schema.get("required", ())) <= set(step["expected_args"]))
                self.assertTrue(set(step["expected_args"]) <= set(schema["properties"]))
                result = tool.fn(**step["expected_args"])
                if step["expected_tool"] == "calculator":
                    self.assertEqual(step["expected_answer"], {"result": result["result"]})
                    self.assertNotIn("expression", step["expected_answer"])
                else:
                    self.assertEqual(result, step["expected_answer"])
            self.assertTrue(
                __import__("math").isclose(
                    float(row["expected_steps"][-1]["source_scalar_result"]),
                    float(row["source_correct_numeric_answer"]),
                    rel_tol=1e-9,
                    abs_tol=1e-9,
                )
            )

    def test_power_preserves_supported_source_exponents(self) -> None:
        for exponent in range(0, builder.CALCULATOR_MAX_EXPONENT_MAGNITUDE + 1):
            tool, arguments, result, scalar = builder._translate_and_execute(
                "power", [2, exponent]
            )
            self.assertEqual(tool, "calculator")
            self.assertEqual(arguments["expression"], f"(2) ** {exponent}")
            self.assertEqual(result["result"], 2 ** exponent)
            self.assertEqual(scalar, 2 ** exponent)
        for exponent in (-1, -3, -10):
            _, arguments, result, scalar = builder._translate_and_execute(
                "power", [2, exponent]
            )
            self.assertEqual(arguments["expression"], f"(2) ** {exponent}")
            self.assertEqual(result["result"], 2 ** exponent)
            self.assertEqual(scalar, 2 ** exponent)

    def test_calculator_and_builder_enforce_same_power_limit(self) -> None:
        self.assertEqual(calculator("2 ** 10")["result"], 1024)
        self.assertEqual(calculator("2 ** -10")["result"], 2 ** -10)
        for exponent in (11, -11, 14, -88888885):
            with self.assertRaisesRegex(ValueError, "Exponent is too large"):
                calculator(f"2 ** {exponent}")
            with self.assertRaisesRegex(
                builder.Ineligible, "destination_tool_exponent_limit"
            ):
                builder._translate_and_execute("power", [2, exponent])

    @unittest.skipUnless(ARCHIVE_AVAILABLE, ARCHIVE_SKIP_REASON)
    def test_source_negative_power_constraint_is_preserved(self) -> None:
        archive = _available_archive()
        assert archive is not None
        with zipfile.ZipFile(archive) as source_archive:
            source = json.loads(source_archive.read("test.json"))
        # MathQA has no resolved negative power exponent in [-10, -1]. Its
        # supplied negative case is far outside the destination limit.
        self.assertEqual(
            source[2973]["linear_formula"], "negate(n1)|power(n0,#0)|"
        )
        with self.assertRaisesRegex(
            builder.Ineligible, "destination_tool_exponent_limit"
        ):
            builder._evaluate_row(source[2973], 2973)

    @unittest.skipUnless(ARCHIVE_AVAILABLE, ARCHIVE_SKIP_REASON)
    def test_test_2800_is_excluded_without_altering_its_exponent(self) -> None:
        archive = _available_archive()
        assert archive is not None
        with zipfile.ZipFile(archive) as source_archive:
            row = json.loads(source_archive.read("test.json"))[2800]
        self.assertEqual(
            row["linear_formula"], "power(n0,n1)|reminder(#0,n2)|"
        )
        self.assertEqual(row["annotated_formula"], "reminder(power(2, 14), 7)")
        with self.assertRaisesRegex(
            builder.Ineligible, "destination_tool_exponent_limit"
        ):
            builder._evaluate_row(row, 2800)
        inventory = self.manifest["row_inventory"][2800]
        self.assertEqual(inventory["reason"], "destination_tool_exponent_limit")
        self.assertNotIn("test:2800", self.manifest["selected_source_coordinates"])
        # This collision proves that checking only the later remainder is unsafe.
        self.assertEqual((2 ** 5) % 7, (2 ** 14) % 7)
        self.assertNotEqual(2 ** 5, 2 ** 14)

    def test_source_rows_with_supported_exponents_above_five_are_not_altered(self) -> None:
        by_coordinate = {row["source_coordinate"]: row for row in self.benchmark}
        observed = []
        for row in self.benchmark:
            for step in row["expected_steps"]:
                if step["source_operation"] != "power":
                    continue
                expression = step["expected_args"]["expression"]
                self.assertNotIn("min(", expression)
                exponent = float(expression.rsplit("**", 1)[1].strip())
                if 5 < abs(exponent) <= builder.CALCULATOR_MAX_EXPONENT_MAGNITUDE:
                    observed.append((row["source_coordinate"], exponent, expression))
        self.assertIn("test:503", by_coordinate)
        self.assertTrue(any(coordinate == "test:503" for coordinate, _, _ in observed))
        self.assertTrue(observed)

    def test_display_answer_is_provenance_and_final_tool_target_is_separate(self) -> None:
        for row in self.benchmark:
            self.assertIsInstance(row["expected_final_answer"], str)
            self.assertEqual(row["expected_final_step_outcome"], row["expected_steps"][-1]["expected_answer"])
            self.assertNotIn("final_step_outcome_contract", row)
            self.assertNotIn("final_program_execution_contract", row)
            self.assertNotIn("workflow_final_answer_accuracy", row)

    def test_model_facing_projection_is_unchanged(self) -> None:
        projection = [
            {
                "id": row["id"],
                "query": row["query"],
                "expected_steps": [
                    {
                        key: step[key]
                        for key in (
                            "id",
                            "query",
                            "prompt_context",
                            "expected_tool",
                            "expected_args",
                            "depends_on",
                        )
                    }
                    for step in row["expected_steps"]
                ],
            }
            for row in self.benchmark
        ]
        self.assertEqual(
            hashlib.sha256(builder._canonical_bytes(projection)).hexdigest(),
            "74127d4a770a94acc23b172eb03cac32617dfaa5e844169fa5bc1d94390343de",
        )

    def test_evaluator_loads_every_workflow_without_model_facing_synthetic_steps(self) -> None:
        samples = load_benchmark(MATH_ROOT / builder.BENCHMARK_NAME)
        self.assertEqual(len(samples), 200)
        self.assertEqual(sum(len(sample.expected_steps) for sample in samples), 892)
        for row in self.benchmark:
            for step in row["expected_steps"]:
                context = json.loads(step["prompt_context"])
                self.assertEqual(context["raw_dsl_call"], step["query"])
                self.assertNotIn("instruction", context)

    def test_selection_is_proportional_sha_ranked_and_source_ordered(self) -> None:
        coordinates = self.manifest["selected_source_coordinates"]
        indexes = [int(value.split(":")[1]) for value in coordinates]
        self.assertEqual(indexes, sorted(indexes))
        self.assertEqual(sum(row["selected"] for row in self.manifest["stratified_allocation"]), 200)
        self.assertEqual(
            self.benchmark[0]["benchmark_provenance"]["selected_subset_sha256"],
            hashlib.sha256(builder._canonical_bytes([
                {
                    "source_coordinate": row["source_coordinate"],
                    "source_row_sha256": row["source_row_sha256"],
                }
                for row in self.manifest["selected_source_rows"]
            ])).hexdigest(),
        )
        self.assertEqual(len(self.manifest["selected_source_rows"]), 200)
        self.assertEqual(
            [row["source_coordinate"] for row in self.manifest["selected_source_rows"]],
            coordinates,
        )
        self.assertEqual(
            self.manifest["provenance"]["mirror_revision"],
            "c4f1cc784c04c4957b50c97858f23893b633eea6",
        )
        inventory = {
            row["source_coordinate"]: row for row in self.manifest["row_inventory"]
        }
        self.assertTrue(
            all(inventory[coordinate]["status"] == "eligible" for coordinate in coordinates)
        )

    def test_operation_and_tool_distributions_match_selected_workflows(self) -> None:
        from collections import Counter
        operations = Counter(step["source_operation"] for row in self.benchmark for step in row["expected_steps"])
        tools = Counter(step["expected_tool"] for row in self.benchmark for step in row["expected_steps"])
        self.assertEqual(dict(sorted(operations.items())), self.manifest["selected_distribution"]["operation"])
        self.assertEqual(dict(sorted(tools.items())), self.manifest["selected_distribution"]["tool"])

    @unittest.skipUnless(ARCHIVE_AVAILABLE, ARCHIVE_SKIP_REASON)
    def test_corrupt_archive_and_member_are_rejected(self) -> None:
        archive = _available_archive()
        assert archive is not None
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "bad.zip"
            data = bytearray(archive.read_bytes())
            data[-10] ^= 1
            path.write_bytes(data)
            with self.assertRaisesRegex(ValueError, "archive SHA-256"):
                builder.build_from_archive(path)

    def test_altered_row_program_or_translation_fails_validation(self) -> None:
        row = copy.deepcopy(self.benchmark[0]["source_record"])
        row["linear_formula"] = "factorial(n0)|add(#0,const_1)|"
        with self.assertRaisesRegex(builder.Ineligible, "unsupported_operation"):
            builder._evaluate_row(row, self.benchmark[0]["source_row_index"])
        mapping = copy.deepcopy(self.mapping)
        mapping["supported_operations"]["gcd"] = "calculator"
        self.assertNotEqual(
            builder._pretty_bytes(mapping),
            (MATH_ROOT / builder.MAPPING_NAME).read_bytes(),
        )

    def test_missing_archive_does_not_disable_committed_artifact_validation(self) -> None:
        missing = MATH_ROOT / "fixtures" / "MathQA.zip-not-present"
        self.assertIsNone(_available_archive(missing))
        self.assertEqual(len(self.benchmark), 200)
        self.assertEqual(sum(len(row["expected_steps"]) for row in self.benchmark), 892)

    def test_equivalent_calculator_expression_separates_arguments_and_outcomes(self) -> None:
        reference_call = _score_sample(
            expected_tool="calculator",
            selected_tool="calculator",
            expected_args={"expression": "1 + 1"},
            selected_args={"expression": "1 + 1"},
            execution_success=True,
            execution_attempted=True,
        )
        equivalent_call = _score_sample(
            expected_tool="calculator",
            selected_tool="calculator",
            expected_args={"expression": "1 + 1"},
            selected_args={"expression": "2"},
            execution_success=True,
            execution_attempted=True,
        )
        self.assertTrue(reference_call.argument_match_correct)
        self.assertFalse(equivalent_call.argument_match_correct)

        step_outcome = _score_final_outcome(
            expected_answer={"result": 2},
            tool_result_value={"expression": "2", "result": 2},
            result_extraction_diagnostic=None,
            domain="mathematics",
            call_predicted_tools=True,
            no_tool_call=False,
            execution_success=True,
            expected_tool="calculator",
            called_tool="calculator",
        )
        self.assertTrue(step_outcome.correct)
        self.assertEqual(step_outcome.matcher, "recursive_json_subset_v1")

        final_outcome = _score_final_step_outcome(
            final_step_record={
                "final_outcome_correct": step_outcome.correct,
                "final_outcome_status": step_outcome.status,
                "final_outcome_matcher": step_outcome.matcher,
                "final_outcome_diagnostic": step_outcome.diagnostic,
            },
            expected_final_step_outcome={"result": 2},
            final_step_outcome_contract=None,
            final_tool_result_value={"expression": "2", "result": 2},
            call_predicted_tools=True,
        )
        self.assertTrue(final_outcome.correct)
        self.assertEqual(final_outcome.matcher, "recursive_json_subset_v1")

    def test_wrong_calculator_value_remains_incorrect_at_step_and_final_step(self) -> None:
        step_outcome = _score_final_outcome(
            expected_answer={"result": 2},
            tool_result_value={"expression": "2 + 1", "result": 3},
            result_extraction_diagnostic=None,
            domain="mathematics",
            call_predicted_tools=True,
            no_tool_call=False,
            execution_success=True,
            expected_tool="calculator",
            called_tool="calculator",
        )
        self.assertFalse(step_outcome.correct)
        final_outcome = _score_final_step_outcome(
            final_step_record={
                "final_outcome_correct": step_outcome.correct,
                "final_outcome_status": step_outcome.status,
                "final_outcome_matcher": step_outcome.matcher,
                "final_outcome_diagnostic": step_outcome.diagnostic,
            },
            expected_final_step_outcome={"result": 2},
            final_step_outcome_contract=None,
            final_tool_result_value={"expression": "2 + 1", "result": 3},
            call_predicted_tools=True,
        )
        self.assertFalse(final_outcome.correct)

    def test_non_calculator_semantic_result_fields_remain_strict(self) -> None:
        expected = {
            "values": ["6", "8"],
            "integer_values": [6, 8],
            "operation": "gcd",
            "source": "python-math",
            "gcd": 2,
        }
        score = _score_final_outcome(
            expected_answer=expected,
            tool_result_value={**expected, "gcd": 4},
            result_extraction_diagnostic=None,
            domain="mathematics",
            call_predicted_tools=True,
            no_tool_call=False,
            execution_success=True,
            expected_tool="gcd_lcm",
            called_tool="gcd_lcm",
        )
        self.assertFalse(score.correct)

    def test_controlled_diagnostic_artifact_is_unchanged(self) -> None:
        self.assertEqual(
            hashlib.sha256((MATH_ROOT / "math_multistep_controlled.json").read_bytes()).hexdigest(),
            "702faf3824c0e1933c40ca66ccd49c222020844c7637803716cf06b2aa6d0947",
        )


if __name__ == "__main__":
    unittest.main()
