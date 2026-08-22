from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from analysis.multistep_scorecard import build_scorecard
from evaluation.evaluate import OUTCOME_METRIC_NAMES


class MultiStepScorecardTests(unittest.TestCase):
    def _run(
        self,
        root: Path,
        *,
        program_accuracy=None,
        dataset: str = "finretrieval",
        headline: bool = True,
        model: str = "model",
        reasoning_mode: str = "direct",
        reasoning_method: str = "none",
        reasoning_effort: str | None = None,
        generation_limit: int = 128,
    ) -> Path:
        run = root / "run"
        dataset_path = run / "domains/finance" / dataset
        dataset_path.mkdir(parents=True)
        (run / "RUN_COMPLETE").write_text("complete\n", encoding="utf-8")
        contract = {
            "outcome_metric_names": list(OUTCOME_METRIC_NAMES),
            "headline_eligible": headline,
        }
        (run / "run_metadata.json").write_text(
            json.dumps(contract), encoding="utf-8"
        )
        summary = {
            **contract,
            "model_name": model,
            "reasoning_mode": reasoning_mode,
            "reasoning_method": reasoning_method,
            "effective_generation_limit": generation_limit,
            "effective_generation_limit_unit": "tokens",
            "tool_selection_accuracy": 0.9,
            "argument_accuracy": 0.8,
            "step_outcome_accuracy": 0.7,
            "step_outcome_scored": 10,
            "step_outcome_status_counts": {"correct": 7, "mismatch": 3},
            "step_outcome_matchers": ["recursive_json_subset_v1"],
            "all_tools_correct_accuracy": 0.6,
            "all_arguments_correct_accuracy": 0.6,
            "all_steps_correct_accuracy": 0.6,
            "all_steps_correct_scored": 10,
            "final_step_outcome_accuracy": 0.5,
            "final_step_outcome_gold": 10,
            "final_step_outcome_scored": 10,
            "final_step_outcome_correct": 5,
            "final_step_outcome_mismatch": 5,
            "final_step_outcome_extraction_error": 0,
            "final_step_outcome_unavailable": 0,
            "final_step_outcome_status_counts": {"correct": 5, "mismatch": 5},
            "final_step_outcome_contracts": ["final_step_expected_outcome"],
            "final_step_outcome_matchers": ["recursive_json_subset_v1"],
        }
        if reasoning_effort is not None:
            summary["reasoning_effort"] = reasoning_effort
        if program_accuracy is not None:
            summary.update(
                {
                    "final_program_execution_accuracy": program_accuracy,
                    "final_program_execution_gold": 10,
                    "final_program_execution_scored": 10,
                    "final_program_execution_correct": 7,
                    "final_program_execution_mismatch": 3,
                    "final_program_execution_extraction_error": 0,
                    "final_program_execution_unavailable": 0,
                    "final_program_execution_status_counts": {
                        "correct": 7,
                        "mismatch": 3,
                    },
                    "final_program_execution_contracts": [
                        "finqa_program_execution"
                    ],
                    "final_program_execution_matchers": [
                        "finqa_program_execution"
                    ],
                }
            )
        (dataset_path / "summary.json").write_text(
            json.dumps(summary), encoding="utf-8"
        )
        return run

    def test_renders_clear_separated_outcome_headings(self) -> None:
        with TemporaryDirectory() as temporary:
            markdown = build_scorecard([self._run(Path(temporary))])
            for heading in (
                "Tool Selection",
                "Argument Accuracy",
                "Step Outcome Accuracy",
                "All Steps Correct",
                "Final Step Outcome",
            ):
                self.assertIn(heading, markdown)
            self.assertNotIn("Final Program Execution", markdown)
            self.assertNotIn("workflow_final_answer_accuracy", markdown)

    def test_program_column_appears_only_when_applicable(self) -> None:
        with TemporaryDirectory() as temporary:
            markdown = build_scorecard(
                [self._run(Path(temporary), program_accuracy=0.75)]
            )
            self.assertIn("Final Program Execution", markdown)
            self.assertIn("75.00%", markdown)

    def test_gpt_oss_harmony_low_condition_is_labeled(self) -> None:
        with TemporaryDirectory() as temporary:
            run = self._run(
                Path(temporary),
                model="openai/gpt-oss-20b",
                reasoning_mode="reasoning",
                reasoning_method="harmony",
                reasoning_effort="low",
                generation_limit=4096,
            )
            markdown = build_scorecard([run])
            self.assertIn("Reasoning — Harmony LOW", markdown)
            self.assertIn("4096 tokens", markdown)

    def test_rejects_historical_scalar_in_metadata(self) -> None:
        with TemporaryDirectory() as temporary:
            run = self._run(Path(temporary))
            metadata_path = run / "run_metadata.json"
            metadata = json.loads(metadata_path.read_text())
            metadata["workflow_final_answer_accuracy"] = 0.0
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Historical scalar"):
                build_scorecard([run])

    def test_rejects_historical_scalar_in_summary(self) -> None:
        with TemporaryDirectory() as temporary:
            run = self._run(Path(temporary))
            summary_path = next(run.glob("domains/*/*/summary.json"))
            summary = json.loads(summary_path.read_text())
            summary["workflow_final_answer_accuracy"] = 0.0
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Historical scalar"):
                build_scorecard([run])

    def test_rejects_missing_corrected_metric_field(self) -> None:
        with TemporaryDirectory() as temporary:
            run = self._run(Path(temporary))
            summary_path = next(run.glob("domains/*/*/summary.json"))
            summary = json.loads(summary_path.read_text())
            del summary["final_step_outcome_matchers"]
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "fields are missing"):
                build_scorecard([run])

    def test_public_mathqa_has_explicit_label(self) -> None:
        with TemporaryDirectory() as temporary:
            markdown = build_scorecard([
                self._run(Path(temporary), dataset="math_public_mathqa_multistep")
            ])
            self.assertIn("MathQA public-derived", markdown)

    def test_non_headline_diagnostic_is_excluded(self) -> None:
        with TemporaryDirectory() as temporary:
            markdown = build_scorecard([
                self._run(Path(temporary), dataset="math_multistep_controlled", headline=False)
            ])
            self.assertNotIn("Math controlled diagnostic", markdown)


if __name__ == "__main__":
    unittest.main()
