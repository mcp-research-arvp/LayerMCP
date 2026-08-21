from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from analysis.multistep_scorecard import build_scorecard
from evaluation.evaluate import WORKFLOW_FINAL_SCORING_VERSION


class MultiStepScorecardTests(unittest.TestCase):
    def _run(self, root: Path, version: str) -> Path:
        run = root / "run"
        dataset = run / "domains/finance/finqa"
        dataset.mkdir(parents=True)
        (run / "RUN_COMPLETE").write_text("complete\n", encoding="utf-8")
        (run / "run_metadata.json").write_text(
            json.dumps({"workflow_final_scoring_version": version}), encoding="utf-8"
        )
        summary = {
            "workflow_final_scoring_version": version,
            "model_name": "model",
            "reasoning_mode": "direct",
            "source_run_identity": "original-run",
        }
        for prefix in (
            "workflow_final_answer",
            "workflow_final_program_execution",
            "workflow_final_tool_result",
        ):
            summary.update(
                {
                    f"{prefix}_accuracy": None,
                    f"{prefix}_gold": 1,
                    f"{prefix}_scored": 0,
                    f"{prefix}_correct": 0,
                    f"{prefix}_mismatch": 0,
                    f"{prefix}_extraction_error": 0,
                    f"{prefix}_unavailable": 1,
                    f"{prefix}_status_counts": {"unsupported": 1},
                    f"{prefix}_contracts": [],
                    f"{prefix}_matchers": [],
                    f"{prefix}_scoring_version": version,
                }
            )
        (dataset / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
        return run

    def test_renders_version_contract_counts_and_source_identity(self) -> None:
        with TemporaryDirectory() as temporary:
            markdown = build_scorecard(
                [self._run(Path(temporary), WORKFLOW_FINAL_SCORING_VERSION)]
            )
            self.assertIn("workflow_final_program_execution", markdown)
            self.assertIn("unavailable", markdown)
            self.assertIn("original-run", markdown)

    def test_rejects_old_scoring_semantics(self) -> None:
        with TemporaryDirectory() as temporary:
            run = self._run(Path(temporary), "workflow_final_metrics_v1")
            with self.assertRaisesRegex(ValueError, "Incompatible"):
                build_scorecard([run])


if __name__ == "__main__":
    unittest.main()
