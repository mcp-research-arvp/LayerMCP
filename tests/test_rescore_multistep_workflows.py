from __future__ import annotations

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from analysis.rescore_multistep_workflows import rescore_multistep_run
from evaluation.evaluate import (
    MULTISTEP_EVALUATION_PROTOCOL,
    WORKFLOW_FINAL_SCORING_VERSION,
)


class RescoreMultiStepWorkflowTests(unittest.TestCase):
    @staticmethod
    def _snapshot(path: Path) -> dict[str, str]:
        return {
            str(item.relative_to(path)): hashlib.sha256(item.read_bytes()).hexdigest()
            for item in sorted(path.rglob("*"))
            if item.is_file()
        }

    def _source(self, root: Path) -> tuple[Path, Path, bytes]:
        benchmark = root / "benchmark/finance/example.json"
        benchmark.parent.mkdir(parents=True)
        rows = [{
            "id": "workflow-1", "expected_final_answer": "9.9%",
            "expected_final_program_result": 0.09864,
            "workflow_final_program_contract": "finqa_execution_v1",
            "expected_steps": [{"id": "step-1"}],
        }]
        benchmark_bytes = (json.dumps(rows, indent=2) + "\n").encode()
        benchmark.write_bytes(benchmark_bytes)
        source = root / "source-run"
        dataset = source / "domains/finance/example"
        dataset.mkdir(parents=True)
        fingerprint = "sha256:" + "a" * 64
        record = {
            "sample_id": "workflow-1", "benchmark_path": str(benchmark),
            "expected_final_answer": "9.9%", "final_step_result_value": {"result": 0.098641},
            "model_name": "model", "prompt_template": "prompt", "reasoning_mode": "direct",
            "reasoning_method": "none", "effective_generation_limit": 128,
            "effective_generation_limit_unit": "tokens", "tool_pool": "full_mcp_registry",
            "tool_count": 60, "tool_registry_fingerprint": fingerprint,
            "tool_registry_fingerprint_version": "registry-v1",
            "evaluation_protocol": MULTISTEP_EVALUATION_PROTOCOL,
            "workflow_execution_mode": "predicted_sequence",
            "benchmark_mode": "grounded_tool_execution",
            "sequence_tool_selection_correct": True,
            "sequence_argument_match_correct": True,
            "sequence_semantic_output_correct": True,
            "steps": [{
                "step_id": "step-1", "execution_success": True,
                "tool_selection_correct": True, "argument_match_correct": True,
                "final_outcome_correct": True,
            }],
            "raw_generation": "preserve me", "selected_args": {"x": 1},
        }
        (dataset / "samples.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
        (dataset / "summary.json").write_text(json.dumps({
            "benchmark_path": str(benchmark), "model_name": "model",
            "prompt_template": "prompt", "reasoning_mode": "direct",
            "reasoning_method": "none", "effective_generation_limit": 128,
            "effective_generation_limit_unit": "tokens",
            "evaluation_protocol": MULTISTEP_EVALUATION_PROTOCOL,
            "total_workflows": 1, "total_steps": 1,
            "benchmark_mode_counts": {"grounded_tool_execution": 1},
            "workflow_execution_modes": ["predicted_sequence"],
            "tool_pool": "full_mcp_registry", "tool_count": 60,
            "tool_registry_fingerprint": fingerprint,
            "tool_registry_fingerprint_version": "registry-v1",
        }), encoding="utf-8")
        (dataset / "evaluation.log").write_text("original log\n", encoding="utf-8")
        metadata = {
            "git_commit": "source-commit", "expected_model_name": "model",
            "prompt_template_id": "prompt", "reasoning_mode": "direct",
            "reasoning_method": "none", "effective_generation_limit": 128,
            "effective_generation_limit_unit": "tokens", "tool_pool": "full_mcp_registry",
            "tool_count": 60, "tool_registry_fingerprint": fingerprint,
            "tool_registry_fingerprint_version": "registry-v1",
            "evaluation_protocol": MULTISTEP_EVALUATION_PROTOCOL,
            "workflow_execution_mode": "predicted_sequence",
            "run_kind": "full", "headline_eligible": True,
            "source_counts": {str(benchmark): {"workflows": 1, "routed_steps": 1}},
            "short_test_selection": {},
        }
        (source / "run_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
        (source / "artifact_index.jsonl").write_text("{}\n", encoding="utf-8")
        (source / "RUN_COMPLETE").write_text("complete\n", encoding="utf-8")
        return source, benchmark, benchmark_bytes

    @patch("analysis.rescore_multistep_workflows.subprocess.check_output", return_value="rescore-commit\n")
    def test_new_artifact_is_immutable_hashed_and_collision_safe(
        self, _head
    ) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, benchmark, blob = self._source(root)
            before = self._snapshot(source)
            output = root / "rescored"
            with patch("analysis.rescore_multistep_workflows._git_blob", return_value=blob):
                rescore_multistep_run(
                    source_run=source, output_run=output,
                    benchmark=benchmark, repository=root,
                )
            self.assertEqual(self._snapshot(source), before)
            self.assertTrue((output / "RUN_COMPLETE").is_file())
            manifest = json.loads((output / "rescore_manifest.json").read_text())
            self.assertEqual(manifest["workflow_final_scoring_version"], WORKFLOW_FINAL_SCORING_VERSION)
            self.assertEqual(manifest["source_artifact_sha256"], before)
            metadata = json.loads((output / "run_metadata.json").read_text())
            self.assertEqual(metadata["benchmark_paths"], [str(benchmark)])
            self.assertEqual(
                metadata["source_counts"][str(benchmark)],
                {"workflows": 1, "routed_steps": 1},
            )
            rescored = json.loads(next(output.glob("domains/*/*/samples.jsonl")).read_text())
            self.assertEqual(rescored["raw_generation"], "preserve me")
            self.assertIsNone(rescored["workflow_final_answer_correct"])
            self.assertTrue(rescored["workflow_final_program_execution_correct"])
            with patch("analysis.rescore_multistep_workflows._git_blob", return_value=blob):
                with self.assertRaises(FileExistsError):
                    rescore_multistep_run(
                        source_run=source, output_run=output,
                        benchmark=benchmark, repository=root,
                    )

    @patch("analysis.rescore_multistep_workflows.validate_and_index_multistep", side_effect=ValueError("invalid"))
    @patch("analysis.rescore_multistep_workflows.subprocess.check_output", return_value="rescore-commit\n")
    def test_failed_validation_cleans_temporary_output(self, _head, _index) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, benchmark, blob = self._source(root)
            output = root / "rescored"
            with patch("analysis.rescore_multistep_workflows._git_blob", return_value=blob):
                with self.assertRaisesRegex(ValueError, "invalid"):
                    rescore_multistep_run(
                        source_run=source, output_run=output,
                        benchmark=benchmark, repository=root,
                    )
            self.assertFalse(output.exists())
            self.assertEqual(list(root.glob(".rescored.rescore-*")), [])

    @patch("analysis.rescore_multistep_workflows.subprocess.check_output", return_value="rescore-commit\n")
    def test_historical_benchmark_may_differ_only_in_scoring_metadata(self, _head) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, benchmark, blob = self._source(root)
            historical = json.loads(blob)
            historical[0]["query"] = "changed model-facing text"
            output = root / "rescored"
            with patch(
                "analysis.rescore_multistep_workflows._git_blob",
                return_value=(json.dumps(historical) + "\n").encode(),
            ):
                with self.assertRaisesRegex(ValueError, "outside versioned"):
                    rescore_multistep_run(
                        source_run=source, output_run=output,
                        benchmark=benchmark, repository=root,
                    )
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
