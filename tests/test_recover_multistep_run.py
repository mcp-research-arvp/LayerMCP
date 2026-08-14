from __future__ import annotations

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from analysis.recover_multistep_run import RECOVERY_METHOD, recover_multistep_run
from analysis.multi_step_run import validate_complete_multistep_run


class RecoverMultiStepRunTests(unittest.TestCase):
    def _source(self, root: Path) -> tuple[Path, Path]:
        benchmark = root / "benchmark.json"
        benchmark.write_text(json.dumps([{
            "id": "workflow-1",
            "domain": "mathematics",
            "task_type": "multi_step_tool_routing",
            "query": "Do two steps",
            "expected_final_answer": {"result": 2},
            "expected_steps": [
                {"id": "step-1", "query": "one", "expected_tool": "calculator", "expected_args": {}, "expected_answer": {"result": 1}},
                {"id": "step-2", "query": "two", "expected_tool": "calculator", "expected_args": {}, "expected_answer": {"result": 2}},
            ],
        }]), encoding="utf-8")
        source = root / "source_123_model_direct_full_math"
        dataset = source / "domains/math/math_multistep_controlled"
        dataset.mkdir(parents=True)
        fingerprint = "sha256:" + "a" * 64
        steps = []
        for index in range(2):
            steps.append({
                "sample_id": "workflow-1", "step_id": f"step-{index + 1}",
                "benchmark_path": str(benchmark), "benchmark_mode": "grounded_tool_execution",
                "workflow_execution_mode": "predicted_sequence", "called_tool": "calculator",
                "execution_success": True, "tool_selection_correct": True,
                "argument_match_correct": True, "final_outcome_correct": True,
                "final_outcome_matcher": "recursive_json_subset_v1", "latency_seconds": 1.0,
            })
        record = {
            "sample_id": "workflow-1", "benchmark_path": str(benchmark),
            "benchmark_mode": "grounded_tool_execution", "workflow_execution_mode": "predicted_sequence",
            "declared_workflow_execution_mode": "isolated_step",
            "evaluation_protocol": "guided_predicted_rollout_v1", "evaluation_protocol_description": "test",
            "model_name": "test/model", "router_id": "test", "router_backend": "test",
            "prompt_template": "test_prompt", "reasoning_mode": "direct", "reasoning_method": "none",
            "effective_generation_limit": 128, "effective_generation_limit_unit": "tokens",
            "tool_pool": "full_mcp_registry", "tool_count": 60,
            "tool_registry_fingerprint": fingerprint,
            "tool_registry_fingerprint_version": "tool_registry_name_schema_description_v1",
            "expected_final_answer": {"result": 2}, "workflow_final_answer_correct": True,
            "sequence_tool_selection_correct": True, "sequence_argument_match_correct": True,
            "sequence_semantic_output_correct": True, "steps": steps,
        }
        (dataset / "samples.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
        (dataset / "evaluation.log").write_text("summary crash\n", encoding="utf-8")
        metadata = {
            "expected_model_name": "test/model", "prompt_template_id": "test_prompt",
            "reasoning_mode": "direct", "reasoning_method": "none",
            "effective_generation_limit": 128, "effective_generation_limit_unit": "tokens",
            "evaluation_protocol": "guided_predicted_rollout_v1", "workflow_execution_mode": "predicted_sequence",
            "tool_pool": "full_mcp_registry", "tool_count": 60,
            "tool_registry_fingerprint": fingerprint,
            "tool_registry_fingerprint_version": "tool_registry_name_schema_description_v1",
            "run_kind": "full", "headline_eligible": True, "slurm_job_id": "123",
            "git_commit": "source-commit", "short_test_selection": {},
        }
        (source / "run_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
        return source, benchmark

    @patch("analysis.recover_multistep_run._git_head", return_value="recovery-commit")
    def test_recovery_preserves_samples_and_builds_valid_bundle(self, _head) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, benchmark = self._source(root)
            source_samples = next(source.glob("domains/*/*/samples.jsonl"))
            original_hash = hashlib.sha256(source_samples.read_bytes()).hexdigest()
            output = root / "recovered"

            recover_multistep_run(
                source_run=source, output_run=output,
                benchmark=benchmark, repository=root,
            )

            recovered_samples = next(output.glob("domains/*/*/samples.jsonl"))
            self.assertEqual(hashlib.sha256(recovered_samples.read_bytes()).hexdigest(), original_hash)
            summary = json.loads(next(output.glob("domains/*/*/summary.json")).read_text())
            self.assertEqual(summary["workflow_final_answer_gold"], 1)
            self.assertEqual(summary["workflow_final_answer_accuracy"], 1.0)
            self.assertTrue(summary["recovered_from_complete_saved_inference"])
            manifest = json.loads((output / "recovery_manifest.json").read_text())
            self.assertEqual(manifest["recovery_method"], RECOVERY_METHOD)
            self.assertEqual(manifest["recovery_commit"], "recovery-commit")
            self.assertEqual(manifest["source_job_id"], "123")
            self.assertTrue((output / "RUN_COMPLETE").is_file())
            validate_complete_multistep_run(
                output / "artifact_index.jsonl", [benchmark], output / "run_metadata.json"
            )

            second_output = root / "recovered-again"
            recover_multistep_run(
                source_run=source, output_run=second_output,
                benchmark=benchmark, repository=root,
            )
            for relative_path in (
                "artifact_index.jsonl",
                "recovery_manifest.json",
                "run_metadata.json",
                "domains/math/math_multistep_controlled/evaluation.log",
                "domains/math/math_multistep_controlled/samples.jsonl",
                "domains/math/math_multistep_controlled/summary.json",
            ):
                self.assertEqual(
                    (output / relative_path).read_bytes(),
                    (second_output / relative_path).read_bytes(),
                    relative_path,
                )

    @patch("analysis.recover_multistep_run._git_head", return_value="recovery-commit")
    def test_recovery_refuses_overwrite_and_incomplete_membership(self, _head) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, benchmark = self._source(root)
            output = root / "recovered"
            output.mkdir()
            with self.assertRaisesRegex(FileExistsError, "already exists"):
                recover_multistep_run(
                    source_run=source, output_run=output,
                    benchmark=benchmark, repository=root,
                )
            output.rmdir()
            samples = next(source.glob("domains/*/*/samples.jsonl"))
            samples.write_text("", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "count"):
                recover_multistep_run(
                    source_run=source, output_run=output,
                    benchmark=benchmark, repository=root,
                )


if __name__ == "__main__":
    unittest.main()
