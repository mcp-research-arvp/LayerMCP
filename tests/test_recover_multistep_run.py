from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from analysis.recover_multistep_run import RECOVERY_METHOD, recover_multistep_run
from analysis.multi_step_run import validate_complete_multistep_run


_MISSING = object()


class RecoverMultiStepRunTests(unittest.TestCase):
    @staticmethod
    def _tree_snapshot(path: Path) -> dict[str, tuple[int, str]]:
        return {
            str(item.relative_to(path)): (
                item.stat().st_size,
                hashlib.sha256(item.read_bytes()).hexdigest(),
            )
            for item in sorted(path.rglob("*"))
            if item.is_file()
        }

    def _source(self, root: Path) -> tuple[Path, Path]:
        root.mkdir(parents=True, exist_ok=True)
        benchmark = root / "benchmark/math/math_multistep_controlled.json"
        benchmark.parent.mkdir(parents=True, exist_ok=True)
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
            "benchmark_paths": [str(benchmark.resolve())],
            "source_counts": {
                str(benchmark.resolve()): {"workflows": 1, "routed_steps": 2}
            },
            "benchmark_mode_distributions": {
                str(benchmark.resolve()): {"grounded_tool_execution": 1}
            },
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
            original_tree = self._tree_snapshot(source)
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
            self.assertEqual(
                manifest["source_run_metadata"]["benchmark_paths"],
                [str(benchmark.resolve())],
            )
            self.assertTrue((output / "RUN_COMPLETE").is_file())
            self.assertEqual(list(root.glob(".recovered.recovery-*")), [])
            self.assertEqual(self._tree_snapshot(source), original_tree)
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

    def test_recovery_rejects_equal_and_nested_destinations_without_source_changes(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            for suffix in (Path(), Path("recovered"), Path("nested/deep/recovered")):
                source, benchmark = self._source(root / f"case-{len(suffix.parts)}")
                before = self._tree_snapshot(source)
                destination = source / suffix
                with self.assertRaisesRegex(ValueError, "must not overlap"):
                    recover_multistep_run(
                        source_run=source,
                        output_run=destination,
                        benchmark=benchmark,
                        repository=root,
                    )
                self.assertEqual(self._tree_snapshot(source), before)

    def test_recovery_rejects_reverse_and_symlink_overlap(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            parent = root / "parent"
            source, benchmark = self._source(parent)
            before = self._tree_snapshot(source)
            with self.assertRaisesRegex(ValueError, "must not overlap"):
                recover_multistep_run(
                    source_run=source,
                    output_run=parent,
                    benchmark=benchmark,
                    repository=root,
                )
            self.assertEqual(self._tree_snapshot(source), before)

            link = root / "source-link"
            link.symlink_to(source, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "must not overlap"):
                recover_multistep_run(
                    source_run=source,
                    output_run=link / "nested",
                    benchmark=benchmark,
                    repository=root,
                )
            self.assertEqual(self._tree_snapshot(source), before)

    def _assert_metadata_failure(
        self, field: str, replacement: object, message: str
    ) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, benchmark = self._source(root)
            metadata_path = source / "run_metadata.json"
            metadata = json.loads(metadata_path.read_text())
            if replacement is _MISSING:
                del metadata[field]
            else:
                metadata[field] = (
                    replacement(benchmark)
                    if callable(replacement)
                    else replacement
                )
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            before = self._tree_snapshot(source)
            output = root / "recovered"
            with self.assertRaisesRegex(ValueError, message):
                recover_multistep_run(
                    source_run=source,
                    output_run=output,
                    benchmark=benchmark,
                    repository=root,
                )
            self.assertFalse(output.exists())
            self.assertEqual(self._tree_snapshot(source), before)

    def test_recovery_rejects_mismatched_source_benchmark_metadata(self) -> None:
        self._assert_metadata_failure(
            "benchmark_paths", ["/different/benchmark.json"], "benchmark_paths"
        )
        self._assert_metadata_failure(
            "source_counts",
            {"/different/benchmark.json": {"workflows": 1, "routed_steps": 2}},
            "source_counts",
        )
        self._assert_metadata_failure(
            "source_counts",
            lambda benchmark: {
                str(benchmark.resolve()): {"workflows": 99, "routed_steps": 2}
            },
            "source_counts",
        )
        self._assert_metadata_failure(
            "source_counts",
            lambda benchmark: {
                str(benchmark.resolve()): {"workflows": 1, "routed_steps": 99}
            },
            "source_counts",
        )
        self._assert_metadata_failure(
            "benchmark_mode_distributions",
            lambda benchmark: {
                str(benchmark.resolve()): {"offline_trace_replay": 1}
            },
            "benchmark_mode_distributions",
        )

    def test_recovery_rejects_missing_source_benchmark_metadata(self) -> None:
        for field in (
            "benchmark_paths", "source_counts", "benchmark_mode_distributions"
        ):
            with self.subTest(field=field):
                self._assert_metadata_failure(field, _MISSING, field)

    @patch("analysis.recover_multistep_run._git_head", return_value="commit")
    def test_cross_worktree_benchmark_requires_same_relative_path_and_hash(
        self, _head
    ) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, source_benchmark = self._source(root / "source-worktree")
            other_benchmark = (
                root / "other-worktree/benchmark/math/math_multistep_controlled.json"
            )
            other_benchmark.parent.mkdir(parents=True)
            other_benchmark.write_bytes(source_benchmark.read_bytes())
            samples = next(source.glob("domains/*/*/samples.jsonl"))
            source_samples_hash = hashlib.sha256(samples.read_bytes()).hexdigest()
            output = root / "recovered"
            recover_multistep_run(
                source_run=source, output_run=output,
                benchmark=other_benchmark, repository=root,
            )
            self.assertTrue((output / "RUN_COMPLETE").is_file())
            recovered_samples = next(output.glob("domains/*/*/samples.jsonl"))
            self.assertEqual(
                hashlib.sha256(recovered_samples.read_bytes()).hexdigest(),
                source_samples_hash,
            )
            original_identity = str(source_benchmark.resolve())
            summary = json.loads(
                next(output.glob("domains/*/*/summary.json")).read_text()
            )
            metadata = json.loads((output / "run_metadata.json").read_text())
            index = json.loads((output / "artifact_index.jsonl").read_text())
            manifest = json.loads((output / "recovery_manifest.json").read_text())
            self.assertEqual(summary["benchmark_path"], original_identity)
            self.assertEqual(metadata["benchmark_paths"], [original_identity])
            self.assertEqual(list(metadata["source_counts"]), [original_identity])
            self.assertEqual(index["benchmark_path"], original_identity)
            self.assertEqual(manifest["original_benchmark_path"], original_identity)
            self.assertEqual(
                manifest["original_benchmark_sha256"],
                hashlib.sha256(source_benchmark.read_bytes()).hexdigest(),
            )
            self.assertEqual(
                manifest["validation_benchmark_path"], str(other_benchmark.resolve())
            )
            self.assertEqual(
                manifest["validation_benchmark_sha256"],
                hashlib.sha256(other_benchmark.read_bytes()).hexdigest(),
            )
            self.assertEqual(samples.read_bytes(), recovered_samples.read_bytes())

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, source_benchmark = self._source(root / "source-worktree")
            other_benchmark = (
                root / "other-worktree/benchmark/math/math_multistep_controlled.json"
            )
            other_benchmark.parent.mkdir(parents=True)
            changed = json.loads(source_benchmark.read_text())
            changed[0]["notes"] = "different benchmark content"
            other_benchmark.write_text(json.dumps(changed), encoding="utf-8")
            source_before = self._tree_snapshot(source)
            output = root / "recovered"
            with self.assertRaisesRegex(ValueError, "benchmark"):
                recover_multistep_run(
                    source_run=source, output_run=output,
                    benchmark=other_benchmark, repository=root,
                )
            self.assertFalse(output.exists())
            self.assertEqual(list(root.glob(".recovered.recovery-*")), [])
            self.assertEqual(self._tree_snapshot(source), source_before)

    @patch("analysis.recover_multistep_run._git_head", return_value="commit")
    def test_cross_worktree_benchmark_rejects_different_relative_path(
        self, _head
    ) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, source_benchmark = self._source(root / "source-worktree")
            other_benchmark = (
                root / "other-worktree/benchmark/math/different_name.json"
            )
            other_benchmark.parent.mkdir(parents=True)
            other_benchmark.write_bytes(source_benchmark.read_bytes())
            source_before = self._tree_snapshot(source)
            output = root / "recovered"
            with self.assertRaisesRegex(ValueError, "benchmark"):
                recover_multistep_run(
                    source_run=source, output_run=output,
                    benchmark=other_benchmark, repository=root,
                )
            self.assertFalse(output.exists())
            self.assertEqual(list(root.glob(".recovered.recovery-*")), [])
            self.assertEqual(self._tree_snapshot(source), source_before)

    def test_module_help_succeeds_from_repository_root(self) -> None:
        repository = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [sys.executable, "-m", "analysis.recover_multistep_run", "--help"],
            cwd=repository,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--source-run", result.stdout)
        self.assertIn("--output-run", result.stdout)
        self.assertIn("--benchmark", result.stdout)
        self.assertIn("--repository", result.stdout)

    @patch("analysis.recover_multistep_run._git_head", side_effect=RuntimeError("git failed"))
    def test_git_failure_leaves_no_destination_and_retry_succeeds(self, _head) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, benchmark = self._source(root)
            before = self._tree_snapshot(source)
            output = root / "recovered"
            with self.assertRaisesRegex(RuntimeError, "git failed"):
                recover_multistep_run(
                    source_run=source, output_run=output,
                    benchmark=benchmark, repository=root,
                )
            self.assertFalse(output.exists())
            self.assertEqual(self._tree_snapshot(source), before)
            with patch("analysis.recover_multistep_run._git_head", return_value="commit"):
                recover_multistep_run(
                    source_run=source, output_run=output,
                    benchmark=benchmark, repository=root,
                )
            self.assertTrue((output / "RUN_COMPLETE").is_file())

    @patch("analysis.recover_multistep_run._git_head", return_value="commit")
    def test_final_validator_failure_is_clean_and_retryable(self, _head) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, benchmark = self._source(root)
            before = self._tree_snapshot(source)
            output = root / "recovered"
            with patch(
                "analysis.recover_multistep_run.validate_complete_multistep_run",
                side_effect=ValueError("final validation failed"),
            ):
                with self.assertRaisesRegex(ValueError, "final validation failed"):
                    recover_multistep_run(
                        source_run=source, output_run=output,
                        benchmark=benchmark, repository=root,
                    )
            self.assertFalse(output.exists())
            self.assertEqual(self._tree_snapshot(source), before)
            self.assertEqual(list(root.glob(".recovered.recovery-*")), [])
            recover_multistep_run(
                source_run=source, output_run=output,
                benchmark=benchmark, repository=root,
            )
            self.assertTrue((output / "RUN_COMPLETE").is_file())

    @patch("analysis.recover_multistep_run._git_head", return_value="commit")
    def test_existing_destination_is_untouched(self, _head) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, benchmark = self._source(root)
            output = root / "recovered"
            output.mkdir()
            marker = output / "keep.txt"
            marker.write_text("untouched", encoding="utf-8")
            before = self._tree_snapshot(output)
            with self.assertRaisesRegex(FileExistsError, "already exists"):
                recover_multistep_run(
                    source_run=source, output_run=output,
                    benchmark=benchmark, repository=root,
                )
            self.assertEqual(self._tree_snapshot(output), before)

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
