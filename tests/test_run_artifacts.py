from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from analysis.run_artifacts import (
    capture_live_registry_metadata,
    safe_path_component,
    validate_and_index_dataset,
    validate_complete_run,
    validate_run_metadata,
)


FINGERPRINT = "sha256:test-registry"
FINGERPRINT_VERSION = "tool_registry_name_schema_description_v1"
TOOL_COUNT = 60
TOOL_POOL = "full_mcp_registry"
MODEL = "test/model"
PROMPT = "structured_tool_call_v1"
REGISTRY_EXPECTATIONS = {
    "expected_registry_fingerprint": FINGERPRINT,
    "expected_registry_fingerprint_version": FINGERPRINT_VERSION,
    "expected_tool_count": TOOL_COUNT,
    "expected_tool_pool": TOOL_POOL,
}


def _write_dataset_artifacts(
    root: Path,
    benchmark: Path,
    *,
    models: tuple[str, ...] = (MODEL, MODEL),
) -> Path:
    dataset_dir = root / "domains" / "coding" / "coding_public"
    dataset_dir.mkdir(parents=True)
    samples = []
    for index, model in enumerate(models):
        samples.append(
            {
                "sample_id": f"sample-{index}",
                "benchmark_path": str(benchmark),
                "model_name": model,
                "prompt_template": PROMPT,
                "evaluation_protocol": "single_step_tool_routing_v1",
                "reasoning_mode": "direct",
                "reasoning_method": "none",
                "effective_generation_limit": 128,
                "effective_generation_limit_unit": "tokens",
                "benchmark_mode": "grounded_tool_execution",
                "tool_pool": TOOL_POOL,
                "tool_count": TOOL_COUNT,
                "tool_registry_fingerprint": FINGERPRINT,
                "tool_registry_fingerprint_version": FINGERPRINT_VERSION,
                "final_outcome_matcher": "recursive_json_subset_v1",
            }
        )
    (dataset_dir / "samples.jsonl").write_text(
        "".join(json.dumps(sample) + "\n" for sample in samples),
        encoding="utf-8",
    )
    (dataset_dir / "summary.json").write_text(
        json.dumps(
            {
                "benchmark_path": str(benchmark),
                "model_name": MODEL,
                "prompt_template": PROMPT,
                "evaluation_protocol": "single_step_tool_routing_v1",
                "reasoning_mode": "direct",
                "reasoning_method": "none",
                "effective_generation_limit": 128,
                "effective_generation_limit_unit": "tokens",
                "benchmark_mode_counts": {"grounded_tool_execution": len(samples)},
                "tool_pool": TOOL_POOL,
                "tool_count": TOOL_COUNT,
                "tool_registry_fingerprint": FINGERPRINT,
                "tool_registry_fingerprint_version": FINGERPRINT_VERSION,
                "total_samples": len(samples),
            }
        ),
        encoding="utf-8",
    )
    (dataset_dir / "evaluation.log").write_text("complete\n", encoding="utf-8")
    return dataset_dir


class RunArtifactTests(unittest.TestCase):
    def _benchmark(self, root: Path) -> Path:
        benchmark = root / "benchmark" / "coding" / "coding_public.json"
        benchmark.parent.mkdir(parents=True)
        benchmark.write_text("[]\n", encoding="utf-8")
        return benchmark

    def test_valid_artifacts_are_indexed_with_deterministic_json(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            benchmark = self._benchmark(root)
            rendered = []
            for directory_name in ("one", "two"):
                run = root / directory_name
                dataset_dir = _write_dataset_artifacts(run, benchmark)
                index = run / "artifact_index.jsonl"

                record = validate_and_index_dataset(
                    dataset_directory=dataset_dir,
                    index_path=index,
                    expected_benchmark=benchmark,
                    expected_model=MODEL,
                    expected_prompt_template=PROMPT,
                    **REGISTRY_EXPECTATIONS,
                )

                self.assertEqual(record["final_outcome_matchers"], [
                    "recursive_json_subset_v1"
                ])
                rendered.append(index.read_text(encoding="utf-8"))
            self.assertEqual(rendered[0], rendered[1])

    def test_path_components_reject_escape_and_empty_values(self) -> None:
        for value in ("", ".", "..", "../escape", "a/b", "a\\b", "two words"):
            with self.subTest(value=value), self.assertRaisesRegex(
                ValueError,
                "Unsafe dataset path component",
            ):
                safe_path_component(value, label="dataset")
        self.assertEqual(
            safe_path_component("llama-3.1-8b-local", label="model"),
            "llama-3.1-8b-local",
        )

    def test_invalid_jsonl_is_rejected(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            benchmark = self._benchmark(root)
            run = root / "run"
            dataset_dir = _write_dataset_artifacts(run, benchmark)
            (dataset_dir / "samples.jsonl").write_text("{broken\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "Invalid JSONL"):
                validate_and_index_dataset(
                    dataset_directory=dataset_dir,
                    index_path=run / "artifact_index.jsonl",
                    expected_benchmark=benchmark,
                    expected_model=MODEL,
                    expected_prompt_template=PROMPT,
                    **REGISTRY_EXPECTATIONS,
                )

    def test_mixed_model_jsonl_is_rejected(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            benchmark = self._benchmark(root)
            run = root / "run"
            dataset_dir = _write_dataset_artifacts(
                run,
                benchmark,
                models=(MODEL, "other/model"),
            )

            with self.assertRaisesRegex(ValueError, "Mixed or unexpected model"):
                validate_and_index_dataset(
                    dataset_directory=dataset_dir,
                    index_path=run / "artifact_index.jsonl",
                    expected_benchmark=benchmark,
                    expected_model=MODEL,
                    expected_prompt_template=PROMPT,
                    **REGISTRY_EXPECTATIONS,
                )

    def test_artifact_path_cannot_be_indexed_twice(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            benchmark = self._benchmark(root)
            run = root / "run"
            dataset_dir = _write_dataset_artifacts(run, benchmark)
            kwargs = {
                "dataset_directory": dataset_dir,
                "index_path": run / "artifact_index.jsonl",
                "expected_benchmark": benchmark,
                "expected_model": MODEL,
                "expected_prompt_template": PROMPT,
                **REGISTRY_EXPECTATIONS,
            }
            validate_and_index_dataset(**kwargs)
            with self.assertRaisesRegex(ValueError, "already indexed"):
                validate_and_index_dataset(**kwargs)

    def test_artifacts_must_be_distinct_and_inside_dataset_directory(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            benchmark = self._benchmark(root)
            run = root / "run"
            dataset_dir = _write_dataset_artifacts(run, benchmark)
            samples_path = dataset_dir / "samples.jsonl"
            samples_path.unlink()
            outside = root / "outside.jsonl"
            outside.write_text("{}\n", encoding="utf-8")
            samples_path.symlink_to(outside)

            with self.assertRaisesRegex(ValueError, "escapes its intended directory"):
                validate_and_index_dataset(
                    dataset_directory=dataset_dir,
                    index_path=run / "artifact_index.jsonl",
                    expected_benchmark=benchmark,
                    expected_model=MODEL,
                    expected_prompt_template=PROMPT,
                    **REGISTRY_EXPECTATIONS,
                )

    def test_artifact_paths_must_be_distinct(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            benchmark = self._benchmark(root)
            run = root / "run"
            dataset_dir = _write_dataset_artifacts(run, benchmark)
            summary_path = dataset_dir / "summary.json"
            summary_path.unlink()
            summary_path.symlink_to(dataset_dir / "samples.jsonl")

            with self.assertRaisesRegex(ValueError, "must be distinct"):
                validate_and_index_dataset(
                    dataset_directory=dataset_dir,
                    index_path=run / "artifact_index.jsonl",
                    expected_benchmark=benchmark,
                    expected_model=MODEL,
                    expected_prompt_template=PROMPT,
                    **REGISTRY_EXPECTATIONS,
                )

    def test_run_completion_requires_exactly_one_index_entry_per_dataset(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            benchmark = self._benchmark(root)
            run = root / "run"
            dataset_dir = _write_dataset_artifacts(run, benchmark)
            index = run / "artifact_index.jsonl"
            validate_and_index_dataset(
                dataset_directory=dataset_dir,
                index_path=index,
                expected_benchmark=benchmark,
                expected_model=MODEL,
                expected_prompt_template=PROMPT,
                **REGISTRY_EXPECTATIONS,
            )

            validate_complete_run(index_path=index, expected_benchmarks=[benchmark])
            with self.assertRaisesRegex(ValueError, "incomplete"):
                validate_complete_run(
                    index_path=index,
                    expected_benchmarks=[benchmark, root / "missing.json"],
                )

    def test_run_metadata_must_match_indexed_reasoning_contract(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            benchmark = self._benchmark(root)
            run = root / "run"
            dataset_dir = _write_dataset_artifacts(run, benchmark)
            index = run / "artifact_index.jsonl"
            validate_and_index_dataset(
                dataset_directory=dataset_dir,
                index_path=index,
                expected_benchmark=benchmark,
                expected_model=MODEL,
                expected_prompt_template=PROMPT,
                **REGISTRY_EXPECTATIONS,
            )
            metadata = {
                "expected_model_name": MODEL,
                "prompt_template_id": PROMPT,
                "reasoning_mode": "direct",
                "reasoning_method": "none",
                "effective_generation_limit": 128,
                "effective_generation_limit_unit": "tokens",
                "evaluation_protocol": "single_step_tool_routing_v1",
                "tool_pool": TOOL_POOL,
                "tool_count": TOOL_COUNT,
                "tool_registry_fingerprint": FINGERPRINT,
                "tool_registry_fingerprint_version": FINGERPRINT_VERSION,
            }
            metadata_path = run / "run_metadata.json"
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            validate_run_metadata(index_path=index, metadata_path=metadata_path)
            metadata["reasoning_mode"] = "reasoning"
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "reasoning_mode"):
                validate_run_metadata(index_path=index, metadata_path=metadata_path)

    def test_single_step_launcher_separates_and_passes_reasoning_mode(self) -> None:
        launcher = (Path(__file__).resolve().parents[1] / "scripts/slurm/run_single_step.sbatch").read_text()
        self.assertIn("${REASONING_COMPONENT}_${RUN_KIND_COMPONENT}", launcher)
        self.assertIn('--reasoning-mode "$REASONING_MODE"', launcher)
        self.assertIn('--run-metadata "$RUN_DIR/run_metadata.json"', launcher)
        self.assertNotRegex(launcher, r"sha256:[0-9a-f]{64}")
        self.assertNotRegex(launcher, r"TOOL_COUNT=[0-9]+")
        for model in (
            "phi-4-local",
            "llama-3.1-8b-local",
            "qwen-3.6-local",
            "gemma-4-local",
            "gpt-oss-local",
        ):
            self.assertIn(model, launcher)
        self.assertLess(
            launcher.index("does not implement reasoning mode"),
            launcher.index("--capture-live-registry"),
        )

    def _validate_launcher_root(
        self,
        launcher_name: str,
        *,
        cwd: Path,
        repo_override: str | None = None,
        submit_dir: str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        root = Path(__file__).resolve().parents[1]
        environment = os.environ.copy()
        environment["LAYERMCP_VALIDATE_REPO_ROOT_ONLY"] = "1"
        environment.pop("LAYERMCP_REPO_ROOT", None)
        environment.pop("SLURM_SUBMIT_DIR", None)
        if repo_override is not None:
            environment["LAYERMCP_REPO_ROOT"] = repo_override
        if submit_dir is not None:
            environment["SLURM_SUBMIT_DIR"] = submit_dir
        return subprocess.run(
            ["bash", str(root / "scripts/slurm" / launcher_name)],
            cwd=cwd,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_launchers_use_repository_root_precedence(self) -> None:
        root = Path(__file__).resolve().parents[1]
        with TemporaryDirectory() as temporary:
            unrelated = Path(temporary)
            for launcher_name in (
                "run_single_step.sbatch",
                "run_multi_step.sbatch",
            ):
                with self.subTest(launcher=launcher_name, source="override"):
                    result = self._validate_launcher_root(
                        launcher_name,
                        cwd=unrelated,
                        repo_override=str(root),
                        submit_dir=str(unrelated),
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(result.stdout.strip(), str(root.resolve()))
                with self.subTest(launcher=launcher_name, source="slurm"):
                    result = self._validate_launcher_root(
                        launcher_name,
                        cwd=unrelated,
                        submit_dir=str(root),
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(result.stdout.strip(), str(root.resolve()))
                with self.subTest(launcher=launcher_name, source="pwd"):
                    result = self._validate_launcher_root(launcher_name, cwd=root)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(result.stdout.strip(), str(root.resolve()))

    def test_launchers_reject_bad_repository_before_expensive_work(self) -> None:
        with TemporaryDirectory() as temporary:
            bad_root = Path(temporary)
            for launcher_name in (
                "run_single_step.sbatch",
                "run_multi_step.sbatch",
            ):
                with self.subTest(launcher=launcher_name):
                    result = self._validate_launcher_root(
                        launcher_name,
                        cwd=bad_root,
                        repo_override=str(bad_root),
                    )
                    self.assertEqual(result.returncode, 2)
                    self.assertIn("not the LayerMCP repository root", result.stderr)
                    self.assertIn("set LAYERMCP_REPO_ROOT explicitly", result.stderr)
                    self.assertNotIn("registry", result.stdout.lower())
                    self.assertNotIn("model weights", result.stdout.lower())

    def test_single_step_launcher_root_and_checkpoint_contract(self) -> None:
        launcher = (
            Path(__file__).resolve().parents[1]
            / "scripts/slurm/run_single_step.sbatch"
        ).read_text()
        self.assertIn('LAUNCHER_PATH="$(realpath "${BASH_SOURCE[0]}")"', launcher)
        self.assertNotIn('dirname "$LAUNCHER_PATH"', launcher)
        self.assertLess(
            launcher.index('if [[ -n "${LAYERMCP_REPO_ROOT:-}" ]]'),
            launcher.index('elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]]'),
        )
        self.assertIn('REPO_ROOT_CANDIDATE="$PWD"', launcher)
        self.assertIn('pwd -P', launcher)
        self.assertIn('cd "$REPO_ROOT"', launcher)
        for required in (
            "pyproject.toml",
            "evaluation/evaluate.py",
            "scripts/slurm/run_single_step.sbatch",
            "scripts/slurm/run_multi_step.sbatch",
        ):
            self.assertIn(required, launcher)
        self.assertNotIn("$SCRATCH/layermcp/LayerMCP", launcher)
        for variable, relative_path in (
            ("LAYERMCP_PHI4_CHECKPOINT", "phi-4"),
            ("LAYERMCP_LLAMA31_8B_CHECKPOINT", "llama-3.1-8b-instruct"),
            ("LAYERMCP_QWEN36_CHECKPOINT", "qwen-3.6"),
            ("LAYERMCP_GEMMA4_CHECKPOINT", "gemma-4"),
            ("LAYERMCP_GPT_OSS_CHECKPOINT", "gpt-oss-20b/original"),
        ):
            self.assertIn(
                f'${{{variable}:=$REPO_ROOT/checkpoints/{relative_path}}}',
                launcher,
            )
        self.assertEqual(launcher.count('LAUNCHER_PATH="$(realpath'), 1)
        self.assertIn('cp "$LAUNCHER_PATH" "$RUN_DIR/launcher.sbatch"', launcher)

    def test_gpt_oss_contract_is_present_in_both_launchers(self) -> None:
        root = Path(__file__).resolve().parents[1]
        for launcher_name in (
            "run_single_step.sbatch",
            "run_multi_step.sbatch",
        ):
            with self.subTest(launcher=launcher_name):
                launcher = (root / "scripts" / "slurm" / launcher_name).read_text()
                self.assertIn("gpt-oss-local", launcher)
                self.assertIn(
                    '${LAYERMCP_GPT_OSS_CHECKPOINT:=$REPO_ROOT/checkpoints/'
                    'gpt-oss-20b/original}',
                    launcher,
                )
                self.assertIn("export LAYERMCP_GPT_OSS_CHECKPOINT", launcher)
                self.assertIn('EXPECTED_MODEL_NAME="openai/gpt-oss-20b"', launcher)
                self.assertIn(
                    'PROMPT_TEMPLATE_ID="harmony_structured_context_sql_v2"',
                    launcher,
                )
                self.assertIn("REQUIRED_CHECKPOINT_FILES=(config.json model.safetensors)", launcher)
                if launcher_name == "run_single_step.sbatch":
                    self.assertIn("gpt-oss-local:reasoning", launcher)
                    self.assertIn(
                        "$MODEL does not implement reasoning mode; use "
                        "REASONING_MODE=direct",
                        launcher,
                    )
                else:
                    self.assertIn(
                        "gpt-oss-local does not implement reasoning mode; use "
                        "REASONING_MODE=direct",
                        launcher,
                    )
                self.assertIn('--router "$MODEL"', launcher)
                self.assertLess(
                    launcher.index('Missing required checkpoint file:'),
                    launcher.index("python -m evaluation.evaluate"),
                )
                self.assertNotIn("dtypes.json", launcher)

    def test_tracked_slurm_launchers_pass_bash_syntax(self) -> None:
        root = Path(__file__).resolve().parents[1]
        for launcher_name in (
            "run_single_step.sbatch",
            "run_multi_step.sbatch",
        ):
            with self.subTest(launcher=launcher_name):
                result = subprocess.run(
                    ["bash", "-n", str(root / "scripts" / "slurm" / launcher_name)],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_live_registry_metadata_uses_evaluator_canonical_implementation(self) -> None:
        tools = [
            SimpleNamespace(
                name="calculator",
                description="Calculate.",
                inputSchema={"type": "object"},
            )
        ]

        class FakeContext:
            async def __aenter__(self):
                return SimpleNamespace(
                    list_tools=AsyncMock(return_value=SimpleNamespace(tools=tools))
                )

            async def __aexit__(self, exc_type, exc, traceback):
                return False

        with patch(
            "evaluation.evaluate._run_server_session",
            return_value=FakeContext(),
        ), patch(
            "evaluation.evaluate._tool_pool_metadata",
            return_value={
                "tool_pool": TOOL_POOL,
                "tool_count": 1,
                "tool_registry_fingerprint": "sha256:" + "a" * 64,
                "tool_registry_fingerprint_version": FINGERPRINT_VERSION,
            },
        ) as canonical_fingerprint:
            import asyncio

            metadata = asyncio.run(
                capture_live_registry_metadata(Path("mcp_server/server.py"))
            )

        canonical_fingerprint.assert_called_once()
        self.assertEqual(metadata["tool_count"], 1)
        self.assertEqual(metadata["tool_pool"], TOOL_POOL)

    def test_registry_metadata_mismatch_is_rejected_before_indexing(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            benchmark = self._benchmark(root)
            run = root / "run"
            dataset_dir = _write_dataset_artifacts(run, benchmark)

            for field, override in (
                ("expected_tool_pool", "other_pool"),
                ("expected_tool_count", 61),
                ("expected_registry_fingerprint", "sha256:" + "b" * 64),
                ("expected_registry_fingerprint_version", "other_version"),
            ):
                expectations = dict(REGISTRY_EXPECTATIONS)
                expectations[field] = override
                with self.subTest(field=field), self.assertRaisesRegex(
                    ValueError,
                    "Unexpected registry",
                ):
                    validate_and_index_dataset(
                        dataset_directory=dataset_dir,
                        index_path=run / "artifact_index.jsonl",
                        expected_benchmark=benchmark,
                        expected_model=MODEL,
                        expected_prompt_template=PROMPT,
                        **expectations,
                    )
                self.assertFalse((run / "artifact_index.jsonl").exists())

    def test_launcher_derives_and_records_live_registry_metadata(self) -> None:
        launcher = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "slurm"
            / "run_single_step.sbatch"
        ).read_text(encoding="utf-8")

        self.assertNotRegex(launcher, r"sha256:[0-9a-f]{64}")
        self.assertIn("--capture-live-registry", launcher)
        self.assertIn("registry_metadata_source", launcher)
        self.assertIn('"tool_pool": "$TOOL_POOL"', launcher)
        self.assertIn('"tool_count": $TOOL_COUNT', launcher)
        self.assertIn('"tool_registry_fingerprint": "$EXPECTED_REGISTRY_FINGERPRINT"', launcher)
        self.assertIn('"tool_registry_fingerprint_version": "$REGISTRY_FINGERPRINT_VERSION"', launcher)
        for option in (
            "--registry-fingerprint",
            "--registry-fingerprint-version",
            "--tool-count",
            "--tool-pool",
        ):
            self.assertIn(option, launcher)


if __name__ == "__main__":
    unittest.main()
