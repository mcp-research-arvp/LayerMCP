from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from analysis.run_artifacts import (
    safe_path_component,
    validate_and_index_dataset,
    validate_complete_run,
)


FINGERPRINT = "sha256:test-registry"
MODEL = "test/model"
PROMPT = "structured_tool_call_v1"


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
                "benchmark_mode": "grounded_tool_execution",
                "tool_pool": "full_mcp_registry",
                "tool_count": 60,
                "tool_registry_fingerprint": FINGERPRINT,
                "tool_registry_fingerprint_version": (
                    "tool_registry_name_schema_description_v1"
                ),
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
                "benchmark_mode_counts": {"grounded_tool_execution": len(samples)},
                "tool_pool": "full_mcp_registry",
                "tool_count": 60,
                "tool_registry_fingerprint": FINGERPRINT,
                "tool_registry_fingerprint_version": (
                    "tool_registry_name_schema_description_v1"
                ),
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
                    expected_registry_fingerprint=FINGERPRINT,
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
                    expected_registry_fingerprint=FINGERPRINT,
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
                    expected_registry_fingerprint=FINGERPRINT,
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
                "expected_registry_fingerprint": FINGERPRINT,
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
                    expected_registry_fingerprint=FINGERPRINT,
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
                    expected_registry_fingerprint=FINGERPRINT,
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
                expected_registry_fingerprint=FINGERPRINT,
            )

            validate_complete_run(index_path=index, expected_benchmarks=[benchmark])
            with self.assertRaisesRegex(ValueError, "incomplete"):
                validate_complete_run(
                    index_path=index,
                    expected_benchmarks=[benchmark, root / "missing.json"],
                )


if __name__ == "__main__":
    unittest.main()
