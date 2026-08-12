from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from analysis.minimal_scorecard import build_scorecard, load_runs
from evaluation.evaluate import DEFAULT_BENCHMARK_MODE


_MISSING = object()


def _sample(
    sample_id: str,
    *,
    model: str,
    domain: str,
    tool: bool,
    args: bool,
    execution: bool,
    final: bool | None,
    failure: str,
    matcher: str = "recursive_json_subset_v1",
    benchmark_mode: str | None | object = DEFAULT_BENCHMARK_MODE,
    reasoning_mode: str = "direct",
    reasoning_method: str = "none",
) -> dict:
    sample = {
        "sample_id": sample_id,
        "model_name": model,
        "domain": domain,
        "evaluation_protocol": "single_step_tool_routing_v1",
        "tool_selection_correct": tool,
        "argument_match_correct": args,
        "execution_success": execution,
        "final_outcome_correct": final,
        "final_outcome_matcher": matcher,
        "failure_category": failure,
        "source": "public_derived",
        "task_type": "single_tool_routing",
        "reasoning_mode": reasoning_mode,
        "reasoning_method": reasoning_method,
    }
    if benchmark_mode is not _MISSING:
        sample["benchmark_mode"] = benchmark_mode
    return sample


def _write_result(
    run: Path,
    name: str,
    *,
    model: str,
    domain: str,
    fingerprint: str,
    samples: list[dict],
    summary_overrides: dict | None = None,
    missing_registry_field: str | None = None,
    reasoning_mode: str = "direct",
    reasoning_method: str = "none",
) -> None:
    artifacts = run / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    samples_path = artifacts / f"{name}_samples.jsonl"
    summary_path = artifacts / f"{name}_summary.json"
    samples_path.write_text(
        "".join(json.dumps(sample) + "\n" for sample in samples),
        encoding="utf-8",
    )
    summary = {
        "model_name": model,
        "benchmark_path": f"benchmark/{domain}/{name}.json",
        "evaluation_protocol": "single_step_tool_routing_v1",
        "reasoning_mode": reasoning_mode,
        "reasoning_method": reasoning_method,
        "total_samples": len(samples),
        "tool_pool": "full_mcp_registry",
        "tool_count": 60,
        "tool_registry_fingerprint": fingerprint,
        "tool_registry_fingerprint_version": (
            "tool_registry_name_schema_description_v1"
        ),
    }
    summary.update(summary_overrides or {})
    if missing_registry_field is not None:
        summary.pop(missing_registry_field)
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    (run / f"{name}.log").write_text(
        f"Results: {samples_path}\nSummary: {summary_path}\n",
        encoding="utf-8",
    )


class MinimalScorecardTests(unittest.TestCase):
    @staticmethod
    def _correct_sample(model: str) -> dict:
        return _sample(
            model,
            model=model,
            domain="coding",
            tool=True,
            args=True,
            execution=True,
            final=True,
            failure="correct",
        )

    def test_generates_minimal_tables_and_diagnostics(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            phi = root / "phi"
            llama = root / "llama"
            phi.mkdir()
            llama.mkdir()
            fingerprint = "sha256:shared"
            _write_result(
                phi,
                "coding_public",
                model="phi",
                domain="coding",
                fingerprint=fingerprint,
                samples=[
                    _sample(
                        "p1",
                        model="phi",
                        domain="coding",
                        tool=True,
                        args=False,
                        execution=True,
                        final=True,
                        failure="wrong_args",
                    ),
                    _sample(
                        "p2",
                        model="phi",
                        domain="coding",
                        tool=False,
                        args=False,
                        execution=False,
                        final=False,
                        failure="wrong_tool",
                    ),
                ],
            )
            _write_result(
                llama,
                "finance_public",
                model="llama",
                domain="finance",
                fingerprint=fingerprint,
                samples=[
                    _sample(
                        "l1",
                        model="llama",
                        domain="finance",
                        tool=True,
                        args=True,
                        execution=True,
                        final=False,
                        failure="correct",
                    )
                ],
            )

            markdown = build_scorecard([llama, phi])

            self.assertIn(
                "| phi | direct | none | grounded_tool_execution | explicit | 2 | 50.0% | "
                "2/2 (100.0%) | 50.0% | — |",
                markdown,
            )
            self.assertIn(
                "| llama | direct | none | grounded_tool_execution | explicit | Finance | 1 | "
                "0.0% | 1/1 (100.0%) | 100.0% | — |",
                markdown,
            )
            self.assertIn("public/source-derived", markdown)
            self.assertIn(
                "| phi | direct | none | grounded_tool_execution | explicit | 0.0% | 50.0% | "
                "0.0% | 1 | 1 | 1 | 0 |",
                markdown,
            )
            self.assertIn("Valid Arguments / Schema-Valid Tool Call (SVCA)", markdown)
            self.assertIn("Exact Reference Argument Match", markdown)
            self.assertNotIn("Exact canonical args", markdown)
            header = (
                "| Model | Reasoning mode | Reasoning method | Benchmark mode | Mode source | N | Final Outcome "
                "Accuracy | Final Outcome Coverage | Tool Selection Accuracy | "
                "Valid Arguments / SVCA |"
            )
            self.assertIn(header, markdown)
            self.assertLess(
                markdown.index("Final Outcome Accuracy"),
                markdown.index("Tool Selection Accuracy"),
            )

    def test_direct_and_reasoning_conditions_cannot_be_combined(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            run = Path(temporary_directory) / "run"
            run.mkdir()
            for mode, method, correct in (
                ("direct", "native_disabled", True),
                ("reasoning", "native_enabled", False),
            ):
                _write_result(
                    run,
                    f"finance_{mode}",
                    model="gemma",
                    domain="finance",
                    fingerprint="sha256:one",
                    reasoning_mode=mode,
                    reasoning_method=method,
                    samples=[
                        _sample(
                            mode,
                            model="gemma",
                            domain="finance",
                            tool=correct,
                            args=correct,
                            execution=correct,
                            final=correct,
                            failure="correct" if correct else "wrong_tool",
                            reasoning_mode=mode,
                            reasoning_method=method,
                        )
                    ],
                )

            markdown = build_scorecard([run])

            self.assertIn("| gemma | direct | native_disabled |", markdown)
            self.assertIn("| gemma | reasoning | native_enabled |", markdown)
            self.assertNotIn(
                "| gemma | direct | native_disabled | grounded_tool_execution | explicit | 2 |",
                markdown,
            )

    def test_sgoa_uses_scored_denominator(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            run = Path(temporary_directory) / "run"
            run.mkdir()
            _write_result(
                run,
                "math_public",
                model="model",
                domain="math",
                fingerprint="sha256:one",
                samples=[
                    _sample(
                        "one",
                        model="model",
                        domain="mathematics",
                        tool=True,
                        args=True,
                        execution=True,
                        final=True,
                        failure="correct",
                    ),
                    _sample(
                        "two",
                        model="model",
                        domain="mathematics",
                        tool=True,
                        args=True,
                        execution=False,
                        final=None,
                        failure="correct",
                    ),
                ],
            )
            markdown = build_scorecard([run])
            self.assertIn(
                "| model | direct | none | grounded_tool_execution | explicit | 2 | 100.0% | "
                "1/2 (50.0%) | 100.0% | — |",
                markdown,
            )

    def test_zero_scored_rows_show_unavailable_accuracy_and_zero_coverage(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            run = Path(temporary_directory) / "run"
            run.mkdir()
            _write_result(
                run,
                "math_public",
                model="model",
                domain="math",
                fingerprint="sha256:one",
                samples=[
                    _sample(
                        sample_id,
                        model="model",
                        domain="mathematics",
                        tool=True,
                        args=True,
                        execution=False,
                        final=None,
                        failure="correct",
                    )
                    for sample_id in ("one", "two")
                ],
            )

            markdown = build_scorecard([run])

            self.assertIn(
                "| model | direct | none | grounded_tool_execution | explicit | 2 | — | "
                "0/2 (0.0%) | 100.0% | — |",
                markdown,
            )

    def test_reports_mixed_pr29_matchers_without_rejecting_run(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            run = root / "mixed"
            other_run = root / "other"
            run.mkdir()
            other_run.mkdir()
            _write_result(
                run,
                "finance_public",
                model="model",
                domain="finance",
                fingerprint="sha256:one",
                samples=[
                    _sample(
                        "finance",
                        model="model",
                        domain="finance",
                        tool=True,
                        args=False,
                        execution=True,
                        final=True,
                        failure="wrong_args",
                        matcher="finance_query_table_rows_v1",
                    ),
                    _sample(
                        "other",
                        model="model",
                        domain="finance",
                        tool=True,
                        args=True,
                        execution=True,
                        final=True,
                        failure="correct",
                        matcher="recursive_json_subset_v1",
                    ),
                ],
            )
            _write_result(
                other_run,
                "coding_public",
                model="other-model",
                domain="coding",
                fingerprint="sha256:one",
                samples=[
                    _sample(
                        "coding",
                        model="other-model",
                        domain="coding",
                        tool=True,
                        args=True,
                        execution=True,
                        final=True,
                        failure="correct",
                        matcher="recursive_json_subset_v1",
                    )
                ],
            )

            markdown = build_scorecard([other_run, run])

            self.assertIn(
                "| model | direct | none | `grounded_tool_execution (explicit)` | "
                "`finance_query_table_rows_v1`, "
                "`recursive_json_subset_v1` |",
                markdown,
            )
            self.assertIn(
                "| other-model | direct | none | `grounded_tool_execution (explicit)` | "
                "`recursive_json_subset_v1` |",
                markdown,
            )

    def test_benchmark_modes_and_default_provenance_are_not_pooled(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            run = Path(temporary_directory) / "run"
            run.mkdir()
            _write_result(
                run,
                "mixed_modes",
                model="model",
                domain="coding",
                fingerprint="sha256:one",
                samples=[
                    _sample(
                        "grounded",
                        model="model",
                        domain="coding",
                        tool=True,
                        args=True,
                        execution=True,
                        final=True,
                        failure="correct",
                        benchmark_mode="grounded_tool_execution",
                    ),
                    _sample(
                        "replay",
                        model="model",
                        domain="coding",
                        tool=False,
                        args=False,
                        execution=False,
                        final=False,
                        failure="wrong_tool",
                        benchmark_mode="offline_trace_replay",
                    ),
                    _sample(
                        "missing",
                        model="model",
                        domain="coding",
                        tool=False,
                        args=False,
                        execution=False,
                        final=False,
                        failure="wrong_tool",
                        benchmark_mode=_MISSING,
                    ),
                    _sample(
                        "null",
                        model="model",
                        domain="coding",
                        tool=True,
                        args=True,
                        execution=True,
                        final=True,
                        failure="correct",
                        benchmark_mode=None,
                    ),
                ],
            )

            markdown = build_scorecard([run])

            self.assertIn(
                "| model | direct | none | grounded_tool_execution | explicit | 1 | 100.0% |",
                markdown,
            )
            self.assertIn(
                "| model | direct | none | grounded_tool_execution | defaulted (missing) | 1 | "
                "0.0% |",
                markdown,
            )
            self.assertIn(
                "| model | direct | none | grounded_tool_execution | defaulted (null) | 1 | "
                "100.0% |",
                markdown,
            )
            self.assertIn(
                "| model | direct | none | offline_trace_replay | explicit | 1 | 0.0% |",
                markdown,
            )
            self.assertIn(
                "`grounded_tool_execution (defaulted (missing))`, "
                "`grounded_tool_execution (defaulted (null))`, "
                "`grounded_tool_execution (explicit)`, "
                "`offline_trace_replay (explicit)`",
                markdown,
            )
            self.assertNotIn("| None |", markdown)
            self.assertEqual(markdown, build_scorecard([run]))

    def test_rejects_incompatible_registry_fingerprints(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            runs = [root / "one", root / "two"]
            for run in runs:
                run.mkdir()
            for run, fingerprint in zip(runs, ("sha256:a", "sha256:b")):
                _write_result(
                    run,
                    "coding_public",
                    model=run.name,
                    domain="coding",
                    fingerprint=fingerprint,
                    samples=[
                        _sample(
                            run.name,
                            model=run.name,
                            domain="coding",
                            tool=True,
                            args=True,
                            execution=True,
                            final=True,
                            failure="correct",
                        )
                    ],
                )
            with self.assertRaisesRegex(ValueError, "incompatible tool registry"):
                load_runs(runs)

    def test_single_run_missing_each_registry_field_is_rejected(self) -> None:
        required_fields = (
            "tool_registry_fingerprint",
            "tool_registry_fingerprint_version",
            "tool_count",
            "tool_pool",
        )
        for field in required_fields:
            with self.subTest(field=field), TemporaryDirectory() as temporary_directory:
                run = Path(temporary_directory) / "run"
                run.mkdir()
                _write_result(
                    run,
                    "coding_public",
                    model="model",
                    domain="coding",
                    fingerprint="sha256:one",
                    samples=[self._correct_sample("model")],
                    missing_registry_field=field,
                )

                with self.assertRaisesRegex(
                    ValueError,
                    rf"coding_public_summary\.json.*{field}.*"
                    "registry compatibility cannot be verified",
                ):
                    load_runs([run])

    def test_verified_run_plus_missing_registry_metadata_is_rejected(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            verified = root / "verified"
            legacy = root / "legacy"
            verified.mkdir()
            legacy.mkdir()
            _write_result(
                verified,
                "coding_public",
                model="verified",
                domain="coding",
                fingerprint="sha256:one",
                samples=[self._correct_sample("verified")],
            )
            _write_result(
                legacy,
                "finance_public",
                model="legacy",
                domain="finance",
                fingerprint="sha256:one",
                samples=[self._correct_sample("legacy")],
                missing_registry_field="tool_pool",
            )

            with self.assertRaisesRegex(
                ValueError,
                "finance_public_summary.json.*tool_pool.*"
                "registry compatibility cannot be verified",
            ):
                load_runs([verified, legacy])

    def test_matching_complete_registry_metadata_is_accepted(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            runs = [root / "one", root / "two"]
            for run in runs:
                run.mkdir()
                _write_result(
                    run,
                    f"{run.name}_coding_public",
                    model=run.name,
                    domain="coding",
                    fingerprint="sha256:same",
                    samples=[self._correct_sample(run.name)],
                )

            loaded = load_runs(runs)

            self.assertEqual(len(loaded.records), 2)
            self.assertEqual(loaded.fingerprints, ("sha256:same",))

    def test_differing_populated_registry_metadata_is_rejected(self) -> None:
        cases = (
            (
                "fingerprint",
                {},
                {"tool_registry_fingerprint": "sha256:different"},
                "incompatible tool registry fingerprints",
            ),
            (
                "fingerprint_version",
                {},
                {"tool_registry_fingerprint_version": "different_version"},
                "incompatible registry fingerprint versions",
            ),
            (
                "tool_count",
                {},
                {"tool_count": 61},
                "different tool counts",
            ),
            (
                "tool_pool",
                {},
                {"tool_pool": "different_pool"},
                "different tool pools",
            ),
        )
        for label, first_overrides, second_overrides, message in cases:
            with self.subTest(field=label), TemporaryDirectory() as temporary_directory:
                root = Path(temporary_directory)
                runs = [root / "one", root / "two"]
                for run, overrides in zip(runs, (first_overrides, second_overrides)):
                    run.mkdir()
                    _write_result(
                        run,
                        f"{run.name}_coding_public",
                        model=run.name,
                        domain="coding",
                        fingerprint="sha256:same",
                        samples=[self._correct_sample(run.name)],
                        summary_overrides=overrides,
                    )

                with self.assertRaisesRegex(ValueError, message):
                    load_runs(runs)

    def test_output_is_deterministic_for_run_order(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            runs = [root / "zeta", root / "alpha"]
            for run in runs:
                run.mkdir()
                _write_result(
                    run,
                    "coding_public",
                    model=run.name,
                    domain="coding",
                    fingerprint="sha256:same",
                    samples=[
                        _sample(
                            run.name,
                            model=run.name,
                            domain="coding",
                            tool=True,
                            args=True,
                            execution=True,
                            final=True,
                            failure="correct",
                        )
                    ],
                )
            self.assertEqual(build_scorecard(runs), build_scorecard(list(reversed(runs))))

    def test_loads_completed_organized_run_from_artifact_index(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            run = Path(temporary_directory) / "run"
            dataset = run / "domains" / "coding" / "coding_public"
            dataset.mkdir(parents=True)
            sample = self._correct_sample("model")
            sample["benchmark_path"] = "benchmark/coding/coding_public.json"
            samples_path = dataset / "samples.jsonl"
            summary_path = dataset / "summary.json"
            samples_path.write_text(json.dumps(sample) + "\n", encoding="utf-8")
            summary_path.write_text(
                json.dumps(
                    {
                        "model_name": "model",
                        "benchmark_path": "benchmark/coding/coding_public.json",
                        "evaluation_protocol": "single_step_tool_routing_v1",
                        "reasoning_mode": "direct",
                        "reasoning_method": "none",
                        "total_samples": 1,
                        "tool_pool": "full_mcp_registry",
                        "tool_count": 60,
                        "tool_registry_fingerprint": "sha256:organized",
                        "tool_registry_fingerprint_version": (
                            "tool_registry_name_schema_description_v1"
                        ),
                    }
                ),
                encoding="utf-8",
            )
            (dataset / "evaluation.log").write_text("complete\n", encoding="utf-8")
            (run / "artifact_index.jsonl").write_text(
                json.dumps(
                    {
                        "samples_path": "domains/coding/coding_public/samples.jsonl",
                        "summary_path": "domains/coding/coding_public/summary.json",
                        "evaluation_log_path": (
                            "domains/coding/coding_public/evaluation.log"
                        ),
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            (run / "RUN_COMPLETE").write_text("complete\n", encoding="utf-8")

            loaded = load_runs([run])

            self.assertEqual(len(loaded.records), 1)
            self.assertEqual(loaded.records[0].model, "model")
            self.assertEqual(loaded.fingerprints, ("sha256:organized",))


if __name__ == "__main__":
    unittest.main()
