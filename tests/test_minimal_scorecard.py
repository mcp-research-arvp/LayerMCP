from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from analysis.minimal_scorecard import build_scorecard, load_runs


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
) -> dict:
    return {
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
    }


def _write_result(
    run: Path,
    name: str,
    *,
    model: str,
    domain: str,
    fingerprint: str,
    samples: list[dict],
) -> None:
    artifacts = run / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    samples_path = artifacts / f"{name}_samples.jsonl"
    summary_path = artifacts / f"{name}_summary.json"
    samples_path.write_text(
        "".join(json.dumps(sample) + "\n" for sample in samples),
        encoding="utf-8",
    )
    summary_path.write_text(
        json.dumps(
            {
                "model_name": model,
                "benchmark_path": f"benchmark/{domain}/{name}.json",
                "evaluation_protocol": "single_step_tool_routing_v1",
                "total_samples": len(samples),
                "tool_pool": "full_mcp_registry",
                "tool_count": 60,
                "tool_registry_fingerprint": fingerprint,
                "tool_registry_fingerprint_version": (
                    "tool_registry_name_schema_description_v1"
                ),
            }
        ),
        encoding="utf-8",
    )
    (run / f"{name}.log").write_text(
        f"Results: {samples_path}\nSummary: {summary_path}\n",
        encoding="utf-8",
    )


class MinimalScorecardTests(unittest.TestCase):
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
                "| phi | 2 | 50.0% | 2/2 (100.0%) | 50.0% | — |",
                markdown,
            )
            self.assertIn(
                "| llama | Finance | 1 | 0.0% | 1/1 (100.0%) | 100.0% | — |",
                markdown,
            )
            self.assertIn("public/source-derived", markdown)
            self.assertIn("| phi | 0.0% | 50.0% | 0.0% | 1 | 1 | 1 | 0 |", markdown)
            self.assertIn("Valid Arguments / Schema-Valid Tool Call (SVCA)", markdown)
            self.assertIn("Exact Reference Argument Match", markdown)
            self.assertNotIn("Exact canonical args", markdown)
            header = (
                "| Model | N | Final Outcome Accuracy | Final Outcome Coverage | "
                "Tool Selection Accuracy | Valid Arguments / SVCA |"
            )
            self.assertIn(header, markdown)
            self.assertLess(
                markdown.index("Final Outcome Accuracy"),
                markdown.index("Tool Selection Accuracy"),
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
                "| model | 2 | 100.0% | 1/2 (50.0%) | 100.0% | — |",
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
                "| model | 2 | — | 0/2 (0.0%) | 100.0% | — |",
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
                "| model | `finance_query_table_rows_v1`, "
                "`recursive_json_subset_v1` |",
                markdown,
            )
            self.assertIn(
                "| other-model | `recursive_json_subset_v1` |",
                markdown,
            )

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


if __name__ == "__main__":
    unittest.main()
