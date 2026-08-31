from __future__ import annotations

import os
from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts" / "slurm"


class SlurmSubmissionScriptTests(unittest.TestCase):
    def run_script(
        self,
        script: str,
        *arguments: str,
        environment: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        execution_environment = os.environ.copy()
        if environment is not None:
            execution_environment.update(environment)
        return subprocess.run(
            ["bash", str(SCRIPTS / script), *arguments],
            cwd=ROOT,
            env=execution_environment,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_all_submission_dry_run_covers_each_supported_condition(self) -> None:
        completed = self.run_script("submit_all.sh", "--dry-run")
        self.assertEqual(completed.returncode, 0, completed.stderr)
        commands = [line for line in completed.stdout.splitlines() if line.startswith("sbatch ")]
        self.assertEqual(len(commands), 56)
        self.assertIn("Submitted 7 single-step job(s).", completed.stdout)
        self.assertIn("Submitted 49 multi-step job(s).", completed.stdout)
        self.assertTrue(any("gpt-oss-local-reasoning-low-single-primary" in line for line in commands))
        self.assertTrue(any("gpt-oss-local-reasoning-low-multi-full-finance_finqa" in line for line in commands))

    def test_single_step_wrapper_can_limit_to_one_model(self) -> None:
        completed = self.run_script(
            "submit_single_step.sh", "--model", "qwen-3.6-local", "--dry-run"
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        commands = [line for line in completed.stdout.splitlines() if line.startswith("sbatch ")]
        self.assertEqual(len(commands), 2)
        self.assertTrue(all("MODEL=qwen-3.6-local" in line for line in commands))
        self.assertIn("Submitted 2 single-step job(s).", completed.stdout)

    def test_multi_step_wrapper_short_test_all_uses_one_job_per_condition(self) -> None:
        completed = self.run_script(
            "submit_multi_step.sh", "--multi-run-kind", "short_test", "--dry-run"
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        commands = [line for line in completed.stdout.splitlines() if line.startswith("sbatch ")]
        self.assertEqual(len(commands), 7)
        self.assertTrue(all("DATASET_GROUP=all" in line for line in commands))
        self.assertIn("Submitted 7 multi-step job(s).", completed.stdout)

    def test_multi_step_wrapper_can_limit_to_one_model_and_group(self) -> None:
        completed = self.run_script(
            "submit_multi_step.sh",
            "--model",
            "gpt-oss-local",
            "--multi-dataset-group",
            "finance_finqa",
            "--dry-run",
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("Submitted 1 multi-step job(s).", completed.stdout)
        self.assertIn("gpt-oss-local-reasoning-low-multi-full-finance_finqa", completed.stdout)
        self.assertIn("REASONING_EFFORT=low", completed.stdout)

    def test_single_step_domains_and_datasets_are_combined(self) -> None:
        completed = self.run_script(
            "submit_single_step.sh",
            "--model",
            "phi-4-local",
            "--domains",
            "math,finance",
            "--datasets",
            "coding_controlled",
            "--dry-run",
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        command = next(line for line in completed.stdout.splitlines() if line.startswith("sbatch "))
        self.assertIn("DATASET_SELECTION=benchmark/math/math_controlled.json:benchmark/math/math_public.json", command)
        self.assertIn("benchmark/finance/finance_finqa_test_single.json", command)
        self.assertIn("benchmark/coding/coding_controlled.json", command)

    def test_multi_step_domains_and_datasets_are_combined(self) -> None:
        completed = self.run_script(
            "submit_multi_step.sh",
            "--model",
            "phi-4-local",
            "--domains",
            "finance",
            "--datasets",
            "math_public_mathqa",
            "--dry-run",
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        commands = [line for line in completed.stdout.splitlines() if line.startswith("sbatch ")]
        self.assertEqual(len(commands), 4)
        self.assertTrue(any("DATASET_GROUP=finance_convfinqa" in line for line in commands))
        self.assertTrue(any("DATASET_GROUP=finance_finqa" in line for line in commands))
        self.assertTrue(any("DATASET_GROUP=finance_finretrieval_replay" in line for line in commands))
        self.assertTrue(any("DATASET_GROUP=math_public_mathqa" in line for line in commands))

    def test_combined_submission_requires_run_type_specific_dataset_selectors(self) -> None:
        completed = self.run_script("submit_all.sh", "--datasets", "math_controlled", "--dry-run")
        self.assertEqual(completed.returncode, 2)
        self.assertIn("--datasets is ambiguous", completed.stderr)

    def test_single_step_selection_rejects_cross_run_kind_datasets(self) -> None:
        for run_kind, dataset in (
            ("primary", "finance_smoke"),
            ("smoke", "finance_finqa_test_single"),
        ):
            with self.subTest(run_kind=run_kind, dataset=dataset):
                completed = self.run_script(
                    "submit_single_step.sh",
                    "--single-run-kind",
                    run_kind,
                    "--datasets",
                    dataset,
                    "--dry-run",
                )
                self.assertEqual(completed.returncode, 2)
                self.assertIn("Unsupported single-step dataset", completed.stderr)

    def test_default_single_step_submission_clears_inherited_dataset_selection(self) -> None:
        completed = self.run_script(
            "submit_all.sh",
            "--dry-run",
            environment={"DATASET_SELECTION": "benchmark/finance/finance_smoke.json"},
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        single_commands = [
            line
            for line in completed.stdout.splitlines()
            if line.startswith("sbatch ") and "run_single_step.sbatch" in line
        ]
        self.assertEqual(len(single_commands), 7)
        self.assertTrue(all("DATASET_SELECTION=" in line for line in single_commands))
        self.assertTrue(
            all(
                "DATASET_SELECTION=benchmark/finance/finance_smoke.json" not in line
                for line in single_commands
            )
        )

    def test_all_model_submission_clears_inherited_reasoning_effort_for_non_gpt(self) -> None:
        completed = self.run_script(
            "submit_all.sh",
            "--dry-run",
            environment={"REASONING_EFFORT": "low"},
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        commands = [line for line in completed.stdout.splitlines() if line.startswith("sbatch ")]
        non_gpt_commands = [line for line in commands if "MODEL=gpt-oss-local" not in line]
        self.assertTrue(non_gpt_commands)
        self.assertTrue(all("REASONING_EFFORT=" in line for line in non_gpt_commands))
        self.assertTrue(all("REASONING_EFFORT=low" not in line for line in non_gpt_commands))
        gpt_command = next(line for line in commands if "MODEL=gpt-oss-local" in line)
        self.assertIn("REASONING_EFFORT=low", gpt_command)

    def test_math_domain_excludes_optional_multistep_controlled_diagnostic(self) -> None:
        completed = self.run_script(
            "submit_multi_step.sh", "--model", "phi-4-local", "--domains", "math", "--dry-run"
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        commands = [line for line in completed.stdout.splitlines() if line.startswith("sbatch ")]
        self.assertEqual(len(commands), 1)
        self.assertIn("DATASET_GROUP=math_public_mathqa", commands[0])
        self.assertNotIn("DATASET_GROUP=math_controlled", commands[0])

        explicit = self.run_script(
            "submit_multi_step.sh",
            "--model",
            "phi-4-local",
            "--datasets",
            "math_controlled",
            "--dry-run",
        )
        self.assertEqual(explicit.returncode, 0, explicit.stderr)
        self.assertIn("DATASET_GROUP=math_controlled", explicit.stdout)

    def test_invalid_model_is_rejected_before_submission(self) -> None:
        completed = self.run_script(
            "submit_single_step.sh", "--model", "not-a-model", "--dry-run"
        )
        self.assertEqual(completed.returncode, 2)
        self.assertIn("Unsupported model: not-a-model", completed.stderr)

    def test_submission_scripts_pass_bash_syntax(self) -> None:
        for path in (
            SCRIPTS / "submit_common.sh",
            SCRIPTS / "submit_all.sh",
            SCRIPTS / "submit_single_step.sh",
            SCRIPTS / "submit_multi_step.sh",
        ):
            with self.subTest(path=path):
                completed = subprocess.run(
                    ["bash", "-n", str(path)],
                    cwd=ROOT,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
