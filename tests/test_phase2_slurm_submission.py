from __future__ import annotations

import os
from pathlib import Path
import shlex
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest


ROOT = Path(__file__).resolve().parents[1]
SUBMITTER = ROOT / "scripts" / "slurm" / "submit_phase2.sh"
SINGLE_LAUNCHER = ROOT / "scripts" / "slurm" / "run_phase2_single_step.sbatch"
MULTI_LAUNCHER = ROOT / "scripts" / "slurm" / "run_phase2_multistep.sbatch"


class Phase2SlurmSubmissionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        checkpoint = self.root / "checkpoint"
        checkpoint.mkdir()
        self.checkpoint = checkpoint
        (checkpoint / "config.json").write_text("{}", encoding="utf-8")
        (checkpoint / "model.safetensors").write_bytes(b"weights")
        tokenizer = self.root / "tokenizer"
        tokenizer.mkdir()
        self.tokenizer = tokenizer
        (tokenizer / "o200k_base.tiktoken").write_bytes(b"asset")
        self.venv = self.root / "external environment"
        (self.venv / "bin").mkdir(parents=True)
        (self.venv / "bin" / "activate").write_text("# test environment\n")
        (self.venv / "bin" / "python").symlink_to(sys.executable)
        self.env_file = self.root / "test.env"
        self.env_file.write_text(
            "\n".join(
                f"{name}={shlex.quote(str(value))}"
                for name, value in (
                    ("LAYERMCP_REPO_ROOT", ROOT),
                    ("LAYERMCP_GPT_OSS_CHECKPOINT", checkpoint),
                    ("TIKTOKEN_ENCODINGS_BASE", tokenizer),
                    ("LAYERMCP_VENV", self.venv),
                )
            ) + "\n",
            encoding="utf-8",
        )
        self.submit_log = self.root / "submitted.txt"
        fake_bin = self.root / "bin"
        fake_bin.mkdir()
        fake_sbatch = fake_bin / "sbatch"
        fake_sbatch.write_text(
            '#!/bin/sh\nprintf "%s\\n" "$*" >> "$SUBMIT_LOG"\nprintf "24000001\\n"\n',
            encoding="utf-8",
        )
        fake_sbatch.chmod(0o750)
        self.environment = os.environ.copy()
        self.environment["PATH"] = f"{fake_bin}{os.pathsep}{self.environment.get('PATH', '')}"
        self.environment["SUBMIT_LOG"] = str(self.submit_log)
        self.environment["WORKFLOW_LIMIT"] = "2"

    def run_submitter(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", str(SUBMITTER), *arguments, "--env-file", str(self.env_file)],
            cwd=self.root,
            env=self.environment,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_dry_run_modes_cover_expected_jobs(self) -> None:
        expected = {"both": 9, "single": 1, "multi": 8}
        for mode, count in expected.items():
            with self.subTest(mode=mode):
                result = self.run_submitter(mode, "--dry-run")
                self.assertEqual(result.returncode, 0, result.stderr)
                commands = [line for line in result.stdout.splitlines() if line.startswith("sbatch ")]
                self.assertEqual(len(commands), count)
                self.assertIn(f"Would submit {count} job(s)", result.stdout)
                self.assertEqual(any("run_phase2_single_step.sbatch" in line for line in commands), mode != "multi")
                self.assertEqual(sum("run_phase2_multistep.sbatch" in line for line in commands), 0 if mode == "single" else 8)
                self.assertEqual(any("DATASET_GROUP=math_controlled" in line for line in commands), mode != "single")
        self.assertFalse(self.submit_log.exists())

    def test_primary_only_excludes_controlled_math(self) -> None:
        result = self.run_submitter("multi", "--primary-only", "--dry-run")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Would submit 7 job(s)", result.stdout)
        self.assertNotIn("DATASET_GROUP=math_controlled", result.stdout)

    def test_no_mode_requires_saved_single_step_run(self) -> None:
        multi = self.run_submitter("multi", "--dry-run")
        self.assertEqual(multi.returncode, 0, multi.stderr)
        single = self.run_submitter("single", "--dry-run")
        self.assertEqual(single.returncode, 0, single.stderr)
        self.assertNotIn("LAYERMCP_GPT_OSS_SOURCE_RUN", single.stdout)

    def test_single_step_dataset_selection_uses_phase1_ids(self) -> None:
        result = self.run_submitter(
            "single", "--domains", "math", "--single-datasets", "finance_finqa_test_single", "--dry-run"
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("benchmark/math/math_controlled.json", result.stdout)
        self.assertIn("benchmark/finance/finance_finqa_test_single.json", result.stdout)
        self.assertNotIn("benchmark/coding/coding_controlled.json", result.stdout)

    def test_submits_with_explicit_configuration_and_clears_short_test_limit(self) -> None:
        result = self.run_submitter(
            "both", "--layer", "2", "--target", "qkv", "--strength", "0.02", "--seed", "42"
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Submitted 9 job(s).", result.stdout)
        calls = self.submit_log.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(calls), 9)
        self.assertTrue(all("LAYER_INDEX=2" in call and "STRENGTH=0.02" in call and "SEED=42" in call for call in calls))
        self.assertEqual(sum("run_phase2_single_step.sbatch" in call for call in calls), 1)
        self.assertEqual(sum("run_phase2_multistep.sbatch" in call for call in calls), 8)
        self.assertTrue(all("RUN_KIND=full" in call for call in calls[1:]))

    def test_invalid_settings_fail_before_submission(self) -> None:
        for arguments in (
            ("unknown",),
            ("both", "--strength", "nan"),
            ("multi", "--target", "invalid"),
            ("single", "--primary-only"),
        ):
            with self.subTest(arguments=arguments):
                result = self.run_submitter(*arguments)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(self.submit_log.exists())

    def test_single_step_launcher_uses_direct_primary_benchmarks(self) -> None:
        environment = self.environment.copy()
        environment.update({
            "SLURM_JOB_ID": "24000001",
            "LAYERMCP_VENV": str(self.venv),
            "LAYERMCP_REPO_ROOT": str(ROOT),
            "LAYERMCP_GPT_OSS_CHECKPOINT": str(self.checkpoint),
            "TIKTOKEN_ENCODINGS_BASE": str(self.tokenizer),
            "PHASE2_PYTHON_LOG": str(self.root / "python_args.txt"),
        })
        command = (
            'module() { :; }; '
            'python() { '
            'if [[ "$1" == "-" ]]; then "$LAYERMCP_VENV/bin/python" "$@"; '
            'else printf "%s\\n" "$*" > "$PHASE2_PYTHON_LOG"; fi; '
            '}; export -f module python; bash "$1"'
        )
        result = subprocess.run(
            ["bash", "-c", command, "bash", str(SINGLE_LAUNCHER)],
            cwd=ROOT, env=environment, capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Evaluating 13 direct single-step benchmark files", result.stdout)
        arguments = (self.root / "python_args.txt").read_text()
        self.assertIn("--benchmarks", arguments)
        self.assertIn("benchmark/math/math_controlled.json", arguments)
        self.assertIn("benchmark/finance/finance_finqa_test_single.json", arguments)
        self.assertNotIn("--source-run-dir", arguments)

    def test_missing_environment_fails_before_submission(self):
        for missing in ("activate", "python"):
            with self.subTest(missing=missing):
                path = self.venv / "bin" / missing
                path.rename(path.with_suffix(".saved"))
                result = self.run_submitter("both")
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("Invalid LAYERMCP_VENV", result.stderr)
                self.assertFalse(self.submit_log.exists())
                path.with_suffix(".saved").rename(path)

    def test_multistep_launcher_uses_external_environment(self):
        environment = self.environment | {
            "SLURM_JOB_ID": "24000002", "LAYERMCP_REPO_ROOT": str(ROOT),
            "LAYERMCP_VENV": str(self.venv),
            "LAYERMCP_GPT_OSS_CHECKPOINT": str(self.checkpoint),
            "TIKTOKEN_ENCODINGS_BASE": str(self.tokenizer),
            "DATASET_GROUP": "math_controlled", "RUN_KIND": "short_test",
        }
        result = subprocess.run(
            ["bash", "-c", 'module() { :; }; python() { printf "%s\\n" "$*"; }; '
             'export -f module python; bash "$1"', "bash", str(MULTI_LAUNCHER)],
            env=environment, capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--workflow-limit 2", result.stdout)

    def test_default_environment_is_checkout_local(self):
        helper = ROOT / "scripts/slurm/phase2_environment.sh"
        environment = self.environment.copy()
        environment.pop("LAYERMCP_VENV", None)
        environment["LAYERMCP_REPO_ROOT"] = str(self.root)
        result = subprocess.run(
            ["bash", "-c", 'source "$1"; validate_phase2_environment', "bash", str(helper)],
            env=environment, capture_output=True, text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(str(self.root / ".venv"), result.stderr)

    def test_launchers_and_submitter_have_valid_shell_syntax(self) -> None:
        for script in (SUBMITTER, SINGLE_LAUNCHER, MULTI_LAUNCHER, ROOT / "scripts/slurm/phase2_environment.sh"):
            result = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
