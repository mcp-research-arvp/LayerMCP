from __future__ import annotations

import os
from pathlib import Path
import json
import shlex
import subprocess
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
        source = self.root / "saved_run"
        source.mkdir()
        self.source = source
        (source / "RUN_COMPLETE").write_text("done\n", encoding="utf-8")
        (source / "run_metadata.json").write_text("{}\n", encoding="utf-8")
        (source / "artifact_index.jsonl").write_text(
            json.dumps({"samples_path": "samples.jsonl"}) + "\n", encoding="utf-8"
        )
        (source / "samples.jsonl").write_text(
            "".join(json.dumps({"sample_id": f"sample-{index}"}) + "\n" for index in range(3)),
            encoding="utf-8",
        )
        self.env_file = self.root / "test.env"
        self.env_file.write_text(
            "\n".join(
                f"{name}={shlex.quote(str(value))}"
                for name, value in (
                    ("LAYERMCP_REPO_ROOT", ROOT),
                    ("LAYERMCP_GPT_OSS_CHECKPOINT", checkpoint),
                    ("LAYERMCP_GPT_OSS_SOURCE_RUN", source),
                    ("TIKTOKEN_ENCODINGS_BASE", tokenizer),
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

    def test_multi_mode_does_not_require_saved_single_step_run(self) -> None:
        (self.source / "RUN_COMPLETE").unlink()
        multi = self.run_submitter("multi", "--dry-run")
        self.assertEqual(multi.returncode, 0, multi.stderr)
        single = self.run_submitter("single", "--dry-run")
        self.assertEqual(single.returncode, 2)
        self.assertIn("missing RUN_COMPLETE", single.stderr)

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

    def test_single_step_launcher_counts_source_rows_at_job_start(self) -> None:
        environment = self.environment.copy()
        environment.update({
            "SLURM_JOB_ID": "24000001",
            "LAYERMCP_REPO_ROOT": str(ROOT),
            "LAYERMCP_GPT_OSS_CHECKPOINT": str(self.checkpoint),
            "LAYERMCP_GPT_OSS_SOURCE_RUN": str(self.source),
            "TIKTOKEN_ENCODINGS_BASE": str(self.tokenizer),
            "PHASE2_PYTHON_LOG": str(self.root / "python_args.txt"),
        })
        command = (
            'module() { :; }; '
            'python() { '
            'if [[ "$1" == "-" ]]; then "$LAYERMCP_REPO_ROOT/.venv/bin/python" "$@"; '
            'else printf "%s\\n" "$*" > "$PHASE2_PYTHON_LOG"; fi; '
            '}; export -f module python; bash "$1"'
        )
        result = subprocess.run(
            ["bash", "-c", command, "bash", str(SINGLE_LAUNCHER)],
            cwd=ROOT, env=environment, capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Evaluating 3 saved single-step examples", result.stdout)
        self.assertIn("--example-limit 3", (self.root / "python_args.txt").read_text())

    def test_launchers_and_submitter_have_valid_shell_syntax(self) -> None:
        for script in (SUBMITTER, SINGLE_LAUNCHER, MULTI_LAUNCHER):
            result = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
