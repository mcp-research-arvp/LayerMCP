from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import unittest

from analysis.multi_step_run import (
    DATASET_GROUPS,
    EMPTY_PLACEHOLDER,
    dataset_counts,
    resolve_dataset_groups,
    select_longest_workflow,
    select_longest_workflows,
    validate_and_index_multistep,
    validate_complete_multistep_run,
    write_short_test_subset,
)
from evaluation.evaluate import (
    MULTISTEP_EVALUATION_PROTOCOL,
    WORKFLOW_FINAL_SCORING_VERSION,
)


ROOT = Path(__file__).resolve().parents[1]
FINGERPRINT = "sha256:" + "a" * 64
VERSION = "tool_registry_name_schema_description_v1"
MODEL = "microsoft/phi-4"
PROMPT = "tool_name_only_v1"


class MultiStepRunTests(unittest.TestCase):
    EXPECTED_LAUNCHER_GROUPS = {
        "coding_sweagent": (
            "benchmark/coding/coding_sweagent_multistep.json",
            "coding",
            "grounded_tool_execution",
            "5",
            "11",
        ),
        "coding_nebius_replay": (
            "benchmark/coding/coding_nebius_sweagent_replay_multistep.json",
            "coding",
            "offline_trace_replay",
            "33",
            "139",
        ),
        "enterprise_tau2": (
            "benchmark/enterprise/enterprise_public_workflows.json",
            "enterprise",
            "grounded_tool_execution",
            "69",
            "350",
        ),
        "finance_convfinqa": (
            "benchmark/finance/finance_convfinqa_multistep.json",
            "finance",
            "grounded_tool_execution",
            "10",
            "35",
        ),
        "finance_finqa": (
            "benchmark/finance/finance_finqa_test_multistep.json",
            "finance",
            "grounded_tool_execution",
            "490",
            "1111",
        ),
        "finance_finretrieval_replay": (
            "benchmark/finance/finance_finretrieval_replay_multistep.json",
            "finance",
            "offline_trace_replay",
            "485",
            "1490",
        ),
        "math_controlled": (
            "benchmark/math/math_multistep_controlled.json",
            "math",
            "grounded_tool_execution",
            "50",
            "105",
        ),
    }

    @staticmethod
    def _metric_summaries(index: Path) -> dict[str, dict[str, object]]:
        entries = [json.loads(line) for line in index.read_text().splitlines()]
        return {
            entry["benchmark_path"]: {
                key: value
                for key, value in entry.items()
                if key == "workflow_final_scoring_version"
                or key.startswith("workflow_final_answer_")
                or key.startswith("workflow_final_program_execution_")
                or key.startswith("workflow_final_tool_result_")
            }
            for entry in entries
        }

    def _validate_launcher_group(
        self,
        group: str | None,
        *,
        run_kind: str = "full",
        launcher: Path | None = None,
    ) -> subprocess.CompletedProcess[str]:
        environment = os.environ.copy()
        environment.update(
            {
                "LAYERMCP_REPO_ROOT": str(ROOT),
                "REASONING_MODE": "direct",
                "RUN_KIND": run_kind,
                "LAYERMCP_VALIDATE_DATASET_GROUPS_ONLY": "1",
            }
        )
        if group is None:
            environment.pop("DATASET_GROUP", None)
        else:
            environment["DATASET_GROUP"] = group
        return subprocess.run(
            ["bash", str(launcher or ROOT / "scripts/slurm/run_multi_step.sbatch")],
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_dataset_mapping_and_counts(self) -> None:
        expected = {
            "coding_sweagent": (5, 11),
            "coding_nebius_replay": (33, 139),
            "enterprise_tau2": (69, 350),
            "finance_convfinqa": (10, 35),
            "finance_finqa": (490, 1111),
            "finance_finretrieval_replay": (485, 1490),
            "math_controlled": (50, 105),
        }
        self.assertEqual(set(DATASET_GROUPS), set(expected))
        for group, counts in expected.items():
            self.assertEqual(dataset_counts(ROOT / DATASET_GROUPS[group]), counts)
        self.assertEqual(sum(value[0] for value in expected.values()), 1142)
        self.assertEqual(sum(value[1] for value in expected.values()), 3241)
        self.assertNotIn(EMPTY_PLACEHOLDER, DATASET_GROUPS.values())
        self.assertEqual(json.loads((ROOT / EMPTY_PLACEHOLDER).read_text()), [])

    def test_full_rejects_all_and_short_test_accepts_it(self) -> None:
        with self.assertRaisesRegex(ValueError, "short_test"):
            resolve_dataset_groups("all", "full")
        self.assertEqual(len(resolve_dataset_groups("all", "short_test")), 7)

    def _benchmark(self, root: Path) -> Path:
        path = root / "source.json"
        path.write_text(json.dumps([
            {"id":"short","domain":"mathematics","task_type":"multi_step_tool_routing","query":"x","expected_steps":[{"id":"s1","query":"x","expected_tool":"calculator","expected_args":{},"expected_answer":1},{"id":"s2","query":"y","expected_tool":"calculator","expected_args":{},"expected_answer":2}]},
            {"id":"long","domain":"mathematics","task_type":"multi_step_tool_routing","query":"long task","expected_steps":[{"id":"a","query":"longest query","expected_tool":"calculator","expected_args":{},"expected_answer":1},{"id":"b","query":"next","expected_tool":"calculator","expected_args":{},"expected_answer":2,"depends_on":["a"]},{"id":"c","query":"final longest query with dependency","expected_tool":"calculator","expected_args":{},"expected_answer":3,"depends_on":["b"]}]},
        ]), encoding="utf-8")
        return path

    def test_short_test_selects_longest_and_records_provenance(self) -> None:
        with TemporaryDirectory() as temporary:
            root=Path(temporary); source=self._benchmark(root)
            subset=root/"subset.json"; provenance=root/"provenance.json"
            original=source.read_bytes()
            metadata=write_short_test_subset(source,subset,provenance,len)
            self.assertEqual(json.loads(subset.read_text())[0]["id"], "long")
            self.assertEqual(metadata["selected_workflow_ids"], ["long"])
            self.assertFalse(metadata["headline_eligible"])
            self.assertEqual(metadata, json.loads(provenance.read_text()))
            self.assertEqual(source.read_bytes(), original)

    def test_gemma_style_short_test_can_select_several_longest(self) -> None:
        with TemporaryDirectory() as temporary:
            source = self._benchmark(Path(temporary))
            selected, metadata = select_longest_workflows(source, len, count=3)
            self.assertEqual([row["id"] for row in selected], ["long", "short"])
            self.assertEqual(metadata["selected_workflow_count"], 2)

    def _artifacts(self, root: Path, benchmark: Path, *, mode="grounded_tool_execution") -> Path:
        data=json.loads(benchmark.read_text()); dataset=root/"domains/math/test"; dataset.mkdir(parents=True)
        records=[]
        for row in data:
            steps=[{"step_id":s["id"],"final_outcome_matcher":"recursive_json_subset_v1"} for s in row["expected_steps"]]
            records.append({"sample_id":row["id"],"benchmark_path":str(benchmark),"model_name":MODEL,"prompt_template":PROMPT,"evaluation_protocol":MULTISTEP_EVALUATION_PROTOCOL,"reasoning_mode":"direct","reasoning_method":"none","effective_generation_limit":128,"effective_generation_limit_unit":"tokens","benchmark_mode":mode,"workflow_execution_mode":"predicted_sequence","steps":steps,"tool_pool":"full_mcp_registry","tool_count":60,"tool_registry_fingerprint":FINGERPRINT,"tool_registry_fingerprint_version":VERSION,"workflow_final_scoring_version":WORKFLOW_FINAL_SCORING_VERSION})
        workflow_metrics = {"workflow_final_scoring_version": WORKFLOW_FINAL_SCORING_VERSION}
        for prefix in ("workflow_final_answer", "workflow_final_program_execution", "workflow_final_tool_result"):
            workflow_metrics.update({f"{prefix}_accuracy":None,f"{prefix}_gold":0,f"{prefix}_scored":0,f"{prefix}_correct":0,f"{prefix}_mismatch":0,f"{prefix}_extraction_error":0,f"{prefix}_unavailable":len(records),f"{prefix}_status_counts":{"unsupported":len(records)},f"{prefix}_contracts":[],f"{prefix}_matchers":[],f"{prefix}_scoring_version":WORKFLOW_FINAL_SCORING_VERSION})
        (dataset/"samples.jsonl").write_text("".join(json.dumps(x)+"\n" for x in records))
        (dataset/"summary.json").write_text(json.dumps({"benchmark_path":str(benchmark),"model_name":MODEL,"prompt_template":PROMPT,"evaluation_protocol":MULTISTEP_EVALUATION_PROTOCOL,"reasoning_mode":"direct","reasoning_method":"none","effective_generation_limit":128,"effective_generation_limit_unit":"tokens","total_workflows":len(records),"total_steps":sum(len(x["steps"]) for x in records),"benchmark_mode_counts":{mode:len(records)},"workflow_execution_modes":["predicted_sequence"],"tool_pool":"full_mcp_registry","tool_count":60,"tool_registry_fingerprint":FINGERPRINT,"tool_registry_fingerprint_version":VERSION,**workflow_metrics}))
        (dataset/"evaluation.log").write_text("complete\n")
        return dataset

    def test_multistep_validation_index_and_collision(self) -> None:
        with TemporaryDirectory() as temporary:
            root=Path(temporary); benchmark=self._benchmark(root); run=root/"run"
            dataset=self._artifacts(run,benchmark); index=run/"artifact_index.jsonl"
            kwargs=dict(dataset_directory=dataset,index_path=index,source_benchmark=benchmark,evaluated_benchmark=benchmark,expected_model=MODEL,expected_prompt_template=PROMPT,expected_registry_fingerprint=FINGERPRINT,expected_registry_fingerprint_version=VERSION,expected_tool_count=60,expected_tool_pool="full_mcp_registry",expected_reasoning_mode="direct",expected_reasoning_method="none",expected_generation_limit=128)
            record=validate_and_index_multistep(**kwargs)
            self.assertEqual(record["workflow_count"],2); self.assertEqual(record["expected_step_count"],5)
            self.assertEqual(record["benchmark_modes"],["grounded_tool_execution"])
            validate_complete_multistep_run(index,[benchmark])
            with self.assertRaisesRegex(ValueError,"already indexed"):
                validate_and_index_multistep(**kwargs)

    def test_wrong_protocol_and_mixed_mode_are_rejected(self) -> None:
        with TemporaryDirectory() as temporary:
            root=Path(temporary); benchmark=self._benchmark(root); run=root/"run"
            dataset=self._artifacts(run,benchmark)
            summary=json.loads((dataset/"summary.json").read_text()); summary["evaluation_protocol"]="teacher_forced_step_routing_v1"; (dataset/"summary.json").write_text(json.dumps(summary))
            with self.assertRaisesRegex(ValueError,"evaluation_protocol"):
                validate_and_index_multistep(dataset_directory=dataset,index_path=run/"artifact_index.jsonl",source_benchmark=benchmark,evaluated_benchmark=benchmark,expected_model=MODEL,expected_prompt_template=PROMPT,expected_registry_fingerprint=FINGERPRINT,expected_registry_fingerprint_version=VERSION,expected_tool_count=60,expected_tool_pool="full_mcp_registry",expected_reasoning_mode="direct",expected_reasoning_method="none",expected_generation_limit=128)

    def test_run_metadata_validates_condition_protocol_and_full_counts(self) -> None:
        with TemporaryDirectory() as temporary:
            root=Path(temporary); benchmark=self._benchmark(root); run=root/"run"
            dataset=self._artifacts(run,benchmark); index=run/"artifact_index.jsonl"
            validate_and_index_multistep(dataset_directory=dataset,index_path=index,source_benchmark=benchmark,evaluated_benchmark=benchmark,expected_model=MODEL,expected_prompt_template=PROMPT,expected_registry_fingerprint=FINGERPRINT,expected_registry_fingerprint_version=VERSION,expected_tool_count=60,expected_tool_pool="full_mcp_registry",expected_reasoning_mode="direct",expected_reasoning_method="none",expected_generation_limit=128)
            metadata={"workflow_final_scoring_version":WORKFLOW_FINAL_SCORING_VERSION,"expected_model_name":MODEL,"prompt_template_id":PROMPT,"reasoning_mode":"direct","reasoning_method":"none","effective_generation_limit":128,"effective_generation_limit_unit":"tokens","evaluation_protocol":MULTISTEP_EVALUATION_PROTOCOL,"workflow_execution_mode":"predicted_sequence","tool_pool":"full_mcp_registry","tool_count":60,"tool_registry_fingerprint":FINGERPRINT,"tool_registry_fingerprint_version":VERSION,"run_kind":"full","headline_eligible":True,"source_counts":{str(benchmark.resolve()):{"workflows":2,"routed_steps":5}},"short_test_selection":{}}
            metadata["workflow_metric_summaries"] = self._metric_summaries(index)
            metadata_path=run/"run_metadata.json";metadata_path.write_text(json.dumps(metadata))
            validate_complete_multistep_run(index,[benchmark],metadata_path)
            metadata["reasoning_method"]="native_enabled";metadata_path.write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError,"reasoning_method"):
                validate_complete_multistep_run(index,[benchmark],metadata_path)

    def test_short_test_is_non_headline_and_selection_is_validated(self) -> None:
        with TemporaryDirectory() as temporary:
            root=Path(temporary); benchmark=self._benchmark(root); run=root/"run"
            dataset=self._artifacts(run,benchmark); index=run/"artifact_index.jsonl"
            provenance={"selected_workflow_ids":["short","long"],"selected_workflow_count":2,"headline_eligible":False}
            provenance_path=run/"short_test/provenance.json";provenance_path.parent.mkdir();provenance_path.write_text(json.dumps(provenance))
            validate_and_index_multistep(dataset_directory=dataset,index_path=index,source_benchmark=benchmark,evaluated_benchmark=benchmark,expected_model=MODEL,expected_prompt_template=PROMPT,expected_registry_fingerprint=FINGERPRINT,expected_registry_fingerprint_version=VERSION,expected_tool_count=60,expected_tool_pool="full_mcp_registry",expected_reasoning_mode="direct",expected_reasoning_method="none",expected_generation_limit=128,short_test_provenance=provenance_path)
            metadata={"workflow_final_scoring_version":WORKFLOW_FINAL_SCORING_VERSION,"expected_model_name":MODEL,"prompt_template_id":PROMPT,"reasoning_mode":"direct","reasoning_method":"none","effective_generation_limit":128,"effective_generation_limit_unit":"tokens","evaluation_protocol":MULTISTEP_EVALUATION_PROTOCOL,"workflow_execution_mode":"predicted_sequence","tool_pool":"full_mcp_registry","tool_count":60,"tool_registry_fingerprint":FINGERPRINT,"tool_registry_fingerprint_version":VERSION,"run_kind":"short_test","headline_eligible":False,"source_counts":{str(benchmark.resolve()):{"workflows":2,"routed_steps":5}},"short_test_selection":{"test":provenance}}
            metadata["workflow_metric_summaries"] = self._metric_summaries(index)
            metadata_path=run/"run_metadata.json";metadata_path.write_text(json.dumps(metadata))
            validate_complete_multistep_run(index,[benchmark],metadata_path)
            metadata["headline_eligible"]=True;metadata_path.write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError,"headline eligibility"):
                validate_complete_multistep_run(index,[benchmark],metadata_path)

    def test_complete_run_rejects_pooled_grounded_and_replay_modes(self) -> None:
        with TemporaryDirectory() as temporary:
            root=Path(temporary); benchmark=self._benchmark(root); run=root/"run"
            dataset=self._artifacts(run,benchmark)
            samples=[json.loads(line) for line in (dataset/"samples.jsonl").read_text().splitlines()]
            samples[1]["benchmark_mode"]="offline_trace_replay"
            (dataset/"samples.jsonl").write_text("".join(json.dumps(x)+"\n" for x in samples))
            summary=json.loads((dataset/"summary.json").read_text()); summary["benchmark_mode_counts"]={"grounded_tool_execution":1,"offline_trace_replay":1}; (dataset/"summary.json").write_text(json.dumps(summary))
            index=run/"artifact_index.jsonl"
            validate_and_index_multistep(dataset_directory=dataset,index_path=index,source_benchmark=benchmark,evaluated_benchmark=benchmark,expected_model=MODEL,expected_prompt_template=PROMPT,expected_registry_fingerprint=FINGERPRINT,expected_registry_fingerprint_version=VERSION,expected_tool_count=60,expected_tool_pool="full_mcp_registry",expected_reasoning_mode="direct",expected_reasoning_method="none",expected_generation_limit=128)
            with self.assertRaisesRegex(ValueError,"pools multiple"):
                validate_complete_multistep_run(index,[benchmark])

    def test_launcher_contract_is_tracked_and_dynamic(self) -> None:
        launcher=(ROOT/"scripts/slurm/run_multi_step.sbatch").read_text()
        self.assertNotRegex(launcher,r"sha256:[0-9a-f]{64}")
        self.assertNotRegex(launcher,r"TOOL_COUNT=[0-9]+")
        self.assertIn("--capture-live-registry --include-catalog",launcher)
        self.assertIn("results/runs/multi_step",launcher)
        self.assertIn("guided_predicted_rollout_v1",launcher)
        self.assertNotIn("teacher_forced_step_routing_v1",launcher)
        self.assertIn('--reasoning-mode "$REASONING_MODE"', launcher)
        self.assertIn("short_test", launcher)
        for model in (
            "phi-4-local",
            "llama-3.1-8b-local",
            "qwen-3.6-local",
            "gemma-4-local",
            "gpt-oss-local",
        ):
            self.assertIn(model, launcher)
        self.assertNotIn("coding_nebius_swerebench_openhands",launcher)
        self.assertIn("PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",launcher)

    def test_launcher_does_not_reuse_bash_groups(self) -> None:
        launcher = (ROOT / "scripts/slurm/run_multi_step.sbatch").read_text()
        self.assertNotRegex(launcher, r"(?m)^\s*GROUPS=")
        self.assertNotIn("${GROUPS", launcher)
        self.assertIn("SELECTED_DATASET_GROUPS", launcher)

    def test_multi_step_launcher_root_and_checkpoint_contract(self) -> None:
        launcher = (ROOT / "scripts/slurm/run_multi_step.sbatch").read_text()
        self.assertIn('LAUNCHER_PATH="$(realpath "${BASH_SOURCE[0]}")"', launcher)
        self.assertNotIn('dirname "$LAUNCHER_PATH"', launcher)
        self.assertLess(
            launcher.index('if [[ -n "${LAYERMCP_REPO_ROOT:-}" ]]'),
            launcher.index('elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]]'),
        )
        self.assertIn('REPO_ROOT_CANDIDATE="$PWD"', launcher)
        self.assertIn("pwd -P", launcher)
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

    def test_each_launcher_group_resolves_without_registry_or_model_work(self) -> None:
        for group, expected in self.EXPECTED_LAUNCHER_GROUPS.items():
            with self.subTest(group=group):
                result = self._validate_launcher_group(group)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(
                    result.stdout.strip().split("\t"),
                    [group, *expected],
                )
                self.assertNotIn("registry", result.stdout.lower())
                self.assertNotIn("model weights", result.stdout.lower())

    def test_launcher_all_resolves_seven_groups_in_order(self) -> None:
        result = self._validate_launcher_group("all", run_kind="short_test")
        self.assertEqual(result.returncode, 0, result.stderr)
        rows = [line.split("\t") for line in result.stdout.splitlines()]
        self.assertEqual(
            [row[0] for row in rows],
            list(self.EXPECTED_LAUNCHER_GROUPS),
        )
        self.assertEqual(len(rows), 7)
        for row in rows:
            self.assertEqual(row[1:], list(self.EXPECTED_LAUNCHER_GROUPS[row[0]]))
            self.assertFalse(row[0].isdigit(), "Unix group ID leaked into selection")

    def test_launcher_rejects_unknown_and_empty_groups_clearly(self) -> None:
        for group in ("not_a_group", None):
            with self.subTest(group=group):
                result = self._validate_launcher_group(group)
                self.assertEqual(result.returncode, 2)
                self.assertIn("Accepted values:", result.stderr)
                for accepted in (*self.EXPECTED_LAUNCHER_GROUPS, "all"):
                    self.assertIn(accepted, result.stderr)

    def test_launcher_missing_group_metadata_fails_before_expensive_work(self) -> None:
        source = (ROOT / "scripts/slurm/run_multi_step.sbatch").read_text()
        modified = source.replace(
            "[math_controlled]=105",
            '[math_controlled]=""',
            1,
        )
        self.assertNotEqual(source, modified)
        with TemporaryDirectory() as temporary:
            launcher = Path(temporary) / "launcher.sbatch"
            launcher.write_text(modified, encoding="utf-8")
            result = self._validate_launcher_group(
                "math_controlled",
                launcher=launcher,
            )
        self.assertEqual(result.returncode, 2)
        self.assertIn("Missing launcher metadata", result.stderr)
        self.assertNotIn("registry", result.stdout.lower())
        self.assertNotIn("model weights", result.stdout.lower())

    def test_group_validation_precedes_registry_and_environment_activation(self) -> None:
        launcher = (ROOT / "scripts/slurm/run_multi_step.sbatch").read_text()
        validation_exit = launcher.index("LAYERMCP_VALIDATE_DATASET_GROUPS_ONLY")
        module_load = launcher.index("module load python/3.11")
        activate = launcher.index('source "$HOME/venvs/layermcp/bin/activate"')
        dataset_content_validation = launcher.index(
            'for dataset_group_name in "${SELECTED_DATASET_GROUPS[@]}"; do',
            module_load,
        )
        registry_capture = launcher.index("--capture-live-registry")
        checkpoint_access = launcher.index("$REPO_ROOT/checkpoints/phi-4")
        self.assertLess(validation_exit, module_load)
        self.assertNotIn("python", launcher[validation_exit:module_load])
        self.assertLess(module_load, activate)
        self.assertLess(activate, dataset_content_validation)
        self.assertLess(dataset_content_validation, registry_capture)
        self.assertLess(dataset_content_validation, checkpoint_access)

    def test_static_group_validation_does_not_require_python_or_environment(self) -> None:
        launcher = (ROOT / "scripts/slurm/run_multi_step.sbatch").read_text()
        static_start = launcher.index("DATASET_GROUP_ORDER=(")
        validation_only = launcher.index(
            'if [[ "${LAYERMCP_VALIDATE_DATASET_GROUPS_ONLY:-0}" == 1 ]]'
        )
        validation_exit = launcher.index("exit 0", validation_only)
        static_section = launcher[static_start:validation_exit]
        self.assertNotIn("python", static_section)
        self.assertNotIn("module load", static_section)
        self.assertNotIn("venvs/layermcp", static_section)
        self.assertNotIn("--capture-live-registry", static_section)

    def test_missing_and_invalid_groups_are_rejected_before_module_load(self) -> None:
        launcher = (ROOT / "scripts/slurm/run_multi_step.sbatch").read_text()
        module_load = launcher.index("module load python/3.11")
        self.assertLess(launcher.index('if [[ -z "${DATASET_GROUP:-}" ]]'), module_load)
        self.assertLess(launcher.index("Unsupported DATASET_GROUP:"), module_load)


if __name__ == "__main__":
    unittest.main()
