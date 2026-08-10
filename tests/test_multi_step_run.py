from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from analysis.multi_step_run import (
    DATASET_GROUPS,
    EMPTY_PLACEHOLDER,
    dataset_counts,
    resolve_dataset_groups,
    select_longest_workflow,
    validate_and_index_multistep,
    validate_complete_multistep_run,
    write_preflight_subset,
)


ROOT = Path(__file__).resolve().parents[1]
FINGERPRINT = "sha256:" + "a" * 64
VERSION = "tool_registry_name_schema_description_v1"
MODEL = "microsoft/phi-4"
PROMPT = "tool_name_only_v1"


class MultiStepRunTests(unittest.TestCase):
    def test_dataset_mapping_and_counts(self) -> None:
        expected = {
            "coding_sweagent": (5, 11),
            "coding_nebius_replay": (33, 139),
            "enterprise_tau2": (69, 350),
            "convfinqa": (10, 35),
            "finqa": (490, 1111),
            "finretrieval_replay": (485, 1490),
            "mathematics": (50, 105),
        }
        self.assertEqual(set(DATASET_GROUPS), set(expected))
        for group, counts in expected.items():
            self.assertEqual(dataset_counts(ROOT / DATASET_GROUPS[group]), counts)
        self.assertEqual(sum(value[0] for value in expected.values()), 1142)
        self.assertEqual(sum(value[1] for value in expected.values()), 3241)
        self.assertNotIn(EMPTY_PLACEHOLDER, DATASET_GROUPS.values())
        self.assertEqual(json.loads((ROOT / EMPTY_PLACEHOLDER).read_text()), [])

    def test_full_rejects_all_and_preflight_accepts_it(self) -> None:
        with self.assertRaisesRegex(ValueError, "preflight"):
            resolve_dataset_groups("all", "full")
        self.assertEqual(len(resolve_dataset_groups("all", "preflight")), 7)

    def _benchmark(self, root: Path) -> Path:
        path = root / "source.json"
        path.write_text(json.dumps([
            {"id":"short","domain":"mathematics","task_type":"multi_step_tool_routing","query":"x","expected_steps":[{"id":"s1","query":"x","expected_tool":"calculator","expected_args":{},"expected_answer":1},{"id":"s2","query":"y","expected_tool":"calculator","expected_args":{},"expected_answer":2}]},
            {"id":"long","domain":"mathematics","task_type":"multi_step_tool_routing","query":"long task","expected_steps":[{"id":"a","query":"longest query","expected_tool":"calculator","expected_args":{},"expected_answer":1},{"id":"b","query":"next","expected_tool":"calculator","expected_args":{},"expected_answer":2,"depends_on":["a"]},{"id":"c","query":"final longest query with dependency","expected_tool":"calculator","expected_args":{},"expected_answer":3,"depends_on":["b"]}]},
        ]), encoding="utf-8")
        return path

    def test_preflight_selects_longest_and_records_provenance(self) -> None:
        with TemporaryDirectory() as temporary:
            root=Path(temporary); source=self._benchmark(root)
            subset=root/"subset.json"; provenance=root/"provenance.json"
            original=source.read_bytes()
            metadata=write_preflight_subset(source,subset,provenance,len)
            self.assertEqual(json.loads(subset.read_text())[0]["id"], "long")
            self.assertEqual(metadata["selected_workflow_id"], "long")
            self.assertFalse(metadata["headline_eligible"])
            self.assertEqual(metadata, json.loads(provenance.read_text()))
            self.assertEqual(source.read_bytes(), original)

    def _artifacts(self, root: Path, benchmark: Path, *, mode="grounded_tool_execution") -> Path:
        data=json.loads(benchmark.read_text()); dataset=root/"domains/math/test"; dataset.mkdir(parents=True)
        records=[]
        for row in data:
            steps=[{"step_id":s["id"],"final_outcome_matcher":"recursive_json_subset_v1"} for s in row["expected_steps"]]
            records.append({"sample_id":row["id"],"benchmark_path":str(benchmark),"model_name":MODEL,"prompt_template":PROMPT,"evaluation_protocol":"teacher_forced_step_routing_v1","benchmark_mode":mode,"workflow_execution_mode":"isolated_step","steps":steps,"tool_pool":"full_mcp_registry","tool_count":60,"tool_registry_fingerprint":FINGERPRINT,"tool_registry_fingerprint_version":VERSION})
        (dataset/"samples.jsonl").write_text("".join(json.dumps(x)+"\n" for x in records))
        (dataset/"summary.json").write_text(json.dumps({"benchmark_path":str(benchmark),"model_name":MODEL,"prompt_template":PROMPT,"evaluation_protocol":"teacher_forced_step_routing_v1","total_workflows":len(records),"total_steps":sum(len(x["steps"]) for x in records),"benchmark_mode_counts":{mode:len(records)},"workflow_execution_modes":["isolated_step"],"tool_pool":"full_mcp_registry","tool_count":60,"tool_registry_fingerprint":FINGERPRINT,"tool_registry_fingerprint_version":VERSION}))
        (dataset/"evaluation.log").write_text("complete\n")
        return dataset

    def test_multistep_validation_index_and_collision(self) -> None:
        with TemporaryDirectory() as temporary:
            root=Path(temporary); benchmark=self._benchmark(root); run=root/"run"
            dataset=self._artifacts(run,benchmark); index=run/"artifact_index.jsonl"
            kwargs=dict(dataset_directory=dataset,index_path=index,source_benchmark=benchmark,evaluated_benchmark=benchmark,expected_model=MODEL,expected_prompt_template=PROMPT,expected_registry_fingerprint=FINGERPRINT,expected_registry_fingerprint_version=VERSION,expected_tool_count=60,expected_tool_pool="full_mcp_registry")
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
            summary=json.loads((dataset/"summary.json").read_text()); summary["evaluation_protocol"]="wrong"; (dataset/"summary.json").write_text(json.dumps(summary))
            with self.assertRaisesRegex(ValueError,"evaluation_protocol"):
                validate_and_index_multistep(dataset_directory=dataset,index_path=run/"artifact_index.jsonl",source_benchmark=benchmark,evaluated_benchmark=benchmark,expected_model=MODEL,expected_prompt_template=PROMPT,expected_registry_fingerprint=FINGERPRINT,expected_registry_fingerprint_version=VERSION,expected_tool_count=60,expected_tool_pool="full_mcp_registry")

    def test_complete_run_rejects_pooled_grounded_and_replay_modes(self) -> None:
        with TemporaryDirectory() as temporary:
            root=Path(temporary); benchmark=self._benchmark(root); run=root/"run"
            dataset=self._artifacts(run,benchmark)
            samples=[json.loads(line) for line in (dataset/"samples.jsonl").read_text().splitlines()]
            samples[1]["benchmark_mode"]="offline_trace_replay"
            (dataset/"samples.jsonl").write_text("".join(json.dumps(x)+"\n" for x in samples))
            summary=json.loads((dataset/"summary.json").read_text()); summary["benchmark_mode_counts"]={"grounded_tool_execution":1,"offline_trace_replay":1}; (dataset/"summary.json").write_text(json.dumps(summary))
            index=run/"artifact_index.jsonl"
            validate_and_index_multistep(dataset_directory=dataset,index_path=index,source_benchmark=benchmark,evaluated_benchmark=benchmark,expected_model=MODEL,expected_prompt_template=PROMPT,expected_registry_fingerprint=FINGERPRINT,expected_registry_fingerprint_version=VERSION,expected_tool_count=60,expected_tool_pool="full_mcp_registry")
            with self.assertRaisesRegex(ValueError,"pools multiple"):
                validate_complete_multistep_run(index,[benchmark])

    def test_launcher_contract_is_tracked_and_dynamic(self) -> None:
        launcher=(ROOT/"scripts/slurm/run_multi_step.sbatch").read_text()
        self.assertNotRegex(launcher,r"sha256:[0-9a-f]{64}")
        self.assertNotRegex(launcher,r"TOOL_COUNT=[0-9]+")
        self.assertIn("--capture-live-registry --include-catalog",launcher)
        self.assertIn("results/runs/multi_step",launcher)
        self.assertIn("teacher_forced_step_routing_v1",launcher)
        self.assertNotIn("coding_nebius_swerebench_openhands",launcher)
        self.assertIn("PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",launcher)


if __name__ == "__main__":
    unittest.main()
