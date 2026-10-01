"""CPU-only saved-artifact tests; no model, checkpoint or MCP server required."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from analysis.phase2_paired_results import (
    RUN_KIND, combine_completed_runs, main, save_summary,
)


def condition(sample_id, correct=True, tool="calculator", parse_status="ok"):
    return {
        "sample_id": sample_id, "selected_tool": tool, "selected_args": {},
        "raw_model_output": str(tool), "parse_status": parse_status,
        "invalid_output": parse_status != "ok", "no_call_outcome": tool is None,
        "tool_selection_correct": correct, "argument_match_correct": correct,
        "execution_success": correct, "final_outcome_correct": correct,
        "query": sample_id, "prompt_context": "", "expected_tool": "calculator",
        "expected_args": {}, "expected_answer": {"result": 1},
        "benchmark_path": "benchmark/math/math_controlled.json",
        "source": "controlled_synthetic", "benchmark_mode": "grounded_tool_execution",
        "tool_registry_fingerprint": "sha256:fixture", "tool_pool": "full_mcp_registry",
        "tool_count": 60, "tool_registry_fingerprint_version": "v1",
        "reasoning_mode": "reasoning", "reasoning_method": "harmony",
        "reasoning_effort": "low", "evaluation_protocol": "single_step_tool_routing_v1",
    }


def pair(layer, seed, sample_id):
    return {
        "kind": RUN_KIND, "layer_index": layer, "seed": seed, "sample_id": sample_id,
        "target": "qkv", "method": "noise", "strength": 0.01,
        "registry_exact_match": True,
        "control_verification": {"enabled": False, "changed_attention_parameter_names": [],
                                 "restored_exactly": None},
        "intervention_verification": {"enabled": True, "restored_exactly": True},
        "control": condition(sample_id), "intervention": condition(sample_id),
    }


class PairedResultsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="paired-smoke-parent-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def write_run(self, name="run", layers=(0,), seeds=(1234,), samples=("success", "failure")):
        path = self.root / name
        path.mkdir()
        metadata = {key: condition("success")[key] for key in (
            "benchmark_mode", "tool_pool", "tool_count", "tool_registry_fingerprint",
            "tool_registry_fingerprint_version", "reasoning_mode", "reasoning_method",
            "reasoning_effort", "evaluation_protocol",
        )}
        metadata.update(model="gpt-oss-local", generation={"temperature": 0.0})
        manifest = {
            "kind": RUN_KIND, "checkpoint_path": "/not-mounted/checkpoint",
            "source_run_directory": "/not-mounted/source", "source_run_metadata": metadata,
            "call_predicted_tools": True, "example_count": len(samples),
            "config": {"layers": list(layers), "seeds": list(seeds), "sample_ids": list(samples),
                       "method": "noise", "target": "qkv", "strength": 0.01},
        }
        records = [pair(layer, seed, sample) for layer in layers for seed in seeds for sample in samples]
        for record in records:
            if record["sample_id"] == "failure":
                record["control"] = condition("failure", False, "wrong_tool")
                record["intervention"] = condition("failure", True)
            else:
                record["intervention"] = condition(record["sample_id"], False, "wrong_tool")
        self.store(path, manifest, records)
        (path / "RUN_COMPLETE").touch()
        return path

    @staticmethod
    def store(path, manifest, records):
        (path / "run_config.json").write_text(json.dumps(manifest), encoding="utf-8")
        (path / "paired_records.jsonl").write_text(
            "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")

    @staticmethod
    def read(path):
        return (json.loads((path / "run_config.json").read_text()),
                [json.loads(line) for line in (path / "paired_records.jsonl").read_text().splitlines()])

    def test_merge_shards_deduplicates_controls_and_reports_repairs_damage(self):
        first = self.write_run("first", seeds=(1234, 5678))
        second = self.write_run("second", layers=(12,))
        before = {path: path.read_bytes() for directory in (first, second) for path in directory.iterdir()}
        result = combine_completed_runs([first, second])
        self.assertEqual(result["unique_sample_count"], 2)
        self.assertEqual(len(result["unique_controls"]), 2)
        self.assertEqual(result["paired_observation_count"], 6)
        self.assertEqual(len(result["rows"]), 3)
        for row in result["rows"]:
            self.assertEqual(row["sample_count"], 2)
            self.assertEqual(row["tool_choice_changes"], 2)
            self.assertEqual(row["tool_choice_repairs"], 1)
            self.assertEqual(row["tool_choice_damage"], 1)
            self.assertEqual(row["final_outcome_repairs"], 1)
            self.assertEqual(row["final_outcome_damage"], 1)
            self.assertEqual(row["tool_choice_repair_rate"], 1.0)
            self.assertEqual(row["benchmark_classification"], "controlled")
        save_summary(result, self.root / "report.json")
        self.assertEqual(before, {path: path.read_bytes() for path in before})

    def test_separates_invalid_valid_no_call_and_valid_wrong(self):
        path = self.write_run(samples=("invalid", "no_call", "wrong"))
        manifest, records = self.read(path)
        records[0]["intervention"] = condition("invalid", False, None, "parse_error")
        records[1]["intervention"] = condition("no_call", False, None)
        records[2]["intervention"] = condition("wrong", False, "wrong_tool")
        records[0]["intervention"]["final_outcome_correct"] = None
        self.store(path, manifest, records)
        row = combine_completed_runs([path])["rows"][0]
        self.assertEqual(row["intervention_invalid_outputs"], 1)
        self.assertEqual(row["intervention_valid_no_call_outputs"], 1)
        self.assertEqual(row["intervention_valid_wrong_tool_calls"], 1)
        self.assertEqual(row["intervention_no_call_outputs"], 2)
        self.assertEqual(row["intervention_final_outcome_scored"], 2)
        self.assertEqual(row["final_outcome_paired_scored"], 2)
        self.assertEqual(row["final_outcome_damage"], 2)

    def test_missing_or_conflicting_completion_markers(self):
        path = self.write_run()
        (path / "RUN_COMPLETE").unlink()
        with self.assertRaisesRegex(ValueError, "RUN_COMPLETE"):
            combine_completed_runs([path])
        (path / "RUN_COMPLETE").touch()
        for marker in ("RUN_IN_PROGRESS", "RUN_FAILED.json"):
            (path / marker).touch()
            with self.assertRaisesRegex(ValueError, "conflicting"):
                combine_completed_runs([path])
            (path / marker).unlink()

    def test_incompatible_manifest_settings(self):
        first = self.write_run("first")
        second = self.write_run("second", layers=(12,))
        manifest, records = self.read(second)
        for field, value in (("checkpoint_path", "/other/checkpoint"),
                             ("source_run_directory", "/other/source"),
                             ("target", "out"), ("method", "replace"), ("strength", 0.003),
                             ("tool_registry_fingerprint", "other"), ("generation", {"temperature": 0.7})):
            with self.subTest(field=field):
                changed = deepcopy(manifest)
                if field in ("target", "method", "strength"):
                    changed["config"][field] = value
                elif field in ("tool_registry_fingerprint", "generation"):
                    changed["source_run_metadata"][field] = value
                else:
                    changed[field] = value
                self.store(second, changed, records)
                with self.assertRaisesRegex(ValueError, "incompatible"):
                    combine_completed_runs([first, second])

    def test_incompatible_sample_panel(self):
        first = self.write_run("first")
        second = self.write_run("second", layers=(12,), samples=("another",))
        with self.assertRaisesRegex(ValueError, "incompatible"):
            combine_completed_runs([first, second])

    def test_duplicates_within_and_between_runs(self):
        first = self.write_run("first")
        second = self.write_run("second")
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            combine_completed_runs([first, second])
        manifest, records = self.read(first)
        self.store(first, manifest, records + [records[0]])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            combine_completed_runs([first])

    def test_requires_literal_exact_restoration(self):
        path = self.write_run()
        manifest, records = self.read(path)
        for value in (False, None, "True", 1):
            with self.subTest(value=value):
                changed = deepcopy(records)
                changed[0]["intervention_verification"]["restored_exactly"] = value
                self.store(path, manifest, changed)
                with self.assertRaisesRegex(ValueError, "exact weight restoration"):
                    combine_completed_runs([path])
        changed[0].pop("intervention_verification")
        self.store(path, manifest, changed)
        with self.assertRaisesRegex(ValueError, "exact weight restoration"):
            combine_completed_runs([path])

    def test_incomplete_grid_despite_marker(self):
        path = self.write_run(seeds=(1234, 5678))
        manifest, records = self.read(path)
        self.store(path, manifest, records[:-1])
        with self.assertRaisesRegex(ValueError, "incomplete"):
            combine_completed_runs([path])

    def test_repeated_control_drift_rejected_but_latency_ignored(self):
        first = self.write_run("first")
        second = self.write_run("second", layers=(12,))
        manifest, records = self.read(second)
        records[0]["control"]["latency_seconds"] = 9.0
        self.store(second, manifest, records)
        combine_completed_runs([first, second])
        records[0]["control"]["raw_model_output"] = "different generation"
        self.store(second, manifest, records)
        with self.assertRaisesRegex(ValueError, "controls disagree"):
            combine_completed_runs([first, second])

    def test_changed_sample_definition_and_registry(self):
        path = self.write_run()
        manifest, records = self.read(path)
        for field, value in (("expected_args", {"expression": "changed"}),
                             ("tool_registry_fingerprint", "other")):
            changed = deepcopy(records)
            changed[0]["intervention"][field] = value
            self.store(path, manifest, changed)
            with self.assertRaisesRegex(ValueError, "provenance differs"):
                combine_completed_runs([path])

    def test_missing_registry_metadata_rejected(self):
        path = self.write_run()
        manifest, records = self.read(path)
        for field in ("tool_pool", "tool_count", "tool_registry_fingerprint", "tool_registry_fingerprint_version"):
            with self.subTest(field=field):
                changed = deepcopy(manifest)
                changed["source_run_metadata"].pop(field)
                self.store(path, changed, records)
                with self.assertRaisesRegex(ValueError, "missing source registry metadata"):
                    combine_completed_runs([path])

    def test_resolved_config_and_implicit_panel_supported(self):
        path = self.write_run()
        manifest, records = self.read(path)
        manifest["resolved_config"] = deepcopy(manifest["config"])
        manifest["resolved_config"]["sample_ids"] = None
        manifest["requested_config"] = {"layers": "all"}
        self.store(path, manifest, records)
        self.assertEqual(combine_completed_runs([path])["unique_sample_count"], 2)

    def test_changed_control_or_bad_saved_flags_rejected(self):
        path = self.write_run()
        manifest, records = self.read(path)
        changed = deepcopy(records)
        changed[0]["control_verification"]["changed_attention_parameter_names"] = ["qkv.weight"]
        self.store(path, manifest, changed)
        with self.assertRaisesRegex(ValueError, "control was not unchanged"):
            combine_completed_runs([path])
        for field, value in (("invalid_output", True), ("tool_selection_correct", "true"),
                             ("tool_choice_correct", False)):
            changed = deepcopy(records)
            changed[0]["control"][field] = value
            self.store(path, manifest, changed)
            with self.assertRaises(ValueError):
                combine_completed_runs([path])

    def test_mixed_benchmark_classes_reported_separately(self):
        path = self.write_run()
        manifest, records = self.read(path)
        for name in ("control", "intervention"):
            records[1][name].update(source="public_derived", benchmark_path="benchmark/math/math_public.json")
        self.store(path, manifest, records)
        result = combine_completed_runs([path])
        self.assertEqual({row["benchmark_classification"] for row in result["rows"]},
                         {"controlled", "public/source-derived"})
        self.assertTrue(all(row["sample_count"] == 1 for row in result["rows"]))

    def test_output_protection_and_cli(self):
        path = self.write_run()
        result = combine_completed_runs([path])
        with self.assertRaisesRegex(ValueError, "outside"):
            save_summary(result, path / "combined.json")
        output = self.root / "report.json"
        self.assertEqual(main(["--run-dir", str(path), "--output", str(output)]), 0)
        with self.assertRaises(FileExistsError):
            save_summary(result, output)
        self.assertEqual(json.loads(output.read_text())["unique_sample_count"], 2)


if __name__ == "__main__":
    unittest.main()
