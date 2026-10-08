from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import torch

from research.phase2.experiment_integrity import (
    checkpoint_identity, verify_identity_inputs, compare_saved_control,
    paired_diagnostics, runtime_identity, gpu_driver_metadata,
)


class ExperimentIntegrityTests(unittest.TestCase):
    def test_added_or_removed_weight_shards_invalidate_fingerprint(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "model.safetensors").write_bytes(b"weights")
            identity = checkpoint_identity(root)
            (root / "extra.safetensors").write_bytes(b"extra")
            with self.assertRaisesRegex(RuntimeError, "weight-file set changed"):
                verify_identity_inputs(identity)
            (root / "extra.safetensors").unlink()
            (root / "model.safetensors").unlink()
            with self.assertRaisesRegex(RuntimeError, "weight-file set changed"):
                verify_identity_inputs(identity)

    def test_driver_metadata_and_unavailable_cases(self):
        with patch("research.phase2.experiment_integrity.subprocess.run", return_value=SimpleNamespace(returncode=0, stdout="570.1\n570.1\n")):
            self.assertEqual(gpu_driver_metadata()["versions"], ["570.1"])
        for error in (FileNotFoundError(), subprocess.TimeoutExpired("nvidia-smi", 5)):
            with patch("research.phase2.experiment_integrity.subprocess.run", side_effect=error):
                self.assertEqual(gpu_driver_metadata()["status"], "unavailable")

    def test_content_fingerprints_and_input_change_detection(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "model.safetensors").write_bytes(b"weights")
            (root / "config.json").write_text("{}")
            (root / "o200k_base.tiktoken").write_bytes(b"tokenizer")
            with patch.dict(os.environ, {"TIKTOKEN_ENCODINGS_BASE": str(root)}):
                first = checkpoint_identity(root)
                self.assertEqual(first["status"], "complete")
                verify_identity_inputs(first)
                for filename in ("config.json", "model.safetensors", "o200k_base.tiktoken"):
                    original = (root / filename).read_bytes()
                    (root / filename).write_bytes(original + b" changed")
                    second = checkpoint_identity(root)
                    self.assertNotEqual(first["fingerprint"], second["fingerprint"])
                    with self.assertRaisesRegex(RuntimeError, "changed during evaluation"):
                        verify_identity_inputs(first)
                    (root / filename).write_bytes(original)
                    first = checkpoint_identity(root)

    def test_missing_assets_are_explicitly_unverifiable(self):
        with TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            identity = checkpoint_identity(Path(directory))
        self.assertEqual(identity["status"], "incomplete")
        self.assertIsNone(identity["fingerprint"])

    def test_saved_control_comparison_does_not_claim_runtime_equivalence(self):
        saved = {"raw_model_output": "answer", "selected_tool": "calculator"}
        self.assertEqual(compare_saved_control(saved, saved)["status"], "matches")
        changed = {**saved, "raw_model_output": "different answer"}
        self.assertEqual(compare_saved_control(saved, changed)["status"], "differs_new_control_baseline")
        self.assertEqual(compare_saved_control({}, saved)["status"], "unavailable")
        self.assertEqual(compare_saved_control(saved, saved)["historical_runtime_equivalence"], "not_established_by_output_matching")

    def test_transitions_validity_and_control_drift_have_distinct_denominators(self):
        def outcome(correct, *, invalid=False, no_call=False, raw="call", tool=None):
            return {"tool_choice_correct": correct, "invalid_output": invalid,
                    "no_call_outcome": no_call, "chosen_tool": tool or ("correct" if correct else "wrong"),
                    "raw_model_output": raw, "selected_tool": tool,
                    "selected_args": {}, "parse_status": "bad" if invalid else "ok"}
        pairs = [
            {"sample_id": "a", "control": outcome(True), "intervention": outcome(True)},
            {"sample_id": "a", "control": outcome(True), "intervention": outcome(False, invalid=True, no_call=True)},
            {"sample_id": "b", "control": outcome(False), "intervention": outcome(True)},
            {"sample_id": "b", "control": outcome(False, raw="drift"), "intervention": outcome(False, no_call=True)},
            {"sample_id": "c", "control": outcome(False), "intervention": outcome(False)},
        ]
        result = paired_diagnostics(pairs)
        self.assertEqual(result["tool_choice_transitions"], {"correct_correct": 1, "correct_wrong": 1, "wrong_correct": 1, "wrong_wrong": 2})
        self.assertEqual(result["unique_control_examples"], 3)
        self.assertEqual(result["paired_observations"], 5)
        for metric in ("argument_match", "execution_success", "final_outcome"):
            self.assertEqual(result["metric_transitions"][metric]["scored_pairs"], 0)
            self.assertEqual(result["metric_transitions"][metric]["unscored_pairs"], 5)
        self.assertEqual(result["control_drift_sample_ids"], ["b"])
        self.assertEqual(result["control_repeatability_status"], "drift_detected")
        self.assertEqual(result["control_observation_counts"], {"a": 2, "b": 2, "c": 1})
        self.assertEqual(paired_diagnostics(pairs[:1])["control_repeatability_status"], "not_tested")
        self.assertEqual(result["output_categories"]["intervention"], {
            "invalid_output": 1, "valid_no_call": 1, "valid_wrong_tool": 1, "valid_correct_tool": 2,
        })

    def test_runtime_is_json_safe_and_contains_current_settings(self):
        identity = runtime_identity(torch.nn.Linear(2, 2))
        json.dumps(identity, allow_nan=False)
        self.assertIn("git_commit", identity)
        self.assertEqual(identity["devices"], ["cpu"])
        self.assertIn("deterministic_algorithms", identity)
        self.assertEqual(identity["generation"]["temperature"], 0.0)

    def test_each_metric_has_independent_paired_transitions(self):
        def outcome(tool, args, execution, final):
            return {"tool_choice_correct": tool, "argument_match_correct": args,
                    "execution_success": execution, "final_outcome_correct": final,
                    "chosen_tool": "calculator", "invalid_output": False,
                    "no_call_outcome": False}
        control = outcome(True, False, True, True)
        intervention = outcome(True, True, False, False)
        pairs = [{"sample_id": "a", "control": control, "intervention": intervention}]
        result = paired_diagnostics(pairs)
        expected = {"tool_choice": "correct_correct", "argument_match": "wrong_correct",
                    "execution_success": "correct_wrong", "final_outcome": "correct_wrong"}
        for metric, transition in expected.items():
            self.assertEqual(result["metric_transitions"][metric], {
                "transitions": {key: int(key == transition) for key in (
                    "correct_correct", "correct_wrong", "wrong_correct", "wrong_wrong",
                )},
                "scored_pairs": 1, "unscored_pairs": 0,
            })
        self.assertEqual(result["repairs"], 0)
        self.assertEqual(result["damage"], 0)

    def test_all_four_transitions_and_unscored_outcomes_are_counted_separately(self):
        pairs = []
        values = ((True, True), (True, False), (False, True), (False, False),
                  (None, True), (False, None), (None, None))
        for index, (control, intervention) in enumerate(values):
            def outcome(value):
                return {"tool_choice_correct": True, "argument_match_correct": value,
                        "execution_success": value, "final_outcome_correct": value,
                        "chosen_tool": "calculator", "invalid_output": False,
                        "no_call_outcome": False}
            pairs.append({"sample_id": str(index), "control": outcome(control),
                          "intervention": outcome(intervention)})
        result = paired_diagnostics(pairs)
        for metric in ("argument_match", "execution_success", "final_outcome"):
            self.assertEqual(result["metric_transitions"][metric], {
                "transitions": {"correct_correct": 1, "correct_wrong": 1,
                                "wrong_correct": 1, "wrong_wrong": 1},
                "scored_pairs": 4, "unscored_pairs": 3,
            })
        self.assertEqual(result["metric_transitions"]["tool_choice"]["scored_pairs"], 7)
        self.assertEqual(result["unique_control_examples"], 7)
        self.assertEqual(result["paired_observations"], 7)

    def test_empty_diagnostics_have_zero_transitions_and_denominators(self):
        result = paired_diagnostics([])
        for metric in result["metric_transitions"].values():
            self.assertEqual(sum(metric["transitions"].values()), 0)
            self.assertEqual(metric["scored_pairs"], 0)
            self.assertEqual(metric["unscored_pairs"], 0)


if __name__ == "__main__":
    unittest.main()
