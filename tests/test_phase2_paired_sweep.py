from __future__ import annotations

import unittest

from research.phase2.paired_sweep import (
    PairedSweepPlanError,
    plan_paired_sweep,
)


class PairedSweepPlanTests(unittest.TestCase):
    def setUp(self) -> None:
        self.available_layers = (0, 1)
        self.available_targets = {
            0: ("attention_all", "qkv", "out"),
            1: ("attention_all", "out"),
        }

    def test_expands_a_small_explicit_grid(self) -> None:
        settings = plan_paired_sweep(
            sample_ids=("success", "failure"),
            layers=(0,),
            targets=("qkv",),
            methods=("noise",),
            strengths=(0.001, 0.01),
            seeds=(1234, 5678),
            available_layers=self.available_layers,
            available_targets_by_layer=self.available_targets,
            max_interventions=10,
        )
        self.assertEqual(len(settings), 8)
        self.assertEqual(settings[0].sample_id, "success")
        self.assertEqual(settings[0].target, "qkv")
        self.assertEqual(settings[-1].seed, 5678)

    def test_all_expands_only_targets_safe_at_each_layer(self) -> None:
        settings = plan_paired_sweep(
            sample_ids=("sample",),
            layers="all",
            targets="all",
            methods=("noise",),
            strengths=(0.01,),
            seeds=(1,),
            available_layers=self.available_layers,
            available_targets_by_layer=self.available_targets,
            max_interventions=5,
        )
        self.assertEqual(
            [(item.layer_index, item.target) for item in settings],
            [(0, "attention_all"), (0, "qkv"), (0, "out"), (1, "attention_all"), (1, "out")],
        )

    def test_large_sweep_requires_explicit_override(self) -> None:
        kwargs = dict(
            sample_ids=("sample",),
            layers="all",
            targets="all",
            methods=("noise", "replace"),
            strengths=(0.01,),
            seeds=(1,),
            available_layers=self.available_layers,
            available_targets_by_layer=self.available_targets,
            max_interventions=5,
        )
        with self.assertRaisesRegex(PairedSweepPlanError, "SWEEP_ALLOW_LARGE=True"):
            plan_paired_sweep(**kwargs)
        self.assertEqual(len(plan_paired_sweep(**kwargs, allow_large=True)), 10)

    def test_rejects_an_unavailable_target_for_a_selected_layer(self) -> None:
        with self.assertRaisesRegex(PairedSweepPlanError, "unavailable at layer 1"):
            plan_paired_sweep(
                sample_ids=("sample",),
                layers=(1,),
                targets=("qkv",),
                methods=("noise",),
                strengths=(0.01,),
                seeds=(1,),
                available_layers=self.available_layers,
                available_targets_by_layer=self.available_targets,
                max_interventions=5,
            )


if __name__ == "__main__":
    unittest.main()
