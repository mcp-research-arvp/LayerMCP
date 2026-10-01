from __future__ import annotations

import unittest

from research.phase2.layer_screen import (
    LayerScreenPlanError,
    plan_layer_screen,
    representative_layers,
    resolve_layer_request,
)


class LayerScreenPlannerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.layers = (0, 1, 2, 3, 4)
        self.targets = {layer: ("attention_all", "qkv", "out") for layer in self.layers}

    def test_representative_layers_uses_discovered_order(self) -> None:
        self.assertEqual(representative_layers((4, 8, 15, 16, 23)), (4, 15, 23))
        self.assertEqual(representative_layers((9,)), (9,))

    def test_all_and_representative_requests_never_assume_model_size(self) -> None:
        self.assertEqual(resolve_layer_request("all", self.layers), self.layers)
        self.assertEqual(resolve_layer_request("representative", self.layers), (0, 2, 4))

    def test_plan_reports_complete_pair_and_generation_counts(self) -> None:
        plan = plan_layer_screen(
            layer_request="representative",
            available_layers=self.layers,
            available_targets_by_layer=self.targets,
            sample_ids=("good", "failure"),
            seeds=(1234, 5678, 9012),
            target="attention_all",
            method="noise",
            strength=0.01,
        )
        self.assertEqual(plan.selected_layers, (0, 2, 4))
        self.assertEqual(plan.pair_count_per_layer, 6)
        self.assertEqual(plan.total_pair_count, 18)
        self.assertEqual(plan.total_generation_count, 36)

    def test_plan_rejects_target_unavailable_on_one_selected_layer(self) -> None:
        targets = dict(self.targets)
        targets[2] = ("attention_all", "out")
        with self.assertRaisesRegex(LayerScreenPlanError, "not safe"):
            plan_layer_screen(
                layer_request=(0, 2, 4),
                available_layers=self.layers,
                available_targets_by_layer=targets,
                sample_ids=("good",),
                seeds=(1234,),
                target="qkv",
                method="noise",
                strength=0.01,
            )
