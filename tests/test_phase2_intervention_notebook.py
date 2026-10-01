from __future__ import annotations

import json
import importlib.util
from pathlib import Path
import unittest


NOTEBOOK = (
    Path(__file__).resolve().parents[1]
    / "research"
    / "phase2"
    / "attention_intervention_demo.ipynb"
)


class Phase2InterventionNotebookTests(unittest.TestCase):
    def test_notebook_is_valid_and_uses_the_public_api(self) -> None:
        notebook = json.loads(NOTEBOOK.read_text())
        self.assertEqual(notebook["nbformat"], 4)
        source = "\n".join(
            "".join(cell["source"])
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        )
        self.assertIn("temporary_attention_intervention", source)
        self.assertIn("available_attention_layers", source)
        self.assertIn("attention_module_for_layer", source)
        self.assertIn("MODEL_CHOICE = 'tiny_cpu'", source)
        self.assertIn("LOAD_CHECKPOINT = False", source)
        self.assertIn("available_attention_targets", source)
        self.assertIn("ATTENTION_TARGET", source)
        self.assertIn("RUN_PAIRED_SAMPLE_COMPARISON = False", source)
        self.assertIn("PAIRED_SAMPLE_ID", source)
        self.assertIn("evaluate_one_saved_example", source)
        self.assertIn("RUN_PAIRED_SWEEP = False", source)
        self.assertIn("plan_paired_sweep", source)
        self.assertIn("SWEEP_MAX_INTERVENTIONS", source)
        self.assertIn("RUN_LAYER_SCREEN_RESULTS_VIEWER = False", source)
        self.assertIn("screen_plan.json", source)
        self.assertIn("RUN_COMPLETE", source)
        self.assertIn("REPO_ROOT", source)
        for choice in (
            "llama31_config_only",
            "gpt_oss_config_only",
            "qwen36_config_only",
            "gemma4_config_only",
            "phi4_config_only",
        ):
            self.assertIn(choice, source)
        self.assertNotIn("from_pretrained", source)
        cell_ids = [cell["id"] for cell in notebook["cells"]]
        self.assertEqual(len(cell_ids), len(set(cell_ids)))

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is required to execute notebook cells")
    def test_default_cpu_cells_execute_without_a_checkpoint(self) -> None:
        notebook = json.loads(NOTEBOOK.read_text())
        namespace = {"__name__": "__notebook_test__"}
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                exec("".join(cell["source"]), namespace)
        self.assertEqual(
            namespace["results"]["disabled"],
            "no-op; identical output and exact values retained",
        )
        self.assertTrue(namespace["results"]["noise"]["restored_after_context"])
        self.assertTrue(namespace["results"]["replace"]["restored_after_context"])
        self.assertFalse(namespace["RUN_PAIRED_SAMPLE_COMPARISON"])
        self.assertFalse(namespace["RUN_PAIRED_SWEEP"])
        self.assertFalse(namespace["RUN_LAYER_SCREEN_RESULTS_VIEWER"])

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is required for architecture views")
    def test_configuration_views_cover_all_repository_families_without_weights(self) -> None:
        notebook = json.loads(NOTEBOOK.read_text())
        namespace = {"__name__": "__notebook_test__"}
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                exec("".join(cell["source"]), namespace)

        views = {
            family: namespace["configuration_view"](family, 0, None)
            for family in ("llama31", "gpt_oss", "qwen36", "gemma4", "phi4")
        }
        self.assertEqual(views["gpt_oss"]["targets"], ("attention_all", "qkv", "out"))
        self.assertEqual(views["phi4"]["targets"], ("attention_all", "o_proj"))
        self.assertEqual(views["qwen36"]["targets"], ("attention_all",))
        self.assertTrue(all(view["layer_ids"] for view in views.values()))


if __name__ == "__main__":
    unittest.main()
