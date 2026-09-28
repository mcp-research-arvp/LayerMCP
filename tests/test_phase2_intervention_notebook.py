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
        self.assertIn("MODEL_CHOICE = \"tiny_cpu\"", source)
        self.assertIn("LOAD_CHECKPOINT = False", source)
        self.assertNotIn("from_pretrained", source)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is required to execute notebook cells")
    def test_default_cpu_cells_execute_without_a_checkpoint(self) -> None:
        notebook = json.loads(NOTEBOOK.read_text())
        namespace = {"__name__": "__notebook_test__"}
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                exec("".join(cell["source"]), namespace)
        self.assertEqual(namespace["results"]["disabled"], "no-op; exact values retained")
        self.assertTrue(namespace["results"]["noise"]["restored_after_context"])
        self.assertTrue(namespace["results"]["replace"]["restored_after_context"])


if __name__ == "__main__":
    unittest.main()
