from __future__ import annotations

import json
import importlib.util
import ast
import contextlib
import io
from pathlib import Path
import unittest


NOTEBOOK = (
    Path(__file__).resolve().parents[1]
    / "research"
    / "phase2"
    / "attention_intervention_demo.ipynb"
)


class Phase2InterventionNotebookTests(unittest.TestCase):
    def test_notebook_has_portable_defaults_and_no_saved_outputs(self) -> None:
        notebook = json.loads(NOTEBOOK.read_text())
        code_cells = [cell for cell in notebook['cells'] if cell['cell_type'] == 'code']
        for cell in code_cells:
            self.assertEqual(cell['outputs'], [])
            self.assertIsNone(cell['execution_count'])
        settings = next(''.join(cell['source']) for cell in code_cells
                        if ''.join(cell['source']).startswith('# Edit this cell'))
        expected = {
            'MODEL_CHOICE': 'tiny_cpu', 'LOAD_CHECKPOINT': False,
            'ATTENTION_TARGET': 'attention_all', 'RUN_REAL_INTERVENTION_PROBE': False,
            'RUN_PAIRED_SAMPLE_COMPARISON': False, 'RUN_PAIRED_SWEEP': False,
            'SWEEP_SEEDS': (1234,), 'SWEEP_MAX_INTERVENTIONS': 24,
            'SWEEP_SHOW_RAW_OUTPUTS': False,
        }
        actual = {statement.targets[0].id: ast.literal_eval(statement.value)
                  for statement in ast.parse(settings).body
                  if isinstance(statement, ast.Assign)
                  and isinstance(statement.targets[0], ast.Name)
                  and statement.targets[0].id in expected}
        self.assertEqual(actual, expected)
        self.assertNotIn('phase2-review-smoke-evidence', NOTEBOOK.read_text())

    def test_measurement_display_is_safe_before_any_sweep(self) -> None:
        notebook = json.loads(NOTEBOOK.read_text())
        source = next(''.join(cell['source']) for cell in notebook['cells']
                      if ''.join(cell['source']).startswith('# Display measurements already collected'))
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            exec(source, {})
        self.assertIn('No sweep records in this kernel yet', output.getvalue())

    def test_measurement_display_reads_existing_records_without_generation(self) -> None:
        notebook = json.loads(NOTEBOOK.read_text())
        source = next(''.join(cell['source']) for cell in notebook['cells']
                      if ''.join(cell['source']).startswith('# Display measurements already collected'))
        displays = []
        namespace = {
            'sweep_records': [{'sample_id': 'example', 'layer_index': 0, 'target': 'qkv',
                'method': 'noise', 'strength': 0.01, 'seed': 7, 'intervention_verification': {
                'restored_exactly': True,
                'perturbation': {'tensors': [{'name': 'qkv.weight', 'changed_elements': 2,
                    'original_rms': 0.0, 'effective_delta_rms': 0.01,
                    'relative_delta_l2': None}]},
            }}],
            'print_table': lambda title, rows: displays.append((title, rows)),
        }
        exec(source, namespace)
        self.assertEqual(displays, [('Measured weight changes', [{
            'sample_id': 'example', 'layer': 0, 'target': 'qkv', 'method': 'noise', 'strength': 0.01,
            'seed': 7, 'tensor': 'qkv.weight', 'changed_values': 2,
            'original_RMS': 0.0, 'change_RMS': 0.01, 'change/original': None,
            'restored': True,
        }])])

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
        self.assertIn("'actual_perturbation': active_probe.perturbation", source)
        self.assertIn("'changed_values': verification['perturbation']['changed_elements']", source)
        self.assertIn("paired_diagnostics(sweep_records)", source)
        self.assertIn("print('Loaded input provenance:'", source)
        self.assertIn("'checkpoint_fingerprint', 'model_binding', 'source_checkpoint_equivalence'", source)
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
