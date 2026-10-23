from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

from models.routers import gpt_oss_local_router as router
from research.phase2.gpt_oss_multistep_intervention_eval import (
    MultistepConfig,
    _paired_results,
    run_evaluation,
)


VALID_CALL = (
    '<|channel|>commentary to=functions.calculator <|constrain|>json'
    '<|message|>{"expression":"2+2"}<|call|>'
)


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        block = nn.Module()
        block.attn = nn.Module()
        block.attn.qkv = nn.Linear(2, 6)
        block.attn.o_proj = nn.Linear(2, 2)
        self.block = nn.ModuleList([block])


class FakeGenerator:
    def __init__(self, model: TinyModel, *, fail_intervention: bool = False) -> None:
        self.model = model
        self.original = model.block[0].attn.qkv.weight.detach().clone()
        self.prompts: list[tuple[bool, str]] = []
        self.fail_intervention = fail_intervention
        self.assistant_action_stop_tokens = [7]

    def render_tool_prompt(self, query, native_tools, *, reasoning_effort):
        return query

    def generate_text(self, *, prompt_tokens, stop_tokens, temperature, max_tokens):
        active = not torch.equal(self.model.block[0].attn.qkv.weight, self.original)
        self.prompts.append((active, prompt_tokens))
        if active and self.fail_intervention:
            raise RuntimeError("injected generation failure")
        return SimpleNamespace(text="invalid model output" if active else VALID_CALL)


@asynccontextmanager
async def fake_server(_server_path):
    class Session:
        async def list_tools(self):
            return SimpleNamespace(tools=[SimpleNamespace(
                name="calculator",
                inputSchema={"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]},
                description="Evaluate an expression.",
            )])

        async def call_tool(self, tool_name, arguments):
            return SimpleNamespace(structuredContent={"result": 4}, content=[], isError=False)

    yield Session()


def write_benchmark(path: Path) -> None:
    path.write_text(json.dumps([{
        "id": "workflow-1",
        "domain": "math",
        "task_type": "multi_step_tool_routing",
        "source": "controlled_test",
        "query": "Compute two steps",
        "expected_steps": [
            {"id": "step-1", "query": "Calculate 2+2", "expected_tool": "calculator", "expected_args": {"expression": "2+2"}, "expected_answer": {"result": 4}},
            {"id": "step-2", "query": "Calculate 2+2 again", "depends_on": ["step-1"], "expected_tool": "calculator", "expected_args": {"expression": "2+2"}, "expected_answer": {"result": 4}},
        ],
    }]), encoding="utf-8")


class MultistepInterventionTests(unittest.TestCase):
    def _run(self, root: Path, generator: FakeGenerator) -> Path:
        benchmark = root / "benchmark.json"
        write_benchmark(benchmark)
        checkpoint = root / "checkpoint"
        checkpoint.mkdir()
        with (
            patch.dict("research.phase2.gpt_oss_multistep_intervention_eval.DATASET_GROUPS", {"test": benchmark}, clear=True),
            patch("evaluation.evaluate._run_server_session", fake_server),
            patch("models.routers.registry.load_router", return_value=router),
        ):
            return run_evaluation(
                dataset_group="test", checkpoint_dir=checkpoint,
                output_dir=root / "output",
                config=MultistepConfig(layer=0, target="qkv", method="noise", strength=0.25, seed=17),
                generator_loader=lambda _: generator,
            )

    def test_paired_rollouts_use_separate_predicted_histories_and_restore(self):
        torch.manual_seed(3)
        model = TinyModel()
        original = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
        generator = FakeGenerator(model)
        with TemporaryDirectory() as directory:
            output = self._run(Path(directory), generator)
            self.assertTrue((output / "RUN_COMPLETE").is_file())
            self.assertFalse((output / "RUN_IN_PROGRESS").exists())
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["paired_workflow_count"], 1)
            self.assertEqual(summary["paired_step_count"], 2)
            self.assertEqual(summary["control"]["tool_selection_accuracy"], 1.0)
            self.assertEqual(summary["intervention"]["tool_selection_accuracy"], 0.0)
            self.assertEqual(summary["metric_transitions"]["tool_selection"], {"correct_wrong": 2})
            self.assertEqual(summary["changed_routed_query_steps"], 1)
            pairs = [json.loads(line) for line in (output / "paired_records.jsonl").read_text().splitlines()]
            self.assertEqual(len(pairs), 1)
            self.assertEqual([step["routed_query_changed"] for step in pairs[0]["steps"]], [False, True])
            self.assertTrue(all(step["intervention_verification"]["restored_exactly"] for step in pairs[0]["steps"]))
            self.assertEqual([active for active, _ in generator.prompts], [False, False, True, True])
            config = json.loads((output / "run_config.json").read_text())
            self.assertEqual(config["raw_benchmark_mode_counts"], {"<absent>": 1})
            self.assertEqual(config["effective_benchmark_mode_counts"], {"grounded_tool_execution": 1})
        for name, parameter in model.named_parameters():
            self.assertTrue(torch.equal(parameter, original[name]), name)

    def test_interrupted_intervention_leaves_failure_marker_and_restores(self):
        model = TinyModel()
        original = model.block[0].attn.qkv.weight.detach().clone()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(RuntimeError, "injected generation failure"):
                self._run(root, FakeGenerator(model, fail_intervention=True))
            self.assertTrue((root / "output" / "RUN_FAILED.json").is_file())
            self.assertFalse((root / "output" / "RUN_COMPLETE").exists())
        self.assertTrue(torch.equal(model.block[0].attn.qkv.weight, original))

    def test_rejects_mismatched_pairs_and_invalid_configuration(self):
        with self.assertRaisesRegex(ValueError, "counts differ"):
            _paired_results([{"steps": []}], [], [], [])
        for strength in (-1.0, float("nan"), float("inf")):
            with self.assertRaisesRegex(ValueError, "finite non-negative"):
                MultistepConfig(0, "qkv", "noise", strength, 17)


if __name__ == "__main__":
    unittest.main()
