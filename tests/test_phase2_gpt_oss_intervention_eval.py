from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import torch
from torch import nn

from evaluation.evaluate import _tool_pool_metadata
from research.phase2.gpt_oss_intervention_eval import (
    EvaluationConfig,
    run_evaluation,
)
from research.phase2.replay import ToolCatalog


VALID_CALL = (
    '<|channel|>commentary to=functions.calculator <|constrain|>json'
    '<|message|>{"expression":"2+2"}<|call|>'
)


class TinyGptBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn = nn.Linear(3, 3)
        self.mlp = nn.Linear(3, 3)


class TinyGptOssModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block = nn.ModuleList([TinyGptBlock(), TinyGptBlock()])


class TinyTargetedGptBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn = nn.Module()
        self.attn.q_proj = nn.Linear(3, 3)
        self.attn.k_proj = nn.Linear(3, 3)
        self.attn.v_proj = nn.Linear(3, 3)
        self.attn.o_proj = nn.Linear(3, 3)
        self.mlp = nn.Linear(3, 3)


class TinyTargetedGptOssModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block = nn.ModuleList([TinyTargetedGptBlock(), TinyTargetedGptBlock()])


class MockGenerator:
    def __init__(self, model: nn.Module, *, fail_on_active_prompt: str | None = None) -> None:
        self.model = model
        self._attention_parameter = next(model.block[0].attn.parameters())
        self._original_attention = self._attention_parameter.detach().clone()
        self.fail_on_active_prompt = fail_on_active_prompt
        self.assistant_action_stop_tokens = [7]

    def render_tool_prompt(self, query, native_tools, *, reasoning_effort):
        self.last_prompt = (query, native_tools, reasoning_effort)
        return query

    def generate_text(self, *, prompt_tokens, stop_tokens, temperature, max_tokens):
        active = not torch.equal(self._attention_parameter, self._original_attention)
        if active and self.fail_on_active_prompt and self.fail_on_active_prompt in prompt_tokens:
            raise RuntimeError("intentional interrupted pair")
        return SimpleNamespace(text="not a valid Harmony call" if active else VALID_CALL)


def _catalog() -> ToolCatalog:
    names = ("calculator",)
    schemas = {
        "calculator": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        }
    }
    descriptions = {"calculator": "Evaluate an arithmetic expression."}
    metadata = _tool_pool_metadata(list(names), schemas, descriptions)
    return ToolCatalog(
        names=names,
        schemas=schemas,
        descriptions=descriptions,
        metadata=metadata,
    )


@asynccontextmanager
async def _session_for_catalog(catalog: ToolCatalog):
    tools = [
        SimpleNamespace(
            name=name,
            inputSchema=catalog.schemas[name],
            description=catalog.descriptions[name],
        )
        for name in catalog.names
    ]

    class FakeSession:
        async def list_tools(self):
            return SimpleNamespace(tools=tools)

        async def call_tool(self, name: str, arguments: dict[str, object]):
            return SimpleNamespace(
                structuredContent={"result": 4},
                content=[],
                isError=False,
            )

    yield FakeSession()


def _write_saved_run(root: Path, catalog: ToolCatalog, queries: list[str]) -> Path:
    source = root / "saved_run"
    samples = source / "samples.jsonl"
    samples.parent.mkdir()
    (source / "run_metadata.json").write_text(
        json.dumps(
            {
                "model": "gpt-oss-local",
                "expected_model_name": "openai/gpt-oss-20b",
                "prompt_template_id": "harmony_structured_context_sql_v2",
                "reasoning_mode": "reasoning",
                "reasoning_effort": "low",
            }
        ),
        encoding="utf-8",
    )
    (source / "artifact_index.jsonl").write_text(
        json.dumps({"samples_path": "samples.jsonl"}) + "\n", encoding="utf-8"
    )
    rows = []
    for index, query in enumerate(queries):
        rows.append(
            {
                "sample_id": f"sample-{index}",
                "query": query,
                "prompt_context": "",
                "benchmark_path": "benchmark/math/test.json",
                "domain": "mathematics",
                "task_type": "single_tool_routing",
                "difficulty": "easy",
                "source": "test",
                "expected_tool": "calculator",
                "expected_args": {"expression": "2+2"},
                "expected_answer": {"result": 4},
                "benchmark_mode": "grounded_tool_execution",
                "tool_names": ["calculator"],
                "model_name": "openai/gpt-oss-20b",
                "router_backend": "local_gpt_oss_pytorch",
                "prompt_template": "harmony_structured_context_sql_v2",
                "reasoning_mode": "reasoning",
                "reasoning_effort": "low",
                **catalog.metadata,
            }
        )
    samples.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return source


class Phase2GptOssInterventionEvaluationTests(unittest.TestCase):
    def _run(
        self,
        directory: Path,
        generator: MockGenerator,
        queries: list[str],
        *,
        method: str = "noise",
        target: str = "attention_all",
        sample_ids: tuple[str, ...] | None = None,
    ) -> Path:
        catalog = _catalog()
        source = _write_saved_run(directory, catalog, queries)
        checkpoint = directory / "checkpoint"
        checkpoint.mkdir()
        return run_evaluation(
            source_run_dir=source,
            checkpoint_dir=checkpoint,
            output_dir=directory / "output",
            config=EvaluationConfig(
                layers=(0,),
                seeds=(17,),
                method=method,
                strength=0.25,
                example_limit=len(sample_ids) if sample_ids is not None else 2,
                target=target,
                sample_ids=sample_ids,
            ),
            generator_loader=lambda _: generator,
            session_factory=lambda: _session_for_catalog(catalog),
        )

    def test_writes_paired_control_and_intervention_records_and_restores_weights(self) -> None:
        torch.manual_seed(6)
        model = TinyGptOssModel()
        before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
        generator = MockGenerator(model)

        with TemporaryDirectory() as temporary:
            output = self._run(Path(temporary), generator, ["first", "second"])
            self.assertTrue((output / "RUN_COMPLETE").is_file())
            self.assertFalse((output / "RUN_IN_PROGRESS").exists())
            records = [
                json.loads(line)
                for line in (output / "paired_records.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(records), 2)
            self.assertTrue(all(record["control"]["tool_choice_correct"] for record in records))
            self.assertTrue(all(not record["control"]["invalid_output"] for record in records))
            self.assertTrue(all(not record["control"]["no_call_outcome"] for record in records))
            self.assertTrue(all(record["intervention"]["invalid_output"] for record in records))
            self.assertTrue(all(record["intervention"]["no_call_outcome"] for record in records))
            self.assertTrue(all(record["control"]["execution_success"] for record in records))
            self.assertTrue(all(record["control"]["final_outcome_correct"] for record in records))
            self.assertTrue(all(not record["intervention"]["execution_success"] for record in records))
            self.assertTrue(all(not record["intervention"]["final_outcome_correct"] for record in records))
            self.assertTrue(all(record["intervention_verification"]["restored_exactly"] for record in records))
            summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["paired_record_count"], 2)
            self.assertEqual(summary["conditions"]["control"]["correct_tool_choices"], 2)
            self.assertEqual(summary["conditions"]["intervention"]["no_call_outcomes"], 2)
            self.assertEqual(summary["conditions"]["control"]["execution_successes"], 2)
            self.assertEqual(summary["conditions"]["control"]["final_outcome_accuracy"], 1.0)

        for name, parameter in model.named_parameters():
            self.assertTrue(torch.equal(parameter, before[name]), name)

    def test_replace_is_recorded_and_restores_weights(self) -> None:
        torch.manual_seed(8)
        model = TinyGptOssModel()
        before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}

        with TemporaryDirectory() as temporary:
            output = self._run(
                Path(temporary), MockGenerator(model), ["one"], method="replace"
            )
            record = json.loads((output / "paired_records.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(record["method"], "replace")
            self.assertTrue(record["intervention_verification"]["restored_exactly"])

        for name, parameter in model.named_parameters():
            self.assertTrue(torch.equal(parameter, before[name]), name)

    def test_explicit_sample_ids_preserve_requested_order(self) -> None:
        model = TinyGptOssModel()
        with TemporaryDirectory() as temporary:
            output = self._run(
                Path(temporary),
                MockGenerator(model),
                ["first", "second"],
                sample_ids=("sample-1", "sample-0"),
            )
            records = [
                json.loads(line)
                for line in (output / "paired_records.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual([record["sample_id"] for record in records], ["sample-1", "sample-0"])

    def test_unknown_explicit_sample_id_fails_before_output_is_created(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(ValueError, "no requested sample IDs"):
                self._run(
                    root,
                    MockGenerator(TinyGptOssModel()),
                    ["first"],
                    sample_ids=("missing-sample",),
                )
            self.assertFalse((root / "output").exists())

    def test_projection_target_is_recorded_and_changes_only_that_projection(self) -> None:
        torch.manual_seed(9)
        model = TinyTargetedGptOssModel()
        before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
        with TemporaryDirectory() as temporary:
            output = self._run(
                Path(temporary), MockGenerator(model), ["one"], target="q_proj"
            )
            record = json.loads((output / "paired_records.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(record["target"], "q_proj")
            self.assertEqual(
                set(record["intervention_verification"]["changed_attention_parameter_names"]),
                {"q_proj.weight", "q_proj.bias"},
            )
        for name, parameter in model.named_parameters():
            self.assertTrue(torch.equal(parameter, before[name]), name)

    def test_interrupted_pair_never_marks_partial_output_complete(self) -> None:
        torch.manual_seed(7)
        model = TinyGptOssModel()
        before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
        generator = MockGenerator(model, fail_on_active_prompt="second")

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(RuntimeError, "intentional interrupted pair"):
                self._run(root, generator, ["first", "second"])
            output = root / "output"
            self.assertFalse((output / "RUN_COMPLETE").exists())
            self.assertTrue((output / "RUN_IN_PROGRESS").is_file())
            failure = json.loads((output / "RUN_FAILED.json").read_text(encoding="utf-8"))
            self.assertEqual(failure["completed_pair_count"], 1)
            self.assertFalse((output / "summary.json").exists())

        for name, parameter in model.named_parameters():
            self.assertTrue(torch.equal(parameter, before[name]), name)


if __name__ == "__main__":
    unittest.main()
