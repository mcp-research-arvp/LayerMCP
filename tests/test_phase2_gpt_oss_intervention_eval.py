from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

from evaluation.evaluate import _tool_pool_metadata
from research.phase2.gpt_oss_intervention_eval import (
    EvaluationConfig,
    RUN_KIND,
    evaluate_one_saved_example,
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
                "raw_model_output": VALID_CALL,
                "selected_tool": "calculator",
                "selected_args": {"expression": "2+2"},
                "parse_status": "ok",
                **catalog.metadata,
            }
        )
    samples.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return source


class Phase2GptOssInterventionEvaluationTests(unittest.TestCase):
    def test_invalid_strength_rejected_when_constructing_config(self):
        for value in (-1, float("nan"), float("inf")):
            with self.assertRaisesRegex(ValueError, "finite non-negative"):
                EvaluationConfig(layers=(0,), seeds=(1,), method="noise", strength=value, example_limit=1)

    def test_fresh_condition_protocol_is_distinct_from_legacy_v1(self):
        self.assertEqual(RUN_KIND, "phase2_gpt_oss_attention_intervention_eval_v2")

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
            self.assertEqual(summary["paired_diagnostics"]["damage"], 2)
            self.assertEqual(summary["paired_diagnostics"]["unique_control_examples"], 2)
            for diagnostics in (
                summary["paired_diagnostics"], summary["by_layer_seed"][0],
            ):
                for metric in ("tool_choice", "argument_match", "execution_success", "final_outcome"):
                    self.assertEqual(diagnostics["metric_transitions"][metric], {
                        "transitions": {"correct_correct": 0, "correct_wrong": 2,
                                        "wrong_correct": 0, "wrong_wrong": 0},
                        "scored_pairs": 2, "unscored_pairs": 0,
                    })
            self.assertEqual(summary["saved_control_comparison_counts"]["matches"], 2)
            self.assertEqual(records[0]["source_checkpoint_equivalence"], "unverifiable")
            self.assertEqual(records[0]["control"]["condition"], "control")
            self.assertGreater(records[0]["intervention_verification"]["perturbation"]["changed_elements"], 0)

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

    def test_one_saved_example_returns_a_visible_complete_pair(self) -> None:
        """The notebook-facing helper reuses the full evaluator without files."""
        torch.manual_seed(10)
        model = TinyGptOssModel()
        before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
        generator = MockGenerator(model)
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            catalog = _catalog()
            source = _write_saved_run(root, catalog, ["one saved query"])
            async def from_notebook_event_loop():
                return evaluate_one_saved_example(
                    generator=generator,
                    source_run_dir=source,
                    sample_id="sample-0",
                    layer_index=0,
                    seed=17,
                    method="noise",
                    target="attention_all",
                    strength=0.25,
                    session_factory=lambda: _session_for_catalog(catalog),
                )

            record = asyncio.run(from_notebook_event_loop())

        self.assertEqual(record["sample_id"], "sample-0")
        self.assertTrue(record["control"]["tool_choice_correct"])
        self.assertTrue(record["control"]["execution_success"])
        self.assertTrue(record["control"]["final_outcome_correct"])
        self.assertTrue(record["intervention"]["invalid_output"])
        self.assertTrue(record["intervention"]["no_call_outcome"])
        self.assertTrue(record["intervention_verification"]["restored_exactly"])
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
            self.assertEqual(failure["failure_stage"], "evaluation")
            self.assertFalse((output / "summary.json").exists())

        for name, parameter in model.named_parameters():
            self.assertTrue(torch.equal(parameter, before[name]), name)

    def test_setup_inspection_and_serialization_failures_write_failure_artifacts(self):
        module = "research.phase2.gpt_oss_intervention_eval"
        cases = (
            ("available_attention_layers", {"side_effect": RuntimeError("layer inspection failed")}, RuntimeError),
            ("available_attention_targets", {"side_effect": RuntimeError("target inspection failed")}, RuntimeError),
            ("asdict", {"return_value": {"not_json_serializable": {1}}}, TypeError),
            ("available_attention_layers", {"side_effect": KeyboardInterrupt("setup interrupted")}, KeyboardInterrupt),
        )
        for function, arguments, error_type in cases:
            with self.subTest(function=function, error=error_type.__name__), TemporaryDirectory() as directory:
                root = Path(directory)
                with patch(f"{module}.{function}", **arguments), self.assertRaises(error_type) as raised:
                    self._run(root, MockGenerator(TinyGptOssModel()), ["one"])
                output = root / "output"
                failure = json.loads((output / "RUN_FAILED.json").read_text())
                self.assertEqual(failure["kind"], RUN_KIND)
                self.assertEqual(failure["error_type"], error_type.__name__)
                self.assertEqual(failure["error"], str(raised.exception))
                self.assertEqual(failure["failure_stage"], "setup")
                self.assertEqual(failure["completed_pair_count"], 0)
                self.assertTrue((output / "RUN_IN_PROGRESS").exists())
                self.assertFalse((output / "RUN_COMPLETE").exists())
                self.assertFalse((output / "paired_records.jsonl").exists())

    def test_setup_io_failures_write_failure_artifacts_when_filesystem_allows(self):
        original_write_text = Path.write_text
        for failed_filename in ("RUN_IN_PROGRESS", "run_config.json"):
            with self.subTest(filename=failed_filename), TemporaryDirectory() as directory:
                root = Path(directory)
                def fail_selected_write(path, *args, **kwargs):
                    if path.name == failed_filename:
                        raise OSError(f"cannot write {failed_filename}")
                    return original_write_text(path, *args, **kwargs)
                with patch.object(Path, "write_text", new=fail_selected_write):
                    with self.assertRaisesRegex(OSError, f"cannot write {failed_filename}"):
                        self._run(root, MockGenerator(TinyGptOssModel()), ["one"])
                output = root / "output"
                failure = json.loads((output / "RUN_FAILED.json").read_text())
                self.assertEqual(failure["error_type"], "OSError")
                self.assertEqual(failure["failure_stage"], "setup")
                self.assertEqual(failure["completed_pair_count"], 0)
                self.assertFalse((output / "RUN_COMPLETE").exists())

    def test_failure_artifact_io_error_does_not_mask_original_error(self):
        original_write_text = Path.write_text
        def fail_output_write(path, *args, **kwargs):
            if path.name in {"run_config.json", "RUN_FAILED.json"}:
                raise OSError(f"cannot write {path.name}")
            return original_write_text(path, *args, **kwargs)
        with TemporaryDirectory() as directory, patch.object(Path, "write_text", new=fail_output_write):
            root = Path(directory)
            with self.assertRaisesRegex(OSError, "cannot write run_config.json") as raised:
                self._run(root, MockGenerator(TinyGptOssModel()), ["one"])
            self.assertIsInstance(raised.exception.__cause__, OSError)
            self.assertEqual(str(raised.exception.__cause__), "cannot write RUN_FAILED.json")
            self.assertFalse((root / "output/RUN_COMPLETE").exists())

    def test_completion_failure_records_completed_pairs_without_success_marker(self):
        original_write_text = Path.write_text
        def fail_summary_write(path, *args, **kwargs):
            if path.name == "summary.json":
                raise OSError("summary write failed")
            return original_write_text(path, *args, **kwargs)
        with TemporaryDirectory() as directory, patch.object(Path, "write_text", new=fail_summary_write):
            root = Path(directory)
            with self.assertRaisesRegex(OSError, "summary write failed"):
                self._run(root, MockGenerator(TinyGptOssModel()), ["one"])
            failure = json.loads((root / "output/RUN_FAILED.json").read_text())
            self.assertEqual(failure["failure_stage"], "completion")
            self.assertEqual(failure["completed_pair_count"], 1)
            self.assertFalse((root / "output/RUN_COMPLETE").exists())

    def test_nonretail_conditions_use_fresh_servers(self):
        """A mutable mock calculator starts clean for every condition."""
        class AlwaysValidGenerator(MockGenerator):
            def generate_text(self, **kwargs):
                return SimpleNamespace(text=VALID_CALL)
        catalog = _catalog()
        counts = []
        @asynccontextmanager
        async def fresh_session():
            async with _session_for_catalog(catalog) as session:
                count = 0
                async def call_tool(name, arguments):
                    nonlocal count
                    count += 1
                    counts.append(count)
                    return SimpleNamespace(structuredContent={"result": 4}, content=[], isError=False)
                session.call_tool = call_tool
                yield session
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = _write_saved_run(root, catalog, ["one"])
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            run_evaluation(
                source_run_dir=source, checkpoint_dir=checkpoint, output_dir=root / "output",
                config=EvaluationConfig(layers=(0, 1), seeds=(1, 2), method="noise", strength=0, example_limit=1),
                generator_loader=lambda _: AlwaysValidGenerator(TinyGptOssModel()),
                session_factory=fresh_session,
            )
            summary = json.loads((root / "output/summary.json").read_text())
            self.assertEqual(summary["no_op_interventions"], 4)
            self.assertEqual(summary["paired_diagnostics"]["unique_control_examples"], 1)
            self.assertEqual(summary["paired_diagnostics"]["control_drift_sample_ids"], [])
        self.assertEqual(counts, [1] * 8)

    def test_condition_registry_drift_fails_instead_of_marking_complete(self):
        catalog = _catalog()
        opens = 0
        @asynccontextmanager
        async def drifting_session():
            nonlocal opens
            opens += 1
            async with _session_for_catalog(catalog) as session:
                if opens > 1:
                    async def list_changed_tools():
                        return SimpleNamespace(tools=[SimpleNamespace(name="calculator", inputSchema={}, description="changed")])
                    session.list_tools = list_changed_tools
                yield session
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = _write_saved_run(root, catalog, ["one"])
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            with self.assertRaisesRegex(ValueError, "registry drifted"):
                run_evaluation(
                    source_run_dir=source, checkpoint_dir=checkpoint, output_dir=root / "output",
                    config=EvaluationConfig(layers=(0,), seeds=(1,), method="noise", strength=.1, example_limit=1),
                    generator_loader=lambda _: MockGenerator(TinyGptOssModel()),
                    session_factory=drifting_session,
                )
            self.assertFalse((root / "output/RUN_COMPLETE").exists())
            self.assertTrue((root / "output/RUN_FAILED.json").exists())


if __name__ == "__main__":
    unittest.main()
