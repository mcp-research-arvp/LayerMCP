from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import torch

from research.phase2.observe import (
    ActivationObserver,
    _write_completed_observation,
    capture_selected_activations,
    select_tool_call_positions,
)
from research.phase2.replay import (
    LLAMA_MODEL_NAME,
    TOKEN_SELECTION_MODE,
    ToolCatalog,
    config_provenance,
    load_replay_config,
    load_saved_example,
    render_llama_prompt,
)


class CharacterTokenizer:
    """A CPU-only tokenizer stub whose IDs decode to one character each."""

    def __init__(self, mapping: dict[int, str]) -> None:
        self.mapping = mapping

    def decode(self, token_ids, skip_special_tokens=False):
        return "".join(self.mapping[token_id] for token_id in token_ids)


class PromptTokenizer(CharacterTokenizer):
    def apply_chat_template(self, messages, tools, add_generation_prompt, tokenize):
        self.last_messages = messages
        self.last_tools = tools
        return "prompt"

    def __call__(self, text, return_tensors=None):
        return {"input_ids": [1, 2, 3]}


class PromptGenerator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.calls = []

    def encode_chat(self, messages, tools=None, fallback_messages=None):
        self.calls.append((messages, tools, fallback_messages))
        return [1, 2, 3]


class FallbackPromptTokenizer(PromptTokenizer):
    def apply_chat_template(self, messages, tools=None, add_generation_prompt=False, tokenize=False):
        if tools is not None:
            raise ValueError("native tools are unavailable")
        self.last_messages = messages
        return "prompt"


class TinyBlock(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = torch.nn.Identity()
        self.mlp = torch.nn.Identity()

    def forward(self, x):
        return x + self.self_attn(x) + self.mlp(x)


class RaisingAttention(torch.nn.Module):
    def forward(self, x):
        raise RuntimeError("intentional observation test error")


class TinyTransformer(torch.nn.Module):
    def __init__(self, raising: bool = False) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList([TinyBlock()])
        if raising:
            self.layers[0].self_attn = RaisingAttention()

    def forward(self, token_ids):
        x = token_ids.to(torch.float32).unsqueeze(-1).repeat(1, 1, 3)
        for layer in self.layers:
            x = layer(x)
        return x


def _saved_record(sample_id: str = "math_1") -> dict:
    return {
        "id": sample_id,
        "model_name": LLAMA_MODEL_NAME,
        "reasoning_mode": "direct",
        "reasoning_method": "none",
        "prompt_template": "structured_tool_call_v1",
        "router_backend": "local_llama31_8b_pytorch",
        "tool_names": ["calculator"],
        "tool_pool": "full_mcp_registry",
        "tool_count": 1,
        "tool_registry_fingerprint": "sha256:test",
        "tool_registry_fingerprint_version": "tool_registry_name_schema_description_v1",
        "routed_query": "Calculate 2 + 2.",
        "effective_generation_limit": 128,
    }


class Phase2ObservabilityTests(unittest.TestCase):
    def _make_source_run(self, directory: Path, record: dict | None = None) -> Path:
        source = directory / "run"
        source.mkdir()
        (source / "run_metadata.json").write_text(
            json.dumps({"generation_settings": {"temperature": 0.0, "max_tokens": 128}}),
            encoding="utf-8",
        )
        (source / "artifact_index.jsonl").write_text(
            json.dumps({"samples_path": "samples.jsonl"}) + "\n", encoding="utf-8"
        )
        (source / "samples.jsonl").write_text(
            json.dumps(record or _saved_record()) + "\n", encoding="utf-8"
        )
        return source

    def test_loads_saved_example_and_validates_replay_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = self._make_source_run(Path(temporary))
            example = load_saved_example(source, "math_1")
        self.assertEqual(example.record["routed_query"], "Calculate 2 + 2.")
        self.assertEqual(example.samples_path.name, "samples.jsonl")

    def test_config_and_provenance_are_declarative(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config_path = Path(temporary) / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "source_run_dir": "/tmp/source",
                        "sample_id": "math_1",
                        "baseline_campaign_commit": "abc123",
                        "layers": [0, 2],
                        "token_selection_mode": TOKEN_SELECTION_MODE,
                        "development_only": True,
                        "require_registry_match": False,
                        "observation_enabled": False,
                    }
                ),
                encoding="utf-8",
            )
            config = load_replay_config(config_path)
        provenance = config_provenance(config)
        self.assertEqual(provenance["observation_layers"], [0, 2])
        self.assertTrue(provenance["development_only"])
        self.assertFalse(provenance["observation_enabled"])
        json.dumps(provenance)

    def test_position_selection_includes_tool_and_argument_tokens_only(self) -> None:
        text = '{"name":"calculator","arguments":{"expression":"2+2"}}'
        tokenizer = CharacterTokenizer({index: character for index, character in enumerate(text)})
        positions = select_tool_call_positions(
            tokenizer,
            list(range(len(text))),
            "calculator",
            {"expression": "2+2"},
        )
        selected_text = "".join(item["decoded_piece"] for item in positions)
        roles = {role for item in positions for role in item["roles"]}
        self.assertIn("calculator", selected_text)
        self.assertIn("expression", selected_text)
        self.assertIn("2+2", selected_text)
        self.assertEqual(roles, {"tool_name", "argument_key", "argument_value"})
        self.assertLess(len(positions), len(text))

    def test_disabled_observation_is_a_noop(self) -> None:
        model = TinyTransformer()
        tokens = torch.tensor([[1, 2, 3]])
        baseline = model(tokens).clone()
        with ActivationObserver(model, (0,), (1,), enabled=False) as observer:
            observed = model(tokens)
        self.assertTrue(torch.equal(baseline, observed))
        self.assertEqual(observer.captured, {})
        self.assertEqual(observer.handles, [])

    def test_hooks_cleanup_after_model_error(self) -> None:
        model = TinyTransformer(raising=True)
        with self.assertRaisesRegex(RuntimeError, "intentional observation"):
            with ActivationObserver(model, (0,), (1,), enabled=True) as observer:
                model(torch.tensor([[1, 2, 3]]))
        self.assertEqual(observer.handles, [])
        self.assertEqual(len(model.layers[0]._forward_hooks), 0)
        self.assertEqual(len(model.layers[0].self_attn._forward_hooks), 0)
        self.assertEqual(len(model.layers[0].mlp._forward_hooks), 0)

    def test_hooks_cleanup_when_installation_fails(self) -> None:
        model = TinyTransformer()
        observer = ActivationObserver(model, (0, 1), (1,), enabled=True)
        with self.assertRaises(IndexError):
            observer.__enter__()
        self.assertEqual(observer.handles, [])
        self.assertEqual(len(model.layers[0]._forward_hooks), 0)
        self.assertEqual(len(model.layers[0].self_attn._forward_hooks), 0)
        self.assertEqual(len(model.layers[0].mlp._forward_hooks), 0)

    def test_teacher_forced_capture_keeps_only_selected_positions(self) -> None:
        model = TinyTransformer()
        captures = capture_selected_activations(
            model,
            (1, 2, 3),
            [4, 5, 6],
            [{"generated_token_index": 1}, {"generated_token_index": 2}],
            (0,),
            torch.device("cpu"),
        )
        self.assertEqual(set(captures["0"]), {"residual_stream", "attention_block", "mlp_block"})
        self.assertEqual(tuple(captures["0"]["residual_stream"].shape), (1, 2, 3))

    def test_disabled_capture_is_empty_and_does_not_install_hooks(self) -> None:
        model = TinyTransformer()
        captures = capture_selected_activations(
            model,
            (1, 2, 3),
            [4, 5],
            [{"generated_token_index": 1}],
            (0,),
            torch.device("cpu"),
            enabled=False,
        )
        self.assertEqual(captures, {})
        self.assertEqual(len(model.layers[0]._forward_hooks), 0)

    def test_disabled_observation_writes_no_activation_tensor_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            _write_completed_observation(
                output,
                {"observation_enabled": False},
                {},
                observation_enabled=False,
            )
            self.assertTrue((output / "provenance.json").is_file())
            self.assertTrue((output / "OBSERVATION_COMPLETE").is_file())
            self.assertFalse((output / "activations.pt").exists())

    def test_rendered_prompt_records_registry_match(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = self._make_source_run(Path(temporary))
            example = load_saved_example(source, "math_1")
            catalog = ToolCatalog(
                names=("calculator",),
                schemas={"calculator": {"type": "object"}},
                descriptions={"calculator": "Calculate."},
                metadata={
                    "tool_pool": "full_mcp_registry",
                    "tool_count": 1,
                    "tool_registry_fingerprint": "sha256:test",
                    "tool_registry_fingerprint_version": "tool_registry_name_schema_description_v1",
                },
            )
            generator = PromptGenerator(PromptTokenizer({}))
            prompt = render_llama_prompt(generator, example, catalog)
        self.assertTrue(prompt.registry_exact_match)
        self.assertEqual(prompt.generation_settings["max_tokens"], 128)
        self.assertEqual(len(generator.calls), 1)

    def test_rendered_prompt_uses_router_fallback_behavior(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = self._make_source_run(Path(temporary))
            example = load_saved_example(source, "math_1")
            catalog = ToolCatalog(
                names=("calculator",),
                schemas={"calculator": {"type": "object"}},
                descriptions={"calculator": "Calculate."},
                metadata={
                    "tool_pool": "full_mcp_registry",
                    "tool_count": 1,
                    "tool_registry_fingerprint": "sha256:test",
                    "tool_registry_fingerprint_version": "tool_registry_name_schema_description_v1",
                },
            )
            tokenizer = FallbackPromptTokenizer({})
            render_llama_prompt(PromptGenerator(tokenizer), example, catalog)
        self.assertIn("Available MCP tools", tokenizer.last_messages[0]["content"])


if __name__ == "__main__":
    unittest.main()
