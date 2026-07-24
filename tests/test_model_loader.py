from __future__ import annotations

import argparse
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import warnings

import torch

from models import model_loader
from models.model_loader import LoadedModelComponents


class ModelLoaderTests(unittest.TestCase):
    def test_resolves_default_model_name(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(
                model_loader.resolve_model_name(),
                "Qwen/Qwen2.5-3B-Instruct",
            )

    def test_resolves_environment_model_name_override(self) -> None:
        with patch.dict(
            "os.environ",
            {"LAYERMCP_MODEL_NAME": "google/gemma-2-2b-it"},
            clear=True,
        ):
            self.assertEqual(
                model_loader.resolve_model_name(),
                "google/gemma-2-2b-it",
            )

    def test_explicit_model_name_takes_precedence(self) -> None:
        with patch.dict(
            "os.environ",
            {"LAYERMCP_MODEL_NAME": "google/gemma-2-2b-it"},
            clear=True,
        ):
            self.assertEqual(
                model_loader.resolve_model_name("meta-llama/Llama-3.2-1B-Instruct"),
                "meta-llama/Llama-3.2-1B-Instruct",
            )

    def test_dtype_option_parsing(self) -> None:
        self.assertEqual(model_loader.resolve_torch_dtype("float16"), torch.float16)
        self.assertEqual(model_loader.resolve_torch_dtype("bfloat16"), torch.bfloat16)
        self.assertEqual(model_loader.resolve_torch_dtype("float32"), torch.float32)

    def test_auto_dtype_uses_cuda_availability(self) -> None:
        with patch("models.model_loader.torch.cuda.is_available", return_value=False):
            self.assertIsNone(model_loader.resolve_torch_dtype("auto"))
        with patch("models.model_loader.torch.cuda.is_available", return_value=True):
            self.assertEqual(model_loader.resolve_torch_dtype("auto"), torch.float16)

    def test_rejects_unknown_dtype(self) -> None:
        with self.assertRaises(ValueError):
            model_loader.resolve_torch_dtype("float8")

    def test_quantization_falls_back_when_bitsandbytes_is_unavailable(self) -> None:
        with patch("models.model_loader.find_spec", return_value=None):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                config = model_loader.build_quantization_config("4bit")

        self.assertIsNone(config)
        self.assertIn("bitsandbytes is not installed", str(caught[0].message))

    def test_model_kwargs_include_hidden_states_without_loading_model(self) -> None:
        with patch("models.model_loader.torch.cuda.is_available", return_value=False):
            kwargs = model_loader.build_model_kwargs(
                dtype="float32",
                output_hidden_states=True,
            )

        self.assertEqual(kwargs["torch_dtype"], torch.float32)
        self.assertTrue(kwargs["output_hidden_states"])
        self.assertTrue(kwargs["low_cpu_mem_usage"])
        self.assertNotIn("device_map", kwargs)


class SharedLoaderIntegrationTests(unittest.TestCase):
    def test_router_uses_shared_loader_through_cache_boundary(self) -> None:
        from models.routers import qwen_hf_router

        fake_components = LoadedModelComponents(
            tokenizer=object(),
            model=object(),
            model_name="fake/model",
        )

        qwen_hf_router._load_model_components.cache_clear()
        try:
            with patch(
                "models.routers.qwen_hf_router.load_model_components",
                return_value=fake_components,
            ) as load_model:
                tokenizer, model = qwen_hf_router._load_model_components()
                cached_tokenizer, cached_model = qwen_hf_router._load_model_components()

            load_model.assert_called_once_with(qwen_hf_router.MODEL_NAME)
            self.assertIs(tokenizer, fake_components.tokenizer)
            self.assertIs(model, fake_components.model)
            self.assertIs(cached_tokenizer, tokenizer)
            self.assertIs(cached_model, model)
        finally:
            qwen_hf_router._load_model_components.cache_clear()

    def test_logit_lens_uses_shared_loader_through_mock(self) -> None:
        from analysis import logit_lens

        fake_model = SimpleNamespace(eval=lambda: None)
        fake_components = LoadedModelComponents(
            tokenizer=object(),
            model=fake_model,
            model_name="fake/model",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            args = argparse.Namespace(
                benchmark=Path("benchmark/tool_routing_phase2_seed.json"),
                model="fake/model",
                output_dir=Path(tmpdir),
                max_examples=0,
                plot=False,
            )

            with patch(
                "analysis.logit_lens.load_model_components",
                return_value=fake_components,
            ) as load_model:
                with patch(
                    "torch.cuda.is_available",
                    return_value=False,
                ):
                    paths = logit_lens.run_analysis(args)

            load_model.assert_called_once_with("fake/model", output_hidden_states=True)
            self.assertTrue(paths["csv"].exists())
            self.assertTrue(paths["summary"].exists())


class RouterRegistryTests(unittest.TestCase):
    def test_local_router_prompt_distinguishes_numeric_and_symbolic_math(self) -> None:
        from mcp_server.server import mcp
        from models.routers.qwen36_local_router import _build_prompt

        tool_names = ["calculator", "simplify_expression", "factor_expression"]
        live_descriptions = {
            name: mcp._tool_manager._tools[name].description for name in tool_names
        }
        prompt = _build_prompt(
            "Compute 55^2 - 45^2.",
            tool_names,
            live_descriptions,
        )

        self.assertIn(
            "calculator: Numerically evaluate an arithmetic expression",
            prompt,
        )
        self.assertIn(
            "Do not use this for a request that only asks for a numeric value",
            prompt,
        )
        self.assertIn(
            "factor_expression: Factor a symbolic expression",
            prompt,
        )

    def test_local_router_prompt_prefers_live_mcp_description(self) -> None:
        from models.routers.qwen36_local_router import _build_prompt

        prompt = _build_prompt(
            "Compute 2 + 2.",
            ["calculator"],
            {"calculator": "Live description supplied by the MCP server."},
        )

        self.assertIn(
            "calculator: Live description supplied by the MCP server.",
            prompt,
        )
        self.assertNotIn("Numerically evaluate", prompt)

    def test_registry_loads_named_router_modules(self) -> None:
        from models.routers.registry import load_router

        qwen_router = load_router("qwen-hf")
        gpt_oss_router = load_router("gpt-oss-local")
        phi4_router = load_router("phi-4-local")
        llama_router = load_router("llama-3.1-8b-local")
        qwen36_router = load_router("qwen-3.6-local")
        gemma4_router = load_router("gemma-4-local")

        self.assertEqual(qwen_router.ROUTER_ID, "qwen_hf_router")
        self.assertEqual(gpt_oss_router.ROUTER_ID, "gpt_oss_local_router")
        self.assertEqual(phi4_router.ROUTER_ID, "phi4_local_router")
        self.assertEqual(llama_router.ROUTER_ID, "llama31_8b_local_router")
        self.assertEqual(qwen36_router.ROUTER_ID, "qwen36_local_router")
        self.assertEqual(gemma4_router.ROUTER_ID, "gemma4_local_router")

    def test_registry_rejects_unknown_router(self) -> None:
        from models.routers.registry import load_router

        with self.assertRaises(ValueError):
            load_router("unknown-router")

    def test_gpt_oss_router_extracts_harmony_tool_name(self) -> None:
        from models.routers.gpt_oss_local_router import _extract_tool_name

        response = (
            "to=calculator to=functions<|channel|><|constrain|>json"
            '<|message|>{"expression": "2 + 2"}<|call|>'
        )

        self.assertEqual(
            _extract_tool_name(response, ["calculator", "github_search"]),
            "calculator",
        )

    def test_gpt_oss_checkpoint_path_uses_environment_override(self) -> None:
        from models.routers.gpt_oss_local_router import resolve_checkpoint_path

        with patch.dict(
            "os.environ",
            {"LAYERMCP_GPT_OSS_CHECKPOINT": "custom/checkpoint"},
        ):
            self.assertEqual(
                resolve_checkpoint_path(),
                Path("custom/checkpoint"),
            )

    def test_phi4_router_extracts_json_tool_name(self) -> None:
        from models.routers.phi4_local_router import _extract_tool_name

        response = '```json\n{"name": "calculator", "arguments": {"expression": "2 + 2"}}\n```'

        self.assertEqual(
            _extract_tool_name(response, ["calculator", "github_search"]),
            "calculator",
        )

    def test_phi4_checkpoint_path_uses_environment_override(self) -> None:
        from models.routers.phi4_local_router import resolve_checkpoint_path

        with patch.dict(
            "os.environ",
            {"LAYERMCP_PHI4_CHECKPOINT": "custom/phi4"},
        ):
            self.assertEqual(
                resolve_checkpoint_path(),
                Path("custom/phi4"),
            )

    def test_llama31_router_extracts_json_tool_name(self) -> None:
        from models.routers.llama31_8b_local_router import _extract_tool_name

        response = '{"tool_name": "github_search"}'

        self.assertEqual(
            _extract_tool_name(response, ["calculator", "github_search"]),
            "github_search",
        )

    def test_llama31_checkpoint_path_uses_environment_override(self) -> None:
        from models.routers.llama31_8b_local_router import resolve_checkpoint_path

        with patch.dict(
            "os.environ",
            {"LAYERMCP_LLAMA31_8B_CHECKPOINT": "custom/llama"},
        ):
            self.assertEqual(
                resolve_checkpoint_path(),
                Path("custom/llama"),
            )

    def test_llama31_router_returns_structured_tool_call(self) -> None:
        from models.routers import llama31_8b_local_router

        generator = Mock()
        generator.encode_chat.return_value = [1, 2, 3]
        generator.stop_tokens = [128001, 128008, 128009]
        generator.generate_text.return_value = (
            '{"name":"factor_expression","arguments":{"expression":"t^2-49"}}'
        )

        with patch.object(llama31_8b_local_router, "_load_generator", return_value=generator):
            selected = llama31_8b_local_router.choose_tool(
                "Factor t^2-49.",
                ["calculator", "factor_expression", "expand_expression"],
            )

        self.assertEqual(selected, "factor_expression")
        generator.generate_text.assert_called_once()

    def test_llama31_parses_parameters_and_structural_tokens(self) -> None:
        from models.routers.structured_tool_call import parse_tool_call

        responses = (
            (
                '<|python_tag|>{"name":"calculator",'
                '"parameters":{"expression":"2+2"}}<|eom_id|>'
            ),
            (
                '<|python_tag|>{"name":"calculator",'
                '"parameters":"{\\"expression\\":\\"2+2\\"}"}<|eom_id|>'
            ),
        )
        for response in responses:
            with self.subTest(response=response):
                prediction = parse_tool_call(response, ["calculator"])
                self.assertEqual(prediction.selected_tool, "calculator")
                self.assertEqual(prediction.selected_args, {"expression": "2+2"})
                self.assertEqual(prediction.raw_output, response)

    def test_llama31_structured_path_receives_live_tools(self) -> None:
        from models.routers import llama31_8b_local_router

        generator = Mock()
        generator.encode_chat.return_value = [1, 2, 3]
        generator.stop_tokens = [128001, 128008, 128009]
        generator.generate_text.return_value = (
            '<|python_tag|>{"name":"calculator",'
            '"parameters":{"expression":"2+2"}}<|eom_id|>'
        )
        schema = {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        }

        with patch.object(
            llama31_8b_local_router,
            "_load_generator",
            return_value=generator,
        ):
            prediction = llama31_8b_local_router.choose_tool_call(
                "Compute 2+2.",
                ["calculator"],
                {"calculator": schema},
                {"calculator": "Evaluate arithmetic."},
            )

        self.assertEqual(prediction.selected_tool, "calculator")
        native_messages = generator.encode_chat.call_args.args[0]
        fallback_messages = generator.encode_chat.call_args.kwargs[
            "fallback_messages"
        ]
        native_tools = generator.encode_chat.call_args.kwargs["tools"]
        self.assertNotIn("Available MCP tools", native_messages[0]["content"])
        self.assertNotIn('"input_schema"', native_messages[0]["content"])
        self.assertIn("Available MCP tools", fallback_messages[0]["content"])
        self.assertIn('"input_schema"', fallback_messages[0]["content"])
        self.assertEqual(native_tools[0]["function"]["parameters"], schema)
        self.assertEqual(
            native_tools[0]["function"]["description"],
            "Evaluate arithmetic.",
        )
        self.assertEqual(
            generator.generate_text.call_args.kwargs["stop_tokens"],
            generator.stop_tokens,
        )

    def test_llama31_rejects_unknown_structured_tool(self) -> None:
        from models.routers.structured_tool_call import parse_tool_call

        response = (
            '<|python_tag|>{"name":"invented_tool",'
            '"parameters":{"expression":"2+2"}}<|eom_id|>'
        )
        prediction = parse_tool_call(response, ["calculator"])

        self.assertEqual(prediction.selected_tool, "unknown_tool")
        self.assertEqual(prediction.parse_status, "unknown_tool")
        self.assertEqual(prediction.attempted_tool, "invented_tool")
        self.assertEqual(prediction.selected_args, {})
        self.assertEqual(prediction.raw_output, response)

    def test_llama31_malformed_json_is_a_parse_error(self) -> None:
        from models.routers.structured_tool_call import parse_tool_call

        response = (
            '<|python_tag|>{"name":"calculator",'
            '"parameters":{"expression":"2+2"}<|eom_id|>'
        )
        prediction = parse_tool_call(response, ["calculator"])

        self.assertEqual(prediction.selected_tool, "parse_error")
        self.assertEqual(prediction.parse_status, "parse_error")
        self.assertIsNone(prediction.attempted_tool)
        self.assertIn("no complete structured", prediction.diagnostic)
        self.assertEqual(prediction.raw_output, response)

    def test_llama31_schema_failure_has_structured_metadata(self) -> None:
        from models.routers.structured_tool_call import parse_tool_call

        response = '{"name":"calculator","parameters":{}}'
        prediction = parse_tool_call(
            response,
            ["calculator"],
            tool_schemas={
                "calculator": {
                    "type": "object",
                    "properties": {"expression": {"type": "string"}},
                    "required": ["expression"],
                }
            },
        )

        self.assertEqual(prediction.selected_tool, "calculator")
        self.assertEqual(prediction.parse_status, "invalid_arguments")
        self.assertIn("missing required arguments", prediction.diagnostic)

    def test_llama31_tokenizer_fallback_uses_serialized_tool_prompt(self) -> None:
        from models.architectures.llama31_8b_pytorch.inference import TokenGenerator

        tokenizer = Mock()
        tokenizer.apply_chat_template.side_effect = [
            TypeError("tools are unsupported"),
            "rendered fallback",
        ]
        tokenizer.return_value = {"input_ids": [1, 2, 3]}
        generator = TokenGenerator.__new__(TokenGenerator)
        generator.tokenizer = tokenizer
        native_messages = [{"role": "user", "content": "Route this query."}]
        fallback_messages = [
            {
                "role": "user",
                "content": "Available MCP tools: serialized fallback",
            }
        ]
        tools = [{"type": "function", "function": {"name": "calculator"}}]

        encoded = generator.encode_chat(
            native_messages,
            tools=tools,
            fallback_messages=fallback_messages,
        )

        self.assertEqual(encoded, [1, 2, 3])
        first_call, second_call = tokenizer.apply_chat_template.call_args_list
        self.assertEqual(first_call.args[0], native_messages)
        self.assertEqual(first_call.kwargs["tools"], tools)
        self.assertEqual(second_call.args[0], fallback_messages)
        self.assertNotIn("tools", second_call.kwargs)

    def test_qwen36_router_extracts_qwen_tool_call(self) -> None:
        from models.routers.qwen36_local_router import _extract_tool_name

        self.assertEqual(
            _extract_tool_name(
                '<tool_call>{"name": "calculator", "arguments": {}}</tool_call>',
                ["calculator", "github_search"],
            ),
            "calculator",
        )

    def test_structured_parser_accepts_qwen_native_tool_call(self) -> None:
        from models.routers.structured_tool_call import parse_tool_call

        response = """Reasoning complete.
</think>
<tool_call>
<function=calculator>
<parameter=expression>
139 + 27 + 23 + 11
</parameter>
</function>
</tool_call><|im_end|>"""

        prediction = parse_tool_call(
            response,
            ["calculator", "simplify_expression"],
        )

        self.assertEqual(prediction.selected_tool, "calculator")
        self.assertEqual(
            prediction.selected_args,
            {"expression": "139 + 27 + 23 + 11"},
        )

    def test_other_local_routers_return_structured_tool_calls(self) -> None:
        from models.routers import (
            gemma4_local_router,
            gpt_oss_local_router,
            phi4_local_router,
            qwen36_local_router,
        )

        cases = []

        qwen_generator = Mock()
        qwen_generator.apply_chat_template.return_value = [1]
        cases.append((qwen36_local_router, qwen_generator))

        gemma_generator = Mock()
        gemma_generator.tokenizer.apply_chat_template.return_value = [1]
        cases.append((gemma4_local_router, gemma_generator))

        gpt_generator = Mock()
        gpt_generator.tokenizer.encode.return_value = [1]
        cases.append((gpt_oss_local_router, gpt_generator))

        phi_generator = Mock()
        phi_generator.tokenizer.chat_template = None
        phi_generator.tokenizer.encode.return_value = [1]
        cases.append((phi4_local_router, phi_generator))

        available_tools = [
            "calculator",
            "factor_expression",
            "expand_expression",
        ]
        for router, generator in cases:
            with self.subTest(router=router.ROUTER_ID):
                generator.generate_text.return_value = SimpleNamespace(
                    text=(
                        '{"name":"factor_expression",'
                        '"arguments":{"expression":"t^2-49"}}'
                    ),
                    tool_call=None,
                )
                with patch.object(router, "_load_generator", return_value=generator):
                    prediction = router.choose_tool_call(
                        "Factor t^2-49.",
                        available_tools,
                        {"factor_expression": {"type": "object"}},
                    )
                self.assertEqual(prediction.selected_tool, "factor_expression")
                self.assertEqual(prediction.selected_args, {"expression": "t^2-49"})
                generator.generate_text.assert_called_once()

    def test_qwen36_checkpoint_path_uses_environment_override(self) -> None:
        from models.routers.qwen36_local_router import resolve_checkpoint_path

        with patch.dict(
            "os.environ",
            {"LAYERMCP_QWEN36_CHECKPOINT": "custom/qwen36"},
        ):
            self.assertEqual(resolve_checkpoint_path(), Path("custom/qwen36"))

    def test_gemma4_router_extracts_gemma_tool_call(self) -> None:
        from models.routers.gemma4_local_router import _extract_tool_name

        self.assertEqual(
            _extract_tool_name(
                '<|tool_call>call:calculator{ expression: <|"|>2 + 2<|"|> }<tool_call|>',
                ["calculator", "github_search"],
            ),
            "calculator",
        )

    def test_gemma4_checkpoint_path_uses_environment_override(self) -> None:
        from models.routers.gemma4_local_router import resolve_checkpoint_path

        with patch.dict(
            "os.environ",
            {"LAYERMCP_GEMMA4_CHECKPOINT": "custom/gemma4"},
        ):
            self.assertEqual(resolve_checkpoint_path(), Path("custom/gemma4"))


if __name__ == "__main__":
    unittest.main()
