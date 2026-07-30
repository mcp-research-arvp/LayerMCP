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

    def test_gpt_oss_finance_validation_rejects_invented_table(self) -> None:
        from models.routers.gpt_oss_local_router import _finance_argument_errors
        from models.routers.structured_tool_call import ToolCallPrediction

        prediction = ToolCallPrediction(
            selected_tool="finance_query_table",
            selected_args={
                "dataset_id": "finqa-public-test-v1",
                "sql": "SELECT numeric_value FROM source_table",
            },
            raw_output="",
        )

        errors = _finance_argument_errors(prediction)
        self.assertEqual(len(errors), 1)
        self.assertIn("only valid SQLite table is data", errors[0])

    def test_gpt_oss_finance_validation_accepts_data_table(self) -> None:
        from models.routers.gpt_oss_local_router import _finance_argument_errors
        from models.routers.structured_tool_call import ToolCallPrediction

        prediction = ToolCallPrediction(
            selected_tool="finance_query_table",
            selected_args={
                "dataset_id": "finqa-public-test-v1",
                "sql": "SELECT AVG(numeric_value) FROM data",
            },
            raw_output="",
        )

        self.assertEqual(_finance_argument_errors(prediction), [])

    def test_structured_parser_accepts_gpt_oss_harmony_variants(self) -> None:
        from models.routers.structured_tool_call import parse_tool_call

        cases = [
            (
                "to=functions.calculator<|channel|><|constrain|>json"
                '<|message|>{"expression":{"left":2,"right":2}}<|call|>',
                {"expression": {"left": 2, "right": 2}},
            ),
            (
                "We must call calculator.\n"
                "to=functions<|channel|><|constrain|>json"
                '<|message|>{"expression":"2 + 2"}<|call|>',
                {"expression": "2 + 2"},
            ),
        ]
        for response, expected_args in cases:
            with self.subTest(response=response):
                prediction = parse_tool_call(
                    response,
                    ["calculator", "github_search"],
                )
                self.assertEqual(prediction.selected_tool, "calculator")
                self.assertEqual(prediction.selected_args, expected_args)

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

    def test_structured_parser_recovers_one_unambiguous_near_tool_name(self) -> None:
        from models.routers.structured_tool_call import parse_tool_call

        native_call = SimpleNamespace(
            function=SimpleNamespace(
                name="calculate",
                arguments='{"expression":"100^4"}',
            )
        )
        prediction = parse_tool_call(
            "",
            ["calculator", "simplify_expression"],
            native_call,
        )

        self.assertEqual(prediction.selected_tool, "calculator")
        self.assertEqual(prediction.selected_args, {"expression": "100^4"})

    def test_structured_parser_rejects_ambiguous_or_distant_tool_name(self) -> None:
        from models.routers.structured_tool_call import parse_tool_call

        native_call = SimpleNamespace(
            function=SimpleNamespace(name="finance_get", arguments="{}")
        )
        prediction = parse_tool_call(
            "",
            [
                "finance_get_company_facts",
                "finance_get_financial_statement",
            ],
            native_call,
        )

        self.assertEqual(prediction.selected_tool, "hallucinated_tool")

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
        gpt_generator.render_tool_prompt.return_value = [1]
        gpt_generator.assistant_action_stop_tokens = [2, 3]
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

    def test_gpt_oss_router_uses_dynamic_harmony_tool_definitions(self) -> None:
        from models.routers import gpt_oss_local_router

        generator = Mock()
        generator.render_tool_prompt.return_value = [1]
        generator.assistant_action_stop_tokens = [2]
        generator.generate_text.return_value = SimpleNamespace(
            text="",
            tool_call=SimpleNamespace(
                function=SimpleNamespace(
                    name="calculator",
                    arguments='{"expression":"2 + 2"}',
                )
            ),
        )

        with patch.object(gpt_oss_local_router, "_load_generator", return_value=generator):
            prediction = gpt_oss_local_router.choose_tool_call(
                "Compute 2 + 2.",
                ["calculator", "factor_expression"],
                {
                    "calculator": {
                        "type": "object",
                        "properties": {"expression": {"type": "string"}},
                        "required": ["expression"],
                    }
                },
                {"calculator": "Evaluate arithmetic."},
            )

        rendered_tools = generator.render_tool_prompt.call_args.args[1]
        self.assertEqual(
            [tool["name"] for tool in rendered_tools],
            ["calculator", "factor_expression"],
        )
        self.assertEqual(prediction.selected_tool, "calculator")
        self.assertEqual(prediction.selected_args, {"expression": "2 + 2"})

    def test_gpt_oss_router_retries_schema_invalid_arguments(self) -> None:
        from models.routers import gpt_oss_local_router

        generator = Mock()
        generator.render_tool_prompt.return_value = [1]
        generator.assistant_action_stop_tokens = [2]
        generator.generate_text.side_effect = [
            SimpleNamespace(
                text="",
                tool_call=SimpleNamespace(
                    function=SimpleNamespace(
                        name="simplify_expression",
                        arguments="{}",
                    )
                ),
            ),
            SimpleNamespace(
                text="",
                tool_call=SimpleNamespace(
                    function=SimpleNamespace(
                        name="simplify_expression",
                        arguments='{"expression":"(x**2-1)/(x-1)"}',
                    )
                ),
            ),
        ]

        schema = {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        }
        with patch.object(gpt_oss_local_router, "_load_generator", return_value=generator):
            prediction = gpt_oss_local_router.choose_tool_call(
                "Simplify (x^2-1)/(x-1).",
                ["simplify_expression"],
                {"simplify_expression": schema},
            )

        self.assertEqual(prediction.selected_tool, "simplify_expression")
        self.assertEqual(
            prediction.selected_args,
            {"expression": "(x**2-1)/(x-1)"},
        )
        self.assertEqual(generator.generate_text.call_count, 2)
        correction_prompt = generator.render_tool_prompt.call_args_list[1].args[0]
        self.assertIn("missing required argument", correction_prompt)

    def test_gpt_oss_router_reconsiders_hallucinated_tool_once(self) -> None:
        from models.routers import gpt_oss_local_router

        generator = Mock()
        generator.render_tool_prompt.return_value = [1]
        generator.assistant_action_stop_tokens = [2]
        generator.generate_text.side_effect = [
            SimpleNamespace(
                text="hallucinated_tool",
                tool_call=None,
            ),
            SimpleNamespace(
                text="",
                tool_call=SimpleNamespace(
                    function=SimpleNamespace(
                        name="finance_get_company_facts",
                        arguments='{"company_identifier":"LMCP"}',
                    )
                ),
            ),
        ]

        with patch.object(gpt_oss_local_router, "_load_generator", return_value=generator):
            prediction = gpt_oss_local_router.choose_tool_call(
                "Retrieve company facts for LMCP.",
                ["finance_get_company_facts"],
                {
                    "finance_get_company_facts": {
                        "type": "object",
                        "properties": {
                            "company_identifier": {"type": "string"},
                        },
                        "required": ["company_identifier"],
                    }
                },
            )

        self.assertEqual(
            prediction.selected_tool,
            "finance_get_company_facts",
        )
        self.assertEqual(
            prediction.selected_args,
            {"company_identifier": "LMCP"},
        )
        self.assertEqual(generator.generate_text.call_count, 2)
        reconsideration_prompt = (
            generator.render_tool_prompt.call_args_list[1].args[0]
        )
        self.assertIn("did not produce a valid call", reconsideration_prompt)

    def test_gpt_oss_router_uses_larger_configurable_generation_budget(self) -> None:
        from models.routers import gpt_oss_local_router

        generator = Mock()
        generator.render_tool_prompt.return_value = [1]
        generator.assistant_action_stop_tokens = [2]
        generator.generate_text.return_value = SimpleNamespace(
            text="",
            tool_call=SimpleNamespace(
                function=SimpleNamespace(
                    name="calculator",
                    arguments='{"expression":"2+2"}',
                )
            ),
        )

        with (
            patch.object(gpt_oss_local_router, "_load_generator", return_value=generator),
            patch.dict(
                "os.environ",
                {"LAYERMCP_GPT_OSS_MAX_TOOL_TOKENS": "640"},
            ),
        ):
            gpt_oss_local_router.choose_tool_call(
                "Compute 2+2.",
                ["calculator"],
                {
                    "calculator": {
                        "type": "object",
                        "properties": {"expression": {"type": "string"}},
                        "required": ["expression"],
                    }
                },
            )

        self.assertEqual(
            generator.generate_text.call_args.kwargs["max_tokens"],
            640,
        )

    def test_schema_validation_handles_optional_any_of_and_extra_args(self) -> None:
        from models.routers.structured_tool_call import validate_tool_arguments

        schema = {
            "type": "object",
            "properties": {
                "company_identifier": {"type": "string"},
                "fiscal_year": {
                    "anyOf": [{"type": "integer"}, {"type": "null"}],
                    "default": None,
                },
            },
            "required": ["company_identifier"],
        }
        self.assertEqual(
            validate_tool_arguments(
                {"company_identifier": "LMCP", "fiscal_year": 2024},
                schema,
            ),
            [],
        )
        self.assertTrue(
            validate_tool_arguments(
                {"company_identifier": "LMCP", "fiscal_year": "FY2024"},
                schema,
            )
        )
        self.assertTrue(
            validate_tool_arguments(
                {"company_identifier": "LMCP", "unsupported": 2024},
                schema,
            )
        )

    def test_gpt_oss_router_repairs_call_from_execution_feedback(self) -> None:
        from models.routers import gpt_oss_local_router
        from models.routers.structured_tool_call import ToolCallPrediction

        generator = Mock()
        generator.render_tool_prompt.return_value = [1]
        generator.assistant_action_stop_tokens = [2]
        generator.generate_text.return_value = SimpleNamespace(
            text="",
            tool_call=SimpleNamespace(
                function=SimpleNamespace(
                    name="convert_units",
                    arguments=(
                        '{"value":6,"from_unit":"feet","to_unit":"meters"}'
                    ),
                )
            ),
        )
        previous = ToolCallPrediction(
            selected_tool="convert_units",
            selected_args={"value": 6, "from_unit": "6", "to_unit": "6"},
            raw_output="first call",
        )

        with patch.object(gpt_oss_local_router, "_load_generator", return_value=generator):
            corrected = gpt_oss_local_router.repair_tool_call(
                "Convert 6 feet to meters.",
                ["convert_units"],
                previous,
                "unsupported unit conversion: 6 to 6",
                {"convert_units": {"type": "object"}},
            )

        self.assertEqual(corrected.selected_tool, "convert_units")
        self.assertEqual(
            corrected.selected_args,
            {"value": 6, "from_unit": "feet", "to_unit": "meters"},
        )
        repair_prompt = generator.render_tool_prompt.call_args.args[0]
        self.assertIn("unsupported unit conversion", repair_prompt)
        self.assertIn('"from_unit": "6"', repair_prompt)

    def test_execution_feedback_correction_is_schema_validated(self) -> None:
        from models.routers import gpt_oss_local_router
        from models.routers.structured_tool_call import ToolCallPrediction

        generator = Mock()
        generator.render_tool_prompt.return_value = [1]
        generator.assistant_action_stop_tokens = [2]
        generator.generate_text.side_effect = [
            SimpleNamespace(
                text="",
                tool_call=SimpleNamespace(
                    function=SimpleNamespace(
                        name="finance_get_company_facts",
                        arguments=(
                            '{"company_identifier":"LMCP",'
                            '"fiscal_year":"FY2024"}'
                        ),
                    )
                ),
            ),
            SimpleNamespace(
                text="",
                tool_call=SimpleNamespace(
                    function=SimpleNamespace(
                        name="finance_get_company_facts",
                        arguments=(
                            '{"company_identifier":"LMCP",'
                            '"fiscal_year":2024}'
                        ),
                    )
                ),
            ),
        ]
        previous = ToolCallPrediction(
            selected_tool="finance_get_company_facts",
            selected_args={"company_identifier": "LayerMCP"},
            raw_output="first call",
        )
        schema = {
            "type": "object",
            "properties": {
                "company_identifier": {"type": "string"},
                "fiscal_year": {
                    "anyOf": [{"type": "integer"}, {"type": "null"}],
                },
            },
            "required": ["company_identifier"],
        }

        with patch.object(gpt_oss_local_router, "_load_generator", return_value=generator):
            corrected = gpt_oss_local_router.repair_tool_call(
                "Get LayerMCP company facts for fiscal 2024.",
                ["finance_get_company_facts"],
                previous,
                "Unknown company_identifier. Available tickers: LMCP, TBLR",
                {"finance_get_company_facts": schema},
            )

        self.assertEqual(
            corrected.selected_args,
            {"company_identifier": "LMCP", "fiscal_year": 2024},
        )
        self.assertEqual(generator.generate_text.call_count, 2)
        second_prompt = generator.render_tool_prompt.call_args_list[1].args[0]
        self.assertIn("still violates its JSON schema", second_prompt)
        self.assertIn("FY2024", second_prompt)

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
