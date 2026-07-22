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
    def test_choice_constraint_only_allows_catalog_prefixes(self) -> None:
        from models.architectures.constrained_decoding import ChoiceConstraint

        tokenizer = Mock()
        tokenizer.encode.side_effect = lambda value, **_: {
            "factor_expression": [10, 11],
            "expand_expression": [20, 21],
        }[value]
        constraint = ChoiceConstraint(
            tokenizer,
            ["factor_expression", "expand_expression"],
            stop_token=99,
        )

        self.assertEqual(constraint.allowed_tokens(()), {10, 20})
        self.assertEqual(constraint.allowed_tokens((10,)), {11})
        self.assertEqual(constraint.allowed_tokens((10, 11)), {99})
        self.assertEqual(constraint.resolve([10, 11, 99]), "factor_expression")

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

    def test_llama31_router_constrains_generation_to_tool_catalog(self) -> None:
        from models.routers import llama31_8b_local_router

        generator = Mock()
        generator.encode_chat.return_value = [1, 2, 3]
        generator.generate_choice.return_value = "factor_expression"

        with patch.object(llama31_8b_local_router, "_load_generator", return_value=generator):
            selected = llama31_8b_local_router.choose_tool(
                "Factor t^2-49.",
                ["calculator", "factor_expression", "expand_expression"],
            )

        self.assertEqual(selected, "factor_expression")
        generator.generate_choice.assert_called_once_with(
            [1, 2, 3],
            ["calculator", "factor_expression", "expand_expression", "hallucinated_tool"],
        )

    def test_qwen36_router_extracts_qwen_tool_call(self) -> None:
        from models.routers.qwen36_local_router import _extract_tool_name

        self.assertEqual(
            _extract_tool_name(
                '<tool_call>{"name": "calculator", "arguments": {}}</tool_call>',
                ["calculator", "github_search"],
            ),
            "calculator",
        )

    def test_other_local_routers_constrain_generation_to_catalog(self) -> None:
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

        expected_choices = [
            "calculator",
            "factor_expression",
            "expand_expression",
            "hallucinated_tool",
        ]
        for router, generator in cases:
            with self.subTest(router=router.ROUTER_ID):
                generator.generate_choice.return_value = "factor_expression"
                with patch.object(router, "_load_generator", return_value=generator):
                    selected = router.choose_tool(
                        "Factor t^2-49.",
                        expected_choices[:-1],
                    )
                self.assertEqual(selected, "factor_expression")
                generator.generate_choice.assert_called_once_with([1], expected_choices)

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
