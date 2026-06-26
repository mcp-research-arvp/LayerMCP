from __future__ import annotations

import argparse
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
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
        from models import qwen_router

        fake_components = LoadedModelComponents(
            tokenizer=object(),
            model=object(),
            model_name="fake/model",
        )

        qwen_router._load_model_components.cache_clear()
        try:
            with patch(
                "models.qwen_router.load_model_components",
                return_value=fake_components,
            ) as load_model:
                tokenizer, model = qwen_router._load_model_components()
                cached_tokenizer, cached_model = qwen_router._load_model_components()

            load_model.assert_called_once_with(qwen_router.MODEL_NAME)
            self.assertIs(tokenizer, fake_components.tokenizer)
            self.assertIs(model, fake_components.model)
            self.assertIs(cached_tokenizer, tokenizer)
            self.assertIs(cached_model, model)
        finally:
            qwen_router._load_model_components.cache_clear()

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


if __name__ == "__main__":
    unittest.main()
