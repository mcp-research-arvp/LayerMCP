from __future__ import annotations

import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from models.architectures.gpt_oss_pytorch.inference import TokenGenerator, Transformer
from models.routers import gpt_oss_local_router as router
from research.phase2.experiment_integrity import checkpoint_identity, native_generator_identity


class GptOssCacheIntegrityTests(unittest.TestCase):
    def test_added_shard_is_rejected_by_cached_public_loader(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text("{}")
            (root / "model.safetensors").write_bytes(b"weights")
            with patch.object(Transformer, "from_checkpoint", return_value=torch.nn.Linear(2, 2)), \
                 patch("models.architectures.gpt_oss_pytorch.inference.get_tokenizer", return_value=SimpleNamespace(encode=lambda *args, **kwargs: [1])):
                router.load_generator(str(root))
                (root / "extra.safetensors").write_bytes(b"extra")
                with self.assertRaisesRegex(RuntimeError, "input files changed"):
                    router.load_generator(str(root))

    def test_native_notebook_provenance_hashes_once_and_reaches_each_pair(self):
        from tests.test_phase2_gpt_oss_intervention_eval import (
            TinyGptOssModel, VALID_CALL, _catalog, _write_saved_run, _session_for_catalog,
        )
        from research.phase2.gpt_oss_intervention_eval import evaluate_one_saved_example

        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text("{}")
            (root / "model.safetensors").write_bytes(b"test-weights")
            (root / "o200k_base.tiktoken").write_bytes(b"tokenizer")
            catalog = _catalog()
            source = _write_saved_run(root, catalog, ["one"])
            tokenizer = SimpleNamespace(encode=lambda *args, **kwargs: [1], stop_tokens_for_assistant_actions=lambda: [7])
            with patch.dict(os.environ, {"TIKTOKEN_ENCODINGS_BASE": str(root)}), \
                 patch.object(Transformer, "from_checkpoint", return_value=TinyGptOssModel()), \
                 patch("models.architectures.gpt_oss_pytorch.inference.get_tokenizer", return_value=tokenizer):
                generator = router.load_generator(str(root))
                with patch.object(generator, "render_tool_prompt", return_value="query"), \
                     patch.object(generator, "generate_text", return_value=SimpleNamespace(text=VALID_CALL)), \
                     patch("research.phase2.experiment_integrity.checkpoint_identity", wraps=checkpoint_identity) as hasher:
                    pairs = [evaluate_one_saved_example(
                        generator=generator, source_run_dir=source, sample_id="sample-0",
                        layer_index=0, seed=seed, method="noise", target="attention_all", strength=.01,
                        session_factory=lambda: _session_for_catalog(catalog),
                    ) for seed in (1, 2)]
                self.assertEqual(hasher.call_count, 1)
                self.assertEqual(pairs[0]["checkpoint_fingerprint"], pairs[1]["checkpoint_fingerprint"])
                self.assertIsNotNone(pairs[0]["checkpoint_fingerprint"])
                self.assertEqual(pairs[0]["model_binding"], "native_loaded_inputs")
                self.assertEqual(pairs[0]["source_checkpoint_equivalence"], "unverifiable")
                self.assertEqual(native_generator_identity(object())["status"], "unverified")

    def setUp(self):
        router._load_generator.cache_clear()
        self.patches = [patch.object(TokenGenerator, name, None) for name in (
            "_model", "_tokenizer", "_checkpoint_path", "_checkpoint_signatures", "_tokenizer_asset_root",
        )]
        for item in self.patches:
            item.start()

    def tearDown(self):
        router._load_generator.cache_clear()
        for item in reversed(self.patches):
            item.stop()

    def test_reuse_same_checkpoint_but_reject_changed_file_path_or_tokenizer(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            first, second = root / "first", root / "second"
            for checkpoint in (first, second):
                checkpoint.mkdir()
                (checkpoint / "config.json").write_text("{}")
                (checkpoint / "model.safetensors").write_bytes(b"test-weights")
            tokenizer = SimpleNamespace(encode=lambda *args, **kwargs: [1])
            with patch.dict(os.environ, {"TIKTOKEN_ENCODINGS_BASE": str(root)}), \
                 patch.object(Transformer, "from_checkpoint", return_value=torch.nn.Linear(2, 2)) as loader, \
                 patch("models.architectures.gpt_oss_pytorch.inference.get_tokenizer", return_value=tokenizer):
                generator = router.load_generator(str(first))
                self.assertIs(router.load_generator(str(first)), generator)
                self.assertEqual(loader.call_count, 1)
                with patch.dict(os.environ, {router.CHECKPOINT_ENV_VAR: str(first)}):
                    router.load_generator()
                    with patch.dict(os.environ, {router.CHECKPOINT_ENV_VAR: str(second)}):
                        with self.assertRaisesRegex(RuntimeError, "checkpoint selection changed"):
                            router.load_generator()
                with self.assertRaisesRegex(RuntimeError, "different/unknown"):
                    router.load_generator(str(second))
                with patch.dict(os.environ, {"TIKTOKEN_ENCODINGS_BASE": str(second)}):
                    with self.assertRaisesRegex(RuntimeError, "tokenizer assets changed"):
                        router.load_generator(str(first))
                (first / "model.safetensors").write_bytes(b"changed-weights")
                with self.assertRaisesRegex(RuntimeError, "input files changed"):
                    router.load_generator(str(first))


if __name__ == "__main__":
    unittest.main()
