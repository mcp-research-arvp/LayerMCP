from __future__ import annotations

import unittest

import torch
from torch import nn

from research.phase2.gpt_oss_attention_outputs import (
    capture_attention_outputs,
    capture_paired_attention_outputs,
    compare_attention_output_captures,
    representation_measurements,
)


class TinyAttention(nn.Module):
    """Small structural stand-in for GPT-OSS ``attn.qkv`` and ``attn.out``."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.qkv = nn.Linear(width, width * 3)
        self.out = nn.Linear(width, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The first third is a convenient stand-in for flattened Attention x V.
        weighted_value = self.qkv(x)[..., : x.shape[-1]]
        return x + self.out(weighted_value)


class TinyBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.attn = TinyAttention(width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.attn(x)


class TinyGptOssModel(nn.Module):
    def __init__(self, *, fail_on_call: int | None = None) -> None:
        super().__init__()
        self.embedding = nn.Embedding(16, 4)
        self.block = nn.ModuleList([TinyBlock(4), TinyBlock(4)])
        self.unembedding = nn.Linear(4, 16)
        self.fail_on_call = fail_on_call
        self.forward_calls = 0

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        self.forward_calls += 1
        hidden = self.embedding(input_ids)
        for block in self.block:
            hidden = block(hidden)
        if self.forward_calls == self.fail_on_call:
            raise RuntimeError("intentional forward failure")
        return self.unembedding(hidden)


class GptOssAttentionOutputTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.model = TinyGptOssModel()
        self.tokens = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)

    def test_capture_records_weighted_value_and_projected_contribution(self) -> None:
        captured = capture_attention_outputs(
            self.model, self.tokens, layers=(0, 1), token_positions=(1, 3)
        )
        self.assertTrue(captured["enabled"])
        self.assertEqual(captured["input"]["token_count"], 4)
        self.assertEqual(set(captured["layers"]), {0, 1})
        for layer in (0, 1):
            weighted_value = captured["layers"][layer]["weighted_value"]
            projected = captured["layers"][layer]["projected_contribution"]
            self.assertEqual(tuple(weighted_value.shape), (1, 2, 4))
            self.assertEqual(tuple(projected.shape), (1, 2, 4))
            expected = self.model.block[layer].attn.out(weighted_value.to(dtype=torch.float32))
            self.assertTrue(torch.allclose(projected, expected, atol=1e-6, rtol=1e-6))
        self.assertEqual(captured["captured_token_positions"], [1, 3])

    def test_disabled_recording_has_no_hooks_and_keeps_logits(self) -> None:
        baseline = self.model(self.tokens).detach().clone()
        captured = capture_attention_outputs(self.model, self.tokens, layers=(0,), enabled=False)
        self.assertFalse(captured["enabled"])
        self.assertEqual(captured["layers"], {})
        self.assertEqual(captured["model_output_shape"], list(baseline.shape))
        self.assertTrue(torch.equal(baseline, self.model(self.tokens)))
        self.assertEqual(len(self.model.block[0].attn.out._forward_hooks), 0)
        self.assertEqual(len(self.model.block[0].attn.out._forward_pre_hooks), 0)

    def test_zero_strength_is_same_fixed_input_and_same_outputs(self) -> None:
        paired = capture_paired_attention_outputs(
            self.model, self.tokens, observed_layers=(0, 1), intervention_layer=0,
            target="qkv", strength=0.0, strength_mode="tensor_rms", seed=1234,
        )
        self.assertTrue(paired["intervention"]["restored_exactly"])
        self.assertEqual(paired["control"]["input"], paired["altered"]["input"])
        self.assertEqual(paired["intervention"]["perturbation"]["status"], "no_op")
        for layer in (0, 1):
            for kind, measurement in paired["measurements"][layer].items():
                self.assertEqual(measurement["normalized_frobenius_distance"], 0.0, kind)
                self.assertAlmostEqual(measurement["cosine_similarity"], 1.0, places=12)

    def test_nonzero_intervention_changes_capture_and_restores_exactly(self) -> None:
        before = self.model.block[0].attn.qkv.weight.detach().clone()
        paired = capture_paired_attention_outputs(
            self.model, self.tokens, observed_layers=(0,), intervention_layer=0,
            target="qkv", strength=0.1, strength_mode="tensor_rms", seed=1234,
        )
        self.assertTrue(paired["intervention"]["restored_exactly"])
        self.assertEqual(paired["intervention"]["perturbation"]["status"], "changed")
        self.assertTrue(torch.equal(before, self.model.block[0].attn.qkv.weight))
        self.assertGreater(
            paired["measurements"][0]["weighted_value"]["normalized_frobenius_distance"], 0.0
        )

    def test_hooks_and_intervention_restore_after_forward_error(self) -> None:
        # The control pass succeeds; the altered pass fails after both hooks run.
        self.model.fail_on_call = 2
        before = self.model.block[0].attn.qkv.weight.detach().clone()
        with self.assertRaisesRegex(RuntimeError, "intentional forward failure"):
            capture_paired_attention_outputs(
                self.model, self.tokens, observed_layers=(0,), intervention_layer=0,
                target="qkv", strength=0.1, strength_mode="tensor_rms",
            )
        self.assertTrue(torch.equal(before, self.model.block[0].attn.qkv.weight))
        self.assertEqual(len(self.model.block[0].attn.out._forward_hooks), 0)
        self.assertEqual(len(self.model.block[0].attn.out._forward_pre_hooks), 0)

    def test_measurements_reject_different_input_identity_and_shapes(self) -> None:
        control = capture_attention_outputs(self.model, self.tokens, layers=(0,))
        other = capture_attention_outputs(self.model, torch.tensor([[1, 2, 3, 5]]), layers=(0,))
        with self.assertRaisesRegex(ValueError, "identical input token IDs"):
            compare_attention_output_captures(control, other)
        with self.assertRaisesRegex(ValueError, "capture shape differs"):
            representation_measurements(torch.zeros((1, 2, 4)), torch.zeros((1, 3, 4)))


if __name__ == "__main__":
    unittest.main()
