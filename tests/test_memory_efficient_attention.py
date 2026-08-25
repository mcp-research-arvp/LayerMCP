from __future__ import annotations

import math
import unittest
from unittest.mock import patch

import torch
from torch.nn.attention import sdpa_kernel
from torch.nn.attention.bias import CausalBias

from models.architectures.attention import (
    CUDA_MEMORY_EFFICIENT_BACKENDS,
    causal_scaled_dot_product_attention,
)
from models.architectures.llama31_8b_pytorch.model import AttentionBlock as LlamaAttention
from models.architectures.phi4_pytorch.model import AttentionBlock as PhiAttention
from models.architectures.qwen36_pytorch.model import (
    FullAttnCache,
    ModelConfigs as QwenConfigs,
    Qwen35Attention,
)


def _eager_reference(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    offset: int,
    scale: float,
) -> torch.Tensor:
    groups = query.shape[1] // key.shape[1]
    key = key.repeat_interleave(groups, dim=1)
    value = value.repeat_interleave(groups, dim=1)
    scores = torch.matmul(query.float(), key.float().transpose(-2, -1)) * scale
    query_positions = torch.arange(offset, offset + query.shape[-2])
    key_positions = torch.arange(key.shape[-2])
    allowed = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
    scores = scores.masked_fill(~allowed, float("-inf"))
    probabilities = torch.softmax(scores, dim=-1).to(value.dtype)
    return torch.matmul(probabilities, value)


class MemoryEfficientAttentionTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)

    def test_prefill_matches_eager_causal_gqa(self) -> None:
        query = torch.randn(2, 4, 6, 8)
        key = torch.randn(2, 2, 6, 8)
        value = torch.randn(2, 2, 6, 8)
        scale = 1 / math.sqrt(8)

        actual = causal_scaled_dot_product_attention(
            query,
            key,
            value,
            offset=0,
            scale=scale,
        )
        expected = _eager_reference(query, key, value, offset=0, scale=scale)

        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)

    def test_bfloat16_prefill_matches_fp32_softmax_reference(self) -> None:
        query = torch.randn(1, 8, 16, 16, dtype=torch.bfloat16)
        key = torch.randn(1, 2, 16, 16, dtype=torch.bfloat16)
        value = torch.randn(1, 2, 16, 16, dtype=torch.bfloat16)
        scale = 0.25

        actual = causal_scaled_dot_product_attention(
            query,
            key,
            value,
            offset=0,
            scale=scale,
        )
        expected = _eager_reference(query, key, value, offset=0, scale=scale)

        torch.testing.assert_close(actual, expected, atol=8e-3, rtol=8e-3)

    def test_cached_chunk_matches_lower_right_causal_reference(self) -> None:
        query = torch.randn(1, 6, 3, 4)
        key = torch.randn(1, 2, 8, 4)
        value = torch.randn(1, 2, 8, 4)
        scale = 0.5

        actual = causal_scaled_dot_product_attention(
            query,
            key,
            value,
            offset=5,
            scale=scale,
        )
        expected = _eager_reference(query, key, value, offset=5, scale=scale)

        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)

    def test_cached_chunk_uses_nonmaterialized_lower_right_causal_bias(self) -> None:
        query = torch.randn(1, 8, 3, 4)
        key = torch.randn(1, 2, 8, 4)
        value = torch.randn(1, 2, 8, 4)
        sentinel = torch.empty_like(query)

        with patch(
            "models.architectures.attention.F.scaled_dot_product_attention",
            return_value=sentinel,
        ) as sdpa:
            result = causal_scaled_dot_product_attention(
                query,
                key,
                value,
                offset=5,
                scale=0.5,
            )

        self.assertIs(result, sentinel)
        passed_query, passed_key, passed_value = sdpa.call_args.args
        self.assertIs(passed_query, query)
        self.assertIs(passed_key, key)
        self.assertIs(passed_value, value)
        bias = sdpa.call_args.kwargs["attn_mask"]
        self.assertIsInstance(bias, CausalBias)
        self.assertEqual((bias.seq_len_q, bias.seq_len_kv), (3, 8))
        self.assertFalse(sdpa.call_args.kwargs["is_causal"])

    def test_cached_one_token_decode_keeps_mask_free_behavior(self) -> None:
        query = torch.randn(1, 8, 1, 4)
        key = torch.randn(1, 2, 6, 4)
        value = torch.randn(1, 2, 6, 4)
        sentinel = torch.empty_like(query)

        with patch(
            "models.architectures.attention.F.scaled_dot_product_attention",
            return_value=sentinel,
        ) as sdpa:
            causal_scaled_dot_product_attention(query, key, value, offset=5)

        self.assertIsNone(sdpa.call_args.kwargs["attn_mask"])
        self.assertFalse(sdpa.call_args.kwargs["is_causal"])

    def test_one_token_cached_continuation_matches_full_prefill(self) -> None:
        query = torch.randn(1, 4, 5, 8)
        key = torch.randn(1, 2, 5, 8)
        value = torch.randn(1, 2, 5, 8)
        scale = 1 / math.sqrt(8)

        full = causal_scaled_dot_product_attention(
            query,
            key,
            value,
            offset=0,
            scale=scale,
        )
        cached_last = causal_scaled_dot_product_attention(
            query[:, :, -1:],
            key,
            value,
            offset=4,
            scale=scale,
        )

        torch.testing.assert_close(cached_last, full[:, :, -1:], atol=2e-6, rtol=2e-6)

    def test_sdpa_receives_unexpanded_gqa_tensors(self) -> None:
        query = torch.randn(1, 32, 17, 8)
        key = torch.randn(1, 8, 17, 8)
        value = torch.randn(1, 8, 17, 8)
        sentinel = torch.empty_like(query)

        with patch(
            "models.architectures.attention.F.scaled_dot_product_attention",
            return_value=sentinel,
        ) as sdpa:
            result = causal_scaled_dot_product_attention(
                query,
                key,
                value,
                offset=0,
            )

        self.assertIs(result, sentinel)
        passed_query, passed_key, passed_value = sdpa.call_args.args
        self.assertEqual(passed_query.shape[1], 32)
        self.assertEqual(passed_key.shape[1], 8)
        self.assertEqual(passed_value.shape[1], 8)
        self.assertTrue(sdpa.call_args.kwargs["enable_gqa"])
        self.assertTrue(sdpa.call_args.kwargs["is_causal"])
        self.assertIsNone(sdpa.call_args.kwargs["attn_mask"])

    def test_cuda_backend_allowlist_excludes_quadratic_math_fallback(self) -> None:
        self.assertNotIn(torch.nn.attention.SDPBackend.MATH, CUDA_MEMORY_EFFICIENT_BACKENDS)

    def test_cuda_backend_allowlist_enters_real_sdpa_context(self) -> None:
        with sdpa_kernel(CUDA_MEMORY_EFFICIENT_BACKENDS):
            pass

    def test_invalid_cache_geometry_fails_before_attention(self) -> None:
        with self.assertRaisesRegex(ValueError, "key length must equal"):
            causal_scaled_dot_product_attention(
                torch.randn(1, 4, 2, 8),
                torch.randn(1, 2, 5, 8),
                torch.randn(1, 2, 5, 8),
                offset=2,
            )


class ArchitectureAttentionWiringTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(11)

    def test_llama_attention_output_matches_eager_reference(self) -> None:
        attention = object.__new__(LlamaAttention)
        attention.num_groups = 2
        attention.sm_scale = 0.5
        query = torch.randn(1, 3, 4, 4)
        key = torch.randn(1, 3, 2, 4)
        value = torch.randn(1, 3, 2, 4)

        actual = attention.sdpa(query, key, value, torch.tensor([0]))
        with patch(
            "models.architectures.llama31_8b_pytorch.model."
            "causal_scaled_dot_product_attention",
            side_effect=_eager_reference,
        ):
            expected = attention.sdpa(query, key, value, torch.tensor([0]))

        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)

    def test_llama_passes_unexpanded_kv_and_cache_offset(self) -> None:
        attention = object.__new__(LlamaAttention)
        attention.num_groups = 4
        attention.sm_scale = 0.25
        query = torch.randn(1, 2, 16, 4)
        key = torch.randn(1, 6, 4, 4)
        value = torch.randn(1, 6, 4, 4)

        with patch(
            "models.architectures.llama31_8b_pytorch.model."
            "causal_scaled_dot_product_attention",
            return_value=torch.zeros(1, 16, 2, 4),
        ) as sdpa:
            output = attention.sdpa(query, key, value, torch.tensor([4]))

        self.assertEqual(output.shape, (1, 2, 64))
        passed_query, passed_key, passed_value = sdpa.call_args.args
        self.assertEqual(passed_query.shape, (1, 16, 2, 4))
        self.assertEqual(passed_key.shape, (1, 4, 6, 4))
        self.assertEqual(passed_value.shape, (1, 4, 6, 4))
        self.assertEqual(sdpa.call_args.kwargs["offset"], 4)

    def test_phi_passes_unexpanded_kv_and_cache_offset(self) -> None:
        class Config:
            head_dim = 4
            num_attention_heads = 4
            num_key_value_heads = 2
            hidden_size = 16

        attention = PhiAttention(Config(), device=torch.device("cpu")).float()
        hidden = torch.randn(1, 2, 16)
        cos = torch.ones(2, 4)
        sin = torch.zeros(2, 4)
        cache = unittest.mock.Mock()
        cache.offset.item.return_value = 3
        cache.extend.side_effect = lambda key, value: (
            torch.cat([torch.zeros(1, 3, 2, 4), key], dim=1),
            torch.cat([torch.zeros(1, 3, 2, 4), value], dim=1),
        )

        with patch(
            "models.architectures.phi4_pytorch.model."
            "causal_scaled_dot_product_attention",
            return_value=torch.zeros(1, 4, 2, 4),
        ) as sdpa:
            output = attention(hidden, cos, sin, cache)

        self.assertEqual(output.shape, (1, 2, 16))
        passed_query, passed_key, passed_value = sdpa.call_args.args
        self.assertEqual(passed_query.shape, (1, 4, 2, 4))
        self.assertEqual(passed_key.shape, (1, 2, 5, 4))
        self.assertEqual(passed_value.shape, (1, 2, 5, 4))
        self.assertEqual(sdpa.call_args.kwargs["offset"], 3)

    def test_phi_attention_output_with_rotary_matches_eager_reference(self) -> None:
        class Config:
            head_dim = 4
            num_attention_heads = 4
            num_key_value_heads = 2
            hidden_size = 16

        attention = PhiAttention(Config(), device=torch.device("cpu")).float()
        hidden = torch.randn(1, 3, 16)
        cos = torch.randn(3, 4).cos()
        sin = torch.randn(3, 4).sin()

        actual = attention(hidden, cos, sin)
        with patch(
            "models.architectures.phi4_pytorch.model."
            "causal_scaled_dot_product_attention",
            side_effect=_eager_reference,
        ):
            expected = attention(hidden, cos, sin)

        torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-6)

    def test_qwen_passes_unexpanded_kv_after_rotary_and_cache_update(self) -> None:
        configs = QwenConfigs(
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=4,
            partial_rotary_factor=0.5,
            layer_types=["full_attention"],
        )
        attention = Qwen35Attention(
            configs,
            layer_idx=0,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        hidden = torch.randn(1, 2, 16)
        cos = torch.ones(1, 2, 2)
        sin = torch.zeros(1, 2, 2)
        cache = FullAttnCache(1, 8, 2, 4, torch.device("cpu"), torch.float32)
        cache.offset = 3

        with patch(
            "models.architectures.qwen36_pytorch.model."
            "causal_scaled_dot_product_attention",
            return_value=torch.zeros(1, 4, 2, 4),
        ) as sdpa:
            output = attention(hidden, (cos, sin), cache=cache)

        self.assertEqual(output.shape, (1, 2, 16))
        passed_query, passed_key, passed_value = sdpa.call_args.args
        self.assertEqual(passed_query.shape, (1, 4, 2, 4))
        self.assertEqual(passed_key.shape, (1, 2, 5, 4))
        self.assertEqual(passed_value.shape, (1, 2, 5, 4))
        self.assertEqual(sdpa.call_args.kwargs["offset"], 3)

    def test_qwen_attention_output_with_rotary_matches_eager_reference(self) -> None:
        configs = QwenConfigs(
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=4,
            partial_rotary_factor=0.5,
            layer_types=["full_attention"],
        )
        attention = Qwen35Attention(
            configs,
            layer_idx=0,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        hidden = torch.randn(1, 3, 16)
        cos = torch.randn(1, 3, 2).cos()
        sin = torch.randn(1, 3, 2).sin()

        actual = attention(hidden, (cos, sin))
        with patch(
            "models.architectures.qwen36_pytorch.model."
            "causal_scaled_dot_product_attention",
            side_effect=_eager_reference,
        ):
            expected = attention(hidden, (cos, sin))

        torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-6)


if __name__ == "__main__":
    unittest.main()
