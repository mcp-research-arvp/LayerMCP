from __future__ import annotations

import os
from pathlib import Path
import unittest
from unittest.mock import patch

import torch

from models.architectures.llama31_8b_pytorch.config import (
    CHECKPOINT_ENV_VAR,
    DEFAULT_CHECKPOINT_PATH,
)
from models.architectures.llama31_8b_pytorch.model import RotaryEmbedding, Transformer


class LlamaRotaryConventionTests(unittest.TestCase):
    def test_rotary_uses_hugging_face_split_half_layout(self) -> None:
        rope = RotaryEmbedding(
            head_dim=4,
            base=10_000.0,
            max_position_embeddings=2,
            rope_scaling=None,
            device=torch.device("cpu"),
        )
        values = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]])
        cos = torch.tensor([[[[0.5, 0.25]]]])
        sin = torch.tensor([[[[0.75, 1.0]]]])

        actual = rope._apply_rotary(values, cos, sin)
        expected = torch.tensor([[[[-1.75, -3.5, 2.25, 3.0]]]])

        torch.testing.assert_close(actual, expected)


@unittest.skipUnless(
    os.environ.get("LAYERMCP_RUN_LLAMA_PARITY") == "1",
    "set LAYERMCP_RUN_LLAMA_PARITY=1 to run local-checkpoint parity",
)
class LlamaCheckpointParityTests(unittest.TestCase):
    """H100 parity ladder; Hugging Face is an oracle, never production."""

    @classmethod
    def setUpClass(cls) -> None:
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is required for Llama checkpoint parity")

        from transformers import AutoModelForCausalLM, AutoTokenizer

        checkpoint = Path(
            os.environ.get(CHECKPOINT_ENV_VAR, str(DEFAULT_CHECKPOINT_PATH))
        )
        if not checkpoint.exists():
            raise unittest.SkipTest(f"Llama checkpoint not found: {checkpoint}")

        cls.device = torch.device("cuda")
        cls.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        cls.input_ids = cls.tokenizer(
            "The capital of France is",
            return_tensors="pt",
        )["input_ids"].to(cls.device)
        cls.hf_model = AutoModelForCausalLM.from_pretrained(
            checkpoint,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
        ).to(cls.device).eval()
        cls.custom_model = Transformer.from_checkpoint(
            str(checkpoint),
            device=cls.device,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        if hasattr(cls, "hf_model"):
            del cls.hf_model
        if hasattr(cls, "custom_model"):
            del cls.custom_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def test_layer_zero_parity_ladder(self) -> None:
        from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

        ids = self.input_ids
        custom_layer = self.custom_model.layers[0]
        hf_layer = self.hf_model.model.layers[0]

        with torch.inference_mode():
            custom_embedding = self.custom_model.embed_tokens(ids)
            hf_embedding = self.hf_model.model.embed_tokens(ids)
            torch.testing.assert_close(custom_embedding, hf_embedding, atol=0, rtol=0)

            custom_norm = custom_layer.input_layernorm(custom_embedding)
            hf_norm = hf_layer.input_layernorm(hf_embedding)
            torch.testing.assert_close(custom_norm, hf_norm, atol=1e-2, rtol=1e-2)

            batch_size, seq_len, _ = custom_norm.shape
            configs = self.custom_model.configs
            custom_q = custom_layer.self_attn.q_proj(custom_norm).view(
                batch_size,
                seq_len,
                configs.num_attention_heads,
                configs.head_dim,
            )
            custom_k = custom_layer.self_attn.k_proj(custom_norm).view(
                batch_size,
                seq_len,
                configs.num_key_value_heads,
                configs.head_dim,
            )
            custom_v = custom_layer.self_attn.v_proj(custom_norm).view(
                batch_size,
                seq_len,
                configs.num_key_value_heads,
                configs.head_dim,
            )
            hf_q = hf_layer.self_attn.q_proj(hf_norm).view_as(custom_q)
            hf_k = hf_layer.self_attn.k_proj(hf_norm).view_as(custom_k)
            hf_v = hf_layer.self_attn.v_proj(hf_norm).view_as(custom_v)
            torch.testing.assert_close(custom_q, hf_q, atol=5e-2, rtol=1e-2)
            torch.testing.assert_close(custom_k, hf_k, atol=5e-2, rtol=1e-2)
            torch.testing.assert_close(custom_v, hf_v, atol=5e-2, rtol=1e-2)

            position_ids = torch.arange(seq_len, device=self.device).unsqueeze(0)
            cos, sin = self.hf_model.model.rotary_emb(hf_embedding, position_ids)
            hf_q_rope, hf_k_rope = apply_rotary_pos_emb(
                hf_q.transpose(1, 2),
                hf_k.transpose(1, 2),
                cos,
                sin,
            )
            zero_offset = torch.zeros(1, dtype=torch.long, device=self.device)
            custom_q_rope, custom_k_rope = custom_layer.self_attn.rope(
                custom_q,
                custom_k,
                zero_offset,
            )
            torch.testing.assert_close(
                custom_q_rope.transpose(1, 2),
                hf_q_rope,
                atol=5e-2,
                rtol=1e-2,
            )
            torch.testing.assert_close(
                custom_k_rope.transpose(1, 2),
                hf_k_rope,
                atol=5e-2,
                rtol=1e-2,
            )

            custom_layer_zero = custom_layer(custom_embedding)
            hf_outputs = self.hf_model.model(
                input_ids=ids,
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
            )
            hf_layer_zero = hf_outputs.hidden_states[1]
            torch.testing.assert_close(
                custom_layer_zero,
                hf_layer_zero,
                atol=2e-1,
                rtol=2e-2,
            )

    def test_final_logits_and_greedy_next_token_parity(self) -> None:
        with torch.inference_mode():
            custom_logits = self.custom_model(self.input_ids)[:, -1, :]
            hf_logits = self.hf_model(
                input_ids=self.input_ids,
                use_cache=False,
                return_dict=True,
            ).logits[:, -1, :].float()

        torch.testing.assert_close(
            custom_logits,
            hf_logits,
            atol=5e-1,
            rtol=5e-2,
        )
        self.assertEqual(
            int(custom_logits.argmax(dim=-1).item()),
            int(hf_logits.argmax(dim=-1).item()),
        )

    def test_custom_runtime_router_smoke(self) -> None:
        from models.architectures.llama31_8b_pytorch.inference import TokenGenerator
        from models.routers import llama31_8b_local_router

        generator = TokenGenerator.__new__(TokenGenerator)
        generator.checkpoint = str(
            os.environ.get(CHECKPOINT_ENV_VAR, str(DEFAULT_CHECKPOINT_PATH))
        )
        generator.device = self.device
        generator.model = self.custom_model
        generator.tokenizer = self.tokenizer
        generator.eos_token_id = self.tokenizer.eos_token_id
        generator.stop_tokens = generator._resolve_stop_tokens()

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
                {"calculator": "Evaluate an arithmetic expression."},
            )

        self.assertEqual(prediction.selected_tool, "calculator")
        self.assertEqual(prediction.selected_args.get("expression"), "2+2")


if __name__ == "__main__":
    unittest.main()
