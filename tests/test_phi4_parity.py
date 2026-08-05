from __future__ import annotations

import math
import os
from pathlib import Path
import unittest
from unittest.mock import patch

import torch

from models.architectures.phi4_pytorch.config import (
    CHECKPOINT_ENV_VAR,
    DEFAULT_CHECKPOINT_PATH,
)
from models.architectures.phi4_pytorch.model import (
    Cache,
    Transformer,
    apply_rotary_pos_emb,
    repeat_kv,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PHI_PRODUCTION_FILES = (
    PROJECT_ROOT / "models" / "architectures" / "phi4_pytorch" / "inference.py",
    PROJECT_ROOT / "models" / "architectures" / "phi4_pytorch" / "model.py",
    PROJECT_ROOT / "models" / "routers" / "phi4_local_router.py",
)


class PhiProductionPathTests(unittest.TestCase):
    def test_hf_causal_lm_is_not_used_by_phi_production_path(self) -> None:
        for path in PHI_PRODUCTION_FILES:
            with self.subTest(path=path):
                self.assertNotIn("AutoModelForCausalLM", path.read_text())


@unittest.skipUnless(
    os.environ.get("LAYERMCP_RUN_PHI4_PARITY") == "1",
    "set LAYERMCP_RUN_PHI4_PARITY=1 to run local-checkpoint parity",
)
class PhiCheckpointParityTests(unittest.TestCase):
    """H100 parity ladder; Hugging Face is an oracle, never production."""

    @classmethod
    def setUpClass(cls) -> None:
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is required for Phi-4 checkpoint parity")

        from transformers import AutoModelForCausalLM, AutoTokenizer

        cls.checkpoint = Path(
            os.environ.get(CHECKPOINT_ENV_VAR, str(DEFAULT_CHECKPOINT_PATH))
        )
        if not cls.checkpoint.exists():
            raise unittest.SkipTest(f"Phi-4 checkpoint not found: {cls.checkpoint}")

        cls.device = torch.device("cuda")
        cls.hf_tokenizer = AutoTokenizer.from_pretrained(
            cls.checkpoint,
            trust_remote_code=True,
            local_files_only=True,
        )
        cls.custom_tokenizer = AutoTokenizer.from_pretrained(
            cls.checkpoint,
            trust_remote_code=True,
            local_files_only=True,
        )
        cls.prompt = "The capital of France is"
        cls.input_ids = cls.hf_tokenizer(
            cls.prompt,
            return_tensors="pt",
        )["input_ids"].to(cls.device)
        cls.hf_model = AutoModelForCausalLM.from_pretrained(
            cls.checkpoint,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
            local_files_only=True,
        ).to(cls.device).eval()
        cls.custom_model = Transformer.from_checkpoint(
            str(cls.checkpoint),
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

    def _layer_zero_projections(self):
        ids = self.input_ids
        custom_layer = self.custom_model.layers[0]
        hf_layer = self.hf_model.model.layers[0]

        custom_embedding = self.custom_model.embed_tokens(ids)
        hf_embedding = self.hf_model.model.embed_tokens(ids)
        custom_norm = custom_layer.input_layernorm(custom_embedding)
        hf_norm = hf_layer.input_layernorm(hf_embedding)
        custom_qkv = custom_layer.self_attn.qkv_proj(custom_norm)
        hf_qkv = hf_layer.self_attn.qkv_proj(hf_norm)

        configs = self.custom_model.configs
        query_size = configs.num_attention_heads * configs.head_dim
        kv_size = configs.num_key_value_heads * configs.head_dim

        def split(qkv):
            batch_size, seq_len, _ = qkv.shape
            query = qkv[..., :query_size].view(
                batch_size,
                seq_len,
                configs.num_attention_heads,
                configs.head_dim,
            )
            key = qkv[..., query_size : query_size + kv_size].view(
                batch_size,
                seq_len,
                configs.num_key_value_heads,
                configs.head_dim,
            )
            value = qkv[..., query_size + kv_size :].view(
                batch_size,
                seq_len,
                configs.num_key_value_heads,
                configs.head_dim,
            )
            return query, key, value

        return {
            "custom_layer": custom_layer,
            "hf_layer": hf_layer,
            "custom_embedding": custom_embedding,
            "hf_embedding": hf_embedding,
            "custom_norm": custom_norm,
            "hf_norm": hf_norm,
            "custom_qkv": custom_qkv,
            "hf_qkv": hf_qkv,
            "custom_split": split(custom_qkv),
            "hf_split": split(hf_qkv),
        }

    def test_tokenizer_input_ids_match(self) -> None:
        self.assertEqual(
            self.custom_tokenizer.encode(self.prompt),
            self.hf_tokenizer.encode(self.prompt),
        )

    def test_embeddings_norm_qkv_and_rope_match(self) -> None:
        from transformers.models.phi3.modeling_phi3 import (
            apply_rotary_pos_emb as hf_apply_rotary_pos_emb,
        )

        with torch.inference_mode():
            values = self._layer_zero_projections()
            torch.testing.assert_close(
                values["custom_embedding"],
                values["hf_embedding"],
                atol=0,
                rtol=0,
            )
            torch.testing.assert_close(
                values["custom_norm"],
                values["hf_norm"],
                atol=1e-2,
                rtol=1e-2,
            )
            torch.testing.assert_close(
                values["custom_qkv"],
                values["hf_qkv"],
                atol=5e-2,
                rtol=1e-2,
            )

            custom_q, custom_k, custom_v = values["custom_split"]
            hf_q, hf_k, hf_v = values["hf_split"]
            self.assertEqual(custom_q.shape, hf_q.shape)
            self.assertEqual(custom_k.shape, hf_k.shape)
            self.assertEqual(custom_v.shape, hf_v.shape)
            torch.testing.assert_close(custom_q, hf_q, atol=5e-2, rtol=1e-2)
            torch.testing.assert_close(custom_k, hf_k, atol=5e-2, rtol=1e-2)
            torch.testing.assert_close(custom_v, hf_v, atol=5e-2, rtol=1e-2)

            seq_len = self.input_ids.shape[1]
            positions = torch.arange(seq_len, device=self.device)
            custom_cos, custom_sin = self.custom_model.rotary(positions)
            custom_q_rope, custom_k_rope = apply_rotary_pos_emb(
                custom_q,
                custom_k,
                custom_cos,
                custom_sin,
            )

            position_ids = positions.unsqueeze(0)
            hf_cos, hf_sin = self.hf_model.model.rotary_emb(
                values["hf_embedding"],
                position_ids,
            )
            hf_q_rope, hf_k_rope = hf_apply_rotary_pos_emb(
                hf_q.transpose(1, 2),
                hf_k.transpose(1, 2),
                hf_cos,
                hf_sin,
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

    def test_attention_mlp_and_layer_zero_residual_match(self) -> None:
        with torch.inference_mode():
            values = self._layer_zero_projections()
            custom_layer = values["custom_layer"]
            hf_layer = values["hf_layer"]
            custom_q, custom_k, custom_v = values["custom_split"]
            hf_q, hf_k, hf_v = values["hf_split"]

            seq_len = self.input_ids.shape[1]
            positions = torch.arange(seq_len, device=self.device)
            custom_cos, custom_sin = self.custom_model.rotary(positions)
            custom_q, custom_k = apply_rotary_pos_emb(
                custom_q,
                custom_k,
                custom_cos,
                custom_sin,
            )

            position_ids = positions.unsqueeze(0)
            hf_cos, hf_sin = self.hf_model.model.rotary_emb(
                values["hf_embedding"],
                position_ids,
            )
            from transformers.models.phi3.modeling_phi3 import (
                apply_rotary_pos_emb as hf_apply_rotary_pos_emb,
            )

            hf_q, hf_k = hf_apply_rotary_pos_emb(
                hf_q.transpose(1, 2),
                hf_k.transpose(1, 2),
                hf_cos,
                hf_sin,
            )

            groups = self.custom_model.configs.num_attention_heads // (
                self.custom_model.configs.num_key_value_heads
            )
            custom_q_heads = custom_q.transpose(1, 2)
            custom_k_heads = repeat_kv(custom_k.transpose(1, 2), groups)
            custom_v_heads = repeat_kv(custom_v.transpose(1, 2), groups)
            hf_k_heads = repeat_kv(hf_k, groups)
            hf_v_heads = repeat_kv(hf_v.transpose(1, 2), groups)
            scaling = 1.0 / math.sqrt(self.custom_model.configs.head_dim)

            # HF performs the QK matmul in BF16. The custom path currently
            # promotes Q/K to FP32 first, so a small BF16 tolerance is expected.
            custom_scores = (
                torch.matmul(
                    custom_q_heads.float(),
                    custom_k_heads.float().transpose(2, 3),
                )
                * scaling
            )
            hf_scores = torch.matmul(hf_q, hf_k_heads.transpose(2, 3)) * scaling
            torch.testing.assert_close(
                custom_scores,
                hf_scores.float(),
                atol=1e-1,
                rtol=2e-2,
            )

            causal_mask = torch.full(
                (seq_len, seq_len),
                float("-inf"),
                device=self.device,
                dtype=torch.float32,
            )
            causal_mask = torch.triu(causal_mask, diagonal=1)
            custom_probs = torch.softmax(
                custom_scores + causal_mask[None, None],
                dim=-1,
            ).to(custom_v_heads.dtype)
            hf_probs = torch.softmax(
                hf_scores.float() + causal_mask[None, None],
                dim=-1,
            ).to(hf_v_heads.dtype)
            custom_attention = torch.matmul(custom_probs, custom_v_heads)
            hf_attention = torch.matmul(hf_probs, hf_v_heads)
            custom_attention = custom_attention.transpose(1, 2).reshape(
                1,
                seq_len,
                -1,
            )
            hf_attention = hf_attention.transpose(1, 2).reshape(
                1,
                seq_len,
                -1,
            )
            custom_attention = custom_layer.self_attn.o_proj(custom_attention)
            hf_attention = hf_layer.self_attn.o_proj(hf_attention)
            torch.testing.assert_close(
                custom_attention,
                hf_attention,
                atol=2e-1,
                rtol=2e-2,
            )

            custom_post_attention = values["custom_embedding"] + custom_attention
            hf_post_attention = values["hf_embedding"] + hf_attention
            torch.testing.assert_close(
                custom_post_attention,
                hf_post_attention,
                atol=2e-1,
                rtol=2e-2,
            )
            custom_post_norm = custom_layer.post_attention_layernorm(
                custom_post_attention
            )
            hf_post_norm = hf_layer.post_attention_layernorm(hf_post_attention)
            torch.testing.assert_close(
                custom_post_norm,
                hf_post_norm,
                atol=2e-1,
                rtol=2e-2,
            )
            custom_mlp = custom_layer.mlp(custom_post_norm)
            hf_mlp = hf_layer.mlp(hf_post_norm)
            torch.testing.assert_close(
                custom_mlp,
                hf_mlp,
                atol=3e-1,
                rtol=3e-2,
            )
            custom_layer_output = custom_post_attention + custom_mlp
            hf_layer_output = hf_post_attention + hf_mlp
            torch.testing.assert_close(
                custom_layer_output,
                hf_layer_output,
                atol=3e-1,
                rtol=3e-2,
            )

    def test_final_logits_and_greedy_next_token_match(self) -> None:
        with torch.inference_mode():
            custom_logits = self.custom_model(self.input_ids)[:, -1, :]
            hf_logits = self.hf_model(
                input_ids=self.input_ids,
                use_cache=False,
                return_dict=True,
            ).logits[:, -1, :].float()

        # Forty BF16 layers can accumulate small eager-kernel differences;
        # exact greedy-token agreement is the stronger routing invariant.
        torch.testing.assert_close(
            custom_logits,
            hf_logits,
            atol=1.0,
            rtol=1e-1,
        )
        self.assertEqual(
            int(custom_logits.argmax(dim=-1).item()),
            int(hf_logits.argmax(dim=-1).item()),
        )

    def test_custom_cache_matches_full_prefix_logits(self) -> None:
        configs = self.custom_model.configs
        caches = [
            Cache(
                batch_size=1,
                n_ctx=self.input_ids.shape[1],
                n_kv_heads=configs.num_key_value_heads,
                d_head=configs.head_dim,
                device=self.device,
            )
            for _ in range(configs.num_hidden_layers)
        ]

        with torch.inference_mode():
            full_logits = self.custom_model(self.input_ids)[:, -1, :]
            self.custom_model(self.input_ids[:, :-1], caches=caches)
            cached_logits = self.custom_model(
                self.input_ids[:, -1:],
                caches=caches,
            )[:, -1, :]

        torch.testing.assert_close(
            cached_logits,
            full_logits,
            atol=2e-1,
            rtol=2e-2,
        )
        self.assertEqual(
            int(cached_logits.argmax(dim=-1).item()),
            int(full_logits.argmax(dim=-1).item()),
        )

    def test_custom_runtime_router_smoke(self) -> None:
        from models.architectures.phi4_pytorch.inference import TokenGenerator
        from models.routers import phi4_local_router

        old_model = TokenGenerator._model
        old_tokenizer = TokenGenerator._tokenizer
        TokenGenerator._model = self.custom_model
        TokenGenerator._tokenizer = self.custom_tokenizer
        try:
            generator = TokenGenerator(
                checkpoint=str(self.checkpoint),
                device=self.device,
            )
            schema = {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            }
            with patch.object(
                phi4_local_router,
                "_load_generator",
                return_value=generator,
            ):
                prediction = phi4_local_router.choose_tool_call(
                    "Compute 2+2.",
                    ["calculator"],
                    {"calculator": schema},
                    {"calculator": "Evaluate an arithmetic expression."},
                )
        finally:
            TokenGenerator._model = old_model
            TokenGenerator._tokenizer = old_tokenizer

        self.assertEqual(
            prediction.selected_tool,
            "calculator",
            prediction.raw_output,
        )
        expression = prediction.selected_args.get("expression")
        self.assertIsInstance(expression, str, prediction.raw_output)
        self.assertTrue(expression.strip(), prediction.raw_output)


if __name__ == "__main__":
    unittest.main()
