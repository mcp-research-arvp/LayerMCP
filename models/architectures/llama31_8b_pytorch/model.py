import math
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.architectures.llama31_8b_pytorch.weights import Checkpoint


# Architecture adapted from Meta's official Llama implementation.
# https://github.com/meta-llama/llama3


@dataclass
class ModelConfigs:
    # Architecture
    num_hidden_layers: int = 32
    hidden_size: int = 4096
    intermediate_size: int = 14336

    # Attention
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128

    # Vocabulary
    vocab_size: int = 128256

    # RoPE
    rope_theta: float = 500000.0
    max_position_embeddings: int = 131072
    rope_scaling: dict[str, Any] | None = field(
        default_factory=lambda: {
            "factor": 8.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_max_position_embeddings": 8192,
            "rope_type": "llama3",
        }
    )

    # RMSNorm / MLP
    rms_norm_eps: float = 1e-5
    norm_eps: float = 1e-5
    hidden_act: str = "silu"

    # Misc
    attention_dropout: float = 0.0
    attention_bias: bool = False
    mlp_bias: bool = False
    tie_word_embeddings: bool = False


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float, device: torch.device | None = None):
        super().__init__()
        self.eps = eps
        self.hidden_size = hidden_size
        self.weight = nn.Parameter(torch.ones(hidden_size, device=device, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == self.hidden_size
        dtype = x.dtype
        t = x.float()
        t = t * torch.rsqrt(torch.mean(t**2, dim=-1, keepdim=True) + self.eps)
        return (t * self.weight).to(dtype)


class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        head_dim: int,
        base: float,
        max_position_embeddings: int,
        rope_scaling: dict[str, Any] | None = None,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.base = base
        self.max_position_embeddings = max_position_embeddings
        self.rope_scaling = rope_scaling
        self.device = device

        inv_freq = self._compute_inv_freq()
        positions = torch.arange(max_position_embeddings, dtype=torch.float32, device=device)
        freqs = torch.einsum("i,j->ij", positions, inv_freq)
        self.register_buffer("cos", freqs.cos(), persistent=False)
        self.register_buffer("sin", freqs.sin(), persistent=False)

    def _compute_inv_freq(self) -> torch.Tensor:
        dim_ids = torch.arange(0, self.head_dim, 2, dtype=torch.float32, device=self.device)
        inv_freq = 1.0 / (self.base ** (dim_ids / self.head_dim))

        scaling = self.rope_scaling
        if not scaling:
            return inv_freq

        rope_type = scaling.get("rope_type", scaling.get("type"))
        if rope_type != "llama3":
            factor = float(scaling.get("factor", 1.0))
            return inv_freq / factor

        factor = float(scaling["factor"])
        low_freq_factor = float(scaling["low_freq_factor"])
        high_freq_factor = float(scaling["high_freq_factor"])
        old_context_len = float(scaling["original_max_position_embeddings"])

        wavelen = 2 * math.pi / inv_freq
        low_freq_wavelen = old_context_len / low_freq_factor
        high_freq_wavelen = old_context_len / high_freq_factor

        inv_freq_llama = torch.where(wavelen > low_freq_wavelen, inv_freq / factor, inv_freq)
        smooth = (old_context_len / wavelen - low_freq_factor) / (
            high_freq_factor - low_freq_factor
        )
        smoothed = (1 - smooth) * inv_freq_llama / factor + smooth * inv_freq_llama
        is_medium = (wavelen <= low_freq_wavelen) & (wavelen >= high_freq_wavelen)
        return torch.where(is_medium, smoothed, inv_freq_llama)

    def _apply_rotary(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        # Hugging Face Llama checkpoints use split-half rotary channels. Meta's
        # interleaved complex convention requires permuted Q/K weights; this
        # loader consumes HF projection tensors unchanged, so it must match the
        # HF rotate_half convention here.
        x_first, x_second = x.chunk(2, dim=-1)
        return torch.cat(
            (
                x_first * cos - x_second * sin,
                x_second * cos + x_first * sin,
            ),
            dim=-1,
        )

    def forward(
        self, query: torch.Tensor, key: torch.Tensor, offset: torch.LongTensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _, num_tokens, _, _ = query.shape
        start = int(offset.item())
        idx = torch.arange(num_tokens, device=query.device, dtype=torch.long) + start
        cos = self.cos.index_select(0, idx).to(query.dtype)
        sin = self.sin.index_select(0, idx).to(query.dtype)
        cos = cos.unsqueeze(0).unsqueeze(2)
        sin = sin.unsqueeze(0).unsqueeze(2)
        return self._apply_rotary(query, cos, sin), self._apply_rotary(key, cos, sin)


class Cache:
    def __init__(
        self,
        batch_size: int,
        n_ctx: int,
        n_kv_heads: int,
        d_head: int,
        device: torch.device | None = None,
    ):
        self.k = torch.zeros((batch_size, n_ctx, n_kv_heads, d_head), dtype=torch.bfloat16, device=device)
        self.v = torch.zeros((batch_size, n_ctx, n_kv_heads, d_head), dtype=torch.bfloat16, device=device)
        self.offset = torch.zeros((1,), dtype=torch.long, device=device)

    def reset(self):
        self.k.zero_()
        self.v.zero_()
        self.offset.zero_()

    def extend(self, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        n_new = k.shape[1]
        start = int(self.offset.item())
        end = start + n_new
        self.k[:, start:end] = k
        self.v[:, start:end] = v
        self.offset += n_new
        return self.k[:, :end], self.v[:, :end]


class AttentionBlock(nn.Module):
    def __init__(self, configs: ModelConfigs, device: torch.device | None = None):
        super().__init__()
        self.head_dim = configs.head_dim
        self.num_attention_heads = configs.num_attention_heads
        self.num_key_value_heads = configs.num_key_value_heads
        self.num_groups = self.num_attention_heads // self.num_key_value_heads
        self.sm_scale = 1 / math.sqrt(configs.head_dim)

        self.q_proj = nn.Linear(
            configs.hidden_size,
            configs.num_attention_heads * configs.head_dim,
            bias=configs.attention_bias,
            device=device,
            dtype=torch.bfloat16,
        )
        self.k_proj = nn.Linear(
            configs.hidden_size,
            configs.num_key_value_heads * configs.head_dim,
            bias=configs.attention_bias,
            device=device,
            dtype=torch.bfloat16,
        )
        self.v_proj = nn.Linear(
            configs.hidden_size,
            configs.num_key_value_heads * configs.head_dim,
            bias=configs.attention_bias,
            device=device,
            dtype=torch.bfloat16,
        )
        self.o_proj = nn.Linear(
            configs.num_attention_heads * configs.head_dim,
            configs.hidden_size,
            bias=configs.attention_bias,
            device=device,
            dtype=torch.bfloat16,
        )
        self.rope = RotaryEmbedding(
            configs.head_dim,
            configs.rope_theta,
            configs.max_position_embeddings,
            configs.rope_scaling,
            device=device,
        )

    def sdpa(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        offset: torch.LongTensor,
    ) -> torch.Tensor:
        batch_size, seq_len, _, _ = q.shape
        n_ctx = k.shape[1]
        offset_int = int(offset.item())

        q = q.transpose(1, 2)
        k = k.repeat_interleave(self.num_groups, dim=2).transpose(1, 2)
        v = v.repeat_interleave(self.num_groups, dim=2).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.sm_scale
        causal_mask = torch.triu(
            attn.new_full((seq_len, n_ctx), -float("inf")),
            diagonal=offset_int + 1,
        )
        attn = attn + causal_mask.unsqueeze(0).unsqueeze(0)
        attn = F.softmax(attn.float(), dim=-1).to(q.dtype)
        out = torch.matmul(attn, v)
        return out.transpose(1, 2).reshape(batch_size, seq_len, -1)

    def forward(self, x: torch.Tensor, cache: Cache | None = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch_size, seq_len, self.num_attention_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)

        if cache is not None:
            offset = cache.offset.clone()
            q, k = self.rope(q, k, offset=offset)
            k, v = cache.extend(k, v)
        else:
            offset = torch.zeros((1,), dtype=torch.long, device=x.device)
            q, k = self.rope(q, k, offset=offset)

        return self.o_proj(self.sdpa(q, k, v, offset))


class MLPBlock(nn.Module):
    def __init__(self, configs: ModelConfigs, device: torch.device | None = None):
        super().__init__()
        self.gate_proj = nn.Linear(
            configs.hidden_size,
            configs.intermediate_size,
            bias=configs.mlp_bias,
            device=device,
            dtype=torch.bfloat16,
        )
        self.up_proj = nn.Linear(
            configs.hidden_size,
            configs.intermediate_size,
            bias=configs.mlp_bias,
            device=device,
            dtype=torch.bfloat16,
        )
        self.down_proj = nn.Linear(
            configs.intermediate_size,
            configs.hidden_size,
            bias=configs.mlp_bias,
            device=device,
            dtype=torch.bfloat16,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, configs: ModelConfigs, device: torch.device | None = None):
        super().__init__()
        self.input_layernorm = RMSNorm(configs.hidden_size, configs.norm_eps, device=device)
        self.self_attn = AttentionBlock(configs, device=device)
        self.post_attention_layernorm = RMSNorm(configs.hidden_size, configs.norm_eps, device=device)
        self.mlp = MLPBlock(configs, device=device)

    def forward(self, x: torch.Tensor, cache: Cache | None = None) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x), cache=cache)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class Transformer(nn.Module):
    def __init__(self, configs: ModelConfigs, device: torch.device | None = None):
        super().__init__()
        self.configs = configs
        self.embed_tokens = nn.Embedding(
            configs.vocab_size,
            configs.hidden_size,
            device=device,
            dtype=torch.bfloat16,
        )
        self.layers = nn.ModuleList(
            [TransformerBlock(configs, device=device) for _ in range(configs.num_hidden_layers)]
        )
        self.norm = RMSNorm(configs.hidden_size, configs.norm_eps, device=device)
        self.lm_head = nn.Linear(
            configs.hidden_size,
            configs.vocab_size,
            bias=False,
            device=device,
            dtype=torch.bfloat16,
        )

    def forward(self, x: torch.Tensor, caches: list[Cache] | None = None) -> torch.Tensor:
        caches = caches or [None] * len(self.layers)
        x = self.embed_tokens(x)
        for layer, cache in zip(self.layers, caches):
            x = layer(x, cache=cache)
        x = self.norm(x)
        return self.lm_head(x).float()

    @staticmethod
    def from_checkpoint(path: str, device: str | torch.device = "mps") -> "Transformer":
        device = torch.device(device)
        cfg = ModelConfigs()
        model = Transformer(cfg, device=device).to(device)
        model.eval()
        ckpt = Checkpoint(path, device)

        with torch.no_grad():
            for name, param in model.named_parameters():
                tensor = ckpt.get(name)
                if tensor is None and name == "lm_head.weight" and cfg.tie_word_embeddings:
                    tensor = ckpt.get("embed_tokens.weight")
                if tensor is None:
                    raise RuntimeError(f"Tensor {name} not found in checkpoint.")
                if tensor.shape != param.shape:
                    raise RuntimeError(
                        f"shape mismatch for {name}: file {tensor.shape} vs model {param.shape}"
                    )
                param.copy_(tensor.to(device=device, dtype=param.dtype))

        return model
