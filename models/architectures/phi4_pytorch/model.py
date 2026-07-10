import os
import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.architectures.phi4_pytorch.weights import Checkpoint
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Phi-4-multimodal *language model* (text backbone) re-implementation.
#
# This file used to host the GPT-OSS-20B MoE transformer. It has been rewritten
# to run the dense Phi-4 decoder, whose architecture matches the HuggingFace
# reference (`modeling_phi4_multimodal.py`):
#
#   - combined qkv_proj / gate_up_proj projections
#   - grouped-query attention (32 query heads, 8 kv heads, head_dim 96)
#   - RMSNorm (pre-attn `input_layernorm`, pre-mlp `post_attention_layernorm`)
#   - SwiGLU MLP: down_proj(up * silu(gate))
#   - LongRoPE position embeddings (short/long factor sets + attention scaling)
#   - plain causal attention (no sinks, no sliding window for this config)
#   - tied lm_head <- embed_tokens
#
# The vision (SigLIP) and audio (conformer) encoders are intentionally NOT
# ported here; this backbone covers pure-text generation, which is what
# `run_batch.py` exercises. To add a modality you would build its encoder +
# projector and splice the resulting embeddings into `inputs_embeds` at the
# `<|image|>` / `<|audio|>` token positions before the decoder loop (mirroring
# `Phi4MultimodalFeatureEmbedding` in the reference).
# ---------------------------------------------------------------------------


@dataclass
class ModelConfigs:
    vocab_size: int = 200064
    hidden_size: int = 3072
    intermediate_size: int = 8192
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 96
    hidden_act: str = "silu"
    rms_norm_eps: float = 1e-5
    max_position_embeddings: int = 131072
    original_max_position_embeddings: int = 4096
    rope_theta: float = 10000.0
    partial_rotary_factor: float = 1.0
    rope_type: str = "longrope"
    rope_short_factor: list | None = None
    rope_long_factor: list | None = None
    pad_token_id: int = 199999
    bos_token_id: int = 199999
    eos_token_id: list | None = None

    # Convenience aliases consumed by inference.py (kept from the GPT-OSS API).
    initial_context_length: int = 4096
    rope_scaling_factor: float = 32.0

    def __post_init__(self):
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads
        if self.eos_token_id is None:
            self.eos_token_id = [199999, 200020]
        # Keep the inference-side context helpers consistent with the rope config.
        self.initial_context_length = self.original_max_position_embeddings
        self.rope_scaling_factor = (
            self.max_position_embeddings / self.original_max_position_embeddings
        )

    @classmethod
    def from_hf_config(cls, config_path: str) -> "ModelConfigs":
        """Build configs from a HuggingFace Phi-4-multimodal `config.json`."""
        with open(config_path, "r") as f:
            cfg = json.load(f)

        # rope params can live under `rope_scaling` (legacy) or `rope_parameters`.
        rope = cfg.get("rope_scaling") or cfg.get("rope_parameters") or {}
        rope_theta = cfg.get("rope_theta", rope.get("rope_theta", 10000.0))
        rope_type = rope.get("type", rope.get("rope_type", "default"))

        head_dim = cfg.get("head_dim") or (
            cfg["hidden_size"] // cfg["num_attention_heads"]
        )

        return cls(
            vocab_size=cfg["vocab_size"],
            hidden_size=cfg["hidden_size"],
            intermediate_size=cfg["intermediate_size"],
            num_hidden_layers=cfg["num_hidden_layers"],
            num_attention_heads=cfg["num_attention_heads"],
            num_key_value_heads=cfg.get("num_key_value_heads", cfg["num_attention_heads"]),
            head_dim=head_dim,
            hidden_act=cfg.get("hidden_act", "silu"),
            rms_norm_eps=cfg.get("rms_norm_eps", 1e-5),
            max_position_embeddings=cfg.get("max_position_embeddings", 131072),
            original_max_position_embeddings=cfg.get("original_max_position_embeddings", 4096),
            rope_theta=rope_theta,
            partial_rotary_factor=cfg.get("partial_rotary_factor", rope.get("partial_rotary_factor", 1.0)),
            rope_type=rope_type,
            rope_short_factor=rope.get("short_factor"),
            rope_long_factor=rope.get("long_factor"),
            pad_token_id=cfg.get("pad_token_id", 199999),
            bos_token_id=cfg.get("bos_token_id", 199999),
            eos_token_id=cfg.get("eos_token_id"),
        )


class RMSNorm(nn.Module):
    """T5/Phi-style RMSNorm. Parameter is named `weight` to match the HF checkpoint."""

    def __init__(self, hidden_size: int, eps: float, device: torch.device | None = None):
        super().__init__()
        self.variance_epsilon = eps
        self.hidden_size = hidden_size
        self.weight = nn.Parameter(torch.ones(hidden_size, device=device, dtype=torch.bfloat16))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * x.to(input_dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims (GPT-NeoX convention used by Phi-4)."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    """
    q, k : (batch, seq_len, n_heads, head_dim)
    cos, sin : (seq_len, rotary_dim)
    Rotary is applied only to the first `rotary_dim` channels; the rest pass through.
    """
    cos = cos[None, :, None, :].to(q.dtype)
    sin = sin[None, :, None, :].to(q.dtype)
    rotary_dim = cos.shape[-1]

    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    q_embed = torch.cat([q_rot * cos + rotate_half(q_rot) * sin, q_pass], dim=-1)
    k_embed = torch.cat([k_rot * cos + rotate_half(k_rot) * sin, k_pass], dim=-1)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """(B, n_kv, S, D) -> (B, n_kv * n_rep, S, D)."""
    batch, num_kv, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_kv * n_rep, slen, head_dim)


class RotaryEmbedding(nn.Module):
    """
    LongRoPE: two inverse-frequency sets (short / long) selected by sequence
    length, plus an `attention_scaling` factor applied to cos/sin. Mirrors
    `_compute_longrope_parameters` + `Phi4MultimodalRotaryEmbedding` from HF.
    """

    def __init__(self, configs: ModelConfigs, device: torch.device | None = None):
        super().__init__()
        partial = configs.partial_rotary_factor
        self.dim = int(configs.head_dim * partial)  # number of rotated channels
        base = configs.rope_theta
        self.original_max = configs.original_max_position_embeddings

        factor = configs.max_position_embeddings / self.original_max
        if configs.rope_type == "default" or factor <= 1.0:
            self.attention_scaling = 1.0
        else:
            self.attention_scaling = math.sqrt(
                1 + math.log(factor) / math.log(self.original_max)
            )

        # base ** (2i / dim)  for i in [0, dim/2)
        exponents = torch.arange(0, self.dim, 2, dtype=torch.int64).float() / self.dim
        powers = base ** exponents  # (dim/2,)

        half = self.dim // 2
        short = configs.rope_short_factor if configs.rope_short_factor is not None else [1.0] * half
        long = configs.rope_long_factor if configs.rope_long_factor is not None else short
        short = torch.tensor(short, dtype=torch.float32)
        long = torch.tensor(long, dtype=torch.float32)

        self.register_buffer("short_inv_freq", (1.0 / (short * powers)).to(device), persistent=False)
        self.register_buffer("long_inv_freq", (1.0 / (long * powers)).to(device), persistent=False)

    @torch.no_grad()
    def forward(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # positions: (seq_len,) absolute indices.
        seq_len = int(positions.max().item()) + 1
        inv_freq = self.long_inv_freq if seq_len > self.original_max else self.short_inv_freq
        inv_freq = inv_freq.to(positions.device)

        freqs = torch.outer(positions.float(), inv_freq)  # (seq_len, dim/2)
        emb = torch.cat((freqs, freqs), dim=-1)           # (seq_len, dim)
        cos = emb.cos() * self.attention_scaling
        sin = emb.sin() * self.attention_scaling
        return cos, sin


class Cache:
    """Simple per-layer KV cache (unchanged from the original implementation)."""

    def __init__(self, batch_size, n_ctx, n_kv_heads, d_head, device: torch.device | None = None):
        self.k = torch.zeros((batch_size, n_ctx, n_kv_heads, d_head), dtype=torch.bfloat16, device=device)
        self.v = torch.zeros((batch_size, n_ctx, n_kv_heads, d_head), dtype=torch.bfloat16, device=device)
        self.offset = torch.zeros((1,), dtype=torch.long, device=device)

    def reset(self):
        self.k.zero_()
        self.v.zero_()
        self.offset.zero_()

    def extend(self, k, v):
        n_new = k.shape[1]
        start = self.offset.item()
        end = start + n_new
        self.k[:, start:end] = k
        self.v[:, start:end] = v
        self.offset += n_new
        return self.k[:, :end], self.v[:, :end]


class AttentionBlock(nn.Module):
    def __init__(self, configs: ModelConfigs, layer_idx: int = 0, device: torch.device | None = None):
        super().__init__()
        self.head_dim = configs.head_dim
        self.num_attention_heads = configs.num_attention_heads
        self.num_key_value_heads = configs.num_key_value_heads
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        self.layer_idx = layer_idx
        self.scaling = self.head_dim ** -0.5

        op_size = configs.num_attention_heads * self.head_dim + 2 * (
            configs.num_key_value_heads * self.head_dim
        )
        self.qkv_proj = nn.Linear(configs.hidden_size, op_size, bias=False, device=device, dtype=torch.bfloat16)
        self.o_proj = nn.Linear(
            configs.num_attention_heads * self.head_dim,
            configs.hidden_size,
            bias=False,
            device=device,
            dtype=torch.bfloat16,
        )

    def forward(self, x, cos, sin, cache: Cache | None = None):
        batch_size, seq_len, _ = x.shape

        qkv = self.qkv_proj(x)
        query_pos = self.num_attention_heads * self.head_dim
        kv_pos = self.num_key_value_heads * self.head_dim

        q = qkv[..., :query_pos]
        k = qkv[..., query_pos : query_pos + kv_pos]
        v = qkv[..., query_pos + kv_pos :]

        q = q.view(batch_size, seq_len, self.num_attention_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)

        # Rotary on (B, S, H, D) layout.
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if cache is not None:
            offset = int(cache.offset.item())
            k, v = cache.extend(k, v)  # full (B, total, n_kv, D)
        else:
            offset = 0

        # -> (B, H, S, D)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        k = repeat_kv(k, self.num_key_value_groups)
        v = repeat_kv(v, self.num_key_value_groups)
        n_ctx = k.shape[2]

        # Attention logits in fp32 for stability (matches HF eager path).
        attn = torch.matmul(q.float(), k.float().transpose(2, 3)) * self.scaling

        # Causal mask aligned to the cache offset: query at local index t maps to
        # absolute position offset + t, so anything strictly future is masked.
        mask = torch.full((seq_len, n_ctx), float("-inf"), device=x.device, dtype=torch.float32)
        mask = torch.triu(mask, diagonal=offset + 1)
        attn = attn + mask[None, None]

        attn = torch.softmax(attn, dim=-1).to(v.dtype)
        out = torch.matmul(attn, v)               # (B, H, S, D)
        out = out.transpose(1, 2).reshape(batch_size, seq_len, -1)
        return self.o_proj(out)


class MLPBlock(nn.Module):
    def __init__(self, configs: ModelConfigs, device: torch.device | None = None):
        super().__init__()
        self.gate_up_proj = nn.Linear(
            configs.hidden_size, 2 * configs.intermediate_size, bias=False, device=device, dtype=torch.bfloat16
        )
        self.down_proj = nn.Linear(
            configs.intermediate_size, configs.hidden_size, bias=False, device=device, dtype=torch.bfloat16
        )
        act = configs.hidden_act
        if act not in ("silu", "swish"):
            raise ValueError(f"Unsupported activation for Phi-4 MLP: {act}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        up_states = self.gate_up_proj(x)
        gate, up_states = up_states.chunk(2, dim=-1)
        up_states = up_states * F.silu(gate)
        return self.down_proj(up_states)


class TransformerBlock(nn.Module):
    def __init__(self, configs: ModelConfigs, layer_idx: int, device: torch.device | None = None):
        super().__init__()
        self.self_attn = AttentionBlock(configs, layer_idx, device)
        self.mlp = MLPBlock(configs, device)
        self.input_layernorm = RMSNorm(configs.hidden_size, configs.rms_norm_eps, device=device)
        self.post_attention_layernorm = RMSNorm(configs.hidden_size, configs.rms_norm_eps, device=device)

    def forward(self, x, cos, sin, cache: Cache | None = None):
        residual = x
        h = self.input_layernorm(x)
        h = self.self_attn(h, cos, sin, cache=cache)
        x = residual + h

        residual = x
        h = self.post_attention_layernorm(x)
        h = self.mlp(h)
        x = residual + h
        return x


class Transformer(nn.Module):
    def __init__(self, configs: ModelConfigs, device: torch.device | None = None):
        super().__init__()
        self.configs = configs

        self.embed_tokens = nn.Embedding(
            configs.vocab_size, configs.hidden_size, configs.pad_token_id, device=device, dtype=torch.bfloat16
        )
        self.layers = nn.ModuleList(
            [TransformerBlock(configs, i, device) for i in range(configs.num_hidden_layers)]
        )
        self.norm = RMSNorm(configs.hidden_size, configs.rms_norm_eps, device=device)
        self.lm_head = nn.Linear(
            configs.hidden_size, configs.vocab_size, bias=False, device=device, dtype=torch.bfloat16
        )

        self.rotary = RotaryEmbedding(configs, device=device)

    def forward(self, x: torch.Tensor, caches: list[Cache] | None = None) -> torch.Tensor:
        caches = caches or [None] * len(self.layers)
        batch_size, seq_len = x.shape

        # Absolute start position from the (pre-extend) cache offset.
        if caches[0] is not None:
            offset = int(caches[0].offset.item())
        else:
            offset = 0
        positions = torch.arange(offset, offset + seq_len, device=x.device, dtype=torch.long)
        cos, sin = self.rotary(positions)

        h = self.embed_tokens(x)
        for layer, cache in zip(self.layers, caches):
            h = layer(h, cos, sin, cache=cache)
        h = self.norm(h)
        logits = self.lm_head(h)
        return logits.float()

    @staticmethod
    def from_checkpoint(path: str, device: str | torch.device = "cpu") -> "Transformer":
        device = torch.device(device)

        cfg = ModelConfigs.from_hf_config(os.path.join(path, "config.json"))
        model = Transformer(cfg, device=device).to(device)
        model.eval()

        ckpt = Checkpoint(path, device)

        tied_lm_head = not ckpt.has("lm_head.weight")
        missing = []
        with torch.no_grad():
            for name, param in model.named_parameters():
                # Map local names to HF checkpoint names: everything except the
                # lm_head lives under the `model.` prefix in the checkpoint.
                hf_key = name if name.startswith("lm_head") else f"model.{name}"

                if not ckpt.has(hf_key):
                    if name == "lm_head.weight" and tied_lm_head:
                        continue  # handled by tying below
                    missing.append(hf_key)
                    continue

                t = ckpt.get(hf_key)
                if t.shape != param.shape:
                    raise RuntimeError(
                        f"shape mismatch for {name}: checkpoint {tuple(t.shape)} vs model {tuple(param.shape)}"
                    )
                param.copy_(t.to(param.dtype).to(device))

        if missing:
            raise RuntimeError("Missing keys when loading checkpoint:\n" + "\n".join(missing))

        # lm_head is tied to the input embeddings in Phi-4.
        if tied_lm_head:
            model.lm_head.weight = model.embed_tokens.weight

        return model
