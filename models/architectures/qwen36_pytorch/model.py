import os
import json
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .weights import Checkpoint

# ---------------------------------------------------------------------------
# This is a from-scratch, text-only re-implementation of the Qwen3.5 language
# model, faithful to the HuggingFace reference (`modeling_qwen3_5.py`).
#
# Key architectural facts (text backbone):
#   * Hybrid layer stack. Most layers use linear attention (a Gated DeltaNet),
#     and every `full_attention_interval`-th layer uses standard (gated) full
#     attention. Default pattern: layer i is "full_attention" iff (i+1) % 4 == 0.
#   * Full attention is GQA with per-head RMSNorm on Q and K, partial rotary
#     embeddings (only the first `head_dim * partial_rotary_factor` dims are
#     rotated) and an output gate (sigmoid).
#   * Linear attention is a Gated DeltaNet with a causal depthwise conv1d,
#     delta-rule recurrence and a gated RMSNorm.
#   * MLP is a dense SwiGLU (gate/up/down), NOT a MoE.
#   * RMSNorm uses the Qwen convention: out = x_normed * (1 + weight).
#
# Module / parameter names mirror HF exactly so that a standard Qwen3.5
# safetensors checkpoint loads directly via `from_checkpoint`.
#
# Vision is intentionally omitted: this is for text generation only.
# ---------------------------------------------------------------------------


@dataclass
class ModelConfigs:
    vocab_size: int = 248320
    hidden_size: int = 4096
    intermediate_size: int = 12288
    num_hidden_layers: int = 32
    num_attention_heads: int = 16
    num_key_value_heads: int = 4
    head_dim: int = 256
    hidden_act: str = "silu"
    max_position_embeddings: int = 32768
    rms_norm_eps: float = 1e-6
    tie_word_embeddings: bool = False
    attention_bias: bool = False
    attention_dropout: float = 0.0

    # RoPE
    rope_theta: float = 5_000_000.0
    partial_rotary_factor: float = 0.25

    # Linear attention (Gated DeltaNet)
    linear_conv_kernel_dim: int = 4
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 32

    # MoE (sparse feed-forward). num_experts == 0 -> dense MLP everywhere.
    num_experts: int = 0
    num_experts_per_tok: int = 0
    moe_intermediate_size: int = 0
    norm_topk_prob: bool = True
    decoder_sparse_step: int = 1
    mlp_only_layers: list[int] | None = None
    shared_expert_intermediate_size: int = 0
    use_shared_expert: bool = True

    # Layer schedule
    full_attention_interval: int = 4
    layer_types: list[str] | None = None

    def is_moe_layer(self, layer_idx: int) -> bool:
        """Mirror Qwen3-MoE: a layer is sparse unless excluded or stepped over."""
        if self.num_experts <= 0:
            return False
        if self.mlp_only_layers and layer_idx in self.mlp_only_layers:
            return False
        step = self.decoder_sparse_step or 1
        return (layer_idx + 1) % step == 0

    def __post_init__(self):
        if self.mlp_only_layers is None:
            self.mlp_only_layers = []
        if self.layer_types is None:
            self.layer_types = [
                "linear_attention"
                if bool((i + 1) % self.full_attention_interval)
                else "full_attention"
                for i in range(self.num_hidden_layers)
            ]

    @property
    def rotary_dim(self) -> int:
        return int(self.head_dim * self.partial_rotary_factor)


# =============================================================================
# Norms
# =============================================================================
class Qwen35RMSNorm(nn.Module):
    """RMSNorm with the Qwen convention: out = x_normed * (1 + weight)."""

    def __init__(self, dim: int, eps: float = 1e-6, device=None):
        super().__init__()
        self.eps = eps
        # Stored centered at 0; HF initialises to zeros and applies (1 + w).
        self.weight = nn.Parameter(torch.zeros(dim, device=device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        out = x * (1.0 + self.weight.float())
        return out.to(dtype)


class Qwen35RMSNormGated(nn.Module):
    """Gated RMSNorm used inside the Gated DeltaNet. weight init = ones (plain)."""

    def __init__(self, hidden_size: int, eps: float = 1e-6, device=None):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, device=device))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = self.weight * hidden_states.to(input_dtype)
        hidden_states = hidden_states * F.silu(gate.to(torch.float32))
        return hidden_states.to(input_dtype)


# =============================================================================
# Rotary embeddings (text-only -> standard partial RoPE)
# =============================================================================
class RotaryEmbedding(nn.Module):
    """
    Standard RoPE over the partial rotary dimension.

    The HF model uses interleaved MRoPE (3 position grids: T/H/W). For pure
    text every grid is identical, which reduces exactly to ordinary RoPE, so we
    implement the text case directly.
    """

    def __init__(self, configs: ModelConfigs, device=None):
        super().__init__()
        dim = configs.rotary_dim  # e.g. 256 * 0.25 = 64
        inv_freq = 1.0 / (
            configs.rope_theta
            ** (torch.arange(0, dim, 2, dtype=torch.float, device=device) / dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.attention_scaling = 1.0

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor):
        # position_ids: (batch, seq_len)
        inv_freq = self.inv_freq[None, :, None].float().expand(
            position_ids.shape[0], -1, 1
        )
        pos = position_ids[:, None, :].float()
        freqs = (inv_freq @ pos).transpose(1, 2)  # (batch, seq, rotary_dim/2)
        emb = torch.cat((freqs, freqs), dim=-1)    # (batch, seq, rotary_dim)
        cos = emb.cos() * self.attention_scaling
        sin = emb.sin() * self.attention_scaling
        return cos.to(x.dtype), sin.to(x.dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """Apply partial rotary embedding; the non-rotary tail passes through."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)

    q_embed = torch.cat([q_embed, q_pass], dim=-1)
    k_embed = torch.cat([k_embed, k_pass], dim=-1)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_kv_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


# =============================================================================
# Caches
# =============================================================================
class FullAttnCache:
    """KV cache for a full-attention layer. Stores (B, n_kv_heads, L, head_dim)."""

    def __init__(self, batch_size, n_ctx, n_kv_heads, head_dim, device, dtype):
        self.k = torch.zeros((batch_size, n_kv_heads, n_ctx, head_dim), dtype=dtype, device=device)
        self.v = torch.zeros((batch_size, n_kv_heads, n_ctx, head_dim), dtype=dtype, device=device)
        self.offset = 0

    def update(self, k, v):
        # k, v: (B, n_kv_heads, seq, head_dim)
        n_new = k.shape[2]
        start, end = self.offset, self.offset + n_new
        self.k[:, :, start:end] = k
        self.v[:, :, start:end] = v
        self.offset = end
        return self.k[:, :, :end], self.v[:, :, :end]


class LinearAttnCache:
    """Conv + recurrent state for a Gated DeltaNet layer."""

    def __init__(self, batch_size, conv_dim, kernel_size, num_v_heads,
                 head_k_dim, head_v_dim, device, dtype):
        self.conv_state = torch.zeros((batch_size, conv_dim, kernel_size), dtype=dtype, device=device)
        self.recurrent_state = torch.zeros(
            (batch_size, num_v_heads, head_k_dim, head_v_dim), dtype=torch.float32, device=device
        )
        self.has_state = False


def build_caches(configs: ModelConfigs, batch_size, n_ctx, device, dtype):
    """Return a per-layer list of the appropriate cache object."""
    conv_dim = (
        configs.linear_key_head_dim * configs.linear_num_key_heads * 2
        + configs.linear_value_head_dim * configs.linear_num_value_heads
    )
    caches = []
    for layer_type in configs.layer_types:
        if layer_type == "full_attention":
            caches.append(
                FullAttnCache(
                    batch_size, n_ctx, configs.num_key_value_heads, configs.head_dim, device, dtype
                )
            )
        else:
            caches.append(
                LinearAttnCache(
                    batch_size,
                    conv_dim,
                    configs.linear_conv_kernel_dim,
                    configs.linear_num_value_heads,
                    configs.linear_key_head_dim,
                    configs.linear_value_head_dim,
                    device,
                    dtype,
                )
            )
    return caches


# =============================================================================
# Linear attention helpers (pure-torch, ported from the HF fallback path)
# =============================================================================
def l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6):
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm


def causal_conv1d_update(hidden_states, conv_state, weight, bias=None):
    """Single-step (decode) causal depthwise conv with in-place state update."""
    _, hidden_size, seq_len = hidden_states.shape
    state_len = conv_state.shape[-1]
    hidden_states_new = torch.cat([conv_state, hidden_states], dim=-1).to(weight.dtype)
    conv_state.copy_(hidden_states_new[:, :, -state_len:])
    out = F.conv1d(hidden_states_new, weight.unsqueeze(1), bias, padding=0, groups=hidden_size)
    out = F.silu(out[:, :, -seq_len:])
    return out.to(hidden_states.dtype)


def chunk_gated_delta_rule(query, key, value, g, beta, chunk_size=64,
                           initial_state=None, output_final_state=False,
                           use_qk_l2norm_in_kernel=False):
    """Chunked delta-rule scan used for prefill. Inputs are (B, L, H, D)."""
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    total_sequence_length = sequence_length + pad_size
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
        for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0)

    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim, dtype=value.dtype, device=value.device)
        if initial_state is None
        else initial_state.to(value)
    )
    core_attn_out = torch.zeros_like(value)

    for i in range(0, total_sequence_length // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn = q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]
        v_prime = (k_cumdecay[:, :, i]) @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        core_attn_out[:, :, i] = attn_inter + attn @ v_new
        last_recurrent_state = (
            last_recurrent_state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
        )

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1])
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


def recurrent_gated_delta_rule(query, key, value, g, beta, initial_state,
                               output_final_state, use_qk_l2norm_in_kernel=False):
    """Step-by-step delta-rule recurrence used for decode. Inputs (B, L, H, D)."""
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    core_attn_out = torch.zeros(
        batch_size, num_heads, sequence_length, v_head_dim, dtype=value.dtype, device=value.device
    )
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim, dtype=value.dtype, device=value.device)
        if initial_state is None
        else initial_state.to(value)
    )

    for i in range(sequence_length):
        q_t = query[:, :, i]
        k_t = key[:, :, i]
        v_t = value[:, :, i]
        g_t = g[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)
        beta_t = beta[:, :, i].unsqueeze(-1)

        last_recurrent_state = last_recurrent_state * g_t
        kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta_t
        last_recurrent_state = last_recurrent_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        core_attn_out[:, :, i] = (last_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2)

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


# =============================================================================
# Linear attention layer (Gated DeltaNet)
# =============================================================================
class Qwen35GatedDeltaNet(nn.Module):
    def __init__(self, configs: ModelConfigs, layer_idx: int, device=None, dtype=None):
        super().__init__()
        self.hidden_size = configs.hidden_size
        self.num_v_heads = configs.linear_num_value_heads
        self.num_k_heads = configs.linear_num_key_heads
        self.head_k_dim = configs.linear_key_head_dim
        self.head_v_dim = configs.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.conv_kernel_size = configs.linear_conv_kernel_dim
        self.layer_idx = layer_idx

        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=False,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=self.conv_kernel_size - 1,
            device=device,
            dtype=dtype,
        )

        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads, device=device, dtype=torch.float32))
        A = torch.empty(self.num_v_heads, device=device).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A).to(torch.float32))

        self.norm = Qwen35RMSNormGated(self.head_v_dim, eps=configs.rms_norm_eps, device=device)

        self.in_proj_qkv = nn.Linear(self.hidden_size, self.key_dim * 2 + self.value_dim, bias=False, device=device, dtype=dtype)
        self.in_proj_z = nn.Linear(self.hidden_size, self.value_dim, bias=False, device=device, dtype=dtype)
        self.in_proj_b = nn.Linear(self.hidden_size, self.num_v_heads, bias=False, device=device, dtype=dtype)
        self.in_proj_a = nn.Linear(self.hidden_size, self.num_v_heads, bias=False, device=device, dtype=dtype)
        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False, device=device, dtype=dtype)

    def forward(self, hidden_states: torch.Tensor, cache: LinearAttnCache | None = None):
        batch_size, seq_len, _ = hidden_states.shape
        use_state = cache is not None and cache.has_state

        mixed_qkv = self.in_proj_qkv(hidden_states).transpose(1, 2)  # (B, conv_dim, L)
        z = self.in_proj_z(hidden_states).reshape(batch_size, seq_len, -1, self.head_v_dim)
        b = self.in_proj_b(hidden_states)
        a = self.in_proj_a(hidden_states)

        if use_state and seq_len == 1:
            mixed_qkv = causal_conv1d_update(
                mixed_qkv, cache.conv_state, self.conv1d.weight.squeeze(1), self.conv1d.bias
            )
        else:
            if use_state:
                mixed_qkv = torch.cat([cache.conv_state, mixed_qkv], dim=-1)
            if cache is not None:
                new_conv_state = F.pad(mixed_qkv, (self.conv_kernel_size - mixed_qkv.shape[-1], 0))
                cache.conv_state.copy_(new_conv_state[:, :, -self.conv_kernel_size :])
            mixed_qkv = F.silu(self.conv1d(mixed_qkv)[:, :, : mixed_qkv.shape[-1]])
            if use_state:
                mixed_qkv = mixed_qkv[:, :, -seq_len:]

        mixed_qkv = mixed_qkv.transpose(1, 2)
        query, key, value = torch.split(
            mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1
        )
        query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
        key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
        value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)

        beta = b.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)

        if self.num_v_heads // self.num_k_heads > 1:
            n = self.num_v_heads // self.num_k_heads
            query = query.repeat_interleave(n, dim=2)
            key = key.repeat_interleave(n, dim=2)

        initial_state = cache.recurrent_state if use_state else None
        if use_state and seq_len == 1:
            core_attn_out, last_state = recurrent_gated_delta_rule(
                query, key, value, g=g, beta=beta,
                initial_state=initial_state, output_final_state=cache is not None,
                use_qk_l2norm_in_kernel=True,
            )
        else:
            core_attn_out, last_state = chunk_gated_delta_rule(
                query, key, value, g=g, beta=beta,
                initial_state=initial_state, output_final_state=cache is not None,
                use_qk_l2norm_in_kernel=True,
            )

        if cache is not None:
            cache.recurrent_state.copy_(last_state.to(cache.recurrent_state.dtype))
            cache.has_state = True

        core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)
        return self.out_proj(core_attn_out)


# =============================================================================
# Full attention layer (gated GQA with q/k norm + partial RoPE)
# =============================================================================
class Qwen35Attention(nn.Module):
    def __init__(self, configs: ModelConfigs, layer_idx: int, device=None, dtype=None):
        super().__init__()
        self.configs = configs
        self.layer_idx = layer_idx
        self.head_dim = configs.head_dim
        self.num_heads = configs.num_attention_heads
        self.num_kv_heads = configs.num_key_value_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.scaling = self.head_dim ** -0.5

        bias = configs.attention_bias
        self.q_proj = nn.Linear(configs.hidden_size, self.num_heads * self.head_dim * 2, bias=bias, device=device, dtype=dtype)
        self.k_proj = nn.Linear(configs.hidden_size, self.num_kv_heads * self.head_dim, bias=bias, device=device, dtype=dtype)
        self.v_proj = nn.Linear(configs.hidden_size, self.num_kv_heads * self.head_dim, bias=bias, device=device, dtype=dtype)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, configs.hidden_size, bias=bias, device=device, dtype=dtype)
        self.q_norm = Qwen35RMSNorm(self.head_dim, eps=configs.rms_norm_eps, device=device)
        self.k_norm = Qwen35RMSNorm(self.head_dim, eps=configs.rms_norm_eps, device=device)

    def forward(self, hidden_states, position_embeddings, attention_mask=None,
                cache: FullAttnCache | None = None):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states, gate = torch.chunk(
            self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2), 2, dim=-1
        )
        gate = gate.reshape(*input_shape, -1)

        query_states = self.q_norm(query_states.reshape(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if cache is not None:
            key_states, value_states = cache.update(key_states, value_states)

        # Eager scaled dot-product attention with GQA.
        key_states = repeat_kv(key_states, self.num_kv_groups)
        value_states = repeat_kv(value_states, self.num_kv_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(*input_shape, -1)
        attn_output = attn_output * torch.sigmoid(gate)
        return self.o_proj(attn_output)


# =============================================================================
# MLP
# =============================================================================
class Qwen35MLP(nn.Module):
    def __init__(self, configs: ModelConfigs, intermediate_size: int | None = None, device=None, dtype=None):
        super().__init__()
        h = configs.hidden_size
        i = intermediate_size if intermediate_size is not None else configs.intermediate_size
        self.gate_proj = nn.Linear(h, i, bias=False, device=device, dtype=dtype)
        self.up_proj = nn.Linear(h, i, bias=False, device=device, dtype=dtype)
        self.down_proj = nn.Linear(i, h, bias=False, device=device, dtype=dtype)
        self.act_fn = F.silu

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class Qwen35FusedExperts(nn.Module):
    """
    Stacked expert weights (Qwen3.5 layout):
        gate_up_proj : (num_experts, 2 * moe_intermediate_size, hidden)
        down_proj    : (num_experts, hidden, moe_intermediate_size)
    Both follow nn.Linear's (out_features, in_features) convention per expert.
    """

    def __init__(self, configs: ModelConfigs, device=None, dtype=None):
        super().__init__()
        e, h, i = configs.num_experts, configs.hidden_size, configs.moe_intermediate_size
        self.gate_up_proj = nn.Parameter(torch.empty(e, 2 * i, h, device=device, dtype=dtype))
        self.down_proj = nn.Parameter(torch.empty(e, h, i, device=device, dtype=dtype))


class Qwen35MoeMLP(nn.Module):
    """Sparse MoE feed-forward with fused routed experts and a shared expert."""

    def __init__(self, configs: ModelConfigs, device=None, dtype=None):
        super().__init__()
        self.num_experts = configs.num_experts
        self.top_k = configs.num_experts_per_tok
        self.norm_topk_prob = configs.norm_topk_prob

        self.gate = nn.Linear(configs.hidden_size, self.num_experts, bias=False, device=device, dtype=dtype)
        self.experts = Qwen35FusedExperts(configs, device, dtype)

        self.use_shared_expert = configs.use_shared_expert
        if self.use_shared_expert:
            shared_i = configs.shared_expert_intermediate_size or configs.moe_intermediate_size
            self.shared_expert = Qwen35MLP(configs, intermediate_size=shared_i, device=device, dtype=dtype)
            self.shared_expert_gate = nn.Linear(configs.hidden_size, 1, bias=False, device=device, dtype=dtype)

    def forward(self, hidden_states):
        batch_size, seq_len, hidden_dim = hidden_states.shape
        x = hidden_states.view(-1, hidden_dim)

        router_logits = self.gate(x)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        if self.norm_topk_prob:
            routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(x.dtype)

        final_hidden_states = torch.zeros_like(x)

        gate_up_proj = self.experts.gate_up_proj
        down_proj = self.experts.down_proj

        # (num_experts, top_k, num_tokens) one-hot of assignments.
        expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)
        hit_experts = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero().flatten().tolist()

        for expert_idx in hit_experts:
            idx, top = torch.where(expert_mask[expert_idx])
            current_state = x[top]
            gate_up = F.linear(current_state, gate_up_proj[expert_idx])
            gate, up = gate_up.chunk(2, dim=-1)
            h = F.silu(gate) * up
            out = F.linear(h, down_proj[expert_idx]) * routing_weights[top, idx, None]
            final_hidden_states.index_add_(0, top, out.to(x.dtype))

        if self.use_shared_expert:
            shared = self.shared_expert(x)
            shared = shared * torch.sigmoid(self.shared_expert_gate(x))
            final_hidden_states = final_hidden_states + shared

        return final_hidden_states.reshape(batch_size, seq_len, hidden_dim)


# =============================================================================
# Decoder layer
# =============================================================================
class Qwen35DecoderLayer(nn.Module):
    def __init__(self, configs: ModelConfigs, layer_idx: int, device=None, dtype=None):
        super().__init__()
        self.layer_type = configs.layer_types[layer_idx]
        if self.layer_type == "linear_attention":
            self.linear_attn = Qwen35GatedDeltaNet(configs, layer_idx, device, dtype)
        else:
            self.self_attn = Qwen35Attention(configs, layer_idx, device, dtype)
        self.mlp = Qwen35MoeMLP(configs, device, dtype) if configs.is_moe_layer(layer_idx) \
            else Qwen35MLP(configs, device=device, dtype=dtype)
        self.input_layernorm = Qwen35RMSNorm(configs.hidden_size, eps=configs.rms_norm_eps, device=device)
        self.post_attention_layernorm = Qwen35RMSNorm(configs.hidden_size, eps=configs.rms_norm_eps, device=device)

    def forward(self, hidden_states, position_embeddings, attention_mask=None, cache=None):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        if self.layer_type == "linear_attention":
            hidden_states = self.linear_attn(hidden_states, cache=cache)
        else:
            hidden_states = self.self_attn(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                cache=cache,
            )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


# =============================================================================
# Backbone + LM head
# =============================================================================
class Qwen35TextModel(nn.Module):
    def __init__(self, configs: ModelConfigs, device=None, dtype=None):
        super().__init__()
        self.configs = configs
        self.embed_tokens = nn.Embedding(configs.vocab_size, configs.hidden_size, device=device, dtype=dtype)
        self.layers = nn.ModuleList(
            [Qwen35DecoderLayer(configs, i, device, dtype) for i in range(configs.num_hidden_layers)]
        )
        self.norm = Qwen35RMSNorm(configs.hidden_size, eps=configs.rms_norm_eps, device=device)
        self.rotary_emb = RotaryEmbedding(configs, device=device)

    def forward(self, input_ids, caches=None, position_ids=None):
        caches = caches or [None] * len(self.layers)
        hidden_states = self.embed_tokens(input_ids)

        batch_size, seq_len = input_ids.shape
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=input_ids.device)[None, :]

        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # Causal mask only needed when there is more than one query token.
        attention_mask = None
        if seq_len > 1:
            min_val = torch.finfo(hidden_states.dtype).min
            mask = torch.full((seq_len, seq_len), min_val, device=input_ids.device, dtype=hidden_states.dtype)
            mask = torch.triu(mask, diagonal=1)
            attention_mask = mask[None, None, :, :]

        for layer, cache in zip(self.layers, caches):
            hidden_states = layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                cache=cache,
            )
        return self.norm(hidden_states)


class Transformer(nn.Module):
    def __init__(self, configs: ModelConfigs, device=None, dtype=torch.bfloat16):
        super().__init__()
        self.configs = configs
        self.model = Qwen35TextModel(configs, device, dtype)
        self.lm_head = nn.Linear(configs.hidden_size, configs.vocab_size, bias=False, device=device, dtype=dtype)

    def forward(self, input_ids, caches=None, position_ids=None):
        hidden = self.model(input_ids, caches=caches, position_ids=position_ids)
        return self.lm_head(hidden).float()

    # ------------------------------------------------------------------ #
    @staticmethod
    def from_checkpoint(path: str, device: str | torch.device = "cpu") -> "Transformer":
        device = torch.device(device)

        with open(os.path.join(path, "config.json"), "r") as f:
            raw = json.load(f)
        cfg = _parse_hf_config(raw)

        # Pick a parameter dtype from the checkpoint metadata if present.
        dtype_str = raw.get("torch_dtype") or raw.get("text_config", {}).get("torch_dtype") or "bfloat16"
        dtype = getattr(torch, dtype_str, torch.bfloat16)

        # Index the checkpoint first so we can adapt the module tree to the
        # actual tensor layout (fused experts, presence of a shared expert).
        ckpt = Checkpoint(path, device)
        available = ckpt.keys()

        if cfg.num_experts > 0:
            cfg.use_shared_expert = any(".mlp.shared_expert." in k for k in available)
            fused = any(k.endswith(".mlp.experts.gate_up_proj") for k in available)
            if not fused and any(".mlp.experts.0." in k for k in available):
                raise RuntimeError(
                    "This checkpoint stores experts per-expert (mlp.experts.N.*), but this "
                    "implementation expects the fused layout (mlp.experts.gate_up_proj). "
                    "Tell me and I'll add the per-expert loading path."
                )
            n_moe = sum(cfg.is_moe_layer(i) for i in range(cfg.num_hidden_layers))
            print(
                f"  MoE detected: {cfg.num_experts} experts, top-{cfg.num_experts_per_tok}, "
                f"moe_intermediate={cfg.moe_intermediate_size}, "
                f"shared_expert={cfg.use_shared_expert}, sparse layers={n_moe}/{cfg.num_hidden_layers}"
            )

        model = Transformer(cfg, device=device, dtype=dtype).to(device)
        model.eval()

        with torch.no_grad():
            for name, param in model.named_parameters():
                candidates = [name]
                # Full multimodal checkpoints nest the text tower under language_model.
                if name.startswith("model."):
                    candidates.append("model.language_model." + name[len("model."):])
                    candidates.append("language_model." + name[len("model."):])
                src = next((c for c in candidates if c in available), None)

                # Tied embeddings fallback for lm_head.
                if src is None and name == "lm_head.weight":
                    for c in ("model.embed_tokens.weight",
                              "model.language_model.embed_tokens.weight"):
                        if c in available:
                            src = c
                            break

                if src is None:
                    # Surface a few real checkpoint keys for the same layer to
                    # make naming mismatches obvious (search the resolved prefix).
                    probe = name
                    if name.startswith("model."):
                        probe = "model.language_model." + name[len("model."):]
                    layer_prefix = probe.split(".mlp.")[0] if ".mlp." in probe else probe.rsplit(".", 1)[0]
                    nearby = sorted(k for k in available if k.startswith(layer_prefix))[:12]
                    raise RuntimeError(
                        f"Could not find a checkpoint tensor for parameter '{name}'.\n"
                        f"Tried: {candidates}\n"
                        f"Available keys under '{layer_prefix}':\n  "
                        + "\n  ".join(nearby or ["<none>"])
                    )

                t = ckpt.get(src)
                if t.shape != param.shape:
                    raise RuntimeError(
                        f"shape mismatch for {name}: checkpoint {tuple(t.shape)} vs model {tuple(param.shape)}"
                    )
                param.copy_(t.to(device=device, dtype=param.dtype))

        return model


def _parse_hf_config(raw: dict) -> ModelConfigs:
    """Build ModelConfigs from an HF config.json (text-only or multimodal)."""
    text = raw.get("text_config", raw)
    rope = text.get("rope_parameters") or text.get("rope_scaling") or {}
    rope_theta = rope.get("rope_theta", text.get("rope_theta", 5_000_000.0))
    partial = rope.get("partial_rotary_factor", text.get("partial_rotary_factor", 0.25))

    return ModelConfigs(
        vocab_size=text.get("vocab_size", 248320),
        hidden_size=text.get("hidden_size", 4096),
        intermediate_size=text.get("intermediate_size", 12288),
        num_hidden_layers=text.get("num_hidden_layers", 32),
        num_attention_heads=text.get("num_attention_heads", 16),
        num_key_value_heads=text.get("num_key_value_heads", 4),
        head_dim=text.get("head_dim", 256),
        hidden_act=text.get("hidden_act", "silu"),
        max_position_embeddings=text.get("max_position_embeddings", 32768),
        rms_norm_eps=text.get("rms_norm_eps", 1e-6),
        tie_word_embeddings=text.get("tie_word_embeddings", raw.get("tie_word_embeddings", False)),
        attention_bias=text.get("attention_bias", False),
        rope_theta=rope_theta,
        partial_rotary_factor=partial,
        linear_conv_kernel_dim=text.get("linear_conv_kernel_dim", 4),
        linear_key_head_dim=text.get("linear_key_head_dim", 128),
        linear_value_head_dim=text.get("linear_value_head_dim", 128),
        linear_num_key_heads=text.get("linear_num_key_heads", 16),
        linear_num_value_heads=text.get("linear_num_value_heads", 32),
        full_attention_interval=text.get("full_attention_interval", 4),
        layer_types=text.get("layer_types", None),
        # MoE (names vary slightly across Qwen MoE configs)
        num_experts=text.get("num_experts", text.get("num_routed_experts", 0)) or 0,
        num_experts_per_tok=text.get("num_experts_per_tok", text.get("moe_topk", 0)) or 0,
        moe_intermediate_size=text.get("moe_intermediate_size", 0) or 0,
        norm_topk_prob=text.get("norm_topk_prob", True),
        decoder_sparse_step=text.get("decoder_sparse_step", 1) or 1,
        mlp_only_layers=text.get("mlp_only_layers", None),
        shared_expert_intermediate_size=text.get("shared_expert_intermediate_size", 0) or 0,
    )


