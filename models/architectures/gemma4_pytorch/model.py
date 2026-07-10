"""
From-scratch PyTorch implementation of the Gemma 4 26B-A4B *text* model
(`google/gemma-4-26B-A4B-it`), written in the same single-file style as the
original GPT-OSS port it replaces.

This is text-only: the vision tower / audio encoder / per-layer-embedding paths
of the multimodal checkpoint are intentionally omitted (the 26B-A4B MoE model
does not use Per-Layer Embeddings, so the text decoder is self-contained).

Architecture summary (from config.json + the HF `modeling_gemma4` reference):
  - 30 decoder layers, hidden_size 2816, vocab 262144, tied embeddings.
  - Hybrid attention: a 5:1 local:global pattern. `layer_types` from the config
    marks layers 5, 11, 17, 23, 29 as "full_attention" (global) and the rest as
    "sliding_attention" (local, window 1024). The final layer is always global.
  - Local layers : 16 query heads / 8 KV heads, head_dim 256, full RoPE theta 1e4.
  - Global layers: 16 query heads / 2 KV heads, head_dim 512, K=V (unified KV),
    and Proportional RoPE (p-RoPE): only the first 25% of head dims are rotated
    (rotary_dim = 128) with theta 1e6, the rest carry no positional signal.
  - Per-head QK-norm (RMSNorm over head_dim) on Q and K before RoPE.
  - Gemma sandwich norms around attention and the feed-forward block.
  - Feed-forward = a dense GeGLU MLP (intermediate 2112, the always-on "3x
    shared expert") summed with a sparse MoE (128 experts, top-8, intermediate
    704, GeGLU).
  - Final logits are tanh-softcapped at 30.

Verified against the HF `modeling_gemma4` reference (text path of the
`gemma-4-26B-A4B-it` checkpoint):
  - attention scaling is 1.0 (queries are per-head RMS-normed by q_norm);
  - V is RMS-normed (no scale) and never receives RoPE;
  - global layers use proportional p-RoPE built as a full-length frequency
    table with the non-rotary tail zeroed (half-offset pairing), NOT a
    leading-slice split;
  - the MoE router normalises (RMSNorm, no scale), scales by router.scale and
    hidden_size**-0.5, top-k weights are renormalised to sum 1 and then scaled
    by per_expert_scale, and routing is decided from the RAW block residual;
  - the five feed-forward norms map as documented in FeedForwardBlock;
  - each layer output is multiplied by the learned layer_scalar.
"""

import os
import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .weights import Checkpoint
from dataclasses import dataclass, field


@dataclass
class ModelConfigs:
    # ---- core dims (gemma-4-26B-A4B-it/config.json -> text_config) ----
    vocab_size: int = 262144
    hidden_size: int = 2816
    num_hidden_layers: int = 30
    num_attention_heads: int = 16

    # local ("sliding") attention
    num_key_value_heads: int = 8
    head_dim: int = 256
    sliding_window: int = 1024

    # global ("full") attention
    num_global_key_value_heads: int = 2
    global_head_dim: int = 512
    attention_k_eq_v: bool = True          # global layers share one K=V projection

    # which layers are global; the rest are local sliding-window layers
    layer_types: list[str] = field(default_factory=lambda: [
        "full_attention" if (i % 6) == 5 else "sliding_attention"
        for i in range(30)
    ])

    # ---- feed-forward / MoE ----
    intermediate_size: int = 2112          # dense (shared) GeGLU MLP
    moe_intermediate_size: int = 704       # each routed expert
    num_experts: int = 128
    top_k_experts: int = 8
    hidden_activation: str = "gelu_pytorch_tanh"

    # ---- norms / rope / misc ----
    rms_norm_eps: float = 1e-6
    rope_theta_local: float = 10000.0
    rope_theta_global: float = 1000000.0
    rope_partial_rotary_factor_global: float = 0.25   # p-RoPE on global layers
    max_position_embeddings: int = 262144
    final_logit_softcapping: float = 30.0
    tie_word_embeddings: bool = True
    # query_pre_attn_scalar is absent from this checkpoint's config -> defaults
    # to the per-layer head_dim (so scaling = head_dim ** -0.5).
    query_pre_attn_scalar: float | None = None

    @staticmethod
    def from_json(path: str) -> "ModelConfigs":
        """Build configs from a HF Gemma 4 config.json (reads `text_config`)."""
        with open(path, "r") as f:
            raw = json.load(f)
        tc = raw.get("text_config", raw)

        rope = tc.get("rope_parameters", {})
        local_rope = rope.get("sliding_attention", {})
        global_rope = rope.get("full_attention", {})

        n_layers = tc.get("num_hidden_layers", 30)
        layer_types = tc.get(
            "layer_types",
            ["full_attention" if (i % 6) == 5 else "sliding_attention" for i in range(n_layers)],
        )

        return ModelConfigs(
            vocab_size=tc.get("vocab_size", 262144),
            hidden_size=tc.get("hidden_size", 2816),
            num_hidden_layers=n_layers,
            num_attention_heads=tc.get("num_attention_heads", 16),
            num_key_value_heads=tc.get("num_key_value_heads", 8),
            head_dim=tc.get("head_dim", 256),
            sliding_window=tc.get("sliding_window", 1024),
            num_global_key_value_heads=tc.get("num_global_key_value_heads", 2),
            global_head_dim=tc.get("global_head_dim", 512),
            attention_k_eq_v=tc.get("attention_k_eq_v", True),
            layer_types=layer_types,
            intermediate_size=tc.get("intermediate_size", 2112),
            moe_intermediate_size=tc.get("moe_intermediate_size", 704),
            num_experts=tc.get("num_experts", 128),
            top_k_experts=tc.get("top_k_experts", 8),
            hidden_activation=tc.get("hidden_activation", "gelu_pytorch_tanh"),
            rms_norm_eps=tc.get("rms_norm_eps", 1e-6),
            rope_theta_local=local_rope.get("rope_theta", 10000.0),
            rope_theta_global=global_rope.get("rope_theta", 1000000.0),
            rope_partial_rotary_factor_global=global_rope.get("partial_rotary_factor", 0.25),
            max_position_embeddings=tc.get("max_position_embeddings", 262144),
            final_logit_softcapping=tc.get("final_logit_softcapping", 30.0),
            tie_word_embeddings=raw.get("tie_word_embeddings", tc.get("tie_word_embeddings", True)),
            query_pre_attn_scalar=tc.get("query_pre_attn_scalar", None),
        )

    # convenience: per-layer geometry
    def is_global(self, layer_idx: int) -> bool:
        return self.layer_types[layer_idx] == "full_attention"

    def layer_head_dim(self, layer_idx: int) -> int:
        return self.global_head_dim if self.is_global(layer_idx) else self.head_dim

    def layer_kv_heads(self, layer_idx: int) -> int:
        return self.num_global_key_value_heads if self.is_global(layer_idx) else self.num_key_value_heads


def _act_fn(name: str):
    if name in ("gelu_pytorch_tanh", "gelu_tanh"):
        return lambda x: F.gelu(x, approximate="tanh")
    if name == "gelu":
        return F.gelu
    if name == "silu":
        return F.silu
    raise ValueError(f"Unsupported activation {name!r}")


# --------------------------------------------------------------------------- #
# Norm
# --------------------------------------------------------------------------- #
class Gemma4RMSNorm(nn.Module):
    """
    Gemma 4 RMSNorm. IMPORTANT: unlike Gemma 1/2/3 (which apply `(1 + weight)`),
    the Gemma 4 reference applies the scale as a *plain* multiply - the saved
    weights are already centred at 1.0. Computation is done in fp32.
    """
    def __init__(self, dim: int, eps: float = 1e-6, with_scale: bool = True,
                 device: torch.device | None = None):
        super().__init__()
        self.eps = eps
        self.with_scale = with_scale
        # In the reference, some norms (the value norm `v_norm` and the MoE
        # router norm) are created with `with_scale=False`: they have NO learnable
        # weight and are pure RMS normalisation.
        if with_scale:
            self.weight = nn.Parameter(torch.ones(dim, device=device, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        t = x.float()
        # torch.pow(.., -0.5) (not rsqrt) to match the reference numerics.
        t = t * torch.pow(t.pow(2).mean(-1, keepdim=True) + self.eps, -0.5)
        if self.with_scale:
            t = t * self.weight.float()
        return t.to(dtype)


# --------------------------------------------------------------------------- #
# RoPE (HF rotate_half convention) with optional partial rotary (p-RoPE)
# --------------------------------------------------------------------------- #
def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding(nn.Module):
    """
    Precomputes cos/sin for `rotary_dim` dimensions. For global layers,
    rotary_dim < head_dim (p-RoPE) so only the leading slice is rotated and the
    remainder is passed through unchanged - this is the "proportional RoPE"
    that preserves low-frequency (semantic) dimensions over long contexts.
    """
    def __init__(self, head_dim: int, rotary_dim: int, theta: float,
                 device: torch.device | None = None):
        super().__init__()
        self.head_dim = head_dim
        self.rotary_dim = rotary_dim
        # IMPORTANT: this mirrors the reference "proportional" RoPE exactly.
        # The frequency table has length head_dim/2 (NOT rotary_dim/2). Only the
        # first rotary_dim/2 entries are nonzero; the rest are 0 so that, after
        # `emb = cat(freqs, freqs)` and the standard half-offset `rotate_half`,
        # the non-rotary dims see cos=1/sin=0 and pass through unchanged.
        #
        # This differs from the usual "split off the first `rotary_dim` columns
        # and rotate those" partial-RoPE: there, dim d pairs with d+rotary_dim/2,
        # whereas here dim d pairs with d+head_dim/2. The trained Q/K projections
        # and per-head QK-norms expect the half-offset pairing, so the table must
        # be built this way for global (p-RoPE) layers to produce correct output.
        half = head_dim // 2
        inv_freq = torch.zeros(half, dtype=torch.float32, device=device)
        n_rot = rotary_dim // 2
        idx = torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=device)
        inv_freq[:n_rot] = 1.0 / (theta ** (idx / rotary_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def apply(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        # x: (B, T, H, head_dim); positions: (T,)
        freqs = torch.outer(positions.float(), self.inv_freq)   # (T, head_dim/2)
        emb = torch.cat((freqs, freqs), dim=-1)                 # (T, head_dim)
        cos = emb.cos().to(x.dtype)[None, :, None, :]
        sin = emb.sin().to(x.dtype)[None, :, None, :]
        # Applied over the FULL head_dim; the zeroed frequencies make the
        # non-rotary dimensions a no-op.
        return x * cos + _rotate_half(x) * sin


# --------------------------------------------------------------------------- #
# KV cache (per-layer geometry: local and global layers differ)
# --------------------------------------------------------------------------- #
class Cache:
    def __init__(self, batch_size, n_ctx, n_kv_heads, d_head, device=None):
        self.k = torch.zeros((batch_size, n_ctx, n_kv_heads, d_head), dtype=torch.bfloat16, device=device)
        self.v = torch.zeros((batch_size, n_ctx, n_kv_heads, d_head), dtype=torch.bfloat16, device=device)
        self.offset = torch.zeros((1,), dtype=torch.long, device=device)

    def reset(self):
        self.k.zero_(); self.v.zero_(); self.offset.zero_()

    def extend(self, k, v):
        n_new = k.shape[1]
        start = int(self.offset.item())
        end = start + n_new
        self.k[:, start:end] = k
        self.v[:, start:end] = v
        self.offset += n_new
        return self.k[:, :end], self.v[:, :end]


# --------------------------------------------------------------------------- #
# Attention
# --------------------------------------------------------------------------- #
class AttentionBlock(nn.Module):
    def __init__(self, configs: ModelConfigs, layer_idx: int, device=None):
        super().__init__()
        self.configs = configs
        self.layer_idx = layer_idx
        self.is_global = configs.is_global(layer_idx)

        self.num_heads = configs.num_attention_heads
        self.head_dim = configs.layer_head_dim(layer_idx)
        self.num_kv_heads = configs.layer_kv_heads(layer_idx)
        self.num_groups = self.num_heads // self.num_kv_heads
        self.k_eq_v = configs.attention_k_eq_v and self.is_global
        self.sliding_window = 0 if self.is_global else configs.sliding_window

        # Gemma4 attention uses scaling = 1.0 (NOT head_dim**-0.5). Queries are
        # already RMS-normalised per head by `q_norm`, and the reference sets
        # `self.scaling = 1.0`. Using head_dim**-0.5 here flattens the softmax
        # (scores shrink ~16-22x) and was a primary cause of garbage output.
        self.scaling = 1.0

        self.input_layernorm = Gemma4RMSNorm(configs.hidden_size, configs.rms_norm_eps, device=device)
        self.q_proj = nn.Linear(configs.hidden_size, self.num_heads * self.head_dim, bias=False, device=device, dtype=torch.bfloat16)
        self.k_proj = nn.Linear(configs.hidden_size, self.num_kv_heads * self.head_dim, bias=False, device=device, dtype=torch.bfloat16)
        # On global layers K==V so there is no separate v_proj in the checkpoint.
        if not self.k_eq_v:
            self.v_proj = nn.Linear(configs.hidden_size, self.num_kv_heads * self.head_dim, bias=False, device=device, dtype=torch.bfloat16)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, configs.hidden_size, bias=False, device=device, dtype=torch.bfloat16)

        # QK-norm over head_dim (per head), like Gemma 3.
        self.q_norm = Gemma4RMSNorm(self.head_dim, configs.rms_norm_eps, device=device)
        self.k_norm = Gemma4RMSNorm(self.head_dim, configs.rms_norm_eps, device=device)
        # V-norm: RMSNorm over head_dim WITHOUT a learnable scale. The reference
        # always normalises the value projection (with_scale=False), including on
        # global k==v layers where V is the *raw* k_proj output. It has no weight,
        # so there is nothing to load for it.
        self.v_norm = Gemma4RMSNorm(self.head_dim, configs.rms_norm_eps, with_scale=False, device=device)

        self.post_attention_layernorm = Gemma4RMSNorm(configs.hidden_size, configs.rms_norm_eps, device=device)

        rotary_dim = self.head_dim
        theta = configs.rope_theta_local
        if self.is_global:
            rotary_dim = int(self.head_dim * configs.rope_partial_rotary_factor_global)
            theta = configs.rope_theta_global
        self.rope = RotaryEmbedding(
            self.head_dim, rotary_dim, theta, device=device,
        )

    def _sdpa(self, q, k, v, offset):
        # q: (B,T,Hq,D)  k,v: (B,Ctx,Hkv,D)
        B, T, Hq, D = q.shape
        Ctx = k.shape[1]
        # expand kv across groups
        k = k.unsqueeze(3).expand(B, Ctx, self.num_kv_heads, self.num_groups, D)
        v = v.unsqueeze(3).expand(B, Ctx, self.num_kv_heads, self.num_groups, D)
        q = q.view(B, T, self.num_kv_heads, self.num_groups, D)

        # logits: (B, Hkv, G, T, Ctx)
        scores = torch.einsum("bthgd,bchgd->bhgtc", q, k) * self.scaling
        # Apply attention soft-capping (cap = 50.0)
        # scores = 50.0 * torch.tanh(scores / 50.0) 

        # causal mask aligned to the cache
        mask = torch.triu(q.new_full((T, Ctx), float("-inf")), diagonal=offset + 1)
        if self.sliding_window > 0:
            mask = mask + torch.tril(
                q.new_full((T, Ctx), float("-inf")), diagonal=offset - self.sliding_window
            )
        scores = scores + mask[None, None, None, :, :]

        probs = torch.softmax(scores.float(), dim=-1).to(v.dtype)
        out = torch.einsum("bhgtc,bchgd->bthgd", probs, v)
        # if self.layer_idx in (0, 29):
        #     w = probs[0, :, :, -1, :]        # last query's weights over the context
        #     print(f"  L{self.layer_idx:02d} global={self.is_global} ctx={w.shape[-1]} "
        #           f"peak={w.max().item():.3f} uniform={1.0 / w.shape[-1]:.3f}")
        return out.reshape(B, T, Hq * D)

    def forward(self, x, cache: Cache | None = None):
        B, T, _ = x.shape
        residual = x
        h = self.input_layernorm(x)

        q = self.q_proj(h).view(B, T, self.num_heads, self.head_dim)
        k = self.k_proj(h).view(B, T, self.num_kv_heads, self.head_dim)
        # Unified KV on global layers: V reuses the *raw* K projection (before
        # k_norm / RoPE). On local layers V comes from its own v_proj.
        v = k if self.k_eq_v else self.v_proj(h).view(B, T, self.num_kv_heads, self.head_dim)

        # QK-norm (per head, over head_dim), applied before RoPE.
        q = self.q_norm(q)
        k = self.k_norm(k)
        # V is normalised (no scale) and never gets RoPE.
        v = self.v_norm(v)

        offset = int(cache.offset.item()) if cache is not None else 0
        positions = torch.arange(T, device=x.device, dtype=torch.long) + offset
        q = self.rope.apply(q, positions)
        k = self.rope.apply(k, positions)

        if cache is not None:
            k, v = cache.extend(k, v)

        attn = self._sdpa(q, k, v, offset)
        attn = self.o_proj(attn)
        attn = self.post_attention_layernorm(attn)   # Gemma sandwich norm
        # return attn # residual + attn
        return residual + attn


# --------------------------------------------------------------------------- #
# Feed-forward: dense GeGLU MLP + sparse MoE, summed
# --------------------------------------------------------------------------- #
class DenseMLP(nn.Module):
    """Always-on GeGLU MLP (the '3x shared expert', intermediate_size 2112)."""
    def __init__(self, configs: ModelConfigs, device=None):
        super().__init__()
        H, I = configs.hidden_size, configs.intermediate_size
        self.gate_proj = nn.Linear(H, I, bias=False, device=device, dtype=torch.bfloat16)
        self.up_proj = nn.Linear(H, I, bias=False, device=device, dtype=torch.bfloat16)
        self.down_proj = nn.Linear(I, H, bias=False, device=device, dtype=torch.bfloat16)
        self.act = _act_fn(configs.hidden_activation)

    def forward(self, x):
        return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))

class MoE(nn.Module):
    """
    Sparse Mixture-of-Experts: 128 experts, top-8 routing, GeGLU experts with
    intermediate 704. Experts are stored as stacked tensors:
        gate_up_proj : (num_experts, hidden, 2 * moe_intermediate)   [transposed at load]
        down_proj    : (num_experts, moe_intermediate, hidden)       [transposed at load]

    Routing (matches Gemma4TextRouter):
        h = rms_norm_no_scale(route_input)
        h = h * router_scale * hidden_size**-0.5
        probs = softmax(proj(h))
        topw, topi = topk(probs, k)
        topw = topw / topw.sum(-1)              # renormalise to sum 1
        topw = topw * per_expert_scale[topi]

    IMPORTANT: the router sees the *raw residual* (the block input), while the
    experts process the `pre_feedforward_layernorm_2`-normed input. These are two
    different views of the same hidden state, so routing and expert inputs are
    passed separately.
    """
    def __init__(self, configs: ModelConfigs, device=None):
        super().__init__()
        self.configs = configs
        self.num_experts = configs.num_experts
        self.top_k = configs.top_k_experts
        H = configs.hidden_size
        I = configs.moe_intermediate_size
        self.act = _act_fn(configs.hidden_activation)

        # Router. `router_norm` is RMSNorm WITHOUT scale (no weight to load).
        self.router_norm = Gemma4RMSNorm(H, configs.rms_norm_eps, with_scale=False, device=device)
        self.router_proj = nn.Linear(H, self.num_experts, bias=False, device=device, dtype=torch.bfloat16)
        self.router_scale = nn.Parameter(torch.ones(H, device=device, dtype=torch.float32))            # router.scale
        self.per_expert_scale = nn.Parameter(torch.ones(self.num_experts, device=device, dtype=torch.float32))
        self.scalar_root_size = H ** -0.5

        self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, H, 2 * I, device=device, dtype=torch.bfloat16))
        self.down_proj = nn.Parameter(torch.empty(self.num_experts, I, H, device=device, dtype=torch.bfloat16))

    def _route(self, x_route):
        flat = x_route.reshape(-1, x_route.shape[-1])
        h = self.router_norm(flat)
        h = h * self.router_scale.to(h.dtype) * self.scalar_root_size
        logits = self.router_proj(h).float()
        probs = torch.softmax(logits, dim=-1)
        topw, topi = probs.topk(self.top_k, dim=-1)                  # (N, k)
        # Renormalise the kept top-k weights to sum to 1, THEN apply per-expert scale.
        topw = topw / topw.sum(dim=-1, keepdim=True)
        topw = topw * self.per_expert_scale[topi]
        return topw, topi

    def forward(self, x_expert, x_route):
        B, T, H = x_expert.shape
        flat = x_expert.reshape(B * T, H)

        topw, topi = self._route(x_route)

        out = torch.zeros_like(flat)
        for slot in range(self.top_k):
            idx = topi[:, slot]            # (N,)  expert id per token
            w = topw[:, slot].unsqueeze(-1).to(flat.dtype)
            for e in idx.unique():
                m = idx == e
                xe = flat[m]               # (n, H)
                gu = xe @ self.gate_up_proj[e]        # (n, 2I)
                gate, up = gu.chunk(2, dim=-1)
                he = self.act(gate) * up              # (n, I)
                ye = he @ self.down_proj[e]           # (n, H)
                out[m] += w[m] * ye
        return out.reshape(B, T, H)



class FeedForwardBlock(nn.Module):
    """
    Dense GeGLU MLP ("shared expert") summed with the sparse MoE, wrapped in
    Gemma sandwich norms. Verified against modeling_gemma4.Gemma4TextDecoderLayer:

      dense  = post_feedforward_layernorm_1( mlp( pre_feedforward_layernorm(x) ) )
      moe    = post_feedforward_layernorm_2( experts( pre_feedforward_layernorm_2(x) ) )
               # routing decided from the RAW residual x (see MoE._route)
      out    = x + post_feedforward_layernorm( dense + moe )
    """
    def __init__(self, configs: ModelConfigs, device=None):
        super().__init__()
        eps, H = configs.rms_norm_eps, configs.hidden_size
        self.pre_dense = Gemma4RMSNorm(H, eps, device=device)        # pre_feedforward_layernorm
        self.pre_moe = Gemma4RMSNorm(H, eps, device=device)          # pre_feedforward_layernorm_2
        self.post_dense = Gemma4RMSNorm(H, eps, device=device)       # post_feedforward_layernorm_1
        self.post_moe = Gemma4RMSNorm(H, eps, device=device)         # post_feedforward_layernorm_2
        self.post_combined = Gemma4RMSNorm(H, eps, device=device)    # post_feedforward_layernorm

        self.mlp = DenseMLP(configs, device=device)
        self.moe = MoE(configs, device=device)

    def forward(self, x):
        residual = x
        dense = self.post_dense(self.mlp(self.pre_dense(x)))
        # Router sees the RAW residual `x`; experts process pre_moe(x).
        moe = self.post_moe(self.moe(self.pre_moe(x), x))
        combined = self.post_combined(dense + moe)
        return residual + combined


# --------------------------------------------------------------------------- #
# Decoder layer + full model
# --------------------------------------------------------------------------- #
class DecoderLayer(nn.Module):
    def __init__(self, configs: ModelConfigs, layer_idx: int, device=None):
        super().__init__()
        self.self_attn = AttentionBlock(configs, layer_idx, device=device)
        self.feed_forward = FeedForwardBlock(configs, device=device)
        # Gemma4 multiplies the whole layer output by a learned `layer_scalar`
        # (initialised to 1.0 and loaded from the checkpoint).
        self.layer_scalar = nn.Parameter(torch.ones((), device=device, dtype=torch.float32))

    def forward(self, x, cache: Cache | None = None):
        # Each sub-block already adds its own residual internally:
        #   self_attn:     x -> x + post_attention_layernorm(attn(input_layernorm(x)))
        #   feed_forward:  x -> x + post_feedforward_layernorm(dense + moe)
        # then the full layer output is scaled by layer_scalar.
        x = self.self_attn(x, cache=cache)
        x = self.feed_forward(x)
        return x * self.layer_scalar.to(x.dtype)


class Transformer(nn.Module):
    def __init__(self, configs: ModelConfigs, device=None):
        super().__init__()
        self.configs = configs
        self.device = device
        self.embed_tokens = nn.Embedding(configs.vocab_size, configs.hidden_size, device=device, dtype=torch.bfloat16)
        self.layers = nn.ModuleList([DecoderLayer(configs, i, device=device) for i in range(configs.num_hidden_layers)])
        self.norm = Gemma4RMSNorm(configs.hidden_size, configs.rms_norm_eps, device=device)
        # tied embeddings -> no separate lm_head; logits computed from embed_tokens.weight
        self.embed_scale = math.sqrt(configs.hidden_size)

    def build_caches(self, batch_size: int, max_len: int, device=None):
        device = device or self.device
        caches = []
        for i in range(self.configs.num_hidden_layers):
            caches.append(Cache(
                batch_size=batch_size,
                n_ctx=max_len,
                n_kv_heads=self.configs.layer_kv_heads(i),
                d_head=self.configs.layer_head_dim(i),
                device=device,
            ))
        return caches

    def forward(self, input_ids: torch.Tensor, caches: list[Cache] | None = None) -> torch.Tensor:
        caches = caches or [None] * len(self.layers)
        x = self.embed_tokens(input_ids)
        # Gemma scales embeddings by sqrt(hidden_size).
        x = x * torch.tensor(self.embed_scale, dtype=x.dtype, device=x.device)

        for layer, cache in zip(self.layers, caches):
            x = layer(x, cache=cache)

        x = self.norm(x)
        logits = F.linear(x, self.embed_tokens.weight)  # tied unembedding

        cap = self.configs.final_logit_softcapping
        if cap:
            logits = cap * torch.tanh(logits / cap)
        # top = torch.topk(logits[0, -1].float(), 5)
        # print("  logits top5 ids:", [int(i) for i in top.indices],
        #       "vals:", [round(float(v), 2) for v in top.values])
        return logits.float()

    # ----------------------------------------------------------------- #
    # Checkpoint loading (bf16 safetensors, HF `model.language_model.*`)
    # ----------------------------------------------------------------- #
    @staticmethod
    def from_checkpoint(path: str, device: str | torch.device = "cpu") -> "Transformer":
        device = torch.device(device)
        cfg = ModelConfigs.from_json(os.path.join(path, "config.json"))
        model = Transformer(cfg, device=device).to(device)
        model.eval()
        ckpt = Checkpoint(path, device)

        P = "model.language_model."

        @torch.no_grad()
        def load(dst: torch.Tensor, name: str, transpose_ok: bool = False):
            t = ckpt.get(name).to(device)
            if t.shape != dst.shape:
                # scalars / 1-D vectors stored with a different but equivalent shape
                if dst.ndim <= 1 and t.numel() == dst.numel():
                    # print(f"[LOAD] reshaping {name}: {tuple(t.shape)} -> {tuple(dst.shape)}")
                    t = t.reshape(dst.shape)
                # stacked expert tensors may be stored transposed on the last two dims
                elif transpose_ok and t.ndim >= 2 and t.transpose(-1, -2).shape == dst.shape:
                    # print(f"[LOAD] transposing {name}: {tuple(t.shape)} -> {tuple(dst.shape)}")
                    t = t.transpose(-1, -2).contiguous()
            if t.shape != dst.shape:
                raise RuntimeError(
                    f"shape mismatch for {name}: ckpt {tuple(t.shape)} vs model {tuple(dst.shape)}"
                )
            dst.copy_(t.to(dst.dtype))

        with torch.no_grad():
            load(model.embed_tokens.weight, P + "embed_tokens.weight")
            load(model.norm.weight, P + "norm.weight")

            for i, layer in enumerate(model.layers):
                lp = f"{P}layers.{i}."
                attn = layer.self_attn
                load(attn.input_layernorm.weight, lp + "input_layernorm.weight")
                load(attn.q_proj.weight, lp + "self_attn.q_proj.weight")
                load(attn.k_proj.weight, lp + "self_attn.k_proj.weight")
                if not attn.k_eq_v:
                    load(attn.v_proj.weight, lp + "self_attn.v_proj.weight")
                load(attn.o_proj.weight, lp + "self_attn.o_proj.weight")
                load(attn.q_norm.weight, lp + "self_attn.q_norm.weight")
                load(attn.k_norm.weight, lp + "self_attn.k_norm.weight")
                load(attn.post_attention_layernorm.weight, lp + "post_attention_layernorm.weight")

                ff = layer.feed_forward
                load(ff.pre_dense.weight, lp + "pre_feedforward_layernorm.weight")
                load(ff.pre_moe.weight, lp + "pre_feedforward_layernorm_2.weight")
                load(ff.post_dense.weight, lp + "post_feedforward_layernorm_1.weight")
                load(ff.post_moe.weight, lp + "post_feedforward_layernorm_2.weight")
                load(ff.post_combined.weight, lp + "post_feedforward_layernorm.weight")

                load(ff.mlp.gate_proj.weight, lp + "mlp.gate_proj.weight")
                load(ff.mlp.up_proj.weight, lp + "mlp.up_proj.weight")
                load(ff.mlp.down_proj.weight, lp + "mlp.down_proj.weight")

                load(ff.moe.router_proj.weight, lp + "router.proj.weight")
                load(ff.moe.router_scale, lp + "router.scale")
                load(ff.moe.per_expert_scale, lp + "router.per_expert_scale")
                # Experts are stacked; allow a transpose in case orientation differs.
                load(ff.moe.gate_up_proj, lp + "experts.gate_up_proj", transpose_ok=True)
                load(ff.moe.down_proj, lp + "experts.down_proj", transpose_ok=True)

                load(layer.layer_scalar, lp + "layer_scalar")

        return model


