"""Shared non-materializing attention for local text-model runtimes."""

from __future__ import annotations

from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.attention.bias import causal_lower_right


CUDA_MEMORY_EFFICIENT_BACKENDS = [
    SDPBackend.FLASH_ATTENTION,
    SDPBackend.EFFICIENT_ATTENTION,
]


def causal_scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    offset: int,
    scale: float | None = None,
) -> torch.Tensor:
    """Apply causal GQA without materializing a per-head score matrix.

    Tensors use ``(batch, heads, sequence, head_dim)`` layout. ``offset`` is
    the number of cached key/value positions preceding the first query. CUDA
    execution is restricted to fused, sub-quadratic-workspace SDPA backends;
    an unsupported kernel therefore fails instead of silently falling back to
    the quadratic math implementation that caused the long-context OOMs.
    """
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("query, key, and value must be rank-four tensors")
    if key.shape != value.shape:
        raise ValueError("key and value shapes must match")
    if query.shape[0] != key.shape[0] or query.shape[-1] != key.shape[-1]:
        raise ValueError("query and key batch/head dimensions are incompatible")
    if offset < 0:
        raise ValueError("cache offset must be non-negative")

    query_length = query.shape[-2]
    key_length = key.shape[-2]
    if key_length != offset + query_length:
        raise ValueError(
            "key length must equal cache offset plus query length: "
            f"{key_length} != {offset} + {query_length}"
        )
    if query.shape[1] % key.shape[1] != 0:
        raise ValueError("query heads must be divisible by key/value heads")

    attention_mask = None
    is_causal = offset == 0
    if offset and query_length > 1:
        # PyTorch 2.13 dispatches this lower-right causal bias directly to
        # fused kernels.  A materialized Boolean QxK mask can disqualify Flash
        # GQA and reintroduce a quadratic allocation during chunked decode.
        attention_mask = causal_lower_right(query_length, key_length)

    backend_context = (
        sdpa_kernel(CUDA_MEMORY_EFFICIENT_BACKENDS)
        if query.device.type == "cuda"
        else nullcontext()
    )
    with backend_context:
        return F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=is_causal,
            scale=scale,
            enable_gqa=query.shape[1] != key.shape[1],
        )
