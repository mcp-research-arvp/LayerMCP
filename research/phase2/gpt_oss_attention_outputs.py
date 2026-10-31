"""Sample-level GPT-OSS attention-output capture for fixed-token comparisons.

This helper deliberately does not run generation or scoring.  Given already
rendered GPT-OSS prompt token IDs, it records two representations at selected
attention layers during one teacher-forced forward pass:

* ``weighted_value``: the input to ``AttentionBlock.out``, i.e. the flattened
  per-head ``Attention × V`` result; and
* ``projected_contribution``: ``AttentionBlock.out(Attention × V)`` before
  ``AttentionBlock.forward`` adds it to the residual stream.

The same token IDs and absolute positions are used for the control and
intervention passes.  This is intentionally a small interactive helper, not a
batch recording protocol or a semantic-information score.
"""

from __future__ import annotations

from collections.abc import Iterable
from contextlib import AbstractContextManager
import hashlib
import math
from typing import Any

import torch

from research.phase2.intervention import (
    AttentionTarget,
    StrengthMode,
    temporary_attention_intervention,
)


CAPTURE_KINDS = ("weighted_value", "projected_contribution")
EPSILON = 1e-12


def _require_input_ids(input_ids: torch.Tensor) -> torch.Tensor:
    if not isinstance(input_ids, torch.Tensor):
        raise TypeError("input_ids must be a torch.Tensor")
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("input_ids must have shape (1, token_count)")
    if input_ids.shape[1] < 1:
        raise ValueError("input_ids must contain at least one token")
    if input_ids.dtype not in (torch.int64, torch.int32, torch.int16, torch.int8, torch.uint8):
        raise TypeError("input_ids must use an integer dtype")
    return input_ids


def input_token_identity(input_ids: torch.Tensor) -> dict[str, Any]:
    """Return a portable identity for one fixed sequence of input token IDs."""
    input_ids = _require_input_ids(input_ids)
    canonical = input_ids.detach().to(device="cpu", dtype=torch.int64).contiguous()
    return {
        "token_count": int(canonical.shape[1]),
        "token_ids_sha256": "sha256:" + hashlib.sha256(canonical.numpy().tobytes()).hexdigest(),
    }


def _capture_positions(token_positions: Iterable[int] | None, token_count: int) -> tuple[int, ...]:
    if token_positions is None:
        return tuple(range(token_count))
    positions = tuple(token_positions)
    if not positions or any(not isinstance(position, int) for position in positions):
        raise ValueError("token_positions must be a non-empty sequence of integer positions")
    if len(set(positions)) != len(positions) or any(position < 0 or position >= token_count for position in positions):
        raise ValueError(f"token_positions must be unique positions in 0..{token_count - 1}")
    return positions


def _attention_out_module(model: Any, layer_index: int) -> Any:
    blocks = getattr(model, "block", None)
    if blocks is None:
        raise TypeError("GPT-OSS capture requires model.block")
    if not isinstance(layer_index, int) or layer_index < 0 or layer_index >= len(blocks):
        raise ValueError(f"Invalid GPT-OSS attention layer {layer_index}")
    attention = getattr(blocks[layer_index], "attn", None)
    output_projection = getattr(attention, "out", None)
    if output_projection is None or not hasattr(output_projection, "register_forward_pre_hook"):
        raise TypeError(
            f"GPT-OSS layer {layer_index} has no hookable attention output projection (.attn.out)"
        )
    return output_projection


def _as_tensor(value: Any, *, label: str) -> torch.Tensor:
    tensor = value[0] if isinstance(value, tuple) else value
    if not isinstance(tensor, torch.Tensor) or tensor.ndim != 3:
        raise TypeError(f"{label} must be a rank-3 tensor, got {type(tensor).__name__}")
    if tensor.shape[0] != 1:
        raise ValueError(f"{label} batch size must be one")
    return tensor


class AttentionOutputObserver(AbstractContextManager["AttentionOutputObserver"]):
    """Temporary hooks for exact GPT-OSS attention-output capture.

    ``weighted_value`` is captured at the input of the output projection.
    ``projected_contribution`` is captured at that projection's output; the
    residual addition happens later in ``AttentionBlock.forward``.
    """

    def __init__(
        self,
        model: Any,
        layers: Iterable[int],
        *,
        token_positions: Iterable[int],
        enabled: bool = True,
    ) -> None:
        normalized_layers = tuple(layers)
        if not normalized_layers or any(not isinstance(layer, int) for layer in normalized_layers):
            raise ValueError("layers must be a non-empty sequence of integer layer indices")
        if len(set(normalized_layers)) != len(normalized_layers):
            raise ValueError("layers must not contain duplicates")
        if not isinstance(enabled, bool):
            raise TypeError("enabled must be a boolean")
        self.model = model
        self.layers = normalized_layers
        self.token_positions = tuple(token_positions)
        self.enabled = enabled
        self.handles: list[Any] = []
        self.captured: dict[int, dict[str, torch.Tensor]] = {}

    def __enter__(self) -> "AttentionOutputObserver":
        if not self.enabled:
            return self
        try:
            for layer_index in self.layers:
                output_projection = _attention_out_module(self.model, layer_index)

                def capture_input(_module: Any, inputs: tuple[Any, ...], *, layer: int = layer_index) -> None:
                    if len(inputs) != 1:
                        raise TypeError("GPT-OSS attention output projection received unexpected inputs")
                    tensor = _as_tensor(inputs[0], label="weighted_value")
                    self.captured.setdefault(layer, {})["weighted_value"] = tensor[
                        :, self.token_positions, :
                    ].detach().to(device="cpu", dtype=torch.float32).clone()

                def capture_output(_module: Any, _inputs: tuple[Any, ...], output: Any, *, layer: int = layer_index) -> None:
                    tensor = _as_tensor(output, label="projected_contribution")
                    self.captured.setdefault(layer, {})["projected_contribution"] = tensor[
                        :, self.token_positions, :
                    ].detach().to(device="cpu", dtype=torch.float32).clone()

                self.handles.extend(
                    [
                        output_projection.register_forward_pre_hook(capture_input),
                        output_projection.register_forward_hook(capture_output),
                    ]
                )
        except Exception:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        for handle in reversed(self.handles):
            handle.remove()
        self.handles.clear()
        return None


def capture_attention_outputs(
    model: Any,
    input_ids: torch.Tensor,
    *,
    layers: Iterable[int],
    token_positions: Iterable[int] | None = None,
    enabled: bool = True,
) -> dict[str, Any]:
    """Teacher-force one fixed prompt and return full selected layer matrices.

    The returned matrices are CPU ``float32`` tensors of shape ``(1,
    selected_tokens, hidden_or_attention_width)``.  No caches are supplied,
    so ``token_positions`` are absolute positions in exactly ``input_ids``.
    Omitting it captures every prompt token; interactive callers should
    normally request a small, documented fixed suffix to bound storage.
    """
    input_ids = _require_input_ids(input_ids)
    identity = input_token_identity(input_ids)
    positions = _capture_positions(token_positions, int(input_ids.shape[1]))
    device = next(model.parameters()).device
    model_input = input_ids.to(device=device, dtype=torch.long).clone()
    with torch.no_grad(), AttentionOutputObserver(
        model, layers, token_positions=positions, enabled=enabled
    ) as observer:
        logits = model(model_input)
    if not enabled:
        return {
            "enabled": False,
            "input": identity,
            "captured_token_positions": list(positions),
            "layers": {},
            "model_output_shape": list(logits.shape),
        }
    expected = set(observer.layers)
    if set(observer.captured) != expected or any(set(values) != set(CAPTURE_KINDS) for values in observer.captured.values()):
        raise RuntimeError("GPT-OSS attention-output hooks did not capture every requested representation")
    return {
        "enabled": True,
        "input": identity,
        "captured_token_positions": list(positions),
        "layers": observer.captured,
        # The full vocabulary logits are deliberately not retained.  This is
        # internal prompt-representation analysis, not continuation scoring,
        # and a prompt-length-by-vocabulary tensor would dominate storage.
        "model_output_shape": list(logits.shape),
    }


def _matrix(tensor: torch.Tensor) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor) or tensor.ndim != 3 or tensor.shape[0] != 1:
        raise ValueError("captured tensor must have shape (1, token_count, width)")
    return tensor.squeeze(0).to(device="cpu", dtype=torch.float64)


def representation_measurements(reference: torch.Tensor, observed: torch.Tensor) -> dict[str, float | int]:
    """Compare equal-shape fixed-token matrices without treating vectors as probabilities.

    Spectral entropy and effective rank use normalized singular values of the
    reference and observed token-by-feature matrices.  Distance and cosine use
    their flattened matrices.  These are structural summaries, not measures of
    semantic information or causal responsibility.
    """
    left, right = _matrix(reference), _matrix(observed)
    if left.shape != right.shape:
        raise ValueError(f"capture shape differs: {tuple(left.shape)} != {tuple(right.shape)}")

    def spectrum(matrix: torch.Tensor, prefix: str) -> dict[str, float]:
        singular_values = torch.linalg.svdvals(matrix)
        positive = singular_values[singular_values > EPSILON]
        if positive.numel() == 0:
            return {f"{prefix}_spectral_entropy": 0.0, f"{prefix}_effective_rank": 0.0}
        probabilities = positive / positive.sum()
        entropy = float(-(probabilities * probabilities.log()).sum().item())
        return {
            f"{prefix}_spectral_entropy": entropy,
            f"{prefix}_effective_rank": float(math.exp(entropy)),
        }

    delta = right - left
    left_flat, right_flat = left.reshape(-1), right.reshape(-1)
    denominator = max(float(torch.linalg.vector_norm(left_flat).item()), EPSILON)
    cosine_denominator = max(
        float(torch.linalg.vector_norm(left_flat).item()) * float(torch.linalg.vector_norm(right_flat).item()),
        EPSILON,
    )
    result: dict[str, float | int] = {
        "token_count": int(left.shape[0]),
        "feature_width": int(left.shape[1]),
        "normalized_frobenius_distance": float(torch.linalg.matrix_norm(delta, ord="fro").item()) / denominator,
        "cosine_similarity": float(torch.dot(left_flat, right_flat).item()) / cosine_denominator,
    }
    result.update(spectrum(left, "reference"))
    result.update(spectrum(right, "observed"))
    return result


def compare_attention_output_captures(
    control: dict[str, Any], intervention: dict[str, Any],
) -> dict[int, dict[str, dict[str, float | int]]]:
    """Measure each capture kind for identical recorded input token IDs."""
    if not control.get("enabled") or not intervention.get("enabled"):
        raise ValueError("both captures must have recording enabled")
    if control.get("input") != intervention.get("input"):
        raise ValueError("control and intervention do not use identical input token IDs")
    if control.get("captured_token_positions") != intervention.get("captured_token_positions"):
        raise ValueError("control and intervention do not use identical captured token positions")
    if set(control["layers"]) != set(intervention["layers"]):
        raise ValueError("control and intervention capture different layers")
    return {
        layer: {
            kind: representation_measurements(control["layers"][layer][kind], intervention["layers"][layer][kind])
            for kind in CAPTURE_KINDS
        }
        for layer in control["layers"]
    }


def capture_paired_attention_outputs(
    model: Any,
    input_ids: torch.Tensor,
    *,
    observed_layers: Iterable[int],
    intervention_layer: int,
    target: AttentionTarget = "qkv",
    method: str = "noise",
    strength: float = 0.01,
    strength_mode: StrengthMode = "tensor_rms",
    seed: int = 1234,
    token_positions: Iterable[int] | None = None,
    recording_enabled: bool = True,
) -> dict[str, Any]:
    """Capture unchanged and temporarily altered outputs on identical token IDs."""
    observed_layers = tuple(observed_layers)
    control = capture_attention_outputs(
        model, input_ids, layers=observed_layers, token_positions=token_positions,
        enabled=recording_enabled,
    )
    with temporary_attention_intervention(
        model,
        intervention_layer,
        target=target,
        method=method,
        strength=strength,
        strength_mode=strength_mode,
        seed=seed,
        enabled=True,
    ) as intervention_context:
        intervention = capture_attention_outputs(
            model, input_ids, layers=observed_layers, token_positions=token_positions,
            enabled=recording_enabled,
        )
    if intervention_context.restored_exactly is not True:
        raise RuntimeError("Attention intervention did not restore exactly after attention-output capture")
    return {
        "input": input_token_identity(input_ids),
        "observed_layers": list(observed_layers),
        "captured_token_positions": control["captured_token_positions"],
        "intervention": {
            "layer": intervention_layer,
            "target": target,
            "method": method,
            "strength": strength,
            "strength_mode": strength_mode,
            "seed": seed,
            "perturbation": intervention_context.perturbation,
            "restored_exactly": intervention_context.restored_exactly,
        },
        "control": control,
        "altered": intervention,
        "measurements": (
            compare_attention_output_captures(control, intervention) if recording_enabled else {}
        ),
    }
