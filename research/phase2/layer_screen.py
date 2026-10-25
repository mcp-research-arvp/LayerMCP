"""Portable planning helpers for one-attention-layer-at-a-time screens.

The planner is intentionally model-neutral.  Callers provide layers and targets
discovered from a loaded model; this module never loads a checkpoint, generates
text, writes an experiment result, or assumes a particular layer count.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import math
from numbers import Real

from research.phase2.intervention import SUPPORTED_ATTENTION_TARGETS, SUPPORTED_METHODS


ALL_LAYERS = "all"
REPRESENTATIVE_LAYERS = "representative"
LayerRequest = tuple[int, ...] | str


class LayerScreenPlanError(ValueError):
    """Raised when a requested screen cannot be run safely."""


@dataclass(frozen=True)
class LayerScreenPlan:
    """Resolved, bounded description of a one-layer-at-a-time screen."""

    requested_layers: str | tuple[int, ...]
    selected_layers: tuple[int, ...]
    sample_ids: tuple[str, ...]
    seeds: tuple[int, ...]
    target: str
    method: str
    strength_mode: str
    strength: float
    pair_count_per_layer: int
    total_pair_count: int
    total_generation_count: int

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _unique(values: Sequence[object]) -> tuple[object, ...]:
    return tuple(dict.fromkeys(values))


def _valid_layers(available_layers: Sequence[int]) -> tuple[int, ...]:
    layers = tuple(_unique(tuple(available_layers)))
    if not layers or any(isinstance(layer, bool) or not isinstance(layer, int) for layer in layers):
        raise LayerScreenPlanError("available_layers must contain one or more integer layer IDs.")
    return layers  # type: ignore[return-value]


def representative_layers(available_layers: Sequence[int]) -> tuple[int, ...]:
    """Return first, midpoint, and last available layers, without size assumptions."""
    layers = _valid_layers(available_layers)
    return _unique((layers[0], layers[len(layers) // 2], layers[-1]))  # type: ignore[return-value]


def resolve_layer_request(request: LayerRequest, available_layers: Sequence[int]) -> tuple[int, ...]:
    """Resolve ``all``/``representative`` or validate explicit model layer IDs."""
    valid = _valid_layers(available_layers)
    if request == ALL_LAYERS:
        return valid
    if request == REPRESENTATIVE_LAYERS:
        return representative_layers(valid)
    if isinstance(request, str):
        raise LayerScreenPlanError(
            f"layers must be comma-separated IDs, {ALL_LAYERS!r}, or {REPRESENTATIVE_LAYERS!r}; got {request!r}."
        )
    selected = tuple(_unique(tuple(request)))
    if not selected or any(isinstance(layer, bool) or not isinstance(layer, int) for layer in selected):
        raise LayerScreenPlanError("layers must contain one or more distinct integer IDs.")
    missing = [layer for layer in selected if layer not in valid]
    if missing:
        raise LayerScreenPlanError(f"Requested unavailable layer IDs {missing}; available: {list(valid)}.")
    return selected  # type: ignore[return-value]


def plan_layer_screen(
    *,
    layer_request: LayerRequest,
    available_layers: Sequence[int],
    available_targets_by_layer: Mapping[int, Sequence[str]],
    sample_ids: Sequence[str],
    seeds: Sequence[int],
    target: str,
    method: str,
    strength_mode: str,
    strength: float,
) -> LayerScreenPlan:
    """Validate and describe a screen before expensive generation begins."""
    selected_layers = resolve_layer_request(layer_request, available_layers)
    samples = tuple(_unique(tuple(sample_ids)))
    if not samples or any(not isinstance(item, str) or not item for item in samples):
        raise LayerScreenPlanError("sample_ids must contain one or more non-empty strings.")
    normalized_seeds = tuple(_unique(tuple(seeds)))
    if not normalized_seeds or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in normalized_seeds):
        raise LayerScreenPlanError("seeds must contain one or more integer values.")
    if target not in SUPPORTED_ATTENTION_TARGETS:
        raise LayerScreenPlanError(f"Unsupported attention target {target!r}.")
    if method not in SUPPORTED_METHODS:
        raise LayerScreenPlanError(f"Unsupported intervention method {method!r}.")
    if strength_mode not in {"absolute", "tensor_rms"}:
        raise LayerScreenPlanError("strength_mode must be 'absolute' or 'tensor_rms'.")
    if isinstance(strength, bool) or not isinstance(strength, Real) or not math.isfinite(float(strength)) or strength < 0:
        raise LayerScreenPlanError("strength must be a finite non-negative number.")
    unavailable = {
        layer: tuple(available_targets_by_layer.get(layer, ()))
        for layer in selected_layers
        if target not in available_targets_by_layer.get(layer, ())
    }
    if unavailable:
        detail = "; ".join(f"layer {layer}: {list(targets)}" for layer, targets in unavailable.items())
        raise LayerScreenPlanError(f"Target {target!r} is not safe for every selected layer; {detail}.")
    pair_count_per_layer = len(samples) * len(normalized_seeds)
    return LayerScreenPlan(
        requested_layers=layer_request,
        selected_layers=selected_layers,
        sample_ids=samples,  # type: ignore[arg-type]
        seeds=normalized_seeds,  # type: ignore[arg-type]
        target=target,
        method=method,
        strength_mode=strength_mode,
        strength=float(strength),
        pair_count_per_layer=pair_count_per_layer,
        total_pair_count=len(selected_layers) * pair_count_per_layer,
        total_generation_count=2 * len(selected_layers) * pair_count_per_layer,
    )
