"""Planning helpers for durable one-layer-at-a-time intervention screens.

The planner is model-neutral: callers supply the layers and safe targets
discovered from a loaded model.  It deliberately does not load weights, run
generation, or write results.  This keeps the requested screen visible before
an expensive GPU evaluation starts.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import math
from numbers import Real

from research.phase2.intervention import SUPPORTED_ATTENTION_TARGETS, SUPPORTED_METHODS


ALL_LAYERS = "all"
REPRESENTATIVE_LAYERS = "representative"


class LayerScreenPlanError(ValueError):
    """Raised when a durable layer-screen request is invalid."""


LayerRequest = tuple[int, ...] | str


@dataclass(frozen=True)
class LayerScreenPlan:
    """Resolved, bounded description of a one-layer-at-a-time screen."""

    requested_layers: str | tuple[int, ...]
    selected_layers: tuple[int, ...]
    sample_ids: tuple[str, ...]
    seeds: tuple[int, ...]
    target: str
    method: str
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
    return layers


def representative_layers(available_layers: Sequence[int]) -> tuple[int, ...]:
    """Return first, midpoint, and last available layers without assumptions.

    A model with fewer than three layers returns each available layer once.  For
    GPT-OSS's current 24 layers this resolves to ``(0, 12, 23)``.
    """
    layers = _valid_layers(available_layers)
    selected = (layers[0], layers[len(layers) // 2], layers[-1])
    return tuple(_unique(selected))  # type: ignore[return-value]


def resolve_layer_request(
    request: LayerRequest,
    available_layers: Sequence[int],
) -> tuple[int, ...]:
    """Resolve ``all``/``representative`` or validate explicit layer IDs."""
    valid_layers = _valid_layers(available_layers)
    if request == ALL_LAYERS:
        return valid_layers
    if request == REPRESENTATIVE_LAYERS:
        return representative_layers(valid_layers)
    if isinstance(request, str):
        raise LayerScreenPlanError(
            f"layers must be comma-separated IDs, {ALL_LAYERS!r}, or {REPRESENTATIVE_LAYERS!r}; "
            f"got {request!r}."
        )
    requested = tuple(_unique(tuple(request)))
    if not requested or any(isinstance(layer, bool) or not isinstance(layer, int) for layer in requested):
        raise LayerScreenPlanError("layers must contain one or more distinct integer IDs.")
    unavailable = [layer for layer in requested if layer not in valid_layers]
    if unavailable:
        raise LayerScreenPlanError(
            f"Requested unavailable layer IDs {unavailable}; available: {list(valid_layers)}."
        )
    return requested  # type: ignore[return-value]


def plan_layer_screen(
    *,
    layer_request: LayerRequest,
    available_layers: Sequence[int],
    available_targets_by_layer: Mapping[int, Sequence[str]],
    sample_ids: Sequence[str],
    seeds: Sequence[int],
    target: str,
    method: str,
    strength: float,
) -> LayerScreenPlan:
    """Validate and describe a batch screen before model generation begins."""
    selected_layers = resolve_layer_request(layer_request, available_layers)
    normalized_samples = tuple(_unique(tuple(sample_ids)))
    if not normalized_samples or any(not isinstance(sample_id, str) or not sample_id for sample_id in normalized_samples):
        raise LayerScreenPlanError("sample_ids must contain one or more non-empty strings.")
    normalized_seeds = tuple(_unique(tuple(seeds)))
    if not normalized_seeds or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in normalized_seeds):
        raise LayerScreenPlanError("seeds must contain one or more integer values.")
    if target not in SUPPORTED_ATTENTION_TARGETS:
        raise LayerScreenPlanError(
            f"Unsupported attention target {target!r}; supported: {sorted(SUPPORTED_ATTENTION_TARGETS)}."
        )
    if method not in SUPPORTED_METHODS:
        raise LayerScreenPlanError(
            f"Unsupported intervention method {method!r}; supported: {sorted(SUPPORTED_METHODS)}."
        )
    if isinstance(strength, bool) or not isinstance(strength, Real) or not math.isfinite(float(strength)) or strength < 0:
        raise LayerScreenPlanError("strength must be a finite non-negative number.")
    unavailable_targets = {
        layer: tuple(available_targets_by_layer.get(layer, ()))
        for layer in selected_layers
        if target not in available_targets_by_layer.get(layer, ())
    }
    if unavailable_targets:
        details = "; ".join(
            f"layer {layer}: {list(targets)}" for layer, targets in unavailable_targets.items()
        )
        raise LayerScreenPlanError(f"Target {target!r} is not safe for every selected layer; {details}.")
    pair_count_per_layer = len(normalized_samples) * len(normalized_seeds)
    total_pair_count = len(selected_layers) * pair_count_per_layer
    return LayerScreenPlan(
        requested_layers=layer_request,
        selected_layers=selected_layers,
        sample_ids=normalized_samples,  # type: ignore[arg-type]
        seeds=normalized_seeds,  # type: ignore[arg-type]
        target=target,
        method=method,
        strength=float(strength),
        pair_count_per_layer=pair_count_per_layer,
        total_pair_count=total_pair_count,
        total_generation_count=2 * total_pair_count,
    )
