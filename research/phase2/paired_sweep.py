"""Planning helpers for small, paired attention-intervention sweeps.

This module deliberately plans only the Cartesian product of user-selected
settings.  It does not load a model, generate text, or write results.  The
notebook uses it to make a small interactive sweep explicit before repeatedly
calling the existing one-sample paired evaluator.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from numbers import Real

from research.phase2.intervention import SUPPORTED_ATTENTION_TARGETS, SUPPORTED_METHODS


ALL_AVAILABLE = "all"


class PairedSweepPlanError(ValueError):
    """Raised when an interactive paired-sweep request is invalid or too large."""


@dataclass(frozen=True)
class PairedSweepSetting:
    """One control/intervention pair to run for one saved example."""

    sample_id: str
    layer_index: int
    target: str
    method: str
    strength: float
    seed: int


def _unique(values: Sequence[object]) -> tuple[object, ...]:
    return tuple(dict.fromkeys(values))


def _sequence_or_all(value: Sequence[object] | str, *, name: str) -> tuple[object, ...] | str:
    if value == ALL_AVAILABLE:
        return ALL_AVAILABLE
    if isinstance(value, str):
        raise PairedSweepPlanError(f"{name} must be a sequence or {ALL_AVAILABLE!r}, not {value!r}.")
    values = _unique(tuple(value))
    if not values:
        raise PairedSweepPlanError(f"{name} must not be empty.")
    return values


def plan_paired_sweep(
    *,
    sample_ids: Sequence[str],
    layers: Sequence[int] | str,
    targets: Sequence[str] | str,
    methods: Sequence[str],
    strengths: Sequence[float],
    seeds: Sequence[int],
    available_layers: Sequence[int],
    available_targets_by_layer: Mapping[int, Sequence[str]],
    max_interventions: int,
    allow_large: bool = False,
) -> tuple[PairedSweepSetting, ...]:
    """Validate and expand a user-selected paired sweep.

    ``layers`` and ``targets`` accept the literal ``"all"``.  ``targets="all"``
    expands only to targets safely available at each chosen layer.  Every
    returned setting represents one complete control/intervention pair, so the
    size guard prevents an accidental interactive run with many model loads of
    the evaluator.  Callers may set ``allow_large=True`` deliberately; this is
    intended for a durable batch runner rather than a short interactive shell.
    """
    if isinstance(max_interventions, bool) or not isinstance(max_interventions, int):
        raise PairedSweepPlanError("max_interventions must be a positive integer.")
    if max_interventions <= 0:
        raise PairedSweepPlanError("max_interventions must be a positive integer.")

    normalized_samples = _sequence_or_all(sample_ids, name="sample_ids")
    if normalized_samples == ALL_AVAILABLE:
        raise PairedSweepPlanError("sample_ids must list explicit saved example IDs.")
    if any(not isinstance(item, str) or not item for item in normalized_samples):
        raise PairedSweepPlanError("sample_ids must contain non-empty strings.")

    valid_layers = tuple(_unique(tuple(available_layers)))
    if not valid_layers or any(isinstance(item, bool) or not isinstance(item, int) for item in valid_layers):
        raise PairedSweepPlanError("available_layers must contain integer layer IDs.")
    requested_layers = _sequence_or_all(layers, name="layers")
    if requested_layers == ALL_AVAILABLE:
        normalized_layers = valid_layers
    else:
        if any(isinstance(item, bool) or not isinstance(item, int) for item in requested_layers):
            raise PairedSweepPlanError("layers must contain integer layer IDs.")
        unavailable_layers = [item for item in requested_layers if item not in valid_layers]
        if unavailable_layers:
            raise PairedSweepPlanError(
                f"layers contains unavailable IDs {unavailable_layers}; available: {valid_layers}."
            )
        normalized_layers = requested_layers

    requested_targets = _sequence_or_all(targets, name="targets")
    if requested_targets != ALL_AVAILABLE:
        unknown_targets = [item for item in requested_targets if item not in SUPPORTED_ATTENTION_TARGETS]
        if unknown_targets:
            raise PairedSweepPlanError(
                f"targets contains unsupported names {unknown_targets}; supported: {sorted(SUPPORTED_ATTENTION_TARGETS)}."
            )

    normalized_methods = _sequence_or_all(methods, name="methods")
    if normalized_methods == ALL_AVAILABLE:
        normalized_methods = tuple(sorted(SUPPORTED_METHODS))
    unknown_methods = [item for item in normalized_methods if item not in SUPPORTED_METHODS]
    if unknown_methods:
        raise PairedSweepPlanError(
            f"methods contains unsupported names {unknown_methods}; supported: {sorted(SUPPORTED_METHODS)}."
        )

    raw_strengths = _sequence_or_all(strengths, name="strengths")
    if raw_strengths == ALL_AVAILABLE:
        raise PairedSweepPlanError("strengths must list explicit non-negative values.")
    normalized_strengths: list[float] = []
    for value in raw_strengths:
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)) or value < 0:
            raise PairedSweepPlanError("strengths must contain finite non-negative numbers.")
        normalized_strengths.append(float(value))
    normalized_strengths = list(_unique(normalized_strengths))

    raw_seeds = _sequence_or_all(seeds, name="seeds")
    if raw_seeds == ALL_AVAILABLE:
        raise PairedSweepPlanError("seeds must list explicit integer values.")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in raw_seeds):
        raise PairedSweepPlanError("seeds must contain integers.")
    normalized_seeds = _unique(raw_seeds)

    settings: list[PairedSweepSetting] = []
    for layer_index in normalized_layers:
        available_targets = tuple(_unique(tuple(available_targets_by_layer.get(layer_index, ()))))
        if not available_targets:
            raise PairedSweepPlanError(f"No safe attention targets are available at layer {layer_index}.")
        if requested_targets == ALL_AVAILABLE:
            layer_targets = available_targets
        else:
            unavailable_targets = [item for item in requested_targets if item not in available_targets]
            if unavailable_targets:
                raise PairedSweepPlanError(
                    f"targets {unavailable_targets} are unavailable at layer {layer_index}; "
                    f"available: {available_targets}."
                )
            layer_targets = requested_targets
        for sample_id in normalized_samples:
            for target in layer_targets:
                for method in normalized_methods:
                    for strength in normalized_strengths:
                        for seed in normalized_seeds:
                            settings.append(
                                PairedSweepSetting(
                                    sample_id=sample_id,
                                    layer_index=layer_index,
                                    target=target,
                                    method=method,
                                    strength=strength,
                                    seed=seed,
                                )
                            )

    if len(settings) > max_interventions and not allow_large:
        raise PairedSweepPlanError(
            f"Sweep plans {len(settings)} control/intervention pairs, exceeding "
            f"SWEEP_MAX_INTERVENTIONS={max_interventions}. Reduce the grid, or set "
            "SWEEP_ALLOW_LARGE=True only for a deliberate longer run."
        )
    return tuple(settings)
