"""Temporary, in-memory attention-parameter interventions.

The API is deliberately structural rather than based on a model-name allowlist.
It supports the layer layouts implemented in this repository and the matching
Hugging Face decoder layout:

* ``model.layers[index].self_attn``;
* ``model.model.layers[index].self_attn`` (or ``.linear_attn`` for the local
  Qwen hybrid implementation); and
* ``model.block[index].attn`` for the local GPT-OSS implementation.

Only registered floating-point parameters below the selected attention module
are changed.  Parameters are copied back in-place on exit, including when the
body of the ``with`` statement raises.  This module never serializes a model or
writes a checkpoint.
"""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
import math
from numbers import Real
from typing import Any, Literal

import torch
from torch import nn


InterventionMethod = Literal["noise", "replace"]
SUPPORTED_METHODS = frozenset({"noise", "replace"})


class AttentionInterventionError(ValueError):
    """Raised when a model cannot be safely targeted by this intervention."""


@dataclass(frozen=True)
class _ParameterSnapshot:
    name: str
    parameter: nn.Parameter
    value: torch.Tensor


def _layer_stack(model: Any) -> tuple[Sequence[nn.Module], str]:
    """Return one explicitly supported decoder-layer container.

    The order is intentional: local Qwen and Hugging Face CausalLM wrappers
    use ``model.model.layers``; local Llama/Phi/Gemma use ``model.layers``;
    and the local GPT-OSS model uses ``model.block``.
    """
    candidates = (
        (getattr(model, "layers", None), "model.layers"),
        (getattr(getattr(model, "model", None), "layers", None), "model.model.layers"),
        (getattr(model, "block", None), "model.block"),
    )
    for layers, path in candidates:
        if isinstance(layers, (nn.ModuleList, list, tuple)):
            if not layers:
                raise AttentionInterventionError(f"{path} is empty; it has no valid layer indices.")
            if not all(isinstance(layer, nn.Module) for layer in layers):
                raise AttentionInterventionError(
                    f"{path} must contain torch.nn.Module decoder layers."
                )
            return layers, path
    raise AttentionInterventionError(
        "Unsupported model structure: expected one of model.layers, "
        "model.model.layers, or model.block to be a non-empty ModuleList."
    )


def _attention_module(layer: nn.Module, layer_path: str) -> tuple[nn.Module, str] | None:
    matches = [
        (name, getattr(layer, name, None))
        for name in ("self_attn", "linear_attn", "attn")
        if isinstance(getattr(layer, name, None), nn.Module)
    ]
    if len(matches) == 1:
        name, module = matches[0]
        return module, f"{layer_path}.{name}"
    if len(matches) > 1:
        names = ", ".join(name for name, _ in matches)
        raise AttentionInterventionError(
            f"Unsupported ambiguous attention structure at {layer_path}: found {names}."
        )
    return None


def _validate_layer_index(layer_index: int) -> None:
    if isinstance(layer_index, bool) or not isinstance(layer_index, int):
        raise TypeError("layer_index must be an integer.")


def available_attention_layers(model: Any) -> tuple[int, ...]:
    """Return layer indices whose attention module is safely identifiable.

    The result is discovered from ``model``; no architecture-specific layer
    count is assumed.  A model with a recognized stack but no recognized
    attention module fails instead of guessing from parameter names.
    """
    layers, stack_path = _layer_stack(model)
    available = tuple(
        index
        for index, layer in enumerate(layers)
        if _attention_module(layer, f"{stack_path}[{index}]") is not None
    )
    if not available:
        raise AttentionInterventionError(
            f"No identifiable attention modules found in {stack_path}. Expected exactly one "
            "of self_attn, linear_attn, or attn per target layer."
        )
    return available


def attention_module_for_layer(model: Any, layer_index: int) -> nn.Module:
    """Return the selected layer's supported attention module.

    This public resolver is useful for inspection and makes the targeting rule
    testable without applying an intervention.
    """
    _validate_layer_index(layer_index)
    layers, stack_path = _layer_stack(model)
    available = available_attention_layers(model)
    if layer_index not in available:
        valid = ", ".join(str(index) for index in available)
        raise AttentionInterventionError(
            f"Layer index {layer_index} has no supported attention module. "
            f"Valid layer indices for this model: {valid}."
        )
    result = _attention_module(layers[layer_index], f"{stack_path}[{layer_index}]")
    if result is None:  # Defensive: available_attention_layers already checked this.
        raise AssertionError("available attention layer unexpectedly lacked an attention module")
    return result[0]


def _validate_model_is_not_quantized(model: Any) -> None:
    if getattr(model, "is_loaded_in_4bit", False) or getattr(model, "is_loaded_in_8bit", False):
        raise AttentionInterventionError(
            "Quantized models are not supported: this API requires safely writable "
            "floating-point attention parameters. Load without 4-bit or 8-bit quantization."
        )


def _snapshots(module: nn.Module) -> list[_ParameterSnapshot]:
    snapshots: list[_ParameterSnapshot] = []
    for name, parameter in module.named_parameters(recurse=True):
        if not parameter.is_floating_point():
            raise AttentionInterventionError(
                f"Attention parameter {name!r} has dtype {parameter.dtype}; only floating-point "
                "parameters can be safely perturbed."
            )
        snapshots.append(
            _ParameterSnapshot(name=name, parameter=parameter, value=parameter.detach().clone())
        )
    if not snapshots:
        raise AttentionInterventionError(
            "The selected attention module has no registered parameters to intervene on."
        )
    return snapshots


def _all_modules(module: nn.Module, path: str = "model") -> list[tuple[str, nn.Module]]:
    """Return module paths without deduplicating aliases.

    ``named_modules`` normally deduplicates module instances.  Keeping aliases
    here lets us reject an attention module that is reused elsewhere in the
    model, because changing it would no longer be a one-layer intervention.
    """
    result = [(path, module)]
    for name, child in module._modules.items():
        if child is not None:
            result.extend(_all_modules(child, f"{path}.{name}"))
    return result


def _ensure_parameters_are_attention_local(
    model: Any,
    attention_module: nn.Module,
    snapshots: list[_ParameterSnapshot],
) -> None:
    """Reject tied parameters that would also change an MLP or another layer."""
    if not isinstance(model, nn.Module):
        raise AttentionInterventionError("The supplied model must be a torch.nn.Module.")
    target_parameter_ids = {id(item.parameter) for item in snapshots}
    attention_descendants = {id(module) for _, module in _all_modules(attention_module)}
    model_modules = _all_modules(model)
    attention_paths = [path for path, module in model_modules if module is attention_module]
    if len(attention_paths) != 1:
        raise AttentionInterventionError(
            "Unsupported shared attention module: it appears at multiple model paths "
            f"({', '.join(attention_paths)}). A one-layer intervention would affect more than one layer."
        )
    for path, module in model_modules:
        if id(module) in attention_descendants:
            continue
        for parameter_name, parameter in module._parameters.items():
            if parameter is not None and id(parameter) in target_parameter_ids:
                raise AttentionInterventionError(
                    "Unsupported tied attention parameter: "
                    f"{path}.{parameter_name} is shared outside the selected attention module. "
                    "A one-layer attention-only intervention would also change another component."
                )


class AttentionParameterIntervention(AbstractContextManager["AttentionParameterIntervention"]):
    """A one-layer, reversible attention-parameter intervention.

    ``noise`` replaces each value ``p`` with ``p + strength * z`` and
    ``replace`` replaces it with ``strength * z``, where every ``z`` is sampled
    independently from the standard normal distribution using a CPU generator
    seeded with ``seed``.  Sampling on CPU gives the same pseudorandom stream
    for CPU and CUDA model parameters; generated values are then cast and copied
    to each parameter's original device and dtype.  Shapes are unchanged.
    """

    def __init__(
        self,
        model: Any,
        layer_index: int,
        *,
        method: InterventionMethod = "noise",
        seed: int = 0,
        strength: float = 0.01,
        enabled: bool = True,
    ) -> None:
        _validate_layer_index(layer_index)
        if method not in SUPPORTED_METHODS:
            supported = ", ".join(sorted(SUPPORTED_METHODS))
            raise ValueError(f"Unsupported intervention method {method!r}. Expected one of: {supported}.")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("seed must be an integer.")
        if isinstance(strength, bool) or not isinstance(strength, Real):
            raise TypeError("strength must be a finite non-negative number.")
        if not math.isfinite(float(strength)) or strength < 0:
            raise ValueError("strength must be a finite non-negative number.")
        if not isinstance(enabled, bool):
            raise TypeError("enabled must be a boolean.")

        self.model = model
        self.layer_index = layer_index
        self.method = method
        self.seed = seed
        self.strength = float(strength)
        self.enabled = enabled
        self._saved: list[_ParameterSnapshot] = []
        self._changed_parameter_names: tuple[str, ...] = ()
        self._restored_exactly: bool | None = None
        self._active = False

    @property
    def changed_parameter_names(self) -> tuple[str, ...]:
        """Names relative to the selected attention module whose values changed."""
        return self._changed_parameter_names

    @property
    def restored_exactly(self) -> bool | None:
        """Whether saved values matched after exit; ``None`` before restoration."""
        return self._restored_exactly

    def _restore(self) -> None:
        with torch.no_grad():
            for item in self._saved:
                item.parameter.copy_(item.value)
        self._restored_exactly = all(
            torch.equal(item.parameter, item.value) for item in self._saved
        )
        self._active = False

    def __enter__(self) -> "AttentionParameterIntervention":
        if not self.enabled:
            return self

        _validate_model_is_not_quantized(self.model)
        module = attention_module_for_layer(self.model, self.layer_index)
        self._saved = _snapshots(module)
        _ensure_parameters_are_attention_local(self.model, module, self._saved)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed)

        try:
            with torch.no_grad():
                for item in self._saved:
                    noise = torch.randn(
                        item.value.shape,
                        generator=generator,
                        device="cpu",
                        dtype=torch.float32,
                    ).to(device=item.parameter.device, dtype=item.parameter.dtype)
                    if self.method == "noise":
                        replacement = item.value + noise * self.strength
                    else:
                        replacement = noise * self.strength
                    item.parameter.copy_(replacement)
            self._changed_parameter_names = tuple(
                item.name
                for item in self._saved
                if not torch.equal(item.parameter, item.value)
            )
            self._active = True
        except Exception:
            self._restore()
            self._saved.clear()
            raise
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self._active:
            self._restore()
        return None


def temporary_attention_intervention(
    model: Any,
    layer_index: int,
    *,
    method: InterventionMethod = "noise",
    seed: int = 0,
    strength: float = 0.01,
    enabled: bool = True,
) -> AttentionParameterIntervention:
    """Create a context manager that temporarily changes one attention module.

    The returned context manager changes no model values until it is entered.
    Set ``enabled=False`` for a true no-op: it does not inspect or mutate the
    model, which allows a caller to run the exact same evaluation code for a
    baseline condition.
    """
    return AttentionParameterIntervention(
        model,
        layer_index,
        method=method,
        seed=seed,
        strength=strength,
        enabled=enabled,
    )
