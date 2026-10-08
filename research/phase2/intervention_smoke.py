"""One saved-example, one-layer attention-intervention smoke test.

This is a functional check for the local Llama runtime used by
``research.phase2.observe``.  It performs two bounded generations from the
same reconstructed prompt, writes a small JSON summary, and never writes model
weights or invokes the baseline evaluator.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import shutil
from typing import Any

from models.architectures.llama31_8b_pytorch.config import Config
from models.architectures.llama31_8b_pytorch.inference import TokenGenerator
from models.routers.llama31_8b_local_router import resolve_checkpoint_path
from research.phase2.intervention import (
    attention_module_for_layer,
    available_attention_layers,
    temporary_attention_intervention,
)
from research.phase2.observe import _require_configured_path
from research.phase2.replay import (
    ReplayConfig,
    load_live_catalog,
    load_replay_config,
    load_saved_example,
    render_llama_prompt,
)


DEFAULT_MAX_TOKENS = 8


def _safe_output_directory(path: Path) -> Path:
    target = path.expanduser().resolve()
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite smoke-test output directory: {target}")
    if target.name in {"", ".", ".."}:
        raise ValueError("Smoke-test output directory has an unsafe final component")
    return target


def _run_generation(generator: Any, prompt_tokens: tuple[int, ...], temperature: float, max_tokens: int) -> list[int]:
    return list(
        generator.generate(
            list(prompt_tokens),
            stop_tokens=generator.stop_tokens,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    )


def run_generation_pair(
    generator: Any,
    prompt_tokens: tuple[int, ...],
    *,
    layer_index: int,
    temperature: float,
    max_tokens: int,
    strength: float = 0.01,
    seed: int = 1234,
) -> dict[str, Any]:
    """Run the disabled and noisy passes and verify their parameter boundaries."""
    model = generator.model
    valid_layers = available_attention_layers(model)
    if layer_index not in valid_layers:
        valid = ", ".join(str(index) for index in valid_layers)
        raise ValueError(f"Layer index {layer_index} is not valid for this model. Valid indices: {valid}.")

    attention_module = attention_module_for_layer(model, layer_index)
    target_parameter_ids = {id(parameter) for parameter in attention_module.parameters()}
    all_parameters = list(model.named_parameters())

    with temporary_attention_intervention(model, layer_index, enabled=False):
        disabled_tokens = _run_generation(generator, prompt_tokens, temperature, max_tokens)

    versions_before = {id(parameter): parameter._version for _, parameter in all_parameters}
    with temporary_attention_intervention(
        model,
        layer_index,
        method="noise",
        strength=strength,
        seed=seed,
        enabled=True,
    ) as intervention:
        noisy_tokens = _run_generation(generator, prompt_tokens, temperature, max_tokens)
        changed_parameter_names = tuple(
            name
            for name, parameter in all_parameters
            if parameter._version != versions_before[id(parameter)]
        )
        changed_parameter_ids = {
            id(parameter)
            for _, parameter in all_parameters
            if parameter._version != versions_before[id(parameter)]
        }

    unexpected_changes = tuple(
        name
        for name, parameter in all_parameters
        if id(parameter) in changed_parameter_ids and id(parameter) not in target_parameter_ids
    )
    missing_target_changes = tuple(
        name
        for name, parameter in attention_module.named_parameters()
        if id(parameter) not in changed_parameter_ids
    )
    selected_parameter_names = tuple(name for name, _ in attention_module.named_parameters())
    all_selected_values_changed = set(intervention.changed_parameter_names) == set(
        selected_parameter_names
    )
    verification = {
        "only_selected_attention_parameters_changed": not unexpected_changes,
        "all_selected_attention_parameters_changed": not missing_target_changes,
        "all_selected_attention_parameter_values_changed": all_selected_values_changed,
        "changed_model_parameter_names": list(changed_parameter_names),
        "changed_attention_parameter_names": list(intervention.changed_parameter_names),
        "unexpected_changed_parameter_names": list(unexpected_changes),
        "unchanged_selected_attention_parameter_names": list(missing_target_changes),
        "restored_exactly": intervention.restored_exactly,
    }
    if not all(
        (
            verification["only_selected_attention_parameters_changed"],
            verification["all_selected_attention_parameters_changed"],
            verification["all_selected_attention_parameter_values_changed"],
            verification["restored_exactly"],
        )
    ):
        raise RuntimeError(f"Attention-intervention verification failed: {verification}")
    return {
        "valid_layer_indices": list(valid_layers),
        "selected_layer_index": layer_index,
        "method": "noise",
        "strength": strength,
        "seed": seed,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "disabled_token_ids": disabled_tokens,
        "noise_token_ids": noisy_tokens,
        "verification": verification,
    }


def run_smoke(
    config: ReplayConfig,
    *,
    checkpoint_dir: Path,
    output_dir: Path,
    layer_index: int | None,
    max_tokens: int,
) -> Path:
    """Load one saved Llama example and write its intervention smoke summary."""
    _require_configured_path(config.source_run_dir, "--source-run-dir")
    target = _safe_output_directory(output_dir)
    checkpoint = resolve_checkpoint_path(checkpoint_dir)
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Llama checkpoint does not exist: {checkpoint}")

    example = load_saved_example(config.source_run_dir, config.sample_id)
    catalog = load_live_catalog()
    generator = TokenGenerator(checkpoint=str(checkpoint), device=Config.device)
    prompt = render_llama_prompt(generator, example, catalog)
    if config.require_registry_match and not prompt.registry_exact_match:
        raise ValueError("Saved and live MCP registry metadata differ; exact replay is required.")

    valid_layers = available_attention_layers(generator.model)
    selected_layer = layer_index if layer_index is not None else valid_layers[0]
    result = run_generation_pair(
        generator,
        prompt.token_ids,
        layer_index=selected_layer,
        temperature=prompt.generation_settings["temperature"],
        max_tokens=max_tokens,
    )
    result.update(
        {
            "kind": "phase2_attention_intervention_smoke_v1",
            "checkpoint_path": str(checkpoint),
            "source_run_directory": str(example.source_run_dir),
            "source_sample_id": config.sample_id,
            "prompt_token_count": len(prompt.token_ids),
            "registry_exact_match": prompt.registry_exact_match,
        }
    )

    target.mkdir(parents=True)
    try:
        summary_path = target / "intervention_smoke.json"
        summary_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (target / "INTERVENTION_SMOKE_COMPLETE").write_text("\n", encoding="utf-8")
    except Exception:
        shutil.rmtree(target)
        raise
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one local-Llama attention-intervention smoke test.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layer-index", type=int)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    args = parser.parse_args()
    if args.max_tokens <= 0:
        parser.error("--max-tokens must be positive")

    config = replace(
        load_replay_config(args.config),
        source_run_dir=args.source_run_dir,
        checkpoint=args.checkpoint,
    )
    destination = run_smoke(
        config,
        checkpoint_dir=args.checkpoint,
        output_dir=args.output_dir,
        layer_index=args.layer_index,
        max_tokens=args.max_tokens,
    )
    print(destination)


if __name__ == "__main__":
    main()
