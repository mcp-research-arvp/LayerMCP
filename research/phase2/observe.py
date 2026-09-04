"""Capture selected Llama residual, attention, and MLP outputs for one replay.

Observation is deliberately a separate, teacher-forced pass after ordinary
generation.  The hooks never alter tensors and are removed in ``finally`` via
the context manager, so disabling observation leaves the generation path alone.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import AbstractContextManager
from dataclasses import asdict
import json
from pathlib import Path
import shutil
from typing import Any, Callable

import torch

from models.architectures.llama31_8b_pytorch.config import Config
from models.architectures.llama31_8b_pytorch.inference import TokenGenerator
from models.routers.llama31_8b_local_router import resolve_checkpoint_path
from models.routers.structured_tool_call import parse_tool_call
from research.phase2.replay import (
    LLAMA_ROUTER_ID,
    LLAMA_PROMPT_TEMPLATE,
    ReplayConfig,
    SavedExample,
    config_provenance,
    load_live_catalog,
    load_replay_config,
    load_saved_example,
    render_llama_prompt,
    sha256_file,
    sha256_text,
)


CAPTURE_MODULES = ("residual_stream", "attention_block", "mlp_block")


class ActivationObserver(AbstractContextManager["ActivationObserver"]):
    """Temporary forward hooks that keep only named token positions on CPU."""

    def __init__(
        self,
        model: Any,
        layers: tuple[int, ...],
        absolute_positions: tuple[int, ...],
        *,
        enabled: bool = True,
    ) -> None:
        self.model = model
        self.layers = layers
        self.absolute_positions = absolute_positions
        self.enabled = enabled
        self.handles: list[Any] = []
        self.captured: dict[str, dict[str, torch.Tensor]] = defaultdict(dict)

    def _capture_hook(self, layer: int, module_name: str) -> Callable[..., None]:
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            tensor = output[0] if isinstance(output, tuple) else output
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"{module_name} hook did not receive a tensor output")
            self.captured[str(layer)][module_name] = (
                tensor[:, self.absolute_positions, :].detach().to("cpu", torch.float32).clone()
            )
        return hook

    def __enter__(self) -> "ActivationObserver":
        if not self.enabled:
            return self
        transformer_layers = getattr(self.model, "layers", None)
        if transformer_layers is None:
            raise TypeError("Observation requires a custom Transformer with .layers")
        try:
            for layer in self.layers:
                block = transformer_layers[layer]
                self.handles.extend(
                    [
                        block.register_forward_hook(self._capture_hook(layer, "residual_stream")),
                        block.self_attn.register_forward_hook(self._capture_hook(layer, "attention_block")),
                        block.mlp.register_forward_hook(self._capture_hook(layer, "mlp_block")),
                    ]
                )
        except Exception:
            for handle in reversed(self.handles):
                handle.remove()
            self.handles.clear()
            raise
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        for handle in reversed(self.handles):
            handle.remove()
        self.handles.clear()
        return None


def _decoded_token_pieces(tokenizer: Any, token_ids: list[int]) -> list[str]:
    previous = ""
    pieces: list[str] = []
    for index in range(len(token_ids)):
        current = tokenizer.decode(token_ids[: index + 1], skip_special_tokens=False)
        pieces.append(current[len(previous) :])
        previous = current
    return pieces


def _character_token_indices(pieces: list[str], target: str) -> set[int]:
    full = "".join(pieces)
    indices: set[int] = set()
    start = 0
    while True:
        start = full.find(target, start)
        if start < 0:
            return indices
        end = start + len(target)
        cursor = 0
        for index, piece in enumerate(pieces):
            next_cursor = cursor + len(piece)
            if cursor < end and next_cursor > start:
                indices.add(index)
            cursor = next_cursor
        start = end


def select_tool_call_positions(
    tokenizer: Any,
    generated_ids: list[int],
    selected_tool: str,
    selected_args: dict[str, Any],
) -> list[dict[str, Any]]:
    """Select only tool-name plus JSON argument key/value token positions."""
    pieces = _decoded_token_pieces(tokenizer, generated_ids)
    roles: dict[int, set[str]] = defaultdict(set)
    targets: list[tuple[str, str]] = [("tool_name", selected_tool)]
    for key, value in selected_args.items():
        targets.append(("argument_key", json.dumps(str(key), ensure_ascii=True)))
        targets.append(("argument_value", json.dumps(value, ensure_ascii=True, sort_keys=True)))
    for role, target in targets:
        for index in _character_token_indices(pieces, target):
            roles[index].add(role)
    return [
        {
            "generated_token_index": index,
            "token_id": int(generated_ids[index]),
            "decoded_piece": pieces[index],
            "roles": sorted(roles[index]),
        }
        for index in sorted(roles)
    ]


def capture_selected_activations(
    model: Any,
    prompt_token_ids: tuple[int, ...],
    generated_ids: list[int],
    selected_positions: list[dict[str, Any]],
    layers: tuple[int, ...],
    device: torch.device,
    *,
    enabled: bool = True,
) -> dict[str, dict[str, torch.Tensor]]:
    """Replay generated tokens teacher-forced and retain selected logits inputs.

    Position ``i`` in generated output is predicted from prompt + generated[:i],
    so the corresponding final-layer representation occurs at absolute input
    position ``len(prompt) + i - 1``.  The first generated token is excluded:
    its representation would be the final prompt token rather than a generated
    tool-call token.
    """
    generated_indices = [
        item["generated_token_index"]
        for item in selected_positions
        if item["generated_token_index"] > 0
    ]
    if not enabled or not generated_indices:
        return {}
    absolute_positions = tuple(len(prompt_token_ids) + index - 1 for index in generated_indices)
    full_tokens = list(prompt_token_ids) + generated_ids[:-1]
    with ActivationObserver(model, layers, absolute_positions, enabled=True) as observer:
        with torch.inference_mode():
            model(torch.tensor([full_tokens], dtype=torch.long, device=device))
    return dict(observer.captured)


def _safe_output_directory(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite an observation directory: {path}")
    if path.name in {"", ".", ".."}:
        raise ValueError("Observation output directory has an unsafe final component")
    return path


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _write_completed_observation(
    target: Path,
    provenance: dict[str, Any],
    captured: dict[str, dict[str, torch.Tensor]],
    *,
    observation_enabled: bool,
) -> None:
    _write_json(target / "provenance.json", provenance)
    if observation_enabled:
        torch.save(captured, target / "activations.pt")
    (target / "OBSERVATION_COMPLETE").write_text("\n", encoding="utf-8")


def run_observation(config: ReplayConfig, output_dir: Path | None = None) -> Path:
    """Load one real checkpoint and write a small, self-contained observation."""
    example: SavedExample = load_saved_example(config.source_run_dir, config.sample_id)
    catalog = load_live_catalog()
    checkpoint = resolve_checkpoint_path(config.checkpoint)
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Llama checkpoint does not exist: {checkpoint}")
    generator = TokenGenerator(checkpoint=str(checkpoint), device=Config.device)
    prompt = render_llama_prompt(generator, example, catalog)
    if config.require_registry_match and not prompt.registry_exact_match:
        raise ValueError(
            "Saved and live MCP registry metadata differ; exact replay is required by config."
        )
    generated_ids = list(
        generator.generate(
            list(prompt.token_ids),
            stop_tokens=generator.stop_tokens,
            temperature=prompt.generation_settings["temperature"],
            max_tokens=prompt.generation_settings["max_tokens"],
        )
    )
    raw_output = generator.tokenizer.decode(generated_ids, skip_special_tokens=False)
    prediction = parse_tool_call(raw_output, example.record["tool_names"], tool_schemas=catalog.schemas)
    selected_positions = select_tool_call_positions(
        generator.tokenizer,
        generated_ids,
        prediction.selected_tool,
        prediction.selected_args,
    )
    captured = capture_selected_activations(
        generator.model,
        prompt.token_ids,
        generated_ids,
        selected_positions,
        config.layers,
        Config.device,
        enabled=config.observation_enabled,
    )
    target = _safe_output_directory(output_dir or config.output_dir or Path("phase2_observation"))
    target.mkdir(parents=True)
    try:
        provenance = {
            **config_provenance(config),
            "source_run_directory": str(example.source_run_dir),
            "source_job_id": example.run_metadata.get("slurm_job_id"),
            "source_sample_id": config.sample_id,
            "source_samples_path": str(example.samples_path),
            "source_samples_sha256": sha256_file(example.samples_path),
            "source_run_commit": example.run_metadata.get("git_commit"),
            "model_name": example.record["model_name"],
            "router_id": LLAMA_ROUTER_ID,
            "prompt_template": LLAMA_PROMPT_TEMPLATE,
            "reasoning_mode": example.record.get("reasoning_mode", "direct"),
            "reasoning_method": example.record.get("reasoning_method", "none"),
            "checkpoint_path": str(checkpoint),
            "prompt_sha256": sha256_text(prompt.text),
            "prompt_token_count": len(prompt.token_ids),
            "generation_settings": prompt.generation_settings,
            "saved_registry": {
                key: example.record.get(key)
                for key in ("tool_pool", "tool_count", "tool_registry_fingerprint", "tool_registry_fingerprint_version")
            },
            "live_registry": catalog.metadata,
            "registry_exact_match": prompt.registry_exact_match,
            "capture_pass": "teacher_forced_replay_of_generated_tokens",
            "captured_modules": list(CAPTURE_MODULES) if config.observation_enabled else [],
            "selected_token_positions": selected_positions,
            "generated_raw_output": raw_output,
            "parsed_prediction": asdict(prediction),
        }
        _write_completed_observation(
            target,
            provenance,
            captured,
            observation_enabled=config.observation_enabled,
        )
    except Exception:
        shutil.rmtree(target)
        raise
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description="Observe selected Llama tool-call activations.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    destination = run_observation(load_replay_config(args.config), args.output_dir)
    print(destination)


if __name__ == "__main__":
    main()
