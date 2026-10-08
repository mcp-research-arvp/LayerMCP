"""Read-only provenance and paired-result diagnostics (no scoring rules)."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
from typing import Any, Sequence

import torch

from research.phase2.replay import sha256_file, sha256_text


def checkpoint_identity(checkpoint: Path) -> dict[str, Any]:
    """Stream file contents once; paths/mtime alone are not fingerprints.

    This identifies files on disk, not an arbitrary caller-supplied in-memory
    model. Missing local tokenizer assets remain explicitly unverifiable.
    """
    root = checkpoint.resolve(strict=True)
    files = {"weights": sorted(root.glob("*.safetensors")),
             "config": [root / "config.json"], "tokenizer": []}
    asset_root = os.environ.get("TIKTOKEN_ENCODINGS_BASE")
    if asset_root:
        files["tokenizer"] = [Path(asset_root) / "o200k_base.tiktoken"]
    manifest = {}
    signatures = {}
    for category, paths in files.items():
        entries = []
        for path in paths:
            if not path.is_file():
                continue
            before = path.stat()
            digest = sha256_file(path)
            after = path.stat()
            signature = (after.st_size, after.st_mtime_ns, after.st_ctime_ns)
            if signature != (before.st_size, before.st_mtime_ns, before.st_ctime_ns):
                raise RuntimeError(f"Provenance input changed while hashing: {path}")
            signatures[str(path.resolve())] = signature
            entries.append({"name": path.name, "bytes": after.st_size, "sha256": digest})
        manifest[category] = entries
    complete = all(manifest.values())
    return {
        "checkpoint_directory": str(root),
        "status": "complete" if complete else "incomplete",
        "fingerprint": sha256_text(json.dumps(manifest, sort_keys=True)) if complete else None,
        "manifest": manifest, "file_signatures": signatures,
        "scope": "checkpoint/config/local Harmony asset files; not an in-memory model digest",
    }


def verify_identity_inputs(identity: dict[str, Any]) -> None:
    """Fail the run if hashed inputs changed subsequently (cheap stat check)."""
    root = Path(identity["checkpoint_directory"])
    expected_weights = {row["name"] for row in identity["manifest"]["weights"]}
    if {path.name for path in root.glob("*.safetensors")} != expected_weights:
        raise RuntimeError("Provenance weight-file set changed during evaluation")
    for filename, expected in identity["file_signatures"].items():
        stat = Path(filename).stat()
        if tuple(expected) != (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns):
            raise RuntimeError(f"Provenance input changed during evaluation: {filename}")


def native_generator_identity(generator: Any, prepared_identity: dict[str, Any] | None = None) -> dict[str, Any]:
    """Bind file identity to the native loader; cache once for notebook sweeps.

    Arbitrary caller-supplied models remain explicitly unverified. This is a
    loader/file binding, not proof against unwrapped in-memory training edits.
    """
    from models.architectures.gpt_oss_pytorch.inference import TokenGenerator
    from models.routers.gpt_oss_local_router import validate_generator_inputs

    if not isinstance(generator, TokenGenerator):
        return {"status": "unverified", "fingerprint": None}
    validate_generator_inputs(generator)
    cached = getattr(generator.model, "_phase2_checkpoint_identity", None)
    identity = prepared_identity or cached
    if identity is None:
        print("Fingerprinting loaded GPT-OSS inputs once (full checkpoint read)...", flush=True)
        identity = checkpoint_identity(Path(generator.checkpoint_path))
    if identity["status"] != "complete":
        raise ValueError("Native GPT-OSS provenance requires weights, config.json and a local Harmony asset via TIKTOKEN_ENCODINGS_BASE")
    verify_identity_inputs(identity)
    if identity["checkpoint_directory"] != generator.checkpoint_path or identity["file_signatures"] != generator.checkpoint_file_signatures:
        raise RuntimeError("Fingerprint inputs do not match the generator's native loaded inputs")
    if cached is not None and cached["fingerprint"] != identity["fingerprint"]:
        raise RuntimeError("Checkpoint identity changed since this model was fingerprinted")
    generator.model._phase2_checkpoint_identity = deepcopy(identity)
    return deepcopy(identity)


def gpu_driver_metadata() -> dict[str, Any]:
    """Read the installed NVIDIA driver without allocation or GPU work."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return {"status": "unavailable", "versions": []}
    versions = sorted(set(line.strip() for line in result.stdout.splitlines() if line.strip()))
    return {"status": "recorded" if result.returncode == 0 and versions else "unavailable",
            "versions": versions if result.returncode == 0 else [], "source": "nvidia-smi"}


def runtime_identity(model: Any) -> dict[str, Any]:
    from models.routers.gpt_oss_local_router import MAX_GENERATED_TOKENS
    root = Path(__file__).resolve().parents[2]
    def git(*args):
        result = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None
    packages = {}
    for name in ("openai-harmony", "safetensors", "mcp"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    devices = sorted({str(p.device) for p in model.parameters()})
    code_manifest = {
        str(path.relative_to(root)): sha256_file(path)
        for directory in ("evaluation", "models", "mcp_server", "research/phase2")
        for path in sorted((root / directory).rglob("*.py"))
    }
    dirty_status = git("status", "--porcelain", "--untracked-files=no")
    return {
        "git_commit": git("rev-parse", "HEAD"),
        "tracked_worktree_dirty": bool(dirty_status) if dirty_status is not None else None,
        "runtime_code_fingerprint": sha256_text(json.dumps(code_manifest, sort_keys=True)),
        "python": platform.python_version(), "torch": str(torch.__version__),
        "cuda_runtime": torch.version.cuda, "packages": packages,
        "cudnn_version": torch.backends.cudnn.version(),
        "devices": devices, "dtypes": sorted({str(p.dtype) for p in model.parameters()}),
        "gpu_names": {device: torch.cuda.get_device_name(device) for device in devices if device.startswith("cuda")},
        "gpu_driver": gpu_driver_metadata() if any(device.startswith("cuda") for device in devices) else {"status": "not_applicable_cpu", "versions": []},
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "generation": {"temperature": 0.0, "reasoning_effort": "low", "max_tokens": MAX_GENERATED_TOKENS},
    }


def compare_saved_control(saved: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    fields = ("raw_model_output", "selected_tool", "selected_args", "parse_status")
    compared = {key: saved[key] == current.get(key) for key in fields if key in saved}
    return {
        "status": "unavailable" if not compared else "matches" if all(compared.values()) else "differs_new_control_baseline",
        "fields": compared,
        "historical_runtime_equivalence": "not_established_by_output_matching",
    }


def paired_diagnostics(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Keep paired transitions and output validity separate from task scoring."""
    metrics = {
        "tool_choice": "tool_choice_correct",
        "argument_match": "argument_match_correct",
        "execution_success": "execution_success",
        "final_outcome": "final_outcome_correct",
    }
    metric_transitions = {
        name: {
            "transitions": {key: 0 for key in (
                "correct_correct", "correct_wrong", "wrong_correct", "wrong_wrong",
            )},
            "scored_pairs": 0,
            "unscored_pairs": 0,
        }
        for name in metrics
    }
    buckets = {condition: Counter({key: 0 for key in (
        "invalid_output", "valid_no_call", "valid_wrong_tool", "valid_correct_tool",
    )}) for condition in ("control", "intervention")}
    controls: dict[str, set[str]] = {}
    control_counts = Counter()
    changes = 0
    for pair in records:
        a, b = pair["control"], pair["intervention"]
        for name, field in metrics.items():
            diagnostic = metric_transitions[name]
            control_value, intervention_value = a.get(field), b.get(field)
            # In particular, a missing final-outcome score is not a failure.
            if control_value is None or intervention_value is None:
                diagnostic["unscored_pairs"] += 1
                continue
            transition = f"{'correct' if control_value else 'wrong'}_{'correct' if intervention_value else 'wrong'}"
            diagnostic["transitions"][transition] += 1
            diagnostic["scored_pairs"] += 1
        changes += a["chosen_tool"] != b["chosen_tool"]
        for name, outcome in (("control", a), ("intervention", b)):
            bucket = ("invalid_output" if outcome["invalid_output"] else
                      "valid_no_call" if outcome["no_call_outcome"] else
                      "valid_correct_tool" if outcome["tool_choice_correct"] else "valid_wrong_tool")
            buckets[name][bucket] += 1
        identity = {key: a.get(key) for key in (
            "raw_model_output", "selected_tool", "selected_args", "parse_status",
            "tool_result", "tool_result_value", "tool_choice_correct", "argument_match_correct",
            "execution_success", "final_outcome_correct",
        )}
        controls.setdefault(pair["sample_id"], set()).add(json.dumps(identity, sort_keys=True))
        control_counts[pair["sample_id"]] += 1
    drift = sorted(key for key, value in controls.items() if len(value) > 1)
    transitions = metric_transitions["tool_choice"]["transitions"]
    return {
        "tool_choice_transitions": dict(transitions),
        "metric_transitions": metric_transitions,
        "repairs": transitions["wrong_correct"], "damage": transitions["correct_wrong"],
        "tool_choice_changes": changes,
        "output_categories": {key: dict(value) for key, value in buckets.items()},
        "unique_control_examples": len(controls), "paired_observations": len(records),
        "control_drift_sample_ids": drift,
        "control_observation_counts": dict(sorted(control_counts.items())),
        "single_observation_sample_ids": sorted(key for key, count in control_counts.items() if count == 1),
        "control_repeatability_status": (
            "drift_detected" if drift else "stable_repeats" if any(count > 1 for count in control_counts.values()) else "not_tested"
        ),
        "denominator_note": "Repeated controls/seeds are paired observations, not independent benchmark examples.",
    }
