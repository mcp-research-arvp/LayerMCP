"""Immutable canonical controls for Phase 2 intervention runs.

The reference is deliberately a small file protocol rather than a database:
one completed control record per selected sample/workflow, a manifest that
describes the comparable protocol, and content hashes linking later
intervention-only chunks to it.  It never supplies tool state to a rollout.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


REFERENCE_KIND = "phase2_canonical_control_reference_v1"
REFERENCE_MANIFEST = "control_reference_manifest.json"
REFERENCE_RECORDS = "control_records.jsonl"


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _require(value: bool, message: str) -> None:
    if not value:
        raise ValueError(message)


def write_reference(
    directory: Path,
    *,
    workflow_type: str,
    compatibility: dict[str, Any],
    controls: Iterable[dict[str, Any]],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Seal a completed control run without retaining any intervention data."""
    records = list(controls)
    ids = [record.get("sample_id") for record in records]
    _require(bool(records), "a canonical reference needs at least one control record")
    _require(all(isinstance(value, str) and value for value in ids), "control records require non-empty sample_id")
    _require(len(ids) == len(set(ids)), "canonical reference has duplicate sample/workflow IDs")
    _require(compatibility.get("workflow_type") == workflow_type, "reference workflow type is inconsistent")
    directory.mkdir(parents=True, exist_ok=False)
    record_bytes = b"".join(canonical_bytes(record) for record in records)
    (directory / REFERENCE_RECORDS).write_bytes(record_bytes)
    manifest = {
        "kind": REFERENCE_KIND,
        "workflow_type": workflow_type,
        "compatibility": compatibility,
        "record_file": REFERENCE_RECORDS,
        "record_sha256": sha256_bytes(record_bytes),
        "selected_ids": ids,
        "control_count": len(records),
        **metadata,
    }
    manifest_bytes = canonical_bytes(manifest)
    (directory / REFERENCE_MANIFEST).write_bytes(manifest_bytes)
    reference = {
        "kind": REFERENCE_KIND,
        "workflow_type": workflow_type,
        "filename": REFERENCE_MANIFEST,
        "sha256": sha256_bytes(manifest_bytes),
        "record_sha256": manifest["record_sha256"],
    }
    (directory / "run_status.json").write_bytes(canonical_bytes({
        "status": "completed", "execution_mode": "reference",
        "control_count": len(records), **{key: metadata[key] for key in ("run_id", "started_at", "completed_at") if key in metadata},
    }))
    (directory / "RUN_COMPLETE").write_text("\n", encoding="utf-8")
    return reference


def load_reference(
    directory: Path | str,
    *,
    workflow_type: str,
    compatibility: dict[str, Any],
    expected_ids: Iterable[str],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Load only a complete, byte-linked, protocol-compatible reference."""
    path = Path(directory).expanduser().resolve(strict=True)
    _require((path / "RUN_COMPLETE").is_file(), f"{path}: canonical control lacks RUN_COMPLETE")
    _require(not (path / "RUN_IN_PROGRESS").exists() and not (path / "RUN_FAILED.json").exists(),
             f"{path}: canonical control has conflicting lifecycle markers")
    _require(not (path / "RUN_IN_PROGRESS").exists() and not (path / "RUN_FAILED.json").exists(),
             f"{path}: canonical control has conflicting lifecycle markers")
    manifest_path = path / REFERENCE_MANIFEST
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    _require(manifest.get("kind") == REFERENCE_KIND, f"{path}: not a canonical-control reference")
    _require(manifest.get("workflow_type") == workflow_type, f"{path}: reference workflow type differs")
    _require(manifest.get("compatibility") == compatibility,
             f"{path}: reference model/input/generation/registry/scoring protocol is incompatible")
    record_name = manifest.get("record_file")
    _require(record_name == REFERENCE_RECORDS, f"{path}: unexpected reference record file")
    record_bytes = (path / record_name).read_bytes()
    _require(sha256_bytes(record_bytes) == manifest.get("record_sha256"), f"{path}: control record hash differs")
    rows = [json.loads(line) for line in record_bytes.decode("utf-8").splitlines() if line.strip()]
    ids = [row.get("sample_id") for row in rows]
    _require(ids == manifest.get("selected_ids"), f"{path}: control record IDs differ from manifest")
    required = list(expected_ids)
    _require(ids == required, f"{path}: reference coverage differs from selected inputs")
    status = json.loads((path / "run_status.json").read_text(encoding="utf-8"))
    _require(status.get("status") == "completed", f"{path}: reference lifecycle is not completed")
    reference = {
        "kind": REFERENCE_KIND, "workflow_type": workflow_type,
        "filename": REFERENCE_MANIFEST, "sha256": sha256_bytes(manifest_bytes),
        "record_sha256": manifest["record_sha256"],
    }
    return reference, {row["sample_id"]: row for row in rows}


def preflight_reference(
    directory: Path | str, *, workflow_type: str, required_compatibility: dict[str, Any], expected_ids: Iterable[str],
) -> None:
    """Reject an obviously wrong/incomplete reference before model weights load.

    Full equality (including live registry/runtime facts) is checked again by
    :func:`load_reference` immediately before records are compared.
    """
    path = Path(directory).expanduser().resolve(strict=True)
    _require((path / "RUN_COMPLETE").is_file(), f"{path}: canonical control lacks RUN_COMPLETE")
    manifest = json.loads((path / REFERENCE_MANIFEST).read_text(encoding="utf-8"))
    _require(manifest.get("kind") == REFERENCE_KIND and manifest.get("workflow_type") == workflow_type,
             f"{path}: wrong canonical-control workflow")
    compatibility = manifest.get("compatibility")
    _require(isinstance(compatibility, dict), f"{path}: missing canonical compatibility")
    for key, expected in required_compatibility.items():
        _require(compatibility.get(key) == expected, f"{path}: incompatible canonical control field {key}")
    _require(manifest.get("selected_ids") == list(expected_ids), f"{path}: canonical-control coverage differs from selected inputs")


def inspect_candidate(path: Path | str) -> dict[str, Any]:
    """Return explicit reuse status; do not infer missing legacy provenance."""
    directory = Path(path).expanduser().resolve(strict=True)
    if (directory / REFERENCE_MANIFEST).is_file():
        try:
            manifest = json.loads((directory / REFERENCE_MANIFEST).read_text(encoding="utf-8"))
            return {"status": "canonical_reference", "workflow_type": manifest.get("workflow_type"),
                    "reference": {"sha256": sha256_bytes((directory / REFERENCE_MANIFEST).read_bytes())}}
        except (OSError, ValueError, json.JSONDecodeError) as error:
            return {"status": "invalid_canonical_reference", "reason": str(error)}
    if ((directory / "RUN_COMPLETE").is_file() and (directory / "paired_records.jsonl").is_file() and
            (directory / "run_config.json").is_file() and (directory / "run_manifest.json").is_file() and
            (directory / "run_status.json").is_file()):
        return {"status": "paired_candidate_requires_registration",
                "reason": "paired controls are present, but no canonical reference manifest exists"}
    return {"status": "unverified_legacy_candidate",
            "reason": "missing canonical manifest and complete linked control records"}
