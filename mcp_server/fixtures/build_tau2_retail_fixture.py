#!/usr/bin/env python3
"""Build the pinned, runtime-local tau2 retail database fixture."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = Path(__file__).with_name("tau2_retail_db.json")
PROVENANCE_PATH = Path(__file__).with_name("tau2_retail_provenance.json")
TAU2_REVISION = "363133ada1936491fb5bcec33cd62c3518a99f65"
TRANSFORMATION_VERSION = "1"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build(tau2_root: Path) -> tuple[str, str]:
    revision = subprocess.check_output(
        ["git", "-C", str(tau2_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != TAU2_REVISION:
        raise RuntimeError(f"{tau2_root} is at {revision}, expected {TAU2_REVISION}")

    source_path = tau2_root / "data" / "tau2" / "domains" / "retail" / "db.json"
    source_bytes = source_path.read_bytes()
    database = json.loads(source_bytes)
    derived_bytes = (
        json.dumps(database, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    OUTPUT_PATH.write_bytes(derived_bytes)

    provenance = {
        "upstream_repository": "https://github.com/sierra-research/tau2-bench",
        "source_revision": TAU2_REVISION,
        "source_path": "data/tau2/domains/retail/db.json",
        "source_sha256": _sha256(source_bytes),
        "derived_fixture_path": "mcp_server/fixtures/tau2_retail_db.json",
        "derived_fixture_sha256": _sha256(derived_bytes),
        "transformation": (
            "Parsed the pinned source JSON and serialized the same retail records "
            "with sorted keys and two-space indentation; no entities or IDs changed."
        ),
        "transformation_script": "mcp_server/fixtures/build_tau2_retail_fixture.py",
        "transformation_version": TRANSFORMATION_VERSION,
        "license": "MIT",
    }
    PROVENANCE_PATH.write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return provenance["source_sha256"], provenance["derived_fixture_sha256"]


def main() -> None:
    default_root = (
        Path(os.environ.get("SCRATCH", ""))
        / "layermcp"
        / "raw_sources"
        / "tau2-bench"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--tau2-root", type=Path, default=default_root)
    args = parser.parse_args()
    source_hash, fixture_hash = build(args.tau2_root.resolve())
    print(f"Wrote {OUTPUT_PATH}")
    print(f"source_sha256={source_hash}")
    print(f"derived_fixture_sha256={fixture_hash}")


if __name__ == "__main__":
    main()
