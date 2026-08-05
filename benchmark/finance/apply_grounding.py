"""Apply deterministic prompt grounding to every finance benchmark call.

The source questions remain unchanged. Grounding is stored as compact JSON in
``prompt_context`` so the evaluator can expose fixture IDs, schemas, source
coordinates, resolved calculator expressions, and recorded trajectory inputs
without pretending that those values were inferable from the source question.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.finance.grounding import (  # noqa: E402
    calculator_prompt_context,
    normalized_call_prompt_context,
    recorded_call_prompt_context,
    table_query_prompt_context,
)
from mcp_server.finance_state import get_finance_fixture  # noqa: E402
from mcp_server.finretrieval_state import (  # noqa: E402
    FINRETRIEVAL_REPLAY_TOOL_NAMES,
)


BENCHMARK_ROOT = Path(__file__).resolve().parent
BENCHMARK_PATTERN = "finance_*.json"


def _with_prompt_context(
    call: dict[str, Any],
    prompt_context: str,
) -> dict[str, Any]:
    """Return a copy with prompt_context directly after the unchanged query."""
    grounded: dict[str, Any] = {}
    inserted = False
    for key, value in call.items():
        if key == "prompt_context":
            continue
        grounded[key] = value
        if key == "query":
            grounded["prompt_context"] = prompt_context
            inserted = True
    if not inserted:
        grounded["prompt_context"] = prompt_context
    return grounded


def _call_prompt_context(
    call: dict[str, Any],
    tables: dict[str, dict[str, Any]],
) -> str:
    expected_tool = str(call["expected_tool"])
    expected_args = call.get("expected_args", {})
    if not isinstance(expected_args, dict):
        raise ValueError("Finance expected_args must be an object.")

    if expected_tool == "finance_query_table":
        dataset_id = expected_args.get("dataset_id")
        if not isinstance(dataset_id, str) or dataset_id not in tables:
            raise ValueError(f"Unknown finance table dataset_id: {dataset_id!r}")
        return table_query_prompt_context(expected_args, tables[dataset_id])
    if expected_tool == "calculator":
        return calculator_prompt_context(expected_args)
    if expected_tool in FINRETRIEVAL_REPLAY_TOOL_NAMES:
        return recorded_call_prompt_context(expected_tool, expected_args)
    return normalized_call_prompt_context(expected_args)


def ground_benchmark_rows(
    rows: list[dict[str, Any]],
    tables: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    grounded_rows: list[dict[str, Any]] = []
    for row in rows:
        grounded_row = dict(row)
        raw_steps = row.get("expected_steps")
        if raw_steps is not None:
            if not isinstance(raw_steps, list):
                raise ValueError(f"{row.get('id')} expected_steps must be a list.")
            grounded_row["expected_steps"] = [
                _with_prompt_context(
                    step,
                    _call_prompt_context(step, tables),
                )
                for step in raw_steps
            ]
        else:
            grounded_row = _with_prompt_context(
                grounded_row,
                _call_prompt_context(grounded_row, tables),
            )
        grounded_rows.append(grounded_row)
    return grounded_rows


def _render_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2) + "\n"


def apply_grounding(*, check: bool = False) -> list[Path]:
    tables = get_finance_fixture()["tables"]
    changed: list[Path] = []
    for path in sorted(BENCHMARK_ROOT.glob(BENCHMARK_PATTERN)):
        rows = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(rows, list) or not all(
            isinstance(row, dict) for row in rows
        ):
            raise ValueError(f"{path} must contain a list of objects.")
        rendered = _render_json(ground_benchmark_rows(rows, tables))
        current = path.read_text(encoding="utf-8")
        if rendered == current:
            continue
        changed.append(path)
        if not check:
            path.write_text(rendered, encoding="utf-8")
    return changed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Apply deterministic context to finance benchmark calls."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Report drift without writing benchmark files.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    changed = apply_grounding(check=args.check)
    if args.check and changed:
        for path in changed:
            print(path.relative_to(PROJECT_ROOT))
        raise SystemExit(1)
    print(f"Grounded finance benchmark files: {len(changed)}")


if __name__ == "__main__":
    main()
