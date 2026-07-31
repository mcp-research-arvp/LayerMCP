#!/usr/bin/env python3
"""Build the pinned DeepMind Mathematics and GSM8K single-step expansion."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random
import re
import subprocess
import sys
from typing import Any, Callable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "benchmark" / "math" / "tool_routing_math_public_derived_expansion.json"
)
DEEPMIND_REVISION = "427f45075f84b8b9774950196ad63867ca20ffb3"
GSM8K_REVISION = "3101c7d5072418e28b9008a6636bde82a006892c"
GENERATOR_SEED = 20260730
GENERATOR_HASH_SEED = "0"

TARGET_COUNTS = {
    ("grade_school_math", "calculator"): 100,
    ("mathematics_dataset", "calculator"): 20,
    ("mathematics_dataset", "simplify_expression"): 25,
    ("mathematics_dataset", "solve_equation"): 35,
    ("mathematics_dataset", "factor_expression"): 15,
    ("mathematics_dataset", "expand_expression"): 25,
    ("mathematics_dataset", "differentiate_expression"): 40,
    ("mathematics_dataset", "convert_units"): 40,
    ("mathematics_dataset", "integer_factorization"): 25,
    ("mathematics_dataset", "gcd_lcm"): 25,
    ("mathematics_dataset", "modular_arithmetic"): 20,
    ("mathematics_dataset", "base_arithmetic"): 30,
}


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _git_head(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def _require_revision(path: Path, expected: str) -> None:
    actual = _git_head(path)
    if actual != expected:
        raise RuntimeError(f"{path} is at {actual}, expected {expected}")


def _tool_functions() -> dict[str, Callable[..., dict[str, Any]]]:
    sys.path.insert(0, str(PROJECT_ROOT))
    from mcp_server.math_tools import (
        base_arithmetic,
        convert_units,
        differentiate_expression,
        expand_expression,
        factor_expression,
        gcd_lcm,
        integer_factorization,
        modular_arithmetic,
        simplify_expression,
        solve_equation,
    )
    from mcp_server.tool_impls import calculator

    return {
        "calculator": calculator,
        "simplify_expression": simplify_expression,
        "solve_equation": solve_equation,
        "factor_expression": factor_expression,
        "expand_expression": expand_expression,
        "differentiate_expression": differentiate_expression,
        "convert_units": convert_units,
        "integer_factorization": integer_factorization,
        "gcd_lcm": gcd_lcm,
        "modular_arithmetic": modular_arithmetic,
        "base_arithmetic": base_arithmetic,
    }


def _row(
    *,
    row_id: str,
    query: str,
    tool: str,
    args: dict[str, Any],
    answer: dict[str, Any],
    dataset: str,
    revision: str,
    source_id: str,
    source_hash: str,
    transformation_notes: str,
    source_answer: str,
    module: str | None = None,
    source_seed: int | None = None,
    generated_index: int | None = None,
    split: str,
) -> dict[str, Any]:
    row = {
        "id": row_id,
        "domain": "mathematics",
        "task_type": "single_tool_routing",
        "difficulty": "medium",
        "source": "public_derived",
        "query": query,
        "expected_tool": tool,
        "expected_args": args,
        "expected_answer": answer,
        "perturbation_type": "source_format_adaptation",
        "notes": "Pinned public-source question adapted to one executable MCP call.",
        "source_dataset": dataset,
        "source_revision": revision,
        "source_id": source_id,
        "source_split": split,
        "source_hash": source_hash,
        "source_answer": source_answer,
        "source_license": "MIT" if dataset == "grade_school_math" else "Apache-2.0",
        "transformation_notes": transformation_notes,
    }
    if module is not None:
        row["source_module"] = module
        row["source_seed"] = source_seed
        row["source_python_hash_seed"] = GENERATOR_HASH_SEED
        row["source_generated_index"] = generated_index
    return row


def _numeric_text(value: str) -> int | float:
    cleaned = value.strip().replace(",", "")
    number = float(cleaned)
    return int(number) if number.is_integer() else number


def _gsm8k_rows(
    gsm_root: Path,
    tools: dict[str, Callable[..., dict[str, Any]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    path = gsm_root / "grade_school_math" / "data" / "test.jsonl"
    for source_index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        source = json.loads(line)
        calculations = re.findall(r"<<(.+?)=([^<>]+)>>", source["answer"])
        final_match = re.search(r"####\s*([-+0-9.,]+)\s*$", source["answer"])
        if not calculations or final_match is None:
            continue
        expression = calculations[-1][0].replace(",", "").strip()
        try:
            expected_final = _numeric_text(final_match.group(1))
            result = tools["calculator"](expression=expression)
        except Exception:
            continue
        if not isinstance(result["result"], (int, float)):
            continue
        if abs(float(result["result"]) - float(expected_final)) > 1e-9:
            continue
        source_hash = _sha256_json(source)
        rows.append(
            _row(
                row_id=f"math_expansion_gsm8k_calculator_{len(rows) + 1:03d}",
                query=source["question"],
                tool="calculator",
                args={"expression": expression},
                answer=result,
                dataset="grade_school_math",
                revision=GSM8K_REVISION,
                source_id=f"test:{source_index}",
                source_hash=source_hash,
                split="test",
                transformation_notes=(
                    "Question text is unchanged. The last contractor-authored inline "
                    "calculation that yields the published #### answer is used as the "
                    "single calculator call."
                ),
                source_answer=final_match.group(1).strip(),
            )
        )
        if len(rows) == TARGET_COUNTS[("grade_school_math", "calculator")]:
            return rows
    raise RuntimeError(f"Only generated {len(rows)} valid GSM8K rows")


def _strip_terminal(text: str) -> str:
    return text.strip().removesuffix(".").strip()


def _deepmind_args(tool: str, question: str) -> dict[str, Any] | None:
    if tool == "calculator":
        match = re.fullmatch(r"Calculate (.+)\.", question)
        return {"expression": match.group(1)} if match else None
    if tool == "simplify_expression":
        match = re.fullmatch(r"Simplify (.+) assuming ([a-z]) is positive\.", question)
        return {"expression": match.group(1)} if match else None
    if tool == "solve_equation":
        match = re.fullmatch(r"Solve (.+) for ([a-z])\.", question)
        return {"equation": match.group(1), "variable": match.group(2)} if match else None
    if tool == "factor_expression":
        match = re.fullmatch(r"Factor (.+)\.", question)
        return {"expression": match.group(1)} if match else None
    if tool == "expand_expression":
        match = re.fullmatch(r"Expand (.+)\.", question)
        return {"expression": match.group(1)} if match else None
    if tool == "differentiate_expression":
        match = re.fullmatch(r"(?:Differentiate|Find the first derivative of) (.+) wrt ([a-z])\.", question)
        if match:
            return {"expression": match.group(1), "variable": match.group(2)}
        return None
    if tool == "convert_units":
        match = re.fullmatch(
            r"Convert (-?[0-9]+(?:\.[0-9]+)?) ([A-Za-z]+) to ([A-Za-z]+)\.",
            question,
        )
        if match:
            return {
                "value": float(match.group(1)),
                "from_unit": match.group(2),
                "to_unit": match.group(3),
            }
        return None
    if tool == "integer_factorization":
        match = re.fullmatch(r"What are the prime factors of (-?[0-9]+)\?", question)
        return {"value": match.group(1)} if match else None
    if tool == "gcd_lcm":
        gcd_match = re.fullmatch(
            r"What is the highest common factor of (-?[0-9]+) and (-?[0-9]+)\?",
            question,
        )
        if gcd_match:
            return {"values": list(gcd_match.groups()), "operation": "gcd"}
        lcm_match = re.fullmatch(
            r"Find the common denominator of (-?[0-9]+)/([0-9]+) and (-?[0-9]+)/([0-9]+)\.",
            question,
        )
        if lcm_match:
            return {
                "values": [lcm_match.group(2), lcm_match.group(4)],
                "operation": "lcm",
            }
        return None
    if tool == "modular_arithmetic":
        match = re.fullmatch(
            r"What is the remainder when (-?[0-9]+) is divided by ([0-9]+)\?",
            question,
        )
        if match:
            return {
                "expression": match.group(1),
                "modulus": int(match.group(2)),
                "operation": "mod",
            }
        return None
    if tool == "base_arithmetic":
        match = re.fullmatch(
            r"In base ([0-9]+), what is (.+)\?",
            question,
        )
        if match:
            return {
                "expression": match.group(2),
                "input_base": int(match.group(1)),
                "output_base": int(match.group(1)),
            }
        return None
    raise KeyError(tool)


DEEPMIND_MODULES = {
    "calculator": ("arithmetic", "add_sub_multiple"),
    "simplify_expression": ("polynomials", "simplify_power"),
    "solve_equation": ("algebra", "linear_1d"),
    "factor_expression": ("algebra", "polynomial_roots"),
    "expand_expression": ("polynomials", "expand"),
    "differentiate_expression": ("calculus", "differentiate"),
    "convert_units": ("measurement", "conversion"),
    "integer_factorization": ("numbers", "list_prime_factors"),
    "gcd_lcm": ("numbers", "gcd"),
    "modular_arithmetic": ("numbers", "div_remainder"),
    "base_arithmetic": ("arithmetic", "add_or_sub_in_base"),
}


def _deepmind_rows(
    deepmind_root: Path,
    tools: dict[str, Callable[..., dict[str, Any]]],
) -> list[dict[str, Any]]:
    sys.path.insert(0, str(deepmind_root))
    from mathematics_dataset.modules import modules

    module_map = modules.test()
    rows: list[dict[str, Any]] = []
    for tool_index, (tool, (family, module_name)) in enumerate(
        DEEPMIND_MODULES.items()
    ):
        module_seed = GENERATOR_SEED + tool_index
        np.random.seed(module_seed)
        random.seed(module_seed)
        target = TARGET_COUNTS[("mathematics_dataset", tool)]
        accepted = 0
        generated_index = 0
        while accepted < target and generated_index < 20_000:
            generated_index += 1
            problem = module_map[family][module_name]()
            question = str(problem.question)
            source_answer = str(problem.answer)
            args = _deepmind_args(tool, question)
            if args is None:
                continue
            try:
                result = tools[tool](**args)
            except Exception:
                continue
            source_hash = _sha256_json(
                {"question": question, "answer": source_answer}
            )
            accepted += 1
            adaptation = (
                "Question and mathematical expression are generated by the pinned "
                f"DeepMind {family}__{module_name} module. Notation is passed to "
                "the equivalent LayerMCP tool and the expected answer is the "
                "deterministic local tool result."
            )
            if tool == "factor_expression":
                adaptation += (
                    " This is a native DeepMind 'Factor' prompt; it is labeled "
                    "public-derived with an MCP-format adaptation, not copied from "
                    "a human-authored factoring benchmark."
                )
            rows.append(
                _row(
                    row_id=f"math_expansion_deepmind_{tool}_{accepted:03d}",
                    query=question,
                    tool=tool,
                    args=args,
                    answer=result,
                    dataset="mathematics_dataset",
                    revision=DEEPMIND_REVISION,
                    source_id=f"interpolate:{family}__{module_name}:{generated_index}",
                    source_hash=source_hash,
                    module=f"{family}__{module_name}",
                    generated_index=generated_index,
                    split="interpolate",
                    transformation_notes=adaptation,
                    source_answer=source_answer,
                    source_seed=module_seed,
                )
            )
        if accepted != target:
            raise RuntimeError(f"Only generated {accepted}/{target} rows for {tool}")
    return rows


def build(raw_root: Path) -> list[dict[str, Any]]:
    deepmind_root = raw_root / "mathematics_dataset"
    gsm_root = raw_root / "grade-school-math"
    _require_revision(deepmind_root, DEEPMIND_REVISION)
    _require_revision(gsm_root, GSM8K_REVISION)
    tools = _tool_functions()
    rows = _gsm8k_rows(gsm_root, tools) + _deepmind_rows(deepmind_root, tools)
    if len(rows) != 400:
        raise RuntimeError(f"Expected 400 rows, generated {len(rows)}")
    counts = Counter((row["source_dataset"], row["expected_tool"]) for row in rows)
    if counts != Counter(TARGET_COUNTS):
        raise RuntimeError(f"Unexpected row distribution: {counts}")
    if len({row["id"] for row in rows}) != len(rows):
        raise RuntimeError("Generated duplicate row IDs")
    return rows


def main() -> None:
    default_raw = Path(
        os.environ.get("SCRATCH", "")
    ) / "layermcp" / "raw_sources"
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=default_raw)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    rows = build(args.raw_root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    if os.environ.get("PYTHONHASHSEED") != GENERATOR_HASH_SEED:
        environment = os.environ.copy()
        environment["PYTHONHASHSEED"] = GENERATOR_HASH_SEED
        os.execve(
            sys.executable,
            [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
            environment,
        )
    main()
