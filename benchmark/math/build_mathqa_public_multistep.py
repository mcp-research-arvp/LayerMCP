"""Build the public-derived MathQA multi-step benchmark deterministically.

The builder consumes the original MathQA ``MathQA.zip`` archive. It does not
download data, interpret rationales, or synthesize program decompositions.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Callable
import zipfile

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mcp_server.math_tools import gcd_lcm, modular_arithmetic
from mcp_server.tool_impls import calculator


SOURCE_URL = "https://math-qa.github.io/math-QA/data/MathQA.zip"
SOURCE_REPOSITORY = "https://github.com/math-qa/math-QA"
SOURCE_REVISION = "original MathQA release archive (immutable content-addressed pin)"
MIRROR_REPOSITORY = "https://huggingface.co/datasets/allenai/math_qa"
MIRROR_REVISION = "c4f1cc784c04c4957b50c97858f23893b633eea6"
PAPER_URL = "https://aclanthology.org/N19-1245/"
LICENSE = "Apache-2.0 (as recorded by the allenai/math_qa dataset card)"
LICENSE_CAVEAT = (
    "The original downloadable archive contains no license file; Apache-2.0 is "
    "recorded by the pinned allenai/math_qa dataset card mirror."
)

ARCHIVE_SHA256 = "7344f30456a7aef3176d4866cc953b35b41bec44eda6b00cdbcfde2876b2f07a"
MEMBER_SHA256 = {
    "test.json": "dfe7bc4691caf26842ccd4cf14e8f978327bab4f2989e0b076a4e6b38a9371d1",
    "train.json": "00e8919347d65dbba9289bf04ed998a6c48dbf451ca909eeb66a35f2419c2bf6",
    "dev.json": "aeca12424fc8f32d0e35c09b7738986a446c12f110c8071f3cc8913712f04bf3",
    "operation_list.txt": "3ea53c71034e52605f8203f1525d7f03bd4a0ddc3e4384c93b3ee1e4ddc1e8ca",
    "constant_list.txt": "611b5d1089f0d10e6486b5ffeaee92c95efc5df9e385b809d4d142181e2bb4a0",
}
EXPECTED_SOURCE_ROWS = 2_985
EXPECTED_ELIGIBLE_WORKFLOWS = 1_498
EXPECTED_ELIGIBLE_STEPS = 6_638
SELECTED_WORKFLOWS = 200
CALCULATOR_MAX_EXPONENT_MAGNITUDE = 10

OUTPUT_DIR = Path(__file__).resolve().parent
BENCHMARK_NAME = "math_public_mathqa_multistep.json"
FIXTURE_NAME = "fixtures/mathqa_test_selected_rows.json"
MANIFEST_NAME = "fixtures/mathqa_test_selection_manifest.json"
MAPPING_NAME = "mathqa_operation_mapping.json"

NUMBER_RE = re.compile(
    r"[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?"
)
CALL_RE = re.compile(r"([a-z_]+)\(([^()]*)\)")
REFERENCE_RE = re.compile(r"(?:n|#)\d+")

SUPPORTED_OPERATIONS = {
    "add", "circle_area", "circumface", "cube_edge_by_volume", "divide",
    "floor", "gcd", "inverse", "lcm", "multiply", "negate", "power",
    "rectangle_area", "rectangle_perimeter", "reminder", "speed", "sqrt",
    "square_area", "square_edge_by_area", "square_perimeter", "subtract",
    "surface_cube", "volume_cube", "volume_cylinder",
    "volume_rectangular_prism",
}
EXPLICITLY_UNSUPPORTED = {
    "choose", "combination", "combined_work", "cosine", "factorial",
    "find_work", "log", "max", "min", "permutation", "sine",
    "surface_cylinder", "tangent",
}

EXCLUSION_CLASS_DESCRIPTIONS = {
    "numeric_option_parsing_limitation": (
        "A ratio or mixed-number reading matches, but the released first-number "
        "option parser does not produce it."
    ),
    "strict_tolerance_mismatch": (
        "The result is within an approximate one-percent comparison but fails "
        "the benchmark's strict comparison."
    ),
    "annotated_formula_linear_formula_disagreement": (
        "The supported annotated formula matches the selected numeric option but "
        "the preserved linear program does not."
    ),
    "program_result_matches_other_option": (
        "The preserved linear program matches a non-selected displayed option."
    ),
    "destination_tool_constraint": (
        "The source operation cannot be represented faithfully within an existing "
        "LayerMCP destination tool's input constraints."
    ),
    "unsupported_operation": "The source program contains an unmapped operation.",
    "invalid_or_forward_reference": (
        "The source program syntax, number reference, constant, or dependency is invalid."
    ),
    "nonfinite_or_tool_execution_failure": (
        "Faithful execution produces a nonfinite value or destination-tool failure."
    ),
    "source_program_selected_answer_disagreement_unresolved": (
        "The source linear program and selected numeric option disagree; neither "
        "is repaired and no causal assertion is made."
    ),
    "insufficient_operations": "The source program has fewer than two operations.",
    "selected_option_not_numeric": "The selected option has no supported numeric target.",
    "split_overlap": "The exact test question also occurs in train or development.",
}


class Ineligible(ValueError):
    """A deterministic eligibility rejection with a stable reason code."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def _exclusion_class(reason: str) -> str:
    if reason.startswith("unsupported_operation:"):
        return "unsupported_operation"
    if reason in {
        "destination_tool_exponent_limit",
        "division_by_zero",
        "negative_square_root_input",
        "negative_radical_input",
        "non_integer_gcd_lcm_argument",
        "non_integer_modular_argument",
        "invalid_modulus",
    }:
        return "destination_tool_constraint"
    if reason.startswith("tool_execution_error:") or reason == "non_finite_intermediate_result":
        return "nonfinite_or_tool_execution_failure"
    if reason in {
        "invalid_program_syntax",
        "invalid_program_arguments",
        "unsupported_argument_token",
        "invalid_constant",
        "forward_or_missing_dependency",
        "missing_question_number",
    }:
        return "invalid_or_forward_reference"
    if reason == "fewer_than_two_operations":
        return "insufficient_operations"
    if reason in {
        "correct_option_marker_missing",
        "correct_option_not_numeric",
        "correct_option_invalid_fraction",
    }:
        return "selected_option_not_numeric"
    if reason == "exact_question_duplicate_train_or_dev":
        return "split_overlap"
    if reason in EXCLUSION_CLASS_DESCRIPTIONS:
        return reason
    raise RuntimeError(f"Unclassified MathQA exclusion reason: {reason}")


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _pretty_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _number_text(value: int | float) -> str:
    number = float(value)
    if not math.isfinite(number):
        raise Ineligible("non_finite_intermediate_result")
    if number.is_integer():
        return str(int(number))
    return repr(number)


def _json_number(value: int | float) -> int | float:
    number = float(value)
    if not math.isfinite(number):
        raise Ineligible("non_finite_intermediate_result")
    return int(number) if number.is_integer() else number


def _extract_numbers(question: str) -> list[float]:
    return [float(item.replace(",", "")) for item in NUMBER_RE.findall(question)]


def _parse_constant(token: str) -> float:
    if token == "const_pi":
        return math.pi
    if token == "const_deg_to_rad":
        return math.pi / 180
    parts = NUMBER_RE.findall(token)
    if len(parts) == 1:
        return float(parts[0])
    if len(parts) == 2:
        return float(parts[0]) + float("0." + parts[1])
    raise Ineligible("invalid_constant")


def _selected_option(row: dict[str, Any]) -> tuple[str, float]:
    options = row["options"]
    start = options.find(f"{row['correct']} )")
    if start < 0:
        raise Ineligible("correct_option_marker_missing")
    tail = options[start:]
    matches = NUMBER_RE.findall(tail)
    if not matches:
        raise Ineligible("correct_option_not_numeric")
    numerator = float(matches[0].replace(",", ""))
    # Preserve the released executor's fixed fraction recognition rule.
    end = start + len(str(numerator)) + 3
    if end < len(options) and options[end] == "/":
        denominator_match = NUMBER_RE.findall(options[end:])
        if not denominator_match:
            raise Ineligible("correct_option_invalid_fraction")
        numerator /= float(denominator_match[0].replace(",", ""))
    next_marker = re.search(r", [a-e] \)", tail[3:])
    selected_text = tail if next_marker is None else tail[: 3 + next_marker.start()]
    return selected_text.strip().rstrip(",").strip(), numerator


def _alternative_ratio_or_mixed_number(text: str) -> float | None:
    """Recover display forms the released first-number parser cannot read.

    This is diagnostic only. Rows needing this interpretation remain excluded;
    the builder never repairs their source answer or promotes them to eligible.
    """
    body = re.sub(r"^[a-e]\s*\)\s*", "", text.strip())
    mixed = re.search(
        r"([-+]?(?:\d+(?:\.\d*)?|\.\d+))\s+"
        r"(\d+(?:\.\d*)?)\s*/\s*(\d+(?:\.\d*)?)",
        body,
    )
    if mixed:
        denominator = float(mixed.group(3))
        if denominator == 0:
            return None
        return float(mixed.group(1)) + float(mixed.group(2)) / denominator
    ratio = re.search(
        r"([-+]?(?:\d+(?:\.\d*)?|\.\d+))\s*[:/]\s*"
        r"([-+]?(?:\d+(?:\.\d*)?|\.\d+))",
        body,
    )
    if ratio:
        denominator = float(ratio.group(2))
        if denominator == 0:
            return None
        return float(ratio.group(1)) / denominator
    return None


def _option_segments(options: str) -> list[tuple[str, str]]:
    markers = list(re.finditer(r"(?:^|,\s*)[a-e]\s*\)\s*", options))
    segments = []
    for index, marker in enumerate(markers):
        label_match = re.search(r"[a-e]", marker.group())
        if label_match is None:
            continue
        end = markers[index + 1].start() if index + 1 < len(markers) else len(options)
        segments.append(
            (label_match.group(), options[marker.end():end].strip(" ,"))
        )
    return segments


def _option_segment_number(text: str) -> float | None:
    alternative = _alternative_ratio_or_mixed_number(f"a ) {text}")
    if alternative is not None:
        return alternative
    matches = NUMBER_RE.findall(text)
    if not matches:
        return None
    return float(matches[0].replace(",", ""))


def _parse_program(linear_formula: str) -> list[tuple[str, list[str], str]]:
    raw_calls = linear_formula.split("|")
    if raw_calls and raw_calls[-1] == "":
        raw_calls.pop()
    if len(raw_calls) < 2:
        raise Ineligible("fewer_than_two_operations")
    parsed = []
    for call in raw_calls:
        match = CALL_RE.fullmatch(call)
        if not match:
            raise Ineligible("invalid_program_syntax")
        operation = match.group(1)
        if operation not in SUPPORTED_OPERATIONS:
            raise Ineligible(f"unsupported_operation:{operation}")
        arguments = match.group(2).split(",") if match.group(2) else []
        if not arguments or any(not argument for argument in arguments):
            raise Ineligible("invalid_program_arguments")
        if any(
            not REFERENCE_RE.fullmatch(argument)
            and not argument.startswith("const_")
            for argument in arguments
        ):
            raise Ineligible("unsupported_argument_token")
        parsed.append((operation, arguments, call))
    return parsed


def _calculator(operation: str, values: list[float]) -> tuple[dict[str, Any], float]:
    v = [_number_text(item) for item in values]
    if operation in {"add", "subtract", "multiply", "divide", "speed", "power"}:
        symbol = {"add": "+", "subtract": "-", "multiply": "*", "divide": "/", "speed": "/", "power": "**"}[operation]
        if (
            operation == "power"
            and abs(values[1]) > CALCULATOR_MAX_EXPONENT_MAGNITUDE
        ):
            raise Ineligible("destination_tool_exponent_limit")
        right = v[1]
        expression = (
            f"({v[0]}) {symbol} {right}"
            if operation == "power"
            else f"{v[0]} {symbol} {right}"
        )
    elif operation in {"inverse", "sqrt", "floor", "negate", "circle_area", "circumface", "cube_edge_by_volume", "square_area", "square_edge_by_area", "square_perimeter", "surface_cube", "volume_cube"}:
        expression = {
            "inverse": f"1 / {v[0]}", "sqrt": f"{v[0]} ** 0.5",
            "floor": f"{v[0]} // 1", "negate": f"-{v[0]}",
            "circle_area": f"{repr(math.pi)} * {v[0]} ** 2",
            "circumface": f"2 * {repr(math.pi)} * {v[0]}",
            "cube_edge_by_volume": f"{v[0]} ** (1 / 3)",
            "square_area": f"{v[0]} ** 2", "square_edge_by_area": f"{v[0]} ** 0.5",
            "square_perimeter": f"4 * {v[0]}", "surface_cube": f"6 * {v[0]} ** 2",
            "volume_cube": f"{v[0]} ** 3",
        }[operation]
    elif operation == "rectangle_area":
        expression = f"{v[0]} * {v[1]}"
    elif operation == "rectangle_perimeter":
        expression = f"2 * ({v[0]} + {v[1]})"
    elif operation == "volume_cylinder":
        expression = f"{repr(math.pi)} * {v[0]} ** 2 * {v[1]}"
    elif operation == "volume_rectangular_prism":
        expression = f"{v[0]} * {v[1]} * {v[2]}"
    else:
        raise Ineligible(f"unsupported_operation:{operation}")
    if operation in {"divide", "speed"} and values[1] == 0:
        raise Ineligible("division_by_zero")
    if operation == "inverse" and values[0] == 0:
        raise Ineligible("division_by_zero")
    if operation == "sqrt" and values[0] < 0:
        raise Ineligible("negative_square_root_input")
    if operation in {"square_edge_by_area", "cube_edge_by_volume"} and values[0] < 0:
        raise Ineligible("negative_radical_input")
    arguments = {"expression": expression}
    result = calculator(**arguments)
    return arguments, float(result["result"])


def _translate_and_execute(
    operation: str, values: list[float]
) -> tuple[str, dict[str, Any], dict[str, Any], float]:
    if operation in {"gcd", "lcm"}:
        if any(not float(value).is_integer() for value in values):
            raise Ineligible("non_integer_gcd_lcm_argument")
        arguments = {
            "values": [_number_text(value) for value in values],
            "operation": operation,
        }
        result = gcd_lcm(**arguments)
        return "gcd_lcm", arguments, result, float(result[operation])
    if operation == "reminder":
        if any(not float(value).is_integer() for value in values):
            raise Ineligible("non_integer_modular_argument")
        if int(values[1]) <= 1:
            raise Ineligible("invalid_modulus")
        arguments = {
            "expression": _number_text(values[0]),
            "modulus": int(values[1]),
            "operation": "mod",
        }
        result = modular_arithmetic(**arguments)
        return "modular_arithmetic", arguments, result, float(result["result"])
    arguments, scalar = _calculator(operation, values)
    result = calculator(**arguments)
    return "calculator", arguments, result, scalar


def _split_nested_arguments(text: str) -> list[str]:
    arguments = []
    start = 0
    depth = 0
    for index, character in enumerate(text):
        if character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
            if depth < 0:
                raise Ineligible("invalid_annotated_formula")
        elif character == "," and depth == 0:
            arguments.append(text[start:index])
            start = index + 1
    if depth != 0:
        raise Ineligible("invalid_annotated_formula")
    arguments.append(text[start:])
    return arguments


def _evaluate_annotated_expression(text: str) -> float:
    """Evaluate a supported annotated formula only for mismatch diagnosis."""
    expression = text.strip()
    match = re.fullmatch(r"([a-z_]+)\((.*)\)", expression)
    if match:
        values = [
            _evaluate_annotated_expression(argument)
            for argument in _split_nested_arguments(match.group(2))
        ]
        return _translate_and_execute(match.group(1), values)[3]
    if expression.startswith("const_"):
        return _parse_constant(expression)
    return float(expression.replace(",", ""))


def _classify_result_mismatch(
    row: dict[str, Any],
    selected_option: str,
    source_answer: float,
    program_result: float,
) -> str:
    alternative = _alternative_ratio_or_mixed_number(selected_option)
    if alternative is not None and math.isclose(
        program_result, alternative, rel_tol=1e-9, abs_tol=1e-9
    ):
        return "numeric_option_parsing_limitation"
    if math.isclose(program_result, source_answer, rel_tol=0.01, abs_tol=0.0):
        return "strict_tolerance_mismatch"
    try:
        annotated_result = _evaluate_annotated_expression(row["annotated_formula"])
    except (ArithmeticError, Ineligible, TypeError, ValueError):
        annotated_result = None
    if annotated_result is not None and math.isclose(
        annotated_result, source_answer, rel_tol=1e-9, abs_tol=1e-9
    ):
        return "annotated_formula_linear_formula_disagreement"
    for label, option_text in _option_segments(row["options"]):
        if label == row["correct"]:
            continue
        option_value = _option_segment_number(option_text)
        if option_value is not None and math.isclose(
            program_result, option_value, rel_tol=1e-9, abs_tol=1e-9
        ):
            return "program_result_matches_other_option"
    return "source_program_selected_answer_disagreement_unresolved"


def _evaluate_row(row: dict[str, Any], index: int) -> dict[str, Any]:
    parsed = _parse_program(row["linear_formula"])
    source_numbers = _extract_numbers(row["Problem"])
    selected_option, source_answer = _selected_option(row)
    scalar_results: list[float] = []
    steps: list[dict[str, Any]] = []
    for step_index, (operation, tokens, raw_call) in enumerate(parsed):
        values: list[float] = []
        dependencies: list[str] = []
        resolved_sources: dict[str, Any] = {}
        for token in tokens:
            if token.startswith("#"):
                reference = int(token[1:])
                if reference >= step_index:
                    raise Ineligible("forward_or_missing_dependency")
                values.append(scalar_results[reference])
                dependencies.append(f"step_{reference:02d}")
                resolved_sources[token] = f"step_{reference:02d}"
            elif token.startswith("n"):
                reference = int(token[1:])
                if reference >= len(source_numbers):
                    raise Ineligible("missing_question_number")
                values.append(source_numbers[reference])
                resolved_sources[token] = _json_number(source_numbers[reference])
            else:
                value = _parse_constant(token)
                values.append(value)
                resolved_sources[token] = _json_number(value)
        try:
            tool, arguments, result, scalar = _translate_and_execute(operation, values)
        except Ineligible:
            raise
        except (ArithmeticError, TypeError, ValueError) as exc:
            raise Ineligible(f"tool_execution_error:{type(exc).__name__}") from exc
        if not math.isfinite(scalar):
            raise Ineligible("non_finite_intermediate_result")
        scalar_results.append(scalar)
        context = {
            "kind": "mathqa_public_source_operation",
            "source_coordinate": f"test:{index}",
            "raw_dsl_call": raw_call,
            "source_operation": operation,
            "source_arguments": tokens,
            "argument_sources": resolved_sources,
        }
        steps.append({
            "id": f"step_{step_index:02d}",
            "query": raw_call,
            "prompt_context": json.dumps(context, ensure_ascii=False, sort_keys=True),
            "expected_tool": tool,
            "expected_args": arguments,
            "expected_answer": result,
            "depends_on": sorted(set(dependencies)),
            "source_operation": operation,
            "source_arguments": tokens,
            "source_raw_dsl_call": raw_call,
            "source_scalar_result": _json_number(scalar),
        })
    if not math.isclose(scalar_results[-1], source_answer, rel_tol=1e-9, abs_tol=1e-9):
        raise Ineligible(
            _classify_result_mismatch(
                row,
                selected_option,
                source_answer,
                scalar_results[-1],
            )
        )
    return {
        "steps": steps,
        "source_numbers": [_json_number(value) for value in source_numbers],
        "selected_option": selected_option,
        "source_answer": _json_number(source_answer),
    }


def _length_bucket(length: int) -> str:
    if length <= 5:
        return str(length)
    if length <= 7:
        return "6-7"
    if length <= 10:
        return "8-10"
    return "11+"


def _mapping_document() -> dict[str, Any]:
    return {
        "mapping_contract": "mechanical_mathqa_dsl_to_existing_layermcp_tools",
        "source_program_semantics": {
            "number_extraction": NUMBER_RE.pattern,
            "constant_rules": ["const_pi", "const_deg_to_rad", "const_<decimal>"],
            "dependency_rule": "#k references the zero-indexed result of source operation k",
            "power_rule": "power(base, exponent) preserves the resolved source exponent",
            "semantic_authority": (
                "MathQA linear_formula, annotated_formula, source rationale, and "
                "standard mathematical operation meaning"
            ),
            "downstream_executor_evidence": {
                "reference": (
                    "google/trax@220a62303ebf4ad18871aa5607b4dda2f064f2d2:"
                    "trax/data/tf_inputs.py"
                ),
                "role": (
                    "corroborating downstream implementation evidence only; "
                    "downstream deviations do not override MathQA source semantics"
                ),
            },
        },
        "supported_operations": {
            operation: (
                "gcd_lcm" if operation in {"gcd", "lcm"}
                else "modular_arithmetic" if operation == "reminder"
                else "calculator"
            )
            for operation in sorted(SUPPORTED_OPERATIONS)
        },
        "calculator_expression_templates": {
            "add": "{a} + {b}",
            "subtract": "{a} - {b}",
            "multiply": "{a} * {b}",
            "divide": "{a} / {b}; require b != 0",
            "speed": "{a} / {b}; require b != 0",
            "power": "({a}) ** {b}; require abs(b) <= 10 for LayerMCP calculator",
            "negate": "-{a}",
            "inverse": "1 / {a}; require a != 0",
            "floor": "{a} // 1",
            "sqrt": "{a} ** 0.5; require a >= 0",
            "rectangle_area": "{a} * {b}",
            "rectangle_perimeter": "2 * ({a} + {b})",
            "square_area": "{a} ** 2",
            "square_edge_by_area": "{a} ** 0.5; require a >= 0",
            "square_perimeter": "4 * {a}",
            "cube_edge_by_volume": "{a} ** (1 / 3); require a >= 0",
            "volume_cube": "{a} ** 3",
            "volume_cylinder": f"{repr(math.pi)} * {{a}} ** 2 * {{b}}",
            "volume_rectangular_prism": "{a} * {b} * {c}",
            "surface_cube": "6 * {a} ** 2",
            "circle_area": f"{repr(math.pi)} * {{a}} ** 2",
            "circumface": f"2 * {repr(math.pi)} * {{a}}",
        },
        "structured_tool_mappings": {
            "gcd": {"tool": "gcd_lcm", "operation": "gcd", "constraint": "resolved arguments are integers"},
            "lcm": {"tool": "gcd_lcm", "operation": "lcm", "constraint": "resolved arguments are integers"},
            "reminder": {"tool": "modular_arithmetic", "operation": "mod", "constraint": "resolved arguments are integers and modulus > 1"},
        },
        "canonical_pi": math.pi,
        "destination_tool_constraints": {
            "calculator_power_maximum_exponent_magnitude": (
                CALCULATOR_MAX_EXPONENT_MAGNITUDE
            ),
            "exclusion_reason": "destination_tool_exponent_limit",
        },
        "explicitly_unsupported_operations": sorted(EXPLICITLY_UNSUPPORTED),
        "other_operations": "unsupported unless explicitly listed in supported_operations",
        "numeric_acceptance": {"relative_tolerance": 1e-9, "absolute_tolerance": 1e-9},
        "mismatch_exclusion_reasons": {
            "numeric_option_parsing_limitation": (
                "program matches a ratio or mixed-number reading not produced by "
                "the released first-number option parser"
            ),
            "strict_tolerance_mismatch": (
                "program misses the strict target but is within the downstream "
                "executor's approximate one-percent tolerance"
            ),
            "annotated_formula_linear_formula_disagreement": (
                "supported annotated_formula matches the selected numeric option "
                "but linear_formula does not"
            ),
            "program_result_matches_other_option": (
                "linear_formula result matches a non-selected displayed option"
            ),
            "source_program_selected_answer_disagreement_unresolved": (
                "linear_formula result and selected numeric option disagree; no "
                "repair or causal assertion is made"
            ),
        },
    }


def _allocate(
    strata: dict[tuple[str, str], list[dict[str, Any]]],
    eligible_workflows: int,
) -> dict[tuple[str, str], int]:
    allocations: dict[tuple[str, str], int] = {}
    remainders = []
    assigned = 0
    for key in sorted(strata):
        quota = len(strata[key]) * SELECTED_WORKFLOWS / eligible_workflows
        base = math.floor(quota)
        allocations[key] = base
        assigned += base
        remainders.append((-(quota - base), key))
    for _, key in sorted(remainders)[: SELECTED_WORKFLOWS - assigned]:
        allocations[key] += 1
    if sum(allocations.values()) != SELECTED_WORKFLOWS:
        raise RuntimeError("Stratified allocation did not produce exactly 200 rows.")
    return allocations


def build_from_archive(archive_path: Path) -> dict[str, Any]:
    archive_bytes = archive_path.read_bytes()
    if _sha256(archive_bytes) != ARCHIVE_SHA256:
        raise ValueError("MathQA archive SHA-256 mismatch.")
    with zipfile.ZipFile(archive_path) as archive:
        members: dict[str, bytes] = {}
        for name, expected_hash in MEMBER_SHA256.items():
            data = archive.read(name)
            if _sha256(data) != expected_hash:
                raise ValueError(f"MathQA member SHA-256 mismatch: {name}")
            members[name] = data
    test_rows = json.loads(members["test.json"])
    train_rows = json.loads(members["train.json"])
    dev_rows = json.loads(members["dev.json"])
    if len(test_rows) != EXPECTED_SOURCE_ROWS:
        raise RuntimeError(f"Expected {EXPECTED_SOURCE_ROWS} test rows, found {len(test_rows)}.")
    non_test_questions = {row["Problem"] for row in train_rows + dev_rows}

    inventory = []
    eligible: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    exclusion_class_counts: Counter[str] = Counter()
    for index, row in enumerate(test_rows):
        coordinate = f"test:{index}"
        row_hash = _sha256(_canonical_bytes(row))
        try:
            if row["Problem"] in non_test_questions:
                raise Ineligible("exact_question_duplicate_train_or_dev")
            evaluated = _evaluate_row(row, index)
        except Ineligible as exc:
            reason_counts[exc.reason] += 1
            exclusion_class = _exclusion_class(exc.reason)
            exclusion_class_counts[exclusion_class] += 1
            inventory.append({
                "source_coordinate": coordinate,
                "source_row_sha256": row_hash,
                "status": "rejected",
                "exclusion_class": exclusion_class,
                "reason": exc.reason,
            })
            continue
        item = {"index": index, "row": row, "row_hash": row_hash, **evaluated}
        item["category"] = row["category"]
        item["length"] = len(evaluated["steps"])
        item["bucket"] = _length_bucket(item["length"])
        item["rank_sha256"] = _sha256(f"{coordinate}\n{row_hash}".encode())
        eligible.append(item)
        inventory.append({"source_coordinate": coordinate, "source_row_sha256": row_hash, "status": "eligible", "operation_count": item["length"]})

    if len(eligible) != EXPECTED_ELIGIBLE_WORKFLOWS or sum(item["length"] for item in eligible) != EXPECTED_ELIGIBLE_STEPS:
        raise RuntimeError(
            "Strict MathQA eligible population changed: expected "
            f"{EXPECTED_ELIGIBLE_WORKFLOWS}/{EXPECTED_ELIGIBLE_STEPS}, found "
            f"{len(eligible)}/{sum(item['length'] for item in eligible)}."
        )
    strata: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in eligible:
        strata[(item["category"], item["bucket"])].append(item)
    allocations = _allocate(strata, len(eligible))
    selected = []
    for key in sorted(strata):
        selected.extend(sorted(strata[key], key=lambda item: item["rank_sha256"])[: allocations[key]])
    selected.sort(key=lambda item: item["index"])
    selected_coordinates = [f"test:{item['index']}" for item in selected]
    subset_identity = [
        {
            "source_coordinate": f"test:{item['index']}",
            "source_row_sha256": item["row_hash"],
        }
        for item in selected
    ]
    subset_hash = _sha256(_canonical_bytes(subset_identity))
    mapping = _mapping_document()
    mapping_hash = _sha256(_pretty_bytes(mapping))
    provenance = {
        "classification": "public-derived deterministic program adaptation",
        "dataset": "MathQA",
        "source_repository": SOURCE_REPOSITORY,
        "source_url": SOURCE_URL,
        "source_revision": SOURCE_REVISION,
        "mirror_repository": MIRROR_REPOSITORY,
        "mirror_revision": MIRROR_REVISION,
        "paper_url": PAPER_URL,
        "license": LICENSE,
        "license_caveat": LICENSE_CAVEAT,
        "archive_sha256": ARCHIVE_SHA256,
        "source_split": "test",
        "source_member": "test.json",
        "source_member_sha256": MEMBER_SHA256["test.json"],
        "operation_list_sha256": MEMBER_SHA256["operation_list.txt"],
        "constant_list_sha256": MEMBER_SHA256["constant_list.txt"],
        "selected_subset_sha256": subset_hash,
        "operation_mapping_sha256": mapping_hash,
        "selection_method": "category/program-length proportional allocation; SHA-256 rank within stratum",
        "llm_generated_content": False,
    }
    benchmark = []
    for item in selected:
        row = item["row"]
        benchmark.append({
            "id": f"math_public_mathqa_test_{item['index']:04d}",
            "domain": "mathematics",
            "task_type": "multi_step_tool_routing",
            "difficulty": "medium" if item["length"] <= 4 else "hard",
            "source": "public_mathqa_program_derived",
            "benchmark_mode": "grounded_tool_execution",
            "workflow_execution_mode": "isolated_step",
            "query": row["Problem"],
            "expected_steps": item["steps"],
            "expected_final_answer": item["selected_option"],
            "final_step_outcome_contract": "exact_normalized_json",
            "expected_final_step_outcome": item["steps"][-1]["expected_answer"],
            "source_coordinate": f"test:{item['index']}",
            "source_row_index": item["index"],
            "source_row_sha256": item["row_hash"],
            "source_record": row,
            "source_extracted_numbers": item["source_numbers"],
            "source_correct_option_text": item["selected_option"],
            "source_correct_numeric_answer": item["source_answer"],
            "source_program_operation_count": item["length"],
            "benchmark_provenance": provenance,
            "notes": "Mechanical adaptation of the preserved MathQA linear_formula; no decomposition was authored in LayerMCP.",
        })
    fixture = {
        "provenance": provenance,
        "selected_source_coordinates": selected_coordinates,
        "rows": [{"source_coordinate": f"test:{item['index']}", "source_row_index": item["index"], "source_row_sha256": item["row_hash"], "source_record": item["row"]} for item in selected],
    }
    eligible_category = Counter(item["category"] for item in eligible)
    eligible_length = Counter(str(item["length"]) for item in eligible)
    selected_category = Counter(item["category"] for item in selected)
    selected_length = Counter(str(item["length"]) for item in selected)
    eligible_operations = Counter(
        step["source_operation"] for item in eligible for step in item["steps"]
    )
    eligible_tools = Counter(
        step["expected_tool"] for item in eligible for step in item["steps"]
    )
    selected_operations = Counter(
        step["source_operation"] for item in selected for step in item["steps"]
    )
    selected_tools = Counter(
        step["expected_tool"] for item in selected for step in item["steps"]
    )
    stratum_rows = []
    for key in sorted(strata):
        quota = len(strata[key]) * SELECTED_WORKFLOWS / len(eligible)
        stratum_rows.append({"category": key[0], "program_length_bucket": key[1], "eligible": len(strata[key]), "quota": quota, "floor": math.floor(quota), "selected": allocations[key]})
    manifest = {
        "provenance": provenance,
        "source_inventory": {"total_rows": len(test_rows), "eligible_workflows": len(eligible), "eligible_steps": sum(item["length"] for item in eligible), "selected_workflows": len(selected), "selected_steps": sum(item["length"] for item in selected)},
        "rejection_counts": dict(sorted(reason_counts.items())),
        "exclusion_class_counts": {
            name: exclusion_class_counts.get(name, 0)
            for name in EXCLUSION_CLASS_DESCRIPTIONS
        },
        "exclusion_class_descriptions": EXCLUSION_CLASS_DESCRIPTIONS,
        "eligible_distribution": {"category": dict(sorted(eligible_category.items())), "program_length": dict(sorted(eligible_length.items(), key=lambda pair: int(pair[0]))), "operation": dict(sorted(eligible_operations.items())), "tool": dict(sorted(eligible_tools.items()))},
        "selected_distribution": {"category": dict(sorted(selected_category.items())), "program_length": dict(sorted(selected_length.items(), key=lambda pair: int(pair[0]))), "operation": dict(sorted(selected_operations.items())), "tool": dict(sorted(selected_tools.items()))},
        "stratified_allocation": stratum_rows,
        "selected_source_coordinates": selected_coordinates,
        "selected_source_rows": [
            {
                "source_coordinate": f"test:{item['index']}",
                "source_row_sha256": item["row_hash"],
                "selection_rank_sha256": item["rank_sha256"],
                "category": item["category"],
                "program_length_bucket": item["bucket"],
            }
            for item in selected
        ],
        "row_inventory": inventory,
    }
    artifact_bytes = {
        BENCHMARK_NAME: _pretty_bytes(benchmark),
        FIXTURE_NAME: _pretty_bytes(fixture),
        MAPPING_NAME: _pretty_bytes(mapping),
    }
    manifest["generated_artifact_sha256"] = {name: _sha256(data) for name, data in sorted(artifact_bytes.items())}
    manifest["generation_sha256"] = _sha256(_canonical_bytes({"provenance": provenance, "artifacts": manifest["generated_artifact_sha256"]}))
    artifact_bytes[MANIFEST_NAME] = _pretty_bytes(manifest)
    return {"benchmark": benchmark, "fixture": fixture, "mapping": mapping, "manifest": manifest, "artifact_bytes": artifact_bytes}


def write_artifacts(built: dict[str, Any], output_dir: Path) -> None:
    for relative_name, data in built["artifact_bytes"].items():
        path = output_dir / relative_name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    built = build_from_archive(args.source_archive)
    if args.check:
        mismatches = [name for name, data in built["artifact_bytes"].items() if not (args.output_dir / name).is_file() or (args.output_dir / name).read_bytes() != data]
        if mismatches:
            raise SystemExit("Generated artifacts differ: " + ", ".join(mismatches))
        print(f"Verified {len(built['benchmark'])} workflows byte-for-byte.")
        return
    write_artifacts(built, args.output_dir)
    steps = sum(len(row["expected_steps"]) for row in built["benchmark"])
    print(f"Wrote {len(built['benchmark'])} workflows / {steps} steps to {args.output_dir}")


if __name__ == "__main__":
    main()
