"""Build deterministic controlled multi-step math routing workflows."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
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


OUTPUT_PATH = Path(__file__).with_name("math_multistep_controlled.json")
TOOLS: dict[str, Callable[..., dict[str, Any]]] = {
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


def _step(
    index: int,
    query: str,
    tool: str,
    arguments: dict[str, Any],
    *,
    depends_on: list[str] | None = None,
    relation: str,
) -> dict[str, Any]:
    step_id = f"step_{index:02d}"
    context = {
        "kind": "math_controlled_current_step_v1",
        "instruction": "Use the authoritative inputs for only this current step.",
        "operation": query,
        "inputs": arguments,
        "sequence_relation": relation,
    }
    return {
        "id": step_id,
        "query": query,
        "prompt_context": json.dumps(context, sort_keys=True),
        "expected_tool": tool,
        "expected_args": arguments,
        "expected_answer": TOOLS[tool](**arguments),
        "depends_on": depends_on or [],
    }


def _workflow(
    number: int,
    category: str,
    query: str,
    steps: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "id": f"math_multistep_controlled_{number:03d}",
        "domain": "mathematics",
        "task_type": "multi_step_tool_routing",
        "difficulty": "medium" if len(steps) == 2 else "hard",
        "source": "controlled_synthetic",
        "benchmark_mode": "grounded_tool_execution",
        "workflow_execution_mode": "isolated_step",
        "query": query,
        "expected_steps": steps,
        "expected_final_answer": steps[-1]["expected_answer"],
        "workflow_final_tool_result_contract": "exact_normalized_json_v1",
        "expected_final_tool_result": steps[-1]["expected_answer"],
        "perturbation_type": "dependent_math_tool_sequence",
        "notes": f"Deterministic controlled {category} workflow; each later call uses an earlier result.",
    }


def build_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    products = [(12, 15), (14, 18), (16, 21), (22, 27), (24, 35)]
    for a, b in products:
        product = a * b
        steps = [
            _step(0, f"Calculate {a} times {b}.", "calculator", {"expression": f"{a} * {b}"}, relation="Produces the integer used by the next step."),
            _step(1, f"Prime-factorize the resulting integer {product}.", "integer_factorization", {"value": str(product)}, depends_on=["step_00"], relation="Uses the numeric result from step_00."),
        ]
        rows.append(_workflow(len(rows) + 1, "arithmetic-to-factorization", f"Compute {a} × {b}, then prime-factorize the product.", steps))

    pairs = [(18, 24), (28, 42), (45, 60), (56, 84), (72, 90)]
    for left, right in pairs:
        first, second = left + 6, right + 6
        steps = [
            _step(0, f"Calculate {left} plus 6.", "calculator", {"expression": f"{left} + 6"}, relation="Produces the first integer for the GCD/LCM step."),
            _step(1, f"Calculate {right} plus 6.", "calculator", {"expression": f"{right} + 6"}, relation="Produces the second integer for the GCD/LCM step."),
            _step(2, f"Compute both the GCD and LCM of {first} and {second}.", "gcd_lcm", {"values": [str(first), str(second)], "operation": "both"}, depends_on=["step_00", "step_01"], relation="Uses both arithmetic results from steps 00 and 01."),
        ]
        rows.append(_workflow(len(rows) + 1, "arithmetic-to-gcd-lcm", f"Evaluate two sums, then compute the GCD and LCM of their results: ({left} + 6) and ({right} + 6).", steps))

    quadratics = [(1, -5, 6), (1, -7, 12), (1, 1, -12), (2, -10, 12), (3, -15, 18)]
    for a, b, c in quadratics:
        expression = f"{a}*x**2 + ({b})*x + ({c})"
        factored = factor_expression(expression)["factored"]
        steps = [
            _step(0, f"Factor {expression}.", "factor_expression", {"expression": expression}, relation="Produces the factored form used to form the equation."),
            _step(1, f"Solve {factored} = 0 for x.", "solve_equation", {"equation": f"{factored} = 0", "variable": "x"}, depends_on=["step_00"], relation="Solves the equation formed from the factorization in step_00."),
        ]
        rows.append(_workflow(len(rows) + 1, "factor-then-solve", f"Factor the polynomial {expression}, then solve the factored equation equal to zero.", steps))

    derivatives = ["x**3 + 3*x**2", "2*x**4 - 8*x", "x**5 + 5*x**2", "3*x**3 - 12*x", "4*x**4 + 8*x**2"]
    for expression in derivatives:
        derivative = differentiate_expression(expression)["derivative"]
        steps = [
            _step(0, f"Differentiate {expression} with respect to x.", "differentiate_expression", {"expression": expression, "variable": "x"}, relation="Produces the derivative used by the next step."),
            _step(1, f"Simplify the derivative {derivative}.", "simplify_expression", {"expression": derivative}, depends_on=["step_00"], relation="Simplifies the derivative returned by step_00."),
        ]
        rows.append(_workflow(len(rows) + 1, "differentiate-then-simplify", f"Differentiate {expression} and simplify the resulting derivative.", steps))

    stationary = ["x**2 - 6*x", "x**3 - 3*x", "2*x**2 + 8*x", "x**3 - 12*x", "3*x**2 - 18*x"]
    for expression in stationary:
        derivative = differentiate_expression(expression)["derivative"]
        steps = [
            _step(0, f"Differentiate {expression} with respect to x.", "differentiate_expression", {"expression": expression, "variable": "x"}, relation="Produces the derivative equation for the next step."),
            _step(1, f"Solve {derivative} = 0 for x.", "solve_equation", {"equation": f"{derivative} = 0", "variable": "x"}, depends_on=["step_00"], relation="Finds stationary points by setting the derivative from step_00 to zero."),
        ]
        rows.append(_workflow(len(rows) + 1, "differentiate-then-solve", f"Differentiate {expression}, then solve where its derivative equals zero.", steps))

    products_to_expand = ["(x + 2)*(x + 5)", "(x - 3)*(x + 4)", "(2*x + 1)*(x - 6)", "(x + 7)**2", "(3*x - 2)*(x + 3)"]
    for expression in products_to_expand:
        expanded = expand_expression(expression)["expanded"]
        steps = [
            _step(0, f"Expand {expression}.", "expand_expression", {"expression": expression}, relation="Produces the expanded polynomial used by the next step."),
            _step(1, f"Differentiate the expanded expression {expanded}.", "differentiate_expression", {"expression": expanded, "variable": "x"}, depends_on=["step_00"], relation="Differentiates the expansion returned by step_00."),
        ]
        rows.append(_workflow(len(rows) + 1, "expand-then-differentiate", f"Expand {expression}, then differentiate the expanded polynomial.", steps))

    base_cases = [("1011 + 110", 2, 10), ("2A + 16", 16, 10), ("123 + 21", 4, 10), ("77 - 12", 8, 10), ("120 * 12", 3, 10)]
    for expression, input_base, output_base in base_cases:
        decimal = base_arithmetic(expression, input_base, output_base)["decimal_result"]
        steps = [
            _step(0, f"Evaluate {expression} in base {input_base} and express it in base {output_base}.", "base_arithmetic", {"expression": expression, "input_base": input_base, "output_base": output_base}, relation="Produces the decimal integer used by the next step."),
            _step(1, f"Prime-factorize the resulting decimal integer {decimal}.", "integer_factorization", {"value": str(decimal)}, depends_on=["step_00"], relation="Factorizes the decimal_result returned by step_00."),
        ]
        rows.append(_workflow(len(rows) + 1, "base-arithmetic-to-factorization", f"Evaluate the base-{input_base} expression {expression}, then prime-factorize its decimal result.", steps))

    modular_cases = [("83", 7, 10), ("145", 12, 9), ("2**10", 17, 5), ("999", 13, 8), ("12345", 19, 4)]
    for expression, modulus, increment in modular_cases:
        residue = modular_arithmetic(expression, modulus)["result"]
        steps = [
            _step(0, f"Compute {expression} modulo {modulus}.", "modular_arithmetic", {"expression": expression, "modulus": modulus, "operation": "mod"}, relation="Produces the residue used by the arithmetic step."),
            _step(1, f"Add {increment} to the residue {residue}.", "calculator", {"expression": f"{residue} + {increment}"}, depends_on=["step_00"], relation="Uses the modular result from step_00."),
        ]
        rows.append(_workflow(len(rows) + 1, "modular-then-arithmetic", f"Find {expression} modulo {modulus}, then add {increment} to the residue.", steps))

    conversions = [(2.5, "kilometer", "meter", 3), (750.0, "centimeter", "meter", 4), (3.0, "hour", "minute", 2), (5.0, "kilogram", "gram", 2), (32.0, "foot", "meter", 10)]
    for value, source_unit, target_unit, multiplier in conversions:
        converted = convert_units(value, source_unit, target_unit)["converted_value"]
        steps = [
            _step(0, f"Convert {value} {source_unit} to {target_unit}.", "convert_units", {"value": value, "from_unit": source_unit, "to_unit": target_unit}, relation="Produces the converted magnitude used by the next step."),
            _step(1, f"Multiply the converted value {converted} by {multiplier}.", "calculator", {"expression": f"{converted} * {multiplier}"}, depends_on=["step_00"], relation="Uses converted_value from step_00."),
        ]
        rows.append(_workflow(len(rows) + 1, "unit-conversion-then-arithmetic", f"Convert {value} {source_unit} to {target_unit}, then multiply the converted magnitude by {multiplier}.", steps))

    rational_equations = [("(2*x + 4)/2", 7), ("3*(x + 2) - 2*x", 11), ("(x**2 - 9)/(x - 3)", 8), ("2*x + 3*x - x", 20), ("(4*x + 8)/4 + x", 10)]
    for expression, target in rational_equations:
        simplified = simplify_expression(expression)["simplified"]
        steps = [
            _step(0, f"Simplify {expression}.", "simplify_expression", {"expression": expression}, relation="Produces the simplified left side for the equation."),
            _step(1, f"Solve {simplified} = {target} for x.", "solve_equation", {"equation": f"{simplified} = {target}", "variable": "x"}, depends_on=["step_00"], relation="Solves an equation using the simplified result from step_00."),
        ]
        rows.append(_workflow(len(rows) + 1, "simplify-then-solve", f"Simplify {expression}, then solve when the simplified expression equals {target}.", steps))

    return rows


def main() -> None:
    rows = build_rows()
    if len(rows) != 50:
        raise RuntimeError(f"Expected 50 workflows, generated {len(rows)}.")
    OUTPUT_PATH.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(rows)} workflows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
