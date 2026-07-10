from __future__ import annotations

import ast
import math
import re
from typing import Any

import sympy
from pint import UnitRegistry
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

_MAX_EXPRESSION_LENGTH = 500
_MAX_INTEGER_ABS_VALUE = 10**18
_UNIT_REGISTRY = UnitRegistry()
_SYMPY_TRANSFORMATIONS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
)


def _normalize_expression(expression: str) -> str:
    normalized = expression.strip()
    if not normalized:
        raise ValueError("expression must not be empty.")
    if len(normalized) > _MAX_EXPRESSION_LENGTH:
        raise ValueError("expression is too long.")
    return normalized


def _parse_expression(expression: str) -> sympy.Expr:
    normalized = _preprocess_math_notation(expression)
    try:
        return parse_expr(
            normalized,
            transformations=_SYMPY_TRANSFORMATIONS,
            evaluate=True,
        )
    except (SympifyError, SyntaxError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid symbolic expression: {expression}") from exc


def _parse_integer_expression(value: str) -> int:
    normalized = _normalize_expression(value)
    parsed = _parse_expression(normalized)
    if parsed.free_symbols:
        raise ValueError("value must be an integer expression without symbols.")

    simplified = sympy.simplify(parsed)
    if not simplified.is_integer:
        raise ValueError("value must evaluate to an integer.")

    integer_value = int(simplified)
    if abs(integer_value) > _MAX_INTEGER_ABS_VALUE:
        raise ValueError("integer expression is too large.")
    return integer_value


def _preprocess_math_notation(expression: str) -> str:
    normalized = expression.strip()
    normalized = normalized.replace("\\left", "").replace("\\right", "")
    normalized = normalized.replace("\\cdot", "*")
    normalized = normalized.replace("^", "**")
    normalized = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"((\1)/(\2))", normalized)
    normalized = re.sub(r"\\sqrt\{([^{}]+)\}", r"sqrt(\1)", normalized)
    normalized = re.sub(r"\*\*\{([^{}]+)\}", r"**(\1)", normalized)
    return normalized


try:
    from sympy.core.sympify import SympifyError
except ImportError:  # pragma: no cover - compatibility fallback
    SympifyError = Exception


def simplify_expression(expression: str) -> dict[str, Any]:
    """
    Simplify a symbolic expression using SymPy.
    """
    normalized = _normalize_expression(expression)
    parsed = _parse_expression(normalized)
    simplified = sympy.simplify(parsed)

    return {
        "expression": normalized,
        "simplified": str(simplified),
        "source": "sympy",
    }


def solve_equation(equation: str, variable: str = "x") -> dict[str, Any]:
    """
    Solve a symbolic equation containing '=' for the requested variable.
    """
    normalized_equation = _normalize_expression(equation)
    normalized_variable = variable.strip()
    if not normalized_variable:
        raise ValueError("variable must not be empty.")
    if "=" not in normalized_equation:
        raise ValueError("equation must contain '='.")

    left_text, right_text = normalized_equation.split("=", 1)
    left = _parse_expression(left_text.strip())
    right = _parse_expression(right_text.strip())
    symbol = sympy.Symbol(normalized_variable)

    try:
        solutions = sympy.solve(sympy.Eq(left, right), symbol)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"could not solve equation for {normalized_variable}.") from exc

    return {
        "equation": normalized_equation,
        "variable": normalized_variable,
        "solutions": [str(solution) for solution in solutions],
        "source": "sympy",
    }


def factor_expression(expression: str) -> dict[str, Any]:
    """
    Factor a symbolic expression using SymPy.
    """
    normalized = _normalize_expression(expression)
    parsed = _parse_expression(normalized)
    factored = sympy.factor(parsed)

    return {
        "expression": normalized,
        "factored": str(factored),
        "source": "sympy",
    }


def expand_expression(expression: str) -> dict[str, Any]:
    """
    Expand a symbolic expression using SymPy.
    """
    normalized = _normalize_expression(expression)
    parsed = _parse_expression(normalized)
    expanded = sympy.expand(parsed)

    return {
        "expression": normalized,
        "expanded": str(expanded),
        "source": "sympy",
    }


def differentiate_expression(expression: str, variable: str = "x") -> dict[str, Any]:
    """
    Differentiate a symbolic expression with respect to a variable using SymPy.
    """
    normalized = _normalize_expression(expression)
    normalized_variable = variable.strip()
    if not normalized_variable:
        raise ValueError("variable must not be empty.")

    parsed = _parse_expression(normalized)
    symbol = sympy.Symbol(normalized_variable)
    derivative = sympy.diff(parsed, symbol)

    return {
        "expression": normalized,
        "variable": normalized_variable,
        "derivative": str(derivative),
        "source": "sympy",
    }


def convert_units(value: float, from_unit: str, to_unit: str) -> dict[str, Any]:
    """
    Convert units using Pint and return a JSON-serializable numeric magnitude.
    """
    source_unit = from_unit.strip()
    target_unit = to_unit.strip()
    if not source_unit:
        raise ValueError("from_unit must not be empty.")
    if not target_unit:
        raise ValueError("to_unit must not be empty.")

    try:
        converted = (value * _UNIT_REGISTRY(source_unit)).to(target_unit)
    except Exception as exc:
        raise ValueError(f"unsupported unit conversion: {from_unit} to {to_unit}.") from exc

    return {
        "value": value,
        "from_unit": source_unit,
        "to_unit": target_unit,
        "converted_value": round(float(converted.magnitude), 6),
        "source": "pint",
    }


def integer_factorization(value: str) -> dict[str, Any]:
    """
    Prime-factorize an integer expression using SymPy.
    """
    integer_value = _parse_integer_expression(value)
    if integer_value == 0:
        raise ValueError("cannot prime-factorize zero.")

    factors = sympy.factorint(abs(integer_value))
    sorted_factors = {
        str(prime): int(factors[prime])
        for prime in sorted(factors)
    }
    factor_list: list[str] = []
    for prime_text, exponent in sorted_factors.items():
        factor_list.extend([prime_text] * exponent)

    prime_values = [int(prime) for prime in sorted_factors]
    return {
        "value": value.strip(),
        "integer_value": integer_value,
        "prime_factors": sorted_factors,
        "factor_list": factor_list,
        "least_prime_factor": min(prime_values) if prime_values else None,
        "greatest_prime_factor": max(prime_values) if prime_values else None,
        "source": "sympy",
    }


def gcd_lcm(values: list[str], operation: str = "both") -> dict[str, Any]:
    """
    Compute the GCD and/or LCM of integer expressions.
    """
    normalized_operation = operation.strip().lower()
    if normalized_operation not in {"gcd", "lcm", "both"}:
        raise ValueError("operation must be one of: gcd, lcm, both.")
    if not values:
        raise ValueError("values must contain at least one integer expression.")

    integer_values = [_parse_integer_expression(value) for value in values]
    result: dict[str, Any] = {
        "values": values,
        "integer_values": integer_values,
        "operation": normalized_operation,
        "source": "python-math",
    }
    if normalized_operation in {"gcd", "both"}:
        result["gcd"] = abs(math.gcd(*integer_values))
    if normalized_operation in {"lcm", "both"}:
        result["lcm"] = abs(math.lcm(*integer_values))
    return result


def modular_arithmetic(
    expression: str,
    modulus: int,
    operation: str = "mod",
    exponent: int | None = None,
) -> dict[str, Any]:
    """
    Compute modular residues, modular powers, or modular inverses.
    """
    if modulus <= 1:
        raise ValueError("modulus must be greater than 1.")
    normalized_operation = operation.strip().lower()
    if normalized_operation not in {"mod", "inverse", "pow"}:
        raise ValueError("operation must be one of: mod, inverse, pow.")

    value = _parse_integer_expression(expression)
    if normalized_operation == "mod":
        result = value % modulus
    elif normalized_operation == "inverse":
        try:
            result = pow(value, -1, modulus)
        except ValueError as exc:
            raise ValueError("modular inverse does not exist.") from exc
    else:
        if exponent is None:
            raise ValueError("exponent is required for pow operation.")
        result = pow(value, exponent, modulus)

    return {
        "operation": normalized_operation,
        "expression": expression.strip(),
        "value": value,
        "modulus": modulus,
        "exponent": exponent,
        "result": result,
        "source": "python-pow",
    }


def _evaluate_base_ast(node: ast.AST) -> int:
    if isinstance(node, ast.Expression):
        return _evaluate_base_ast(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_evaluate_base_ast(node.operand)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd):
        return _evaluate_base_ast(node.operand)
    if isinstance(node, ast.BinOp):
        left = _evaluate_base_ast(node.left)
        right = _evaluate_base_ast(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
    raise ValueError("base arithmetic supports only +, -, *, and parentheses.")


def _format_in_base(value: int, base: int) -> str:
    if value == 0:
        return "0"

    digits = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    sign = "-" if value < 0 else ""
    remaining = abs(value)
    output = []
    while remaining:
        remaining, digit = divmod(remaining, base)
        output.append(digits[digit])
    return sign + "".join(reversed(output))


def _rewrite_base_expression(expression: str, base: int) -> str:
    if not 2 <= base <= 36:
        raise ValueError("base must be between 2 and 36.")
    if not re.fullmatch(r"[0-9A-Za-z+\-*\s()]+", expression):
        raise ValueError("base expression contains unsupported characters.")

    def replace_token(match: re.Match[str]) -> str:
        token = match.group(0)
        try:
            return str(int(token, base))
        except ValueError as exc:
            raise ValueError(f"invalid base-{base} digit in {token}.") from exc

    return re.sub(r"[0-9A-Za-z]+", replace_token, expression)


def base_arithmetic(
    expression: str,
    input_base: int,
    output_base: int | None = None,
) -> dict[str, Any]:
    """
    Evaluate simple arithmetic in a non-decimal base.
    """
    normalized = _normalize_expression(expression)
    target_base = input_base if output_base is None else output_base
    if not 2 <= input_base <= 36:
        raise ValueError("input_base must be between 2 and 36.")
    if not 2 <= target_base <= 36:
        raise ValueError("output_base must be between 2 and 36.")

    decimal_expression = _rewrite_base_expression(normalized, input_base)
    try:
        parsed = ast.parse(decimal_expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError("invalid base arithmetic expression.") from exc
    decimal_result = _evaluate_base_ast(parsed)

    return {
        "expression": normalized,
        "decimal_result": decimal_result,
        "base_result": _format_in_base(decimal_result, target_base),
        "input_base": input_base,
        "output_base": target_base,
        "source": "python-ast",
    }
