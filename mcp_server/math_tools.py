from __future__ import annotations

from typing import Any

import sympy
from pint import UnitRegistry

_MAX_EXPRESSION_LENGTH = 500
_UNIT_REGISTRY = UnitRegistry()


def _normalize_expression(expression: str) -> str:
    normalized = expression.strip()
    if not normalized:
        raise ValueError("expression must not be empty.")
    if len(normalized) > _MAX_EXPRESSION_LENGTH:
        raise ValueError("expression is too long.")
    return normalized


def _parse_expression(expression: str) -> sympy.Expr:
    try:
        return sympy.sympify(expression)
    except (SympifyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid symbolic expression: {expression}") from exc


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
