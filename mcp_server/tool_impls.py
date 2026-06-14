from __future__ import annotations

import ast
import operator
import re
from typing import Any

_ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_ALLOWED_UNARYOPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

_CUSTOMER_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,32}$")


def _safe_eval(node: ast.AST) -> int | float:
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)

    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value

    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _ALLOWED_BINOPS:
            raise ValueError(f"Unsupported operator: {op_type.__name__}")

        left = _safe_eval(node.left)
        right = _safe_eval(node.right)

        if op_type is ast.Pow and abs(right) > 10:
            raise ValueError("Exponent is too large for this research fixture.")

        try:
            return _ALLOWED_BINOPS[op_type](left, right)
        except ZeroDivisionError as exc:
            raise ValueError("Division by zero is not allowed.") from exc

    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _ALLOWED_UNARYOPS:
            raise ValueError(f"Unsupported unary operator: {op_type.__name__}")
        return _ALLOWED_UNARYOPS[op_type](_safe_eval(node.operand))

    raise ValueError("Expression contains unsupported syntax.")


def calculator(expression: str) -> dict[str, Any]:
    """
    Safely evaluate an arithmetic expression for research experiments.
    """
    normalized = expression.strip()
    if not normalized:
        raise ValueError("Expression must not be empty.")
    if len(normalized) > 200:
        raise ValueError("Expression is too long.")

    tree = ast.parse(normalized, mode="eval")
    result = _safe_eval(tree)

    return {
        "expression": normalized,
        "result": result,
    }


def customer_lookup(customer_id: str) -> dict[str, Any]:
    """
    Return deterministic mock customer data for offline MCP experiments.
    """
    normalized = customer_id.strip()
    if not _CUSTOMER_ID_PATTERN.fullmatch(normalized):
        raise ValueError(
            "customer_id must be 1-32 characters using letters, digits, _ or -."
        )

    last_digit = next((char for char in reversed(normalized) if char.isdigit()), "0")
    tier = "premium" if int(last_digit) % 2 else "standard"

    return {
        "customer_id": normalized,
        "status": tier,
        "source": "offline-fixture",
    }


def github_search(query: str) -> dict[str, Any]:
    """
    Return deterministic mock GitHub-style search results without network access.
    """
    normalized = query.strip()
    if not normalized:
        raise ValueError("query must not be empty.")

    keywords = re.findall(r"[A-Za-z0-9_-]+", normalized.lower())
    summary = " ".join(keywords[:3]) or "general"

    return {
        "query": normalized,
        "source": "offline-fixture",
        "results": [
            {
                "repository": "example/research-mcp",
                "title": f"Issue related to {summary}",
            },
            {
                "repository": "example/tool-routing",
                "title": f"Discussion about {summary}",
            },
        ],
    }
