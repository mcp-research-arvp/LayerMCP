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

_STOCK_PRICES = {
    "AAPL": 214.35,
    "MSFT": 497.12,
    "TSLA": 182.44,
    "NVDA": 141.67,
}

_CODE_FILES = {
    "src/auth.py": "def authenticate(token):\n    return token == 'offline-demo-token'\n",
    "src/payments.py": "def calculate_invoice_total(items):\n    return sum(item['price'] for item in items)\n",
    "tests/test_auth.py": "def test_authenticate_accepts_demo_token():\n    assert authenticate('offline-demo-token')\n",
    "README.md": "# Example Project\n\nOffline fixture used for code-routing benchmarks.\n",
}


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
    Numerically evaluate an arithmetic expression and return its value. Use
    this for requests that ask to compute, evaluate, or find a numeric value.
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


def stock_price_api(ticker: str) -> dict[str, Any]:
    """
    Return deterministic fake stock prices for offline finance routing experiments.
    """
    normalized = ticker.strip().upper()
    if normalized not in _STOCK_PRICES:
        raise ValueError(
            "ticker must be one of: " + ", ".join(sorted(_STOCK_PRICES))
        )

    return {
        "ticker": normalized,
        "price": _STOCK_PRICES[normalized],
        "currency": "USD",
        "source": "offline-fixture",
    }


def unit_converter(value: float, from_unit: str, to_unit: str) -> dict[str, Any]:
    """
    Convert supported distance or temperature units using deterministic formulas.
    """
    source_unit = from_unit.strip().lower()
    target_unit = to_unit.strip().lower()
    conversions = {
        ("km", "miles"): value * 0.621371,
        ("kilometers", "miles"): value * 0.621371,
        ("miles", "km"): value / 0.621371,
        ("miles", "kilometers"): value / 0.621371,
        ("celsius", "fahrenheit"): (value * 9 / 5) + 32,
        ("fahrenheit", "celsius"): (value - 32) * 5 / 9,
    }

    key = (source_unit, target_unit)
    if key not in conversions:
        raise ValueError(
            "supported conversions are km/miles and Celsius/Fahrenheit."
        )

    return {
        "value": value,
        "from_unit": source_unit,
        "to_unit": target_unit,
        "converted_value": round(conversions[key], 4),
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


def read_code_file(path: str) -> dict[str, Any]:
    """
    Return deterministic fake repository file contents for offline coding tasks.
    """
    normalized = path.strip().replace("\\", "/")
    if normalized not in _CODE_FILES:
        raise ValueError(
            "path must be one of: " + ", ".join(sorted(_CODE_FILES))
        )

    return {
        "path": normalized,
        "content": _CODE_FILES[normalized],
        "source": "offline-fixture",
    }


def ticket_router(issue: str) -> dict[str, Any]:
    """
    Route an enterprise support issue to a deterministic offline ticket category.
    """
    normalized = issue.strip()
    if not normalized:
        raise ValueError("issue must not be empty.")

    lowered = normalized.lower()
    if any(word in lowered for word in ("invoice", "billing", "refund", "charge")):
        category = "billing"
    elif any(word in lowered for word in ("password", "login", "account", "profile")):
        category = "account"
    elif any(word in lowered for word in ("breach", "security", "phishing", "mfa")):
        category = "security"
    else:
        category = "technical_support"

    return {
        "issue": normalized,
        "category": category,
        "source": "offline-fixture",
    }
