from __future__ import annotations

import hashlib
import re
from typing import Any

_VALID_ORDER_STATUSES = {
    "pending",
    "processing",
    "shipped",
    "delivered",
    "cancelled",
    "refunded",
}

_VALID_PRIORITIES = {"low", "normal", "high", "urgent"}

_ORDERS = {
    "ORD-1001": {
        "order_id": "ORD-1001",
        "customer_id": "CUST-1001",
        "status": "processing",
        "items": [
            {"sku": "SKU-ALPHA", "quantity": 2, "unit_price": 19.5},
            {"sku": "SKU-BETA", "quantity": 1, "unit_price": 44.0},
        ],
        "total": 83.0,
    },
    "ORD-1002": {
        "order_id": "ORD-1002",
        "customer_id": "CUST-1002",
        "status": "shipped",
        "items": [
            {"sku": "SKU-GAMMA", "quantity": 1, "unit_price": 129.99},
        ],
        "total": 129.99,
    },
    "ORD-1003": {
        "order_id": "ORD-1003",
        "customer_id": "CUST-1003",
        "status": "delivered",
        "items": [
            {"sku": "SKU-DELTA", "quantity": 3, "unit_price": 8.25},
        ],
        "total": 24.75,
    },
}

_KNOWLEDGE_BASE = [
    {
        "article_id": "KB-001",
        "title": "Reset a customer password",
        "category": "account",
        "keywords": {"reset", "password", "login", "account"},
    },
    {
        "article_id": "KB-002",
        "title": "Refund policy for duplicate charges",
        "category": "billing",
        "keywords": {"refund", "invoice", "charge", "billing", "duplicate"},
    },
    {
        "article_id": "KB-003",
        "title": "Investigate suspicious sign-in alerts",
        "category": "security",
        "keywords": {"security", "phishing", "mfa", "alert", "suspicious"},
    },
]


def _normalize_identifier(value: str, field_name: str) -> str:
    normalized = value.strip().upper()
    if not normalized:
        raise ValueError(f"{field_name} must not be empty.")
    if not re.fullmatch(r"[A-Z0-9_-]{1,32}", normalized):
        raise ValueError(f"{field_name} must be 1-32 characters using letters, digits, _ or -.")
    return normalized


def _copy_order(order: dict[str, Any]) -> dict[str, Any]:
    return {
        "order_id": order["order_id"],
        "customer_id": order["customer_id"],
        "status": order["status"],
        "items": [dict(item) for item in order["items"]],
        "total": order["total"],
        "source": "offline-fixture",
    }


def get_order(order_id: str) -> dict[str, Any]:
    """
    Return deterministic offline order data.
    """
    normalized = _normalize_identifier(order_id, "order_id")
    if normalized not in _ORDERS:
        raise ValueError("order_id must be one of: " + ", ".join(sorted(_ORDERS)))

    return _copy_order(_ORDERS[normalized])


def update_order_status(order_id: str, status: str) -> dict[str, Any]:
    """
    Return a deterministic simulated order status update.
    """
    normalized_order_id = _normalize_identifier(order_id, "order_id")
    normalized_status = status.strip().lower()
    if normalized_order_id not in _ORDERS:
        raise ValueError("order_id must be one of: " + ", ".join(sorted(_ORDERS)))
    if normalized_status not in _VALID_ORDER_STATUSES:
        raise ValueError(
            "status must be one of: " + ", ".join(sorted(_VALID_ORDER_STATUSES))
        )

    previous_status = _ORDERS[normalized_order_id]["status"]
    return {
        "order_id": normalized_order_id,
        "previous_status": previous_status,
        "status": normalized_status,
        "updated": previous_status != normalized_status,
        "source": "offline-fixture",
    }


def create_support_ticket(
    customer_id: str,
    issue: str,
    priority: str = "normal",
) -> dict[str, Any]:
    """
    Create a deterministic offline support ticket fixture.
    """
    normalized_customer_id = _normalize_identifier(customer_id, "customer_id")
    normalized_issue = issue.strip()
    normalized_priority = priority.strip().lower()
    if not normalized_issue:
        raise ValueError("issue must not be empty.")
    if normalized_priority not in _VALID_PRIORITIES:
        raise ValueError(
            "priority must be one of: " + ", ".join(sorted(_VALID_PRIORITIES))
        )

    digest = hashlib.sha256(
        f"{normalized_customer_id}|{normalized_issue.lower()}|{normalized_priority}".encode(
            "utf-8"
        )
    ).hexdigest()[:8].upper()

    return {
        "ticket_id": f"TCK-{digest}",
        "customer_id": normalized_customer_id,
        "issue": normalized_issue,
        "priority": normalized_priority,
        "status": "open",
        "source": "offline-fixture",
    }


def search_knowledge_base(query: str) -> dict[str, Any]:
    """
    Search deterministic offline knowledge-base articles by keyword overlap.
    """
    normalized = query.strip()
    if not normalized:
        raise ValueError("query must not be empty.")

    query_terms = set(re.findall(r"[A-Za-z0-9_-]+", normalized.lower()))
    results = []
    for article in _KNOWLEDGE_BASE:
        score = len(query_terms & article["keywords"])
        if score:
            results.append(
                {
                    "article_id": article["article_id"],
                    "title": article["title"],
                    "category": article["category"],
                    "score": score,
                }
            )

    results.sort(key=lambda item: (-item["score"], item["article_id"]))
    return {
        "query": normalized,
        "results": results,
        "source": "offline-fixture",
    }


def check_policy(action: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
    """
    Check an action against a small deterministic enterprise policy rule set.
    """
    normalized_action = action.strip().lower()
    if not normalized_action:
        raise ValueError("action must not be empty.")

    policy_context = context or {}
    if not isinstance(policy_context, dict):
        raise ValueError("context must be an object when provided.")

    if normalized_action == "refund":
        amount = float(policy_context.get("amount", 0))
        allowed = amount <= 100
        reason = "refund_amount_within_limit" if allowed else "refund_amount_exceeds_limit"
    elif normalized_action == "cancel_order":
        status = str(policy_context.get("status", "")).strip().lower()
        allowed = status in {"pending", "processing"}
        reason = "order_can_be_cancelled" if allowed else "order_already_fulfilled"
    elif normalized_action == "view_customer_profile":
        role = str(policy_context.get("role", "")).strip().lower()
        allowed = role in {"support_agent", "manager"}
        reason = "role_authorized" if allowed else "role_not_authorized"
    else:
        allowed = False
        reason = "unknown_action"

    return {
        "action": normalized_action,
        "allowed": allowed,
        "reason": reason,
        "source": "offline-fixture",
    }
