from __future__ import annotations

from collections import Counter
from copy import deepcopy
from typing import Any

from mcp_server.retail_state import get_retail_state


_CANCEL_REASONS = {"no longer needed", "ordered by mistake"}


def _normalize(value: str, field_name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be empty.")
    return normalized


def _address(
    address1: str,
    address2: str,
    city: str,
    state: str,
    country: str,
    zip: str,
) -> dict[str, str]:
    return {
        "address1": _normalize(address1, "address1"),
        "address2": address2.strip(),
        "city": _normalize(city, "city"),
        "state": _normalize(state, "state"),
        "country": _normalize(country, "country"),
        "zip": _normalize(zip, "zip"),
    }


def _state() -> dict[str, Any]:
    return get_retail_state()


def _get_user(user_id: str) -> dict[str, Any]:
    user_id = _normalize(user_id, "user_id")
    try:
        return _state()["users"][user_id]
    except KeyError as exc:
        raise ValueError(f"User {user_id} not found") from exc


def _get_order(order_id: str) -> dict[str, Any]:
    order_id = _normalize(order_id, "order_id")
    if order_id.startswith("W") and order_id[1:].isdigit():
        order_id = f"#{order_id}"
    try:
        return _state()["orders"][order_id]
    except KeyError as exc:
        raise ValueError(f"Order {order_id} not found") from exc


def _get_product(product_id: str) -> dict[str, Any]:
    product_id = _normalize(product_id, "product_id")
    try:
        return _state()["products"][product_id]
    except KeyError as exc:
        raise ValueError(f"Product {product_id} not found") from exc


def _find_variant(item_id: str) -> tuple[str, dict[str, Any]]:
    item_id = _normalize(item_id, "item_id")
    for product_id, product in _state()["products"].items():
        if item_id in product["variants"]:
            return product_id, product["variants"][item_id]
    raise ValueError(f"Item {item_id} not found")


def _get_payment_method(
    user: dict[str, Any], payment_method_id: str
) -> dict[str, Any]:
    payment_method_id = _normalize(payment_method_id, "payment_method_id")
    try:
        return user["payment_methods"][payment_method_id]
    except KeyError as exc:
        raise ValueError(
            f"Payment method {payment_method_id} does not belong to the order user"
        ) from exc


def _validate_item_counts(order: dict[str, Any], item_ids: list[str]) -> list[str]:
    if not item_ids:
        raise ValueError("item_ids must not be empty.")
    normalized = [_normalize(item_id, "item_id") for item_id in item_ids]
    order_counts = Counter(item["item_id"] for item in order["items"])
    requested_counts = Counter(normalized)
    for item_id, count in requested_counts.items():
        if count > order_counts[item_id]:
            raise ValueError(f"Number of {item_id} not found")
    return normalized


def _is_gift_card(payment_method: dict[str, Any]) -> bool:
    return payment_method["source"] == "gift_card"


def find_user_id_by_email(email: str) -> str:
    """Find a tau2 retail user ID by email address."""
    email = _normalize(email, "email")
    for user_id, user in _state()["users"].items():
        if user["email"].lower() == email.lower():
            return user_id
    raise ValueError("User not found")


def find_user_id_by_name_zip(first_name: str, last_name: str, zip: str) -> str:
    """Find a tau2 retail user ID by first name, last name, and ZIP code."""
    first_name = _normalize(first_name, "first_name")
    last_name = _normalize(last_name, "last_name")
    zip = _normalize(zip, "zip")
    for user_id, user in _state()["users"].items():
        if (
            user["name"]["first_name"].lower() == first_name.lower()
            and user["name"]["last_name"].lower() == last_name.lower()
            and user["address"]["zip"] == zip
        ):
            return user_id
    raise ValueError("User not found")


def get_user_details(user_id: str) -> dict[str, Any]:
    """Return a tau2 retail user, including payment methods and order IDs."""
    return deepcopy(_get_user(user_id))


def get_order_details(order_id: str) -> dict[str, Any]:
    """Return the native tau2 retail order representation."""
    return deepcopy(_get_order(order_id))


def get_product_details(product_id: str) -> dict[str, Any]:
    """Return the native tau2 retail product and keyed variants."""
    return deepcopy(_get_product(product_id))


def cancel_pending_order(order_id: str, reason: str) -> dict[str, Any]:
    """Cancel a pending tau2 retail order and record its refunds."""
    order = _get_order(order_id)
    reason = _normalize(reason, "reason")
    if order["status"] != "pending":
        raise ValueError("Non-pending order cannot be cancelled")
    if reason not in _CANCEL_REASONS:
        raise ValueError("Invalid reason")

    user = _get_user(order["user_id"])
    refunds = []
    for payment in order["payment_history"]:
        refund = {
            "transaction_type": "refund",
            "amount": payment["amount"],
            "payment_method_id": payment["payment_method_id"],
        }
        refunds.append(refund)
        payment_method = _get_payment_method(user, payment["payment_method_id"])
        if _is_gift_card(payment_method):
            payment_method["balance"] = round(
                payment_method["balance"] + payment["amount"], 2
            )

    order["status"] = "cancelled"
    order["cancel_reason"] = reason
    order["payment_history"].extend(refunds)
    return deepcopy(order)


def modify_pending_order_items(
    order_id: str,
    item_ids: list[str],
    new_item_ids: list[str],
    payment_method_id: str,
) -> dict[str, Any]:
    """Replace pending-order items with available variants of the same products."""
    order = _get_order(order_id)
    if order["status"] != "pending":
        raise ValueError("Non-pending order cannot be modified")
    item_ids = _validate_item_counts(order, item_ids)
    new_item_ids = [_normalize(item_id, "new_item_id") for item_id in new_item_ids]
    if len(item_ids) != len(new_item_ids):
        raise ValueError("The number of items to be exchanged should match")

    replacements: list[tuple[dict[str, Any], dict[str, Any]]] = []
    price_difference = 0.0
    for item_id, new_item_id in zip(item_ids, new_item_ids):
        if item_id == new_item_id:
            raise ValueError("The new item id should be different from the old item id")
        old_item = next(
            (item for item in order["items"] if item["item_id"] == item_id), None
        )
        if old_item is None:
            raise ValueError(f"Item {item_id} not found")
        product = _get_product(old_item["product_id"])
        try:
            new_variant = product["variants"][new_item_id]
        except KeyError as exc:
            raise ValueError(
                f"Item {new_item_id} is not a variant of {old_item['product_id']}"
            ) from exc
        if not new_variant["available"]:
            raise ValueError(f"New item {new_item_id} not found or available")
        price_difference += new_variant["price"] - old_item["price"]
        replacements.append((old_item, new_variant))

    price_difference = round(price_difference, 2)
    user = _get_user(order["user_id"])
    payment_method = _get_payment_method(user, payment_method_id)
    if _is_gift_card(payment_method) and payment_method["balance"] < price_difference:
        raise ValueError("Insufficient gift card balance to pay for the new item")

    order["payment_history"].append(
        {
            "transaction_type": "payment" if price_difference > 0 else "refund",
            "amount": abs(price_difference),
            "payment_method_id": payment_method_id,
        }
    )
    if _is_gift_card(payment_method):
        payment_method["balance"] = round(
            payment_method["balance"] - price_difference, 2
        )
    for old_item, new_variant in replacements:
        old_item["item_id"] = new_variant["item_id"]
        old_item["price"] = new_variant["price"]
        old_item["options"] = deepcopy(new_variant["options"])
    order["status"] = "pending (item modified)"
    return deepcopy(order)


def modify_pending_order_address(
    order_id: str,
    address1: str,
    address2: str,
    city: str,
    state: str,
    country: str,
    zip: str,
) -> dict[str, Any]:
    """Modify the shipping address of a pending tau2 retail order."""
    order = _get_order(order_id)
    if "pending" not in order["status"]:
        raise ValueError("Non-pending order cannot be modified")
    order["address"] = _address(
        address1, address2, city, state, country, zip
    )
    return deepcopy(order)


def modify_user_address(
    user_id: str,
    address1: str,
    address2: str,
    city: str,
    state: str,
    country: str,
    zip: str,
) -> dict[str, Any]:
    """Modify a tau2 retail user's default address."""
    user = _get_user(user_id)
    user["address"] = _address(
        address1, address2, city, state, country, zip
    )
    return deepcopy(user)


def return_delivered_order_items(
    order_id: str,
    item_ids: list[str],
    payment_method_id: str,
) -> dict[str, Any]:
    """Request a return of items from a delivered tau2 retail order."""
    order = _get_order(order_id)
    if order["status"] != "delivered":
        raise ValueError("Non-delivered order cannot be returned")
    item_ids = _validate_item_counts(order, item_ids)
    user = _get_user(order["user_id"])
    payment_method = _get_payment_method(user, payment_method_id)
    if (
        not _is_gift_card(payment_method)
        and payment_method_id != order["payment_history"][0]["payment_method_id"]
    ):
        raise ValueError("Payment method should be the original payment method")
    order["status"] = "return requested"
    order["return_items"] = sorted(item_ids)
    order["return_payment_method_id"] = payment_method_id
    return deepcopy(order)


def exchange_delivered_order_items(
    order_id: str,
    item_ids: list[str],
    new_item_ids: list[str],
    payment_method_id: str,
) -> dict[str, Any]:
    """Request an exchange for delivered tau2 retail items."""
    order = _get_order(order_id)
    if order["status"] != "delivered":
        raise ValueError("Non-delivered order cannot be exchanged")
    item_ids = _validate_item_counts(order, item_ids)
    new_item_ids = [_normalize(item_id, "new_item_id") for item_id in new_item_ids]
    if len(item_ids) != len(new_item_ids):
        raise ValueError("The number of items to be exchanged should match")

    price_difference = 0.0
    for item_id, new_item_id in zip(item_ids, new_item_ids):
        old_item = next(
            (item for item in order["items"] if item["item_id"] == item_id), None
        )
        if old_item is None:
            raise ValueError(f"Item {item_id} not found")
        product = _get_product(old_item["product_id"])
        try:
            new_variant = product["variants"][new_item_id]
        except KeyError as exc:
            raise ValueError(
                f"Item {new_item_id} is not a variant of {old_item['product_id']}"
            ) from exc
        if not new_variant["available"]:
            raise ValueError(f"New item {new_item_id} not found or available")
        price_difference += new_variant["price"] - old_item["price"]
    price_difference = round(price_difference, 2)

    user = _get_user(order["user_id"])
    payment_method = _get_payment_method(user, payment_method_id)
    if _is_gift_card(payment_method) and payment_method["balance"] < price_difference:
        raise ValueError(
            "Insufficient gift card balance to pay for the price difference"
        )
    order["status"] = "exchange requested"
    order["exchange_items"] = sorted(item_ids)
    order["exchange_new_items"] = sorted(new_item_ids)
    order["exchange_payment_method_id"] = payment_method_id
    order["exchange_price_difference"] = price_difference
    return deepcopy(order)


def transfer_to_human_agents(summary: str) -> str:
    """Transfer the retail request to a human agent."""
    _normalize(summary, "summary")
    return "Transfer successful"
