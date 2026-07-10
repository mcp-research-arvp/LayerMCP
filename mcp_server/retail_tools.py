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
    normalized = _normalize(user_id, "user_id").upper()
    users = _state()["users"]
    if normalized not in users:
        raise ValueError("user_id must be one of: " + ", ".join(sorted(users)))
    return users[normalized]


def _get_order(order_id: str) -> dict[str, Any]:
    normalized = _normalize(order_id, "order_id").upper()
    orders = _state()["orders"]
    if normalized not in orders:
        raise ValueError("order_id must be one of: " + ", ".join(sorted(orders)))
    return orders[normalized]


def _get_product(product_id: str) -> dict[str, Any]:
    normalized = _normalize(product_id, "product_id").upper()
    products = _state()["products"]
    if normalized not in products:
        raise ValueError("product_id must be one of: " + ", ".join(sorted(products)))
    return products[normalized]


def _find_variant(item_id: str) -> tuple[str, dict[str, Any]]:
    normalized = _normalize(item_id, "item_id").upper()
    for product_id, product in _state()["products"].items():
        variants = product["variants"]
        if normalized in variants:
            return product_id, variants[normalized]
    raise ValueError(f"item_id not found: {item_id}")


def _get_payment_method(user: dict[str, Any], payment_method_id: str) -> dict[str, Any]:
    normalized = _normalize(payment_method_id, "payment_method_id").upper()
    payment_methods = user["payment_methods"]
    if normalized not in payment_methods:
        raise ValueError("payment_method_id does not belong to the order user.")
    return payment_methods[normalized]


def _is_pending_status(status: str) -> bool:
    return "pending" in status


def _payment_summary(payment_method: dict[str, Any]) -> dict[str, Any]:
    summary = {
        "payment_method_id": payment_method["payment_method_id"],
        "type": payment_method["type"],
    }
    if payment_method["type"] == "credit_card":
        summary["brand"] = payment_method["brand"]
        summary["last_four"] = payment_method["last_four"]
    if payment_method["type"] == "gift_card":
        summary["balance"] = payment_method["balance"]
    return summary


def _order_total(order: dict[str, Any]) -> float:
    return round(sum(item["price"] for item in order["items"]), 2)


def _validate_item_counts(order: dict[str, Any], item_ids: list[str]) -> list[str]:
    if not item_ids:
        raise ValueError("item_ids must not be empty.")
    normalized = [_normalize(item_id, "item_id").upper() for item_id in item_ids]
    order_counts = Counter(item["item_id"] for item in order["items"])
    requested_counts = Counter(normalized)
    for item_id, count in requested_counts.items():
        if count > order_counts[item_id]:
            raise ValueError(f"requested item is not present in the order: {item_id}")
    return normalized


def _replace_one_item(order: dict[str, Any], old_item_id: str, new_variant: dict[str, Any], product_id: str) -> float:
    for index, item in enumerate(order["items"]):
        if item["item_id"] == old_item_id:
            old_price = item["price"]
            order["items"][index] = {
                "name": _state()["products"][product_id]["name"],
                "product_id": product_id,
                "item_id": new_variant["item_id"],
                "price": new_variant["price"],
                "options": deepcopy(new_variant["options"]),
            }
            return round(new_variant["price"] - old_price, 2)
    raise ValueError(f"requested item is not present in the order: {old_item_id}")


def _apply_price_difference(payment_method: dict[str, Any], price_difference: float) -> None:
    if payment_method["type"] != "gift_card":
        return
    if price_difference > payment_method["balance"]:
        raise ValueError("gift card balance is insufficient for the price difference.")
    payment_method["balance"] = round(payment_method["balance"] - price_difference, 2)


def _refund_to_payment_method(payment_method: dict[str, Any], amount: float) -> None:
    if payment_method["type"] == "gift_card":
        payment_method["balance"] = round(payment_method["balance"] + amount, 2)


def find_user_id_by_email(email: str) -> str:
    """
    Find a Retail fixture user ID by email address.
    """
    normalized_email = _normalize(email, "email").lower()
    for user in _state()["users"].values():
        if user["email"].lower() == normalized_email:
            return user["user_id"]
    raise ValueError("User not found")


def find_user_id_by_name_zip(first_name: str, last_name: str, zip: str) -> str:
    """
    Find a Retail fixture user ID by first name, last name, and ZIP code.
    """
    normalized_first = _normalize(first_name, "first_name").lower()
    normalized_last = _normalize(last_name, "last_name").lower()
    normalized_zip = _normalize(zip, "zip")
    for user in _state()["users"].values():
        name = user["name"]
        if (
            name["first_name"].lower() == normalized_first
            and name["last_name"].lower() == normalized_last
            and user["address"]["zip"] == normalized_zip
        ):
            return user["user_id"]
    raise ValueError("User not found")


def get_user_details(user_id: str) -> dict[str, Any]:
    """
    Return Retail fixture user profile details, payment summaries, and order IDs.
    """
    user = _get_user(user_id)
    return {
        "user_id": user["user_id"],
        "name": deepcopy(user["name"]),
        "email": user["email"],
        "address": deepcopy(user["address"]),
        "payment_methods": [
            _payment_summary(payment_method)
            for payment_method in user["payment_methods"].values()
        ],
        "order_ids": list(user["order_ids"]),
        "source": "retail-fixture",
    }


def get_order_details(order_id: str) -> dict[str, Any]:
    """
    Return Retail fixture order status, items, address, payments, and mutation metadata.
    """
    order = _get_order(order_id)
    result = deepcopy(order)
    result["total"] = _order_total(order)
    result["source"] = "retail-fixture"
    return result


def get_product_details(product_id: str) -> dict[str, Any]:
    """
    Return Retail fixture product variants, options, availability, and prices.
    """
    product = _get_product(product_id)
    return {
        "product_id": product["product_id"],
        "name": product["name"],
        "variants": [
            deepcopy(variant)
            for variant in product["variants"].values()
        ],
        "source": "retail-fixture",
    }


def cancel_pending_order(order_id: str, reason: str) -> dict[str, Any]:
    """
    Cancel a pending Retail fixture order and record refunds.
    """
    order = _get_order(order_id)
    normalized_reason = _normalize(reason, "reason").lower()
    if order["status"] != "pending":
        raise ValueError("only pending orders can be cancelled.")
    if normalized_reason not in _CANCEL_REASONS:
        raise ValueError("reason must be one of: no longer needed, ordered by mistake.")

    user = _get_user(order["user_id"])
    refunds = []
    for payment in list(order["payment_history"]):
        if payment["transaction_type"] != "payment":
            continue
        payment_method = _get_payment_method(user, payment["payment_method_id"])
        refund = {
            "transaction_type": "refund",
            "amount": payment["amount"],
            "payment_method_id": payment["payment_method_id"],
        }
        refunds.append(refund)
        _refund_to_payment_method(payment_method, payment["amount"])

    order["status"] = "cancelled"
    order["cancel_reason"] = normalized_reason
    order["payment_history"].extend(refunds)
    return get_order_details(order["order_id"])


def modify_pending_order_items(
    order_id: str,
    item_ids: list[str],
    new_item_ids: list[str],
    payment_method_id: str,
) -> dict[str, Any]:
    """
    Modify items in a pending Retail fixture order to available variants of the same products.
    """
    order = _get_order(order_id)
    if order["status"] != "pending":
        raise ValueError("only pending orders can have items modified.")
    normalized_items = _validate_item_counts(order, item_ids)
    normalized_new_items = [_normalize(item_id, "new_item_id").upper() for item_id in new_item_ids]
    if len(normalized_items) != len(normalized_new_items):
        raise ValueError("item_ids and new_item_ids must have the same length.")

    user = _get_user(order["user_id"])
    payment_method = _get_payment_method(user, payment_method_id)
    working_order = deepcopy(order)
    modifications = []
    total_difference = 0.0

    for old_item_id, new_item_id in zip(normalized_items, normalized_new_items):
        if old_item_id == new_item_id:
            raise ValueError("new_item_id must be different from the old item_id.")
        old_item = next((item for item in working_order["items"] if item["item_id"] == old_item_id), None)
        if old_item is None:
            raise ValueError(f"requested item is not present in the order: {old_item_id}")
        new_product_id, new_variant = _find_variant(new_item_id)
        if new_product_id != old_item["product_id"]:
            raise ValueError("replacement item must belong to the same product.")
        if not new_variant["available"]:
            raise ValueError("replacement item is not available.")
        difference = _replace_one_item(working_order, old_item_id, new_variant, new_product_id)
        total_difference = round(total_difference + difference, 2)
        modifications.append(
            {
                "old_item_id": old_item_id,
                "new_item_id": new_item_id,
                "price_difference": difference,
            }
        )

    _apply_price_difference(payment_method, total_difference)
    order.update(working_order)
    order["status"] = "pending (item modified)"
    order["modified_items"] = modifications
    order["modification_payment_method_id"] = payment_method["payment_method_id"]
    order["modification_price_difference"] = total_difference
    if total_difference:
        order["payment_history"].append(
            {
                "transaction_type": "payment" if total_difference > 0 else "refund",
                "amount": abs(total_difference),
                "payment_method_id": payment_method["payment_method_id"],
            }
        )
    return get_order_details(order["order_id"])


def modify_pending_order_address(
    order_id: str,
    address1: str,
    address2: str,
    city: str,
    state: str,
    country: str,
    zip: str,
) -> dict[str, Any]:
    """
    Modify only the shipping address of a pending Retail fixture order.
    """
    order = _get_order(order_id)
    if not _is_pending_status(order["status"]):
        raise ValueError("only pending orders can have addresses modified.")
    order["address"] = _address(address1, address2, city, state, country, zip)
    order["address_modified"] = True
    return get_order_details(order["order_id"])


def modify_user_address(
    user_id: str,
    address1: str,
    address2: str,
    city: str,
    state: str,
    country: str,
    zip: str,
) -> dict[str, Any]:
    """
    Modify only a Retail fixture user's default profile address.
    """
    user = _get_user(user_id)
    user["address"] = _address(address1, address2, city, state, country, zip)
    return get_user_details(user["user_id"])


def return_delivered_order_items(
    order_id: str,
    item_ids: list[str],
    payment_method_id: str,
) -> dict[str, Any]:
    """
    Request return of items in a delivered Retail fixture order.
    """
    order = _get_order(order_id)
    if order["status"] != "delivered":
        raise ValueError("only delivered orders can be returned.")
    normalized_items = _validate_item_counts(order, item_ids)
    user = _get_user(order["user_id"])
    payment_method = _get_payment_method(user, payment_method_id)
    original_payment_method_id = order["payment_history"][0]["payment_method_id"]
    if payment_method["type"] != "gift_card" and payment_method["payment_method_id"] != original_payment_method_id:
        raise ValueError("payment method should be the original payment method or a gift card.")
    order["status"] = "return requested"
    order["return_items"] = sorted(normalized_items)
    order["return_payment_method_id"] = payment_method["payment_method_id"]
    return get_order_details(order["order_id"])


def exchange_delivered_order_items(
    order_id: str,
    item_ids: list[str],
    new_item_ids: list[str],
    payment_method_id: str,
) -> dict[str, Any]:
    """
    Request exchange of delivered Retail fixture items for available variants of the same products.
    """
    order = _get_order(order_id)
    if order["status"] != "delivered":
        raise ValueError("only delivered orders can be exchanged.")
    normalized_items = _validate_item_counts(order, item_ids)
    normalized_new_items = [_normalize(item_id, "new_item_id").upper() for item_id in new_item_ids]
    if len(normalized_items) != len(normalized_new_items):
        raise ValueError("item_ids and new_item_ids must have the same length.")

    user = _get_user(order["user_id"])
    payment_method = _get_payment_method(user, payment_method_id)
    total_difference = 0.0
    exchanges = []
    for old_item_id, new_item_id in zip(normalized_items, normalized_new_items):
        old_item = next((item for item in order["items"] if item["item_id"] == old_item_id), None)
        if old_item is None:
            raise ValueError(f"requested item is not present in the order: {old_item_id}")
        new_product_id, new_variant = _find_variant(new_item_id)
        if new_product_id != old_item["product_id"]:
            raise ValueError("replacement item must belong to the same product.")
        if not new_variant["available"]:
            raise ValueError("replacement item is not available.")
        difference = round(new_variant["price"] - old_item["price"], 2)
        total_difference = round(total_difference + difference, 2)
        exchanges.append(
            {
                "old_item_id": old_item_id,
                "new_item_id": new_item_id,
                "price_difference": difference,
            }
        )

    if payment_method["type"] == "gift_card" and payment_method["balance"] < total_difference:
        raise ValueError("gift card balance is insufficient for the price difference.")
    order["status"] = "exchange requested"
    order["exchange_items"] = sorted(normalized_items)
    order["exchange_new_items"] = sorted(normalized_new_items)
    order["exchange_payment_method_id"] = payment_method["payment_method_id"]
    order["exchange_price_difference"] = total_difference
    order["exchange_details"] = exchanges
    return get_order_details(order["order_id"])


def transfer_to_human_agents(summary: str) -> dict[str, Any]:
    """
    Return a deterministic Retail human-transfer request response.
    """
    normalized = _normalize(summary, "summary")
    return {
        "transfer_requested": True,
        "summary": normalized,
        "status": "pending_human_agent",
        "source": "retail-fixture",
    }
