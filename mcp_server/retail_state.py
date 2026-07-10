from __future__ import annotations

from copy import deepcopy
from typing import Any


def _address(
    address1: str,
    address2: str,
    city: str,
    state: str,
    country: str,
    zip: str,
) -> dict[str, str]:
    return {
        "address1": address1,
        "address2": address2,
        "city": city,
        "state": state,
        "country": country,
        "zip": zip,
    }


_INITIAL_RETAIL_STATE: dict[str, Any] = {
    "products": {
        "PROD-KEYBOARD": {
            "product_id": "PROD-KEYBOARD",
            "name": "Mechanical Keyboard",
            "variants": {
                "ITEM-KB-LINEAR": {
                    "item_id": "ITEM-KB-LINEAR",
                    "options": {"switch": "linear", "color": "black"},
                    "available": True,
                    "price": 120.0,
                },
                "ITEM-KB-CLICKY": {
                    "item_id": "ITEM-KB-CLICKY",
                    "options": {"switch": "clicky", "color": "black"},
                    "available": True,
                    "price": 135.0,
                },
                "ITEM-KB-SILENT": {
                    "item_id": "ITEM-KB-SILENT",
                    "options": {"switch": "silent", "color": "white"},
                    "available": False,
                    "price": 150.0,
                },
            },
        },
        "PROD-BOTTLE": {
            "product_id": "PROD-BOTTLE",
            "name": "Water Bottle",
            "variants": {
                "ITEM-BOTTLE-500": {
                    "item_id": "ITEM-BOTTLE-500",
                    "options": {"capacity": "500ml", "material": "steel"},
                    "available": True,
                    "price": 20.0,
                },
                "ITEM-BOTTLE-1000": {
                    "item_id": "ITEM-BOTTLE-1000",
                    "options": {"capacity": "1000ml", "material": "steel"},
                    "available": True,
                    "price": 32.0,
                },
                "ITEM-BOTTLE-GLASS": {
                    "item_id": "ITEM-BOTTLE-GLASS",
                    "options": {"capacity": "750ml", "material": "glass"},
                    "available": False,
                    "price": 28.0,
                },
            },
        },
        "PROD-LAMP": {
            "product_id": "PROD-LAMP",
            "name": "Desk Lamp",
            "variants": {
                "ITEM-LAMP-USB": {
                    "item_id": "ITEM-LAMP-USB",
                    "options": {"power": "USB", "brightness": "standard"},
                    "available": True,
                    "price": 45.0,
                },
                "ITEM-LAMP-BATTERY": {
                    "item_id": "ITEM-LAMP-BATTERY",
                    "options": {"power": "battery", "brightness": "bright"},
                    "available": True,
                    "price": 55.0,
                },
            },
        },
    },
    "users": {
        "USER-YUSUF": {
            "user_id": "USER-YUSUF",
            "name": {"first_name": "Yusuf", "last_name": "Rossi"},
            "email": "yusuf.rossi@example.com",
            "address": _address("123 Pine Street", "Apt 4", "Philadelphia", "PA", "US", "19122"),
            "payment_methods": {
                "CARD-YUSUF": {
                    "payment_method_id": "CARD-YUSUF",
                    "type": "credit_card",
                    "brand": "visa",
                    "last_four": "4242",
                },
                "GIFT-YUSUF": {
                    "payment_method_id": "GIFT-YUSUF",
                    "type": "gift_card",
                    "balance": 200.0,
                },
                "PAYPAL-YUSUF": {
                    "payment_method_id": "PAYPAL-YUSUF",
                    "type": "paypal",
                    "email": "yusuf.paypal@example.com",
                },
            },
            "order_ids": ["RET-1001", "RET-1002", "RET-1004"],
        },
        "USER-MEI": {
            "user_id": "USER-MEI",
            "name": {"first_name": "Mei", "last_name": "Kovacs"},
            "email": "mei.kovacs@example.com",
            "address": _address("88 Lake Road", "", "Charlotte", "NC", "US", "28236"),
            "payment_methods": {
                "CARD-MEI": {
                    "payment_method_id": "CARD-MEI",
                    "type": "credit_card",
                    "brand": "mastercard",
                    "last_four": "1111",
                },
                "GIFT-MEI": {
                    "payment_method_id": "GIFT-MEI",
                    "type": "gift_card",
                    "balance": 25.0,
                },
                "PAYPAL-MEI": {
                    "payment_method_id": "PAYPAL-MEI",
                    "type": "paypal",
                    "email": "mei.paypal@example.com",
                },
            },
            "order_ids": ["RET-2001", "RET-2002"],
        },
    },
    "orders": {
        "RET-1001": {
            "order_id": "RET-1001",
            "user_id": "USER-YUSUF",
            "address": _address("123 Pine Street", "Apt 4", "Philadelphia", "PA", "US", "19122"),
            "items": [
                {
                    "name": "Mechanical Keyboard",
                    "product_id": "PROD-KEYBOARD",
                    "item_id": "ITEM-KB-LINEAR",
                    "price": 120.0,
                    "options": {"switch": "linear", "color": "black"},
                },
                {
                    "name": "Water Bottle",
                    "product_id": "PROD-BOTTLE",
                    "item_id": "ITEM-BOTTLE-500",
                    "price": 20.0,
                    "options": {"capacity": "500ml", "material": "steel"},
                },
            ],
            "status": "pending",
            "payment_history": [
                {"transaction_type": "payment", "amount": 140.0, "payment_method_id": "CARD-YUSUF"}
            ],
        },
        "RET-1002": {
            "order_id": "RET-1002",
            "user_id": "USER-YUSUF",
            "address": _address("500 Market Street", "Suite 9", "Philadelphia", "PA", "US", "19123"),
            "items": [
                {
                    "name": "Desk Lamp",
                    "product_id": "PROD-LAMP",
                    "item_id": "ITEM-LAMP-USB",
                    "price": 45.0,
                    "options": {"power": "USB", "brightness": "standard"},
                }
            ],
            "status": "pending",
            "payment_history": [
                {"transaction_type": "payment", "amount": 45.0, "payment_method_id": "GIFT-YUSUF"}
            ],
        },
        "RET-1004": {
            "order_id": "RET-1004",
            "user_id": "USER-YUSUF",
            "address": _address("123 Pine Street", "Apt 4", "Philadelphia", "PA", "US", "19122"),
            "items": [
                {
                    "name": "Mechanical Keyboard",
                    "product_id": "PROD-KEYBOARD",
                    "item_id": "ITEM-KB-LINEAR",
                    "price": 120.0,
                    "options": {"switch": "linear", "color": "black"},
                }
            ],
            "status": "processed",
            "payment_history": [
                {"transaction_type": "payment", "amount": 120.0, "payment_method_id": "PAYPAL-YUSUF"}
            ],
        },
        "RET-2001": {
            "order_id": "RET-2001",
            "user_id": "USER-MEI",
            "address": _address("88 Lake Road", "", "Charlotte", "NC", "US", "28236"),
            "items": [
                {
                    "name": "Mechanical Keyboard",
                    "product_id": "PROD-KEYBOARD",
                    "item_id": "ITEM-KB-LINEAR",
                    "price": 120.0,
                    "options": {"switch": "linear", "color": "black"},
                },
                {
                    "name": "Water Bottle",
                    "product_id": "PROD-BOTTLE",
                    "item_id": "ITEM-BOTTLE-500",
                    "price": 20.0,
                    "options": {"capacity": "500ml", "material": "steel"},
                },
            ],
            "status": "delivered",
            "payment_history": [
                {"transaction_type": "payment", "amount": 140.0, "payment_method_id": "CARD-MEI"}
            ],
        },
        "RET-2002": {
            "order_id": "RET-2002",
            "user_id": "USER-MEI",
            "address": _address("88 Lake Road", "", "Charlotte", "NC", "US", "28236"),
            "items": [
                {
                    "name": "Desk Lamp",
                    "product_id": "PROD-LAMP",
                    "item_id": "ITEM-LAMP-USB",
                    "price": 45.0,
                    "options": {"power": "USB", "brightness": "standard"},
                }
            ],
            "status": "cancelled",
            "payment_history": [
                {"transaction_type": "payment", "amount": 45.0, "payment_method_id": "PAYPAL-MEI"}
            ],
            "cancel_reason": "ordered by mistake",
        },
    },
}

_retail_state: dict[str, Any] = deepcopy(_INITIAL_RETAIL_STATE)


def get_retail_state() -> dict[str, Any]:
    return _retail_state


def reset_retail_state() -> dict[str, Any]:
    _retail_state.clear()
    _retail_state.update(deepcopy(_INITIAL_RETAIL_STATE))
    return _retail_state


def snapshot_retail_state() -> dict[str, Any]:
    return deepcopy(_retail_state)
