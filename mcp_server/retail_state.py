from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any


RETAIL_FIXTURE_PATH = (
    Path(__file__).resolve().parent / "fixtures" / "tau2_retail_db.json"
)


def _load_initial_state() -> dict[str, Any]:
    with RETAIL_FIXTURE_PATH.open("r", encoding="utf-8") as handle:
        state = json.load(handle)
    required = {"products", "users", "orders"}
    if set(state) != required:
        raise ValueError(
            f"Invalid tau2 retail fixture sections: expected {sorted(required)}, "
            f"got {sorted(state)}"
        )
    return state


_INITIAL_RETAIL_STATE: dict[str, Any] = _load_initial_state()
_retail_state: dict[str, Any] = deepcopy(_INITIAL_RETAIL_STATE)


def get_retail_state() -> dict[str, Any]:
    return _retail_state


def reset_retail_state() -> dict[str, Any]:
    _retail_state.clear()
    _retail_state.update(deepcopy(_INITIAL_RETAIL_STATE))
    return _retail_state


def snapshot_retail_state() -> dict[str, Any]:
    return deepcopy(_retail_state)
