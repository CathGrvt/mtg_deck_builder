from __future__ import annotations

import os
from typing import Any, Optional


TRUTHY_VALUES = {"1", "true", "yes", "on"}
FALSY_VALUES = {"0", "false", "no", "off"}


def parse_bool_value(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    lowered = str(value).strip().lower()
    if not lowered:
        return default
    if lowered in TRUTHY_VALUES:
        return True
    if lowered in FALSY_VALUES:
        return False
    return default


def parse_bool_env(name: str, default: bool = False) -> bool:
    return parse_bool_value(os.getenv(name), default=default)


def parse_int_value(
    value: Any,
    default: int,
    *,
    minimum: Optional[int] = None,
    maximum: Optional[int] = None,
) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = int(default)

    if minimum is not None:
        parsed = max(int(minimum), parsed)
    if maximum is not None:
        parsed = min(int(maximum), parsed)
    return parsed


def parse_str_env(name: str, default: str = "", strip: bool = True) -> str:
    value = os.getenv(name, default)
    if strip:
        return str(value).strip()
    return str(value)
