"""Canonical JSON for snapshots and reproducibility hashes."""

import json
from datetime import date, datetime, time
from enum import Enum
from math import isfinite
from typing import Any

from pydantic import BaseModel

from mce.models.time import datetime_to_ns, ns_to_iso


def _jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return _jsonable(value.model_dump(mode="python"))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return ns_to_iso(datetime_to_ns(value))
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, time):
        if value.tzinfo is not None:
            raise ValueError("canonical local time must be timezone-naive")
        return value.isoformat()
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("canonical JSON object keys must be strings")
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (set, frozenset)):
        normalized = [_jsonable(item) for item in value]
        return sorted(normalized, key=lambda item: json.dumps(item, sort_keys=True))
    if isinstance(value, float) and not isfinite(value):
        raise ValueError("canonical JSON cannot contain NaN or Infinity")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unsupported canonical JSON value: {type(value).__name__}")


def canonical_json_str(value: Any) -> str:
    """Serialize to compact, UTF-8-compatible JSON with sorted object keys."""
    return json.dumps(
        _jsonable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def canonical_json_bytes(value: Any) -> bytes:
    return canonical_json_str(value).encode("utf-8")
