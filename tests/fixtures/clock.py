"""Explicit UTC test timestamps, never dependent on machine-local time."""

from datetime import UTC, datetime

from mce.models.time import datetime_to_ns


def utc_ns(value: str) -> int:
    instant = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if instant.tzinfo is None:
        raise ValueError("test timestamp must be timezone-aware")
    return datetime_to_ns(instant.astimezone(UTC))
