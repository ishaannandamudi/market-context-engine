"""UTC nanosecond time semantics, independent of machine timezone."""

from datetime import UTC, datetime, timedelta
from typing import Annotated

from pydantic import Field

TimestampNs = Annotated[int, Field(ge=0, strict=True)]

_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
_NS_PER_SECOND = 1_000_000_000


def ns_to_datetime(timestamp_ns: int) -> datetime:
    """Convert to UTC datetime, truncating sub-microsecond precision only in the view."""
    if timestamp_ns < 0:
        raise ValueError("timestamp_ns must be nonnegative")
    seconds, nanoseconds = divmod(timestamp_ns, _NS_PER_SECOND)
    return _EPOCH + timedelta(seconds=seconds, microseconds=nanoseconds // 1_000)


def datetime_to_ns(value: datetime) -> int:
    """Convert an aware instant exactly to nanoseconds (datetime has microsecond precision)."""
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("datetime must be timezone-aware")
    delta = value.astimezone(UTC) - _EPOCH
    result = (delta.days * 86_400 + delta.seconds) * _NS_PER_SECOND
    result += delta.microseconds * 1_000
    if result < 0:
        raise ValueError("datetime must be at or after Unix epoch")
    return result


def ns_to_iso(timestamp_ns: int) -> str:
    """Return a deterministic ISO-8601 UTC string retaining all nine fractional digits."""
    value = ns_to_datetime(timestamp_ns)
    return f"{value:%Y-%m-%dT%H:%M:%S}.{timestamp_ns % _NS_PER_SECOND:09d}Z"
