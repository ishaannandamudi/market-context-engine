from datetime import UTC, datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from mce.models.context import MarketContextSnapshot
from mce.models.time import datetime_to_ns, ns_to_datetime, ns_to_iso


def test_ns_to_datetime_is_utc_and_keeps_integer_precision() -> None:
    value = 1_700_000_000_123_456_789
    view = ns_to_datetime(value)
    assert view.tzinfo is UTC
    assert view.microsecond == 123_456
    assert ns_to_iso(value).endswith(".123456789Z")


def test_datetime_to_ns_uses_aware_offset_without_float_rounding() -> None:
    instant = datetime(2024, 1, 2, 10, 0, 0, 123456, tzinfo=timezone(timedelta(hours=-5)))
    assert datetime_to_ns(instant) == datetime_to_ns(instant.astimezone(UTC))
    assert datetime_to_ns(instant) % 1_000 == 0


def test_epoch_and_boundaries() -> None:
    assert datetime_to_ns(datetime(1970, 1, 1, tzinfo=UTC)) == 0
    assert ns_to_iso(0) == "1970-01-01T00:00:00.000000000Z"
    with pytest.raises(ValueError):
        ns_to_datetime(-1)
    with pytest.raises(ValueError):
        datetime_to_ns(datetime(1969, 12, 31, tzinfo=UTC))
    with pytest.raises(ValidationError):
        MarketContextSnapshot(as_of_ns=-1)


def test_naive_datetime_rejected() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        datetime_to_ns(datetime(2024, 1, 1))


def test_as_of_iso_is_checked_against_nanoseconds() -> None:
    with pytest.raises(ValidationError, match="does not match"):
        MarketContextSnapshot(as_of_ns=1, as_of_iso="1970-01-01T00:00:00.000000000Z")
