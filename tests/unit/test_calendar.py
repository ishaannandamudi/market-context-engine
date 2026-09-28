from datetime import date

import pytest

from mce.calendar import MarketSessionService
from mce.errors import CalendarRangeError
from mce.models.enums import MarketSession
from mce.serialization import canonical_json_str
from tests.fixtures.clock import utc_ns


@pytest.mark.parametrize(
    "instant,expected",
    [
        ("2024-01-03T08:59:59Z", MarketSession.CLOSED),
        ("2024-01-03T09:00:00Z", MarketSession.PRE_MARKET),
        ("2024-01-03T14:29:59Z", MarketSession.PRE_MARKET),
        ("2024-01-03T14:30:00Z", MarketSession.REGULAR),
        ("2024-01-03T17:00:00Z", MarketSession.REGULAR),
        ("2024-01-03T21:00:00Z", MarketSession.AFTER_HOURS),
        ("2024-01-04T01:00:00Z", MarketSession.CLOSED),
        ("2024-01-06T17:00:00Z", MarketSession.CLOSED),
        ("2024-07-04T17:00:00Z", MarketSession.CLOSED),
    ],
)
def test_session_boundaries(
    calendar: MarketSessionService, instant: str, expected: MarketSession
) -> None:
    assert calendar.session_for_timestamp(utc_ns(instant)).session == expected


def test_winter_summer_and_dst_transition(calendar: MarketSessionService) -> None:
    assert calendar.regular_session_bounds(date(2024, 1, 3))[0] == utc_ns("2024-01-03T14:30:00Z")
    assert calendar.regular_session_bounds(date(2024, 7, 2))[0] == utc_ns("2024-07-02T13:30:00Z")
    assert calendar.regular_session_bounds(date(2024, 3, 8))[0] == utc_ns("2024-03-08T14:30:00Z")
    assert calendar.regular_session_bounds(date(2024, 3, 11))[0] == utc_ns("2024-03-11T13:30:00Z")
    assert calendar.session_for_timestamp(utc_ns("2024-03-10T14:30:00Z")).session == (
        MarketSession.CLOSED
    )


def test_holiday_early_close_and_exchange_vs_federal_holiday(
    calendar: MarketSessionService,
) -> None:
    assert not calendar.is_trading_day(date(2024, 7, 4))
    assert calendar.is_trading_day(date(2024, 10, 14))  # Columbus Day is not an XNYS close.
    open_ns, close_ns = calendar.regular_session_bounds(date(2024, 11, 29))
    assert open_ns == utc_ns("2024-11-29T14:30:00Z")
    assert close_ns == utc_ns("2024-11-29T18:00:00Z")
    before = calendar.session_for_timestamp(close_ns - 1)
    after = calendar.session_for_timestamp(close_ns)
    assert before.session == MarketSession.REGULAR
    assert after.session == MarketSession.AFTER_HOURS
    assert before.is_early_close and after.is_early_close


def test_calendar_day_navigation_and_utc_midnight(calendar: MarketSessionService) -> None:
    assert calendar.previous_trading_day(date(2024, 7, 8)) == date(2024, 7, 5)
    assert calendar.next_trading_day(date(2024, 7, 3)) == date(2024, 7, 5)
    assert calendar.previous_trading_day(date(2024, 7, 4)) == date(2024, 7, 3)
    assert calendar.next_trading_day(date(2024, 7, 4)) == date(2024, 7, 5)
    assert calendar.trading_date_for_timestamp(utc_ns("2024-01-03T00:00:00Z")) == date(2024, 1, 2)
    assert calendar.trading_date_for_timestamp(utc_ns("2024-07-04T17:00:00Z")) is None


def test_calendar_unknown_and_invalid_dates(calendar: MarketSessionService) -> None:
    info = calendar.session_for_timestamp(utc_ns("2040-01-03T17:00:00Z"))
    assert info.session == MarketSession.UNKNOWN
    with pytest.raises(CalendarRangeError):
        calendar.is_trading_day(date(2040, 1, 3))
    with pytest.raises(ValueError):
        calendar.regular_session_bounds(date(2024, 7, 4))
    with pytest.raises(CalendarRangeError):
        calendar.previous_trading_day(date(2000, 1, 3))
    with pytest.raises(CalendarRangeError):
        calendar.next_trading_day(date(2035, 12, 31))


def test_snapshot_completion_times_and_serialization(calendar: MarketSessionService) -> None:
    times = calendar.snapshot_times(date(2024, 11, 29), 5 * 60_000_000_000)
    assert times[0] == utc_ns("2024-11-29T14:35:00Z")
    assert times[-1] == utc_ns("2024-11-29T18:00:00Z")
    assert len(times) == 42
    info = calendar.session_for_timestamp(times[0])
    assert '"calendar_name":"XNYS"' in canonical_json_str(info)
    assert type(info).model_validate_json(canonical_json_str(info)) == info
    with pytest.raises(ValueError):
        calendar.snapshot_times(date(2024, 11, 29), 0)
