"""XNYS-backed session classification with explicit UTC nanosecond boundaries."""

from datetime import date
from zoneinfo import ZoneInfo

import exchange_calendars as xcals  # type: ignore[import-untyped]

from mce.config.settings import MarketSettings
from mce.errors import CalendarRangeError
from mce.models._base import FrozenModel
from mce.models.enums import MarketSession
from mce.models.time import TimestampNs, ns_to_datetime


class MarketSessionInfo(FrozenModel):
    timestamp_ns: TimestampNs
    trading_date: date | None
    session: MarketSession
    regular_open_ns: TimestampNs | None
    regular_close_ns: TimestampNs | None
    is_trading_day: bool
    is_early_close: bool
    calendar_name: str


class MarketSessionService:
    """Classify by New York local date and the exchange's actual regular schedule.

    Extended hours are configurable local clock conventions, applied only on
    trading dates. Regular [open, close); pre/post intervals are also half-open.
    """

    def __init__(self, settings: MarketSettings | None = None) -> None:
        self.settings = settings or MarketSettings()
        self._zone = ZoneInfo(self.settings.exchange_timezone)
        self._calendar = xcals.get_calendar(
            self.settings.calendar_name,
            start=self.settings.calendar_start.isoformat(),
            end=self.settings.calendar_end.isoformat(),
            side="left",
        )
        self._early_close_dates = frozenset(item.date() for item in self._calendar.early_closes)

    def _in_range(self, day: date) -> bool:
        return self.settings.calendar_start <= day <= self.settings.calendar_end

    def _require_range(self, day: date) -> None:
        if not self._in_range(day):
            raise CalendarRangeError(f"{day} is outside configured calendar range")

    def is_trading_day(self, day: date) -> bool:
        self._require_range(day)
        return bool(self._calendar.is_session(day.isoformat()))

    def regular_session_bounds(self, day: date) -> tuple[int, int]:
        self._require_range(day)
        if not self.is_trading_day(day):
            raise ValueError(f"{day} is not a trading day")
        label = day.isoformat()
        return (
            int(self._calendar.session_open(label).value),
            int(self._calendar.session_close(label).value),
        )

    def previous_trading_day(self, day: date) -> date:
        self._require_range(day)
        if day <= self._calendar.first_session.date():
            raise CalendarRangeError("no prior session in configured range")
        label = self._calendar.date_to_session(day.isoformat(), direction="previous")
        if label.date() == day:
            label = self._calendar.previous_session(label)
        return date.fromisoformat(str(label.date()))

    def next_trading_day(self, day: date) -> date:
        self._require_range(day)
        if day >= self._calendar.last_session.date():
            raise CalendarRangeError("no next session in configured range")
        label = self._calendar.date_to_session(day.isoformat(), direction="next")
        if label.date() == day:
            label = self._calendar.next_session(label)
        return date.fromisoformat(str(label.date()))

    def session_for_timestamp(self, as_of_ns: int) -> MarketSessionInfo:
        local = ns_to_datetime(as_of_ns).astimezone(self._zone)
        day = local.date()
        if not self._in_range(day):
            return MarketSessionInfo(
                timestamp_ns=as_of_ns,
                trading_date=None,
                session=MarketSession.UNKNOWN,
                regular_open_ns=None,
                regular_close_ns=None,
                is_trading_day=False,
                is_early_close=False,
                calendar_name=self.settings.calendar_name,
            )
        if not self.is_trading_day(day):
            return MarketSessionInfo(
                timestamp_ns=as_of_ns,
                trading_date=None,
                session=MarketSession.CLOSED,
                regular_open_ns=None,
                regular_close_ns=None,
                is_trading_day=False,
                is_early_close=False,
                calendar_name=self.settings.calendar_name,
            )
        regular_open_ns, regular_close_ns = self.regular_session_bounds(day)
        local_time = local.time()
        if regular_open_ns <= as_of_ns < regular_close_ns:
            session = MarketSession.REGULAR
        elif local_time >= self.settings.pre_market_start_local and as_of_ns < regular_open_ns:
            session = MarketSession.PRE_MARKET
        elif as_of_ns >= regular_close_ns and local_time < self.settings.after_hours_end_local:
            session = MarketSession.AFTER_HOURS
        else:
            session = MarketSession.CLOSED
        return MarketSessionInfo(
            timestamp_ns=as_of_ns,
            trading_date=day,
            session=session,
            regular_open_ns=regular_open_ns,
            regular_close_ns=regular_close_ns,
            is_trading_day=True,
            is_early_close=day in self._early_close_dates,
            calendar_name=self.settings.calendar_name,
        )

    def trading_date_for_timestamp(self, as_of_ns: int) -> date | None:
        return self.session_for_timestamp(as_of_ns).trading_date

    def snapshot_times(self, day: date, cadence_ns: int) -> tuple[int, ...]:
        """Regular-session completion instants: open+cadence through close inclusive."""
        if cadence_ns <= 0:
            raise ValueError("cadence_ns must be positive")
        regular_open_ns, regular_close_ns = self.regular_session_bounds(day)
        return tuple(range(regular_open_ns + cadence_ns, regular_close_ns + 1, cadence_ns))
