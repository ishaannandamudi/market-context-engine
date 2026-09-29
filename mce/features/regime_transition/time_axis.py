"""XNYS regular-session time coordinates, including holidays and early closes."""

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from datetime import date, timedelta
from zoneinfo import ZoneInfo

from mce.calendar.service import MarketSessionService
from mce.errors import CalendarRangeError
from mce.models.time import ns_to_datetime


def local_date(timestamp_ns: int, calendar: MarketSessionService) -> date:
    return (
        ns_to_datetime(timestamp_ns)
        .astimezone(ZoneInfo(calendar.settings.exchange_timezone))
        .date()
    )


def window_start(calendar: MarketSessionService, as_of_ns: int, sessions: int) -> int:
    info = calendar.session_for_timestamp(as_of_ns)
    day = local_date(as_of_ns, calendar)
    if not info.is_trading_day or info.regular_open_ns is None or as_of_ns < info.regular_open_ns:
        day = calendar.previous_trading_day(day)
    for _ in range(sessions - 1):
        try:
            day = calendar.previous_trading_day(day)
        except CalendarRangeError:
            break
    return calendar.regular_session_bounds(day)[0]


@dataclass(frozen=True, slots=True)
class SessionAxis:
    dates: tuple[date, ...]
    opens: tuple[int, ...]
    closes: tuple[int, ...]
    cumulative_ns: tuple[int, ...]

    @classmethod
    def build(cls, calendar: MarketSessionService, start_ns: int, end_ns: int) -> "SessionAxis":
        first = local_date(start_ns, calendar)
        last = local_date(end_ns, calendar)
        dates: list[date] = []
        opens: list[int] = []
        closes: list[int] = []
        cumulative = [0]
        day = first
        while day <= last:
            if calendar.is_trading_day(day):
                opened, closed = calendar.regular_session_bounds(day)
                dates.append(day)
                opens.append(opened)
                closes.append(closed)
                cumulative.append(cumulative[-1] + closed - opened)
            day += timedelta(days=1)
        return cls(tuple(dates), tuple(opens), tuple(closes), tuple(cumulative))

    def trading_coordinate(self, timestamp_ns: int) -> int:
        index = bisect_right(self.opens, timestamp_ns) - 1
        if index < 0:
            return 0
        return self.cumulative_ns[index] + min(
            self.closes[index] - self.opens[index],
            max(0, timestamp_ns - self.opens[index]),
        )

    def session_coordinate(self, timestamp_ns: int) -> float:
        index = bisect_right(self.opens, timestamp_ns) - 1
        if index < 0:
            return 0.0
        length = self.closes[index] - self.opens[index]
        partial = min(length, max(0, timestamp_ns - self.opens[index]))
        return index + partial / length

    def trading_ns_between(self, start_ns: int, end_ns: int) -> int:
        return max(0, self.trading_coordinate(end_ns) - self.trading_coordinate(start_ns))

    def session_equivalents_between(self, start_ns: int, end_ns: int) -> float:
        return max(0.0, self.session_coordinate(end_ns) - self.session_coordinate(start_ns))

    def sessions_spanned(self, start_ns: int, end_ns: int, calendar: MarketSessionService) -> int:
        start_day = local_date(start_ns, calendar)
        end_day = local_date(end_ns, calendar)
        return max(0, bisect_right(self.dates, end_day) - bisect_left(self.dates, start_day))
