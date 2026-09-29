"""Provider-neutral regular-session prices and trading-day reference closes."""

from datetime import date

from mce.calendar.service import MarketSessionService
from mce.errors import CalendarRangeError
from mce.models.market_data import BarObservation
from mce.replay.store import CausalDataStore


class BreadthPriceReader:
    def __init__(self, store: CausalDataStore, calendar: MarketSessionService, source: str) -> None:
        self.store = store
        self.calendar = calendar
        self.source = source
        self._bounds: dict[date, tuple[int, int]] = {}

    def _session_bounds(self, day: date) -> tuple[int, int]:
        if day not in self._bounds:
            self._bounds[day] = self.calendar.regular_session_bounds(day)
        return self._bounds[day]

    def _regular_bars(self, symbol: str, day: date, as_of_ns: int) -> tuple[BarObservation, ...]:
        opened, closed = self._session_bounds(day)
        return tuple(
            bar
            for bar in self.store.bars(symbol, opened, closed + 1, as_of_ns)
            if bar.source == self.source
            and bar.bar_start_ns is not None
            and bar.bar_end_ns is not None
            and opened <= bar.bar_start_ns < bar.bar_end_ns <= closed
            and bar.bar_end_ns <= as_of_ns
            and bar.close > 0
        )

    def current(self, symbol: str, day: date, as_of_ns: int) -> BarObservation | None:
        bars = self._regular_bars(symbol, day, as_of_ns)
        return (
            max(
                bars,
                key=lambda bar: (
                    bar.bar_end_ns or 0,
                    bar.effective_timestamp_ns,
                    bar.bar_start_ns or 0,
                    bar.close,
                ),
            )
            if bars
            else None
        )

    def session_close(self, symbol: str, day: date, as_of_ns: int) -> float | None:
        _, closed = self._session_bounds(day)
        bars = tuple(
            bar for bar in self._regular_bars(symbol, day, as_of_ns) if bar.bar_end_ns == closed
        )
        return (
            max(bars, key=lambda bar: (bar.effective_timestamp_ns, bar.close)).close
            if bars
            else None
        )

    def prior_closes(
        self, symbol: str, day: date, count: int, as_of_ns: int
    ) -> tuple[float | None, ...]:
        if count <= 0:
            raise ValueError("count must be positive")
        result: list[float | None] = []
        cursor = day
        for _ in range(count):
            try:
                cursor = self.calendar.previous_trading_day(cursor)
            except CalendarRangeError:
                break
            result.append(self.session_close(symbol, cursor, as_of_ns))
        return tuple(result)

    def closes_on_days(
        self, symbol: str, days: tuple[date, ...], as_of_ns: int
    ) -> tuple[float | None, ...]:
        return tuple(self.session_close(symbol, day, as_of_ns) for day in days)
