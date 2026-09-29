"""Exact-cadence regular-session returns; missing closes never become zeros."""

from dataclasses import dataclass
from datetime import date
from math import log

from mce.calendar.service import MarketSessionService
from mce.errors import CalendarRangeError
from mce.models.market_data import BarObservation
from mce.replay.store import CausalDataStore


@dataclass(frozen=True)
class ReturnSeries:
    values: dict[int, float]
    expected: int
    latest_available: bool
    latest_effective_ns: int | None
    latest_price_available: bool = False


class CorrelationReturnReader:
    def __init__(
        self,
        store: CausalDataStore,
        calendar: MarketSessionService,
        source: str,
        cadence_ns: int,
    ) -> None:
        self.store = store
        self.calendar = calendar
        self.source = source
        self.cadence_ns = cadence_ns
        self._bars: dict[tuple[str, date, int], tuple[BarObservation, ...]] = {}

    def _regular(self, symbol: str, day: date, as_of_ns: int) -> tuple[BarObservation, ...]:
        key = (symbol, day, as_of_ns)
        if key not in self._bars:
            opened, closed = self.calendar.regular_session_bounds(day)
            self._bars[key] = tuple(
                bar
                for bar in self.store.bars(symbol, opened, closed + 1, as_of_ns)
                if bar.source == self.source
                and bar.bar_start_ns is not None
                and bar.bar_end_ns is not None
                and opened <= bar.bar_start_ns < bar.bar_end_ns <= closed
                and bar.bar_end_ns <= as_of_ns
                and bar.close > 0
            )
        return self._bars[key]

    def intraday(self, symbol: str, day: date, as_of_ns: int, window_bars: int) -> ReturnSeries:
        opened, closed = self.calendar.regular_session_bounds(day)
        end = min(as_of_ns, closed)
        last = opened + max(0, (end - opened) // self.cadence_ns) * self.cadence_ns
        completed = (last - opened) // self.cadence_ns
        if completed <= window_bars:
            return ReturnSeries({}, window_bars, False, None)
        slots = tuple(
            opened + index * self.cadence_ns
            for index in range(completed - window_bars + 1, completed + 1)
        )
        first = slots[0] - self.cadence_ns
        prices: dict[int, BarObservation] = {}
        regular = self._regular(symbol, day, as_of_ns)
        for bar in regular:
            start, finish = bar.bar_start_ns, bar.bar_end_ns
            if (
                start is None
                or finish is None
                or finish - start != self.cadence_ns
                or start < opened
                or (start - opened) % self.cadence_ns
                or not first <= finish <= last
            ):
                continue
            prior = prices.get(finish)
            if prior is None or (bar.effective_timestamp_ns, bar.close) > (
                prior.effective_timestamp_ns,
                prior.close,
            ):
                prices[finish] = bar
        values = {
            finish: log(prices[finish].close / prices[finish - self.cadence_ns].close)
            for finish in slots
            if finish in prices and finish - self.cadence_ns in prices
        }
        latest = prices.get(last)
        latest_observed = max(
            (
                bar.effective_timestamp_ns
                for bar in regular
                if bar.bar_end_ns is not None
                and bar.bar_end_ns <= last
                and bar.bar_end_ns - (bar.bar_start_ns or 0) == self.cadence_ns
            ),
            default=None,
        )
        return ReturnSeries(
            values,
            window_bars,
            last in values,
            latest_observed,
            latest is not None,
        )

    def daily_close(self, symbol: str, day: date, as_of_ns: int) -> BarObservation | None:
        opened, closed = self.calendar.regular_session_bounds(day)
        if closed > as_of_ns:
            return None
        candidates = (
            bar
            for bar in self._regular(symbol, day, as_of_ns)
            if bar.bar_end_ns == closed
            and (bar.bar_start_ns == opened or bar.bar_start_ns == closed - self.cadence_ns)
        )
        return max(
            candidates, key=lambda item: (item.effective_timestamp_ns, item.close), default=None
        )

    def daily(self, symbol: str, end_day: date, as_of_ns: int, sessions: int) -> ReturnSeries:
        days = [end_day]
        for _ in range(sessions):
            try:
                days.append(self.calendar.previous_trading_day(days[-1]))
            except CalendarRangeError:
                return ReturnSeries({}, sessions, False, None)
        days.reverse()
        closes = {day: self.daily_close(symbol, day, as_of_ns) for day in days}
        values = {
            index: log(current.close / previous.close)
            for index, (earlier, later) in enumerate(zip(days, days[1:], strict=False))
            if (previous := closes[earlier]) is not None and (current := closes[later]) is not None
        }
        latest = closes[end_day]
        return ReturnSeries(
            values,
            sessions,
            sessions - 1 in values,
            latest.effective_timestamp_ns if latest else None,
            latest is not None,
        )
