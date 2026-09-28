"""Causal completed-session OHLC extraction and prior-only distribution statistics."""

from datetime import date
from math import isfinite, sqrt
from statistics import median

from mce.calendar.service import MarketSessionService
from mce.errors import CalendarRangeError
from mce.features.volatility.estimators import SessionOHLC
from mce.replay.store import CausalDataStore


def prior_distribution(
    current: float, prior: tuple[float, ...], minimum: int
) -> tuple[float, float | None, float] | None:
    """Midrank percentile, robust z, and upward rank shock; current excluded from prior."""
    if (
        minimum <= 0
        or len(prior) < minimum
        or not isfinite(current)
        or any(not isfinite(value) for value in prior)
    ):
        return None
    percentile = (
        sum(value < current for value in prior) + 0.5 * sum(value == current for value in prior)
    ) / len(prior)
    center = median(prior)
    mad = median(abs(value - center) for value in prior)
    robust_z = (
        (current - center) / (1.4826 * mad) if mad > 0 else (0.0 if current == center else None)
    )
    if robust_z is not None and not isfinite(robust_z):
        robust_z = None
    shock = max(0.0, 2 * percentile - 1)
    return percentile, robust_z, shock


def estimator_consensus(values: tuple[float, ...]) -> tuple[float, float, float, float] | None:
    """Median, range, population std, and reciprocal relative-dispersion agreement."""
    if not values or any(value < 0 or not isfinite(value) for value in values):
        return None
    middle = median(values)
    spread = max(values) - min(values)
    average = sum(value / len(values) for value in values)
    std = sqrt(sum((value - average) * (value - average) for value in values) / len(values))
    if not all(isfinite(item) for item in (middle, spread, std)):
        return None
    agreement = 1 / (1 + std / middle) if middle > 0 else (1.0 if std == 0 else 0.0)
    return middle, spread, std, agreement


class CompletedSessionReader:
    """Daily OHLC only from an effective full-session bar or complete intraday coverage."""

    def __init__(
        self, store: CausalDataStore, calendar: MarketSessionService, source: str, cadence_ns: int
    ) -> None:
        self.store = store
        self.calendar = calendar
        self.source = source
        self.cadence_ns = cadence_ns

    def session(self, symbol: str, day: date, as_of_ns: int) -> SessionOHLC | None:
        opened, closed = self.calendar.regular_session_bounds(day)
        if closed > as_of_ns:
            return None
        bars = tuple(
            bar
            for bar in self.store.bars(symbol, opened, closed + 1, as_of_ns)
            if bar.source == self.source
            and bar.bar_start_ns is not None
            and bar.bar_end_ns is not None
            and bar.bar_end_ns <= closed
        )
        daily = [bar for bar in bars if bar.bar_start_ns == opened and bar.bar_end_ns == closed]
        if daily:
            chosen = sorted(daily, key=lambda bar: (bar.effective_timestamp_ns, bar.close))[-1]
            return SessionOHLC(
                day,
                chosen.open,
                chosen.high,
                chosen.low,
                chosen.close,
                chosen.effective_timestamp_ns,
            )
        intraday = sorted(
            (
                bar
                for bar in bars
                if bar.bar_start_ns is not None
                and bar.bar_end_ns is not None
                and bar.bar_end_ns - bar.bar_start_ns == self.cadence_ns
            ),
            key=lambda bar: (bar.bar_start_ns, bar.effective_timestamp_ns, bar.close),
        )
        expected = (closed - opened) // self.cadence_ns
        if expected <= 0 or len(intraday) != expected:
            return None
        if any(
            bar.bar_start_ns != opened + index * self.cadence_ns
            or bar.bar_end_ns != opened + (index + 1) * self.cadence_ns
            for index, bar in enumerate(intraday)
        ):
            return None
        return SessionOHLC(
            day,
            intraday[0].open,
            max(bar.high for bar in intraday),
            min(bar.low for bar in intraday),
            intraday[-1].close,
            max(bar.effective_timestamp_ns for bar in intraday),
        )

    def history(
        self, symbol: str, end_day: date, count: int, as_of_ns: int
    ) -> tuple[SessionOHLC | None, ...]:
        """Consecutive trading-session slots, oldest first; missing slots stay None."""
        if count <= 0:
            raise ValueError("count must be positive")
        slots: list[SessionOHLC | None] = []
        day = end_day
        for _ in range(count):
            slots.append(self.session(symbol, day, as_of_ns))
            try:
                day = self.calendar.previous_trading_day(day)
            except CalendarRangeError:
                break
        return tuple(reversed(slots))


def contiguous_tail(
    slots: tuple[SessionOHLC | None, ...], count: int
) -> tuple[SessionOHLC, ...] | None:
    if count <= 0 or len(slots) < count or any(item is None for item in slots[-count:]):
        return None
    return tuple(item for item in slots[-count:] if item is not None)


def prior_rolling_volatilities(
    slots: tuple[SessionOHLC | None, ...],
    short_sessions: int,
    baseline_sessions: int,
    annual_sessions: int,
) -> tuple[float, ...]:
    """Prior endpoint values only, each from short_sessions close returns."""
    from mce.features.volatility.estimators import close_to_close

    values: list[float] = []
    # Last slot is the current complete-session endpoint, never part of the baseline.
    for end in range(max(0, len(slots) - baseline_sessions - 1), len(slots) - 1):
        start = end - short_sessions
        if start < 0:
            continue
        window = slots[start : end + 1]
        if any(item is None for item in window):
            continue
        estimate = close_to_close(
            tuple(item for item in window if item is not None), annual_sessions
        )
        if estimate is not None:
            values.append(estimate.annualized_volatility)
    return tuple(values)


def previous_short_volatility(
    slots: tuple[SessionOHLC | None, ...], short_sessions: int, annual_sessions: int
) -> float | None:
    from mce.features.volatility.estimators import close_to_close

    previous = contiguous_tail(slots[:-1], short_sessions + 1)
    result = close_to_close(previous, annual_sessions) if previous is not None else None
    return result.annualized_volatility if result is not None else None
