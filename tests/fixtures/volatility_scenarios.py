"""Fixed daily and intraday volatility paths; no stochastic simulation."""

from datetime import date

from mce.calendar.service import MarketSessionService
from mce.models.market_data import BarObservation
from mce.replay.store import CausalDataStore

MINUTE = 60_000_000_000


def prior_days(calendar: MarketSessionService, end: date, count: int) -> tuple[date, ...]:
    days = [end]
    for _ in range(count - 1):
        days.append(calendar.previous_trading_day(days[-1]))
    return tuple(reversed(days))


def daily_bar(
    calendar: MarketSessionService,
    day: date,
    *,
    symbol: str = "SPY",
    opening: float = 100.0,
    closing: float = 101.0,
    high: float | None = None,
    low: float | None = None,
    delay_ns: int = 0,
) -> BarObservation:
    opened, closed = calendar.regular_session_bounds(day)
    return BarObservation(
        symbol=symbol,
        source="synthetic",
        bar_start_ns=opened,
        bar_end_ns=closed,
        event_timestamp_ns=opened,
        effective_timestamp_ns=closed + delay_ns,
        open=opening,
        high=high if high is not None else max(opening, closing),
        low=low if low is not None else min(opening, closing),
        close=closing,
        volume=1.0,
    )


def daily_store(
    calendar: MarketSessionService,
    *,
    end: date = date(2024, 6, 28),
    count: int = 90,
    symbols: tuple[str, ...] = ("SPY", "QQQ", "IWM"),
    amplitudes: tuple[float, ...] = (0.001, 0.001, 0.001),
    gap: float = 0.0,
) -> tuple[CausalDataStore, int]:
    days = prior_days(calendar, end, count)
    store = CausalDataStore()
    for symbol, amplitude in zip(symbols, amplitudes, strict=True):
        previous = 100.0
        for index, day in enumerate(days):
            opening = previous * (1 + gap * (1 if index % 2 else -1))
            closing = opening * (1 + amplitude * (1 if index % 2 else -1))
            store.add(
                daily_bar(
                    calendar,
                    day,
                    symbol=symbol,
                    opening=opening,
                    closing=closing,
                    high=max(opening, closing) * 1.001,
                    low=min(opening, closing) * 0.999,
                )
            )
            previous = closing
    return store, calendar.regular_session_bounds(end)[1]


def patterned_daily_store(
    calendar: MarketSessionService,
    amplitudes: tuple[float, ...],
    *,
    end: date = date(2024, 6, 28),
    gaps: tuple[float, ...] | None = None,
) -> tuple[CausalDataStore, int]:
    """Alternating signed daily moves with explicit time-varying amplitudes and gaps."""
    if gaps is not None and len(gaps) != len(amplitudes):
        raise ValueError("gaps and amplitudes must have the same length")
    store = CausalDataStore()
    previous = 100.0
    for index, day in enumerate(prior_days(calendar, end, len(amplitudes))):
        sign = 1 if index % 2 else -1
        opening = previous * (1 + sign * (gaps[index] if gaps is not None else 0))
        closing = opening * (1 + sign * amplitudes[index])
        store.add(
            daily_bar(
                calendar,
                day,
                opening=opening,
                closing=closing,
                high=max(opening, closing),
                low=min(opening, closing),
            )
        )
        previous = closing
    return store, calendar.regular_session_bounds(end)[1]


def intraday_bars(
    calendar: MarketSessionService,
    day: date,
    returns: tuple[float, ...],
    *,
    symbol: str = "SPY",
    cadence_ns: int = MINUTE,
    missing: frozenset[int] = frozenset(),
    delayed_last_ns: int = 0,
) -> tuple[BarObservation, ...]:
    opened, closed = calendar.regular_session_bounds(day)
    price = 100.0
    bars = []
    for index, change in enumerate(returns):
        start = opened + index * cadence_ns
        end = start + cadence_ns
        if end > closed:
            raise ValueError("path exceeds regular session")
        new_price = price * (1 + change)
        if index not in missing:
            bars.append(
                BarObservation(
                    symbol=symbol,
                    source="synthetic",
                    bar_start_ns=start,
                    bar_end_ns=end,
                    event_timestamp_ns=start,
                    effective_timestamp_ns=end
                    + (delayed_last_ns if index == len(returns) - 1 else 0),
                    open=price,
                    high=max(price, new_price),
                    low=min(price, new_price),
                    close=new_price,
                    volume=0.0,
                )
            )
        price = new_price
    return tuple(bars)
