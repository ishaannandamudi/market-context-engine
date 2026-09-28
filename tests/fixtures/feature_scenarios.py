"""Deterministic trend paths used only by feature tests and the benchmark."""

from datetime import date

from mce.calendar.service import MarketSessionService
from mce.models.market_data import BarObservation
from mce.replay.store import CausalDataStore

MINUTE = 60_000_000_000


def bars_for_day(
    calendar: MarketSessionService,
    day: date,
    symbol: str = "SPY",
    prices: tuple[float, ...] = (),
    delay_ns: int = 0,
) -> tuple[BarObservation, ...]:
    opened, closed = calendar.regular_session_bounds(day)
    result: list[BarObservation] = []
    for index, close in enumerate(prices):
        start = opened + index * MINUTE
        end = start + MINUTE
        if end > closed:
            raise ValueError("price path exceeds session")
        opening = prices[index - 1] if index else close
        result.append(
            BarObservation(
                symbol=symbol,
                source="synthetic",
                bar_start_ns=start,
                bar_end_ns=end,
                event_timestamp_ns=start,
                effective_timestamp_ns=end + delay_ns,
                open=opening,
                high=max(opening, close),
                low=min(opening, close),
                close=close,
                volume=0.0,
            )
        )
    return tuple(result)


def trend_store(
    calendar: MarketSessionService,
    *,
    symbols: tuple[str, ...] = ("SPY", "QQQ", "IWM"),
    direction: int = 1,
    points: int = 61,
    missing: frozenset[int] = frozenset(),
    delayed_last_ns: int = 0,
) -> tuple[CausalDataStore, int]:
    day = date(2024, 1, 3)
    opened, _ = calendar.regular_session_bounds(day)
    store = CausalDataStore()
    for symbol in symbols:
        for index in range(points):
            if index in missing:
                continue
            price = 100.0 + direction * index * 0.05
            delay = delayed_last_ns if index == points - 1 else 0
            bar = BarObservation(
                symbol=symbol,
                source="synthetic",
                bar_start_ns=opened + index * MINUTE,
                bar_end_ns=opened + (index + 1) * MINUTE,
                event_timestamp_ns=opened + index * MINUTE,
                effective_timestamp_ns=opened + (index + 1) * MINUTE + delay,
                open=price,
                high=price,
                low=price,
                close=price,
                volume=0.0,
            )
            store.add(bar)
    return store, opened + points * MINUTE
