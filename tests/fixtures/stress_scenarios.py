"""Deterministic bars and quotes for liquidity/stress reference scenarios."""

from datetime import date

from mce.calendar.service import MarketSessionService
from mce.models.market_data import BarObservation, QuoteObservation
from mce.replay.store import CausalDataStore
from tests.fixtures.breadth_scenarios import MINUTE, days_before

SYMBOLS = ("T1", "T2", "T3", "E1", "E2", "E3")


def stress_store(
    calendar: MarketSessionService,
    *,
    volumes: tuple[float, ...] = (100,) * 6,
    returns: tuple[float, ...] = (0.001,) * 6,
    gaps: tuple[float, ...] = (0,) * 6,
    prior_volume: float = 100,
    prior_sessions: int = 6,
    quote_width: float | None = 0.02,
    missing: frozenset[str] = frozenset(),
    stale: frozenset[str] = frozenset(),
    day: date = date(2024, 6, 28),
) -> tuple[CausalDataStore, int]:
    if not len(volumes) == len(returns) == len(gaps) == len(SYMBOLS):
        raise ValueError("one volume, return and gap per symbol")
    store = CausalDataStore(max_observations=100_000)

    def add_bar(symbol: str, start: int, end: int, old: float, new: float, volume: float) -> None:
        store.add(
            BarObservation(
                symbol=symbol,
                source="synthetic",
                event_timestamp_ns=end,
                effective_timestamp_ns=end,
                bar_start_ns=start,
                bar_end_ns=end,
                open=old,
                high=max(old, new),
                low=min(old, new),
                close=new,
                volume=volume,
            )
        )

    for prior in days_before(calendar, day, prior_sessions):
        opened, closed = calendar.regular_session_bounds(prior)
        for symbol in SYMBOLS:
            if symbol in missing:
                continue
            for minute in (1, 2, 3):
                old = 100.0
                new = 100.1 if minute == 3 else old
                add_bar(
                    symbol,
                    opened + (minute - 1) * MINUTE,
                    opened + minute * MINUTE,
                    old,
                    new,
                    prior_volume,
                )
            add_bar(symbol, closed - MINUTE, closed, 100.1, 100, prior_volume)
    opened, _ = calendar.regular_session_bounds(day)
    as_of = opened + 3 * MINUTE
    for symbol, volume, movement, gap in zip(SYMBOLS, volumes, returns, gaps, strict=True):
        if symbol in missing or symbol in stale:
            continue
        base = 100 * (1 + gap)
        for minute in (1, 2, 3):
            old = base
            new = base * (1 + movement) if minute == 3 else old
            add_bar(
                symbol, opened + (minute - 1) * MINUTE, opened + minute * MINUTE, old, new, volume
            )
        if quote_width is not None:
            midpoint = base * (1 + movement)
            store.add(
                QuoteObservation(
                    symbol=symbol,
                    source="synthetic",
                    event_timestamp_ns=as_of,
                    effective_timestamp_ns=as_of,
                    bid=midpoint - quote_width / 2,
                    ask=midpoint + quote_width / 2,
                )
            )
    return store, as_of
