"""Deterministic constituent and index-proxy histories for breadth tests."""

from datetime import date

from mce.calendar.service import MarketSessionService
from mce.features.breadth.membership import MembershipTimeline, UniverseMembership
from mce.models.market_data import BarObservation
from mce.replay.store import CausalDataStore

MINUTE = 60_000_000_000


def days_before(calendar: MarketSessionService, day: date, count: int) -> tuple[date, ...]:
    days: list[date] = []
    for _ in range(count):
        day = calendar.previous_trading_day(day)
        days.append(day)
    return tuple(reversed(days))


def membership(
    symbols: tuple[str, ...], weights: tuple[float | None, ...] | None = None
) -> MembershipTimeline:
    if weights is None:
        weights = (None,) * len(symbols)
    return MembershipTimeline(
        tuple(
            UniverseMembership(symbol=symbol, weight=weight)
            for symbol, weight in zip(symbols, weights, strict=True)
        )
    )


def breadth_store(
    calendar: MarketSessionService,
    symbols: tuple[str, ...],
    returns: tuple[float, ...],
    *,
    day: date = date(2024, 6, 28),
    history_sessions: int = 65,
    history_changes: tuple[float, ...] | None = None,
    index_return: float | None = 0.01,
    missing: frozenset[str] = frozenset(),
    stale: frozenset[str] = frozenset(),
    cadence_ns: int = MINUTE,
) -> tuple[CausalDataStore, int]:
    if len(symbols) != len(returns):
        raise ValueError("symbols and returns must have equal length")
    if history_changes is not None and len(history_changes) != history_sessions:
        raise ValueError("history changes must have history_sessions values")
    store = CausalDataStore()
    prices = dict.fromkeys((*symbols, "SPY"), 100.0)
    for index, prior_day in enumerate(days_before(calendar, day, history_sessions)):
        opened, closed = calendar.regular_session_bounds(prior_day)
        movement = history_changes[index] if history_changes is not None else 0.0
        for symbol in prices:
            old = prices[symbol]
            new = old * (1 + movement) if symbol != "SPY" else old
            store.add(
                BarObservation(
                    symbol=symbol,
                    source="synthetic",
                    event_timestamp_ns=opened,
                    effective_timestamp_ns=closed,
                    bar_start_ns=opened,
                    bar_end_ns=closed,
                    open=old,
                    high=max(old, new),
                    low=min(old, new),
                    close=new,
                    volume=1.0,
                )
            )
            prices[symbol] = new
    opened, _ = calendar.regular_session_bounds(day)
    as_of_ns = opened + 30 * MINUTE
    for symbol, movement in zip(symbols, returns, strict=True):
        if symbol in missing:
            continue
        start = opened if symbol in stale else as_of_ns - cadence_ns
        old = prices[symbol]
        new = old * (1 + movement)
        store.add(
            BarObservation(
                symbol=symbol,
                source="synthetic",
                event_timestamp_ns=start,
                effective_timestamp_ns=start + cadence_ns,
                bar_start_ns=start,
                bar_end_ns=start + cadence_ns,
                open=old,
                high=max(old, new),
                low=min(old, new),
                close=new,
                volume=1.0,
            )
        )
    if "SPY" not in symbols and index_return is not None:
        old = prices["SPY"]
        new = old * (1 + index_return)
        store.add(
            BarObservation(
                symbol="SPY",
                source="synthetic",
                event_timestamp_ns=as_of_ns - cadence_ns,
                effective_timestamp_ns=as_of_ns,
                bar_start_ns=as_of_ns - cadence_ns,
                bar_end_ns=as_of_ns,
                open=old,
                high=max(old, new),
                low=min(old, new),
                close=new,
                volume=1.0,
            )
        )
    return store, as_of_ns
