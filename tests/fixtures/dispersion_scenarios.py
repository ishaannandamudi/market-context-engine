"""Deterministic regular-session panels for dispersion causality tests."""

from datetime import date

from mce.calendar.service import MarketSessionService
from mce.models.market_data import BarObservation
from mce.replay.store import CausalDataStore
from tests.fixtures.breadth_scenarios import MINUTE, days_before

SYMBOLS = ("T1", "T2", "T3", "E1", "E2", "E3")
GROUPS = {"technology": SYMBOLS[:3], "energy": SYMBOLS[3:]}


def dispersion_store(
    calendar: MarketSessionService,
    returns: tuple[float, ...],
    *,
    days: int = 12,
    day: date = date(2024, 6, 28),
    missing: frozenset[str] = frozenset(),
    stale: frozenset[str] = frozenset(),
    scale: float = 1.0,
) -> tuple[CausalDataStore, int]:
    if len(returns) != len(SYMBOLS):
        raise ValueError("one current return required per symbol")
    store = CausalDataStore(max_observations=100_000)

    def add(symbol: str, start: int, end: int, old: float, new: float) -> None:
        store.add(
            BarObservation(
                symbol=symbol,
                source="synthetic",
                event_timestamp_ns=start,
                effective_timestamp_ns=end,
                bar_start_ns=start,
                bar_end_ns=end,
                open=old,
                high=max(old, new),
                low=min(old, new),
                close=new,
                volume=1,
            )
        )

    for index, prior_day in enumerate(days_before(calendar, day, days)):
        _, closed = calendar.regular_session_bounds(prior_day)
        for symbol_index, symbol in enumerate(SYMBOLS):
            if symbol in missing:
                continue
            base = 100 * scale
            movement = (symbol_index - 2.5) * (0.001 + index * 0.00005)
            add(symbol, closed - 3 * MINUTE, closed - 2 * MINUTE, base, base)
            add(symbol, closed - 2 * MINUTE, closed - MINUTE, base, base)
            add(symbol, closed - MINUTE, closed, base, base * (1 + movement))
    opened, _ = calendar.regular_session_bounds(day)
    as_of_ns = opened + 3 * MINUTE
    for symbol, movement in zip(SYMBOLS, returns, strict=True):
        if symbol in missing:
            continue
        base = 100 * scale
        if symbol in stale:
            continue
        add(symbol, opened, opened + MINUTE, base, base)
        add(symbol, opened + MINUTE, opened + 2 * MINUTE, base, base)
        add(symbol, opened + 2 * MINUTE, as_of_ns, base, base * (1 + movement))
    return store, as_of_ns
