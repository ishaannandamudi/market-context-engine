"""Deterministic point-in-time sector and proxy price scenarios."""

from datetime import date

from mce.calendar.service import MarketSessionService
from mce.features.sectors.membership import SectorMembership, SectorMembershipTimeline
from mce.models.market_data import BarObservation
from mce.replay.store import CausalDataStore
from tests.fixtures.breadth_scenarios import MINUTE, days_before

GROUPS = {
    "technology": ("T1", "T2", "T3"),
    "energy": ("E1", "E2", "E3"),
    "utilities": ("U1", "U2", "U3"),
}
PROXIES = {"technology": "XLK", "energy": "XLE", "utilities": "XLU"}


def sector_membership(
    groups: dict[str, tuple[str, ...]] | None = None,
    weights: dict[str, float | None] | None = None,
) -> SectorMembershipTimeline:
    groups = groups or GROUPS
    weights = weights or {}
    return SectorMembershipTimeline(
        tuple(
            SectorMembership(sector=sector, symbol=symbol, weight=weights.get(symbol))
            for sector, symbols in groups.items()
            for symbol in symbols
        )
    )


def sector_store(
    calendar: MarketSessionService,
    *,
    groups: dict[str, tuple[str, ...]] | None = None,
    history_sessions: int = 70,
    history_changes: dict[str, tuple[float, ...]] | None = None,
    market_history: tuple[float, ...] | None = None,
    current: dict[str, float] | None = None,
    market_current: float = 0.0,
    proxy_current: dict[str, float] | None = None,
    missing: frozenset[str] = frozenset(),
    stale: frozenset[str] = frozenset(),
    day: date = date(2024, 6, 28),
) -> tuple[CausalDataStore, int]:
    groups = groups or GROUPS
    history_changes = history_changes or {}
    current = current or {}
    proxy_current = proxy_current or {}
    symbols = tuple(symbol for members in groups.values() for symbol in members)
    for changes in history_changes.values():
        if len(changes) != history_sessions:
            raise ValueError("historical changes must match history_sessions")
    if market_history is not None and len(market_history) != history_sessions:
        raise ValueError("market history must match history_sessions")
    sector_of = {symbol: sector for sector, members in groups.items() for symbol in members}
    sector_of.update({proxy: sector for sector, proxy in PROXIES.items()})
    prices = dict.fromkeys((*symbols, *PROXIES.values(), "SPY"), 100.0)
    store = CausalDataStore(max_observations=100_000)
    for index, prior_day in enumerate(days_before(calendar, day, history_sessions)):
        opened, closed = calendar.regular_session_bounds(prior_day)
        for symbol, old in tuple(prices.items()):
            movement = (
                (market_history[index] if market_history is not None else 0.0)
                if symbol == "SPY"
                else history_changes.get(sector_of[symbol], (0.0,) * history_sessions)[index]
            )
            new = old * (1 + movement)
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
                    volume=1,
                )
            )
            prices[symbol] = new
    opened, _ = calendar.regular_session_bounds(day)
    as_of = opened + 30 * MINUTE
    for symbol, old in tuple(prices.items()):
        if symbol in missing:
            continue
        movement = (
            market_current
            if symbol == "SPY"
            else proxy_current.get(symbol, 0.0)
            if symbol in PROXIES.values()
            else current.get(symbol, 0.0)
        )
        minutes = (1,) if symbol in stale else tuple(range(1, 31))
        previous_price = old
        for minute in minutes:
            end = opened + minute * MINUTE
            new = old * (1 + movement) ** (minute / 30)
            store.add(
                BarObservation(
                    symbol=symbol,
                    source="synthetic",
                    event_timestamp_ns=end - MINUTE,
                    effective_timestamp_ns=end,
                    bar_start_ns=end - MINUTE,
                    bar_end_ns=end,
                    open=previous_price,
                    high=max(previous_price, new),
                    low=min(previous_price, new),
                    close=new,
                    volume=1,
                )
            )
            previous_price = new
    return store, as_of
