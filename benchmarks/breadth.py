"""Manual 500-member, 60-session breadth benchmark; never a CI timing gate."""

import platform
from datetime import date
from time import perf_counter

from mce.calendar.service import MarketSessionService
from mce.config.settings import (
    BreadthSettings,
    FeatureSettings,
    MarketSettings,
    MCEConfig,
    UniverseSettings,
)
from mce.features.breadth.engine import BreadthFeatureEngine
from mce.features.breadth.membership import MembershipTimeline, UniverseMembership
from mce.models.market_data import BarObservation
from mce.replay.store import CausalDataStore

MINUTE = 60_000_000_000


def previous_days(calendar: MarketSessionService, final_day: date, count: int) -> tuple[date, ...]:
    days: list[date] = []
    cursor = final_day
    for _ in range(count):
        cursor = calendar.previous_trading_day(cursor)
        days.append(cursor)
    return tuple(reversed(days))


def main() -> None:
    calendar = MarketSessionService()
    day = date(2024, 6, 28)
    symbols = tuple(f"B{i:04d}" for i in range(500))
    members = MembershipTimeline(
        tuple(UniverseMembership(symbol=symbol, weight=1.0) for symbol in symbols)
    )
    store = CausalDataStore(max_observations=100_000)
    prices = dict.fromkeys((*symbols, "SPY"), 100.0)
    for index, prior_day in enumerate(previous_days(calendar, day, 60)):
        opened, closed = calendar.regular_session_bounds(prior_day)
        for number, symbol in enumerate(prices):
            start = prices[symbol]
            change = (1 if (index + number) % 2 else -1) * (0.001 + number * 0.000001)
            end = start * (1 + change)
            store.add(
                BarObservation(
                    symbol=symbol,
                    source="synthetic",
                    bar_start_ns=opened,
                    bar_end_ns=closed,
                    event_timestamp_ns=opened,
                    effective_timestamp_ns=closed,
                    open=start,
                    high=max(start, end),
                    low=min(start, end),
                    close=end,
                    volume=1,
                )
            )
            prices[symbol] = end
    opened, _ = calendar.regular_session_bounds(day)
    for minute in range(21, 31):
        start_ns = opened + (minute - 1) * MINUTE
        for number, symbol in enumerate(prices):
            start = prices[symbol]
            change = (1 if number % 3 else -1) * 0.0002
            end = start * (1 + change)
            store.add(
                BarObservation(
                    symbol=symbol,
                    source="synthetic",
                    bar_start_ns=start_ns,
                    bar_end_ns=start_ns + MINUTE,
                    event_timestamp_ns=start_ns,
                    effective_timestamp_ns=start_ns + MINUTE,
                    open=start,
                    high=max(start, end),
                    low=min(start, end),
                    close=end,
                    volume=1,
                )
            )
            prices[symbol] = end
    config = MCEConfig(
        _env_file=None,
        market=MarketSettings(universe=UniverseSettings(breadth=symbols)),
        features=FeatureSettings(breadth=BreadthSettings(price_adjustment_status="adjusted")),
    )
    engine = BreadthFeatureEngine(config, members, calendar)
    times = tuple(opened + minute * MINUTE for minute in range(21, 31))
    began = perf_counter()
    results = [engine.compute(store, as_of_ns) for as_of_ns in times]
    elapsed = perf_counter() - began
    print(f"Python {platform.python_version()} on {platform.platform()}")
    print(f"{len(symbols)} members; 60 prior sessions; {len(store)} bars")
    print(
        f"{len(results)} snapshots; {elapsed:.3f} s total; "
        f"{elapsed / len(results) * 1000:.1f} ms/snapshot"
    )


if __name__ == "__main__":
    main()
