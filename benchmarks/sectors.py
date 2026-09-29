"""Manual eleven-sector relative strength, breadth and volatility benchmark."""

import platform
from datetime import date
from time import perf_counter

from mce.calendar.service import MarketSessionService
from mce.config.settings import FeatureSettings, MCEConfig, SectorSettings
from mce.features.sectors.engine import SectorFeatureEngine
from mce.features.sectors.membership import SectorMembership, SectorMembershipTimeline
from mce.models.market_data import BarObservation
from mce.replay.store import CausalDataStore

MINUTE = 60_000_000_000


def main() -> None:
    calendar = MarketSessionService()
    day = date(2024, 6, 28)
    groups = {
        f"sector_{index:02d}": tuple(f"S{index:02d}{member:03d}" for member in range(30))
        for index in range(11)
    }
    provider = SectorMembershipTimeline(
        tuple(
            SectorMembership(sector=sector, symbol=symbol, weight=1)
            for sector, symbols in groups.items()
            for symbol in symbols
        )
    )
    sector_of = {
        symbol: index for index, members in enumerate(groups.values()) for symbol in members
    }
    prices = dict.fromkeys((*sector_of, "SPY"), 100.0)
    store = CausalDataStore(max_observations=100_000)
    prior_days: list[date] = []
    cursor = day
    for _ in range(65):
        cursor = calendar.previous_trading_day(cursor)
        prior_days.append(cursor)
    for index, previous_day in enumerate(reversed(prior_days)):
        opened, closed = calendar.regular_session_bounds(previous_day)
        for symbol, old in tuple(prices.items()):
            sector = sector_of.get(symbol, 5)
            movement = (1 if (index + sector) % 2 else -1) * (0.002 + sector * 0.0002)
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
    for minute in range(1, 31):
        end = opened + minute * MINUTE
        for symbol, old in tuple(prices.items()):
            sector = sector_of.get(symbol, 5)
            movement = (sector - 5) * 0.00002
            new = old * (1 + movement)
            store.add(
                BarObservation(
                    symbol=symbol,
                    source="synthetic",
                    event_timestamp_ns=end - MINUTE,
                    effective_timestamp_ns=end,
                    bar_start_ns=end - MINUTE,
                    bar_end_ns=end,
                    open=old,
                    high=max(old, new),
                    low=min(old, new),
                    close=new,
                    volume=1,
                )
            )
            prices[symbol] = new
    config = MCEConfig(
        _env_file=None,
        features=FeatureSettings(
            sectors=SectorSettings(groups=groups, price_adjustment_status="adjusted")
        ),
    )
    engine = SectorFeatureEngine(config, provider, calendar)
    times = tuple(opened + minute * MINUTE for minute in range(26, 31))
    began = perf_counter()
    results = [engine.compute(store, as_of_ns) for as_of_ns in times]
    elapsed = perf_counter() - began
    print(f"Python {platform.python_version()} on {platform.platform()}")
    print(f"{len(sector_of)} constituents; {len(groups)} sectors; 65 prior sessions")
    print(f"{len(store)} bars")
    print(
        f"{len(results)} snapshots; {elapsed:.3f} s total; "
        f"{elapsed / len(results) * 1000:.1f} ms/snapshot"
    )


if __name__ == "__main__":
    main()
