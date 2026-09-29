"""Manual 500-name, 11-sector causal dispersion benchmark."""

import argparse
import platform
from datetime import date
from math import sin
from time import perf_counter

from mce.calendar.service import MarketSessionService
from mce.config.settings import DispersionSettings, FeatureSettings, MCEConfig
from mce.features.breadth.membership import MembershipTimeline, UniverseMembership
from mce.features.dispersion.engine import DispersionFeatureEngine
from mce.features.sectors.membership import SectorMembershipTimeline
from mce.features.volatility.models import RealizedVolatilityFeatureSet, SymbolVolatilityFeatures
from mce.models.enums import DataQuality
from mce.models.market_data import BarObservation
from mce.replay.store import CausalDataStore

MINUTE = 60_000_000_000


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", type=int, default=500)
    parser.add_argument("--sectors", type=int, default=11)
    parser.add_argument("--snapshots", type=int, default=5)
    parser.add_argument("--prior-sessions", type=int, default=22)
    args = parser.parse_args()
    if not 3 <= args.symbols <= 1000 or not 2 <= args.sectors <= args.symbols:
        parser.error("symbols must be 3–1000 and sectors 2–symbols")
    if not 1 <= args.snapshots <= 10 or not 1 <= args.prior_sessions <= 30:
        parser.error("snapshots must be 1–10 and prior sessions 1–30")
    calendar = MarketSessionService()
    day = date(2024, 6, 28)
    opened, _ = calendar.regular_session_bounds(day)
    symbols = tuple(f"D{index:04d}" for index in range(args.symbols))
    groups = {
        f"sector_{sector:02d}": tuple(
            symbol for index, symbol in enumerate(symbols) if index % args.sectors == sector
        )
        for sector in range(args.sectors)
    }
    membership = MembershipTimeline(
        tuple(
            UniverseMembership(symbol=symbol, weight=float(index % 10 + 1))
            for index, symbol in enumerate(symbols)
        )
    )
    sectors = SectorMembershipTimeline.from_groups(groups)
    store = CausalDataStore(max_observations=args.symbols * (args.prior_sessions * 2 + 32))

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

    prior_days = []
    cursor = day
    for _ in range(args.prior_sessions):
        cursor = calendar.previous_trading_day(cursor)
        prior_days.append(cursor)
    for day_index, prior in enumerate(reversed(prior_days)):
        _, closed = calendar.regular_session_bounds(prior)
        for index, symbol in enumerate(symbols):
            base = 100.0
            change = 0.0005 * sin(index + day_index) + 0.0005 * sin(index % args.sectors)
            add(symbol, closed - 31 * MINUTE, closed - 30 * MINUTE, base, base)
            add(symbol, closed - MINUTE, closed, base, base * (1 + change))
    for index, symbol in enumerate(symbols):
        previous = 100.0
        for minute in range(1, 32):
            end = opened + minute * MINUTE
            change = 0.00004 * sin(index * 0.7 + minute * 0.2)
            current = previous * (1 + change)
            add(symbol, end - MINUTE, end, previous, current)
            previous = current
    config = MCEConfig(
        _env_file=None,
        features=FeatureSettings(
            dispersion=DispersionSettings(
                universe=symbols,
                historical_baseline_sessions=args.prior_sessions,
                minimum_baseline_samples=min(10, args.prior_sessions),
                price_adjustment_status="adjusted",
            )
        ),
    )
    engine = DispersionFeatureEngine(config, membership, sectors, calendar)
    times = tuple(opened + (31 - index) * MINUTE for index in range(args.snapshots))
    volatility_contexts = tuple(
        RealizedVolatilityFeatureSet(
            as_of_ns=as_of,
            per_symbol={
                symbol: SymbolVolatilityFeatures(
                    symbol=symbol,
                    as_of_ns=as_of,
                    intraday={},
                    daily={},
                    rv_medium=0.1 + 0.01 * (index % args.sectors),
                    baseline_sample_count=0,
                    sample_count=30,
                    expected_count=30,
                    coverage_ratio=1,
                )
                for index, symbol in enumerate(symbols)
            },
            market_realized_volatility=0.15,
            index_percentiles={},
            available_symbol_count=len(symbols),
            expected_symbol_count=len(symbols),
            coverage_ratio=1,
            data_quality=DataQuality.GOOD,
        )
        for as_of in times
    )
    began = perf_counter()
    results = [
        engine.compute(store, as_of, volatility_context=context)
        for as_of, context in zip(times, volatility_contexts, strict=True)
    ]
    elapsed = perf_counter() - began
    print(f"Python {platform.python_version()} on {platform.platform()}")
    print(
        f"members={len(symbols)} sectors={len(groups)} observations={len(store)} "
        f"sessions={args.prior_sessions + 1} snapshots={len(results)}"
    )
    print(
        f"valid_members={results[0].valid_member_count} "
        f"baseline_samples={results[0].baseline_sample_count} "
        f"return_dispersion={results[0].return_dispersion} "
        f"volatility_dispersion={results[0].volatility_dispersion}"
    )
    print(f"total_seconds={elapsed:.3f} seconds_per_snapshot={elapsed / len(results):.3f}")


if __name__ == "__main__":
    main()
