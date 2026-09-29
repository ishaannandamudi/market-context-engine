"""Manual 500-member causal liquidity and precomputed-context stress benchmark."""

import argparse
import platform
from datetime import date
from time import perf_counter

from mce.calendar.service import MarketSessionService
from mce.config.settings import FeatureSettings, LiquiditySettings, MCEConfig, StressSettings
from mce.features.breadth.membership import MembershipTimeline, UniverseMembership
from mce.features.breadth.models import BreadthFeatureSet
from mce.features.correlation.models import CorrelationFeatureSet
from mce.features.dispersion.models import DispersionFeatureSet
from mce.features.sectors.models import SectorContext, SectorFeatureSet
from mce.features.stress.engine import LiquidityStressFeatureEngine
from mce.features.volatility.models import RealizedVolatilityFeatureSet
from mce.models.market_data import BarObservation, QuoteObservation
from mce.replay.store import CausalDataStore

MINUTE = 60_000_000_000


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", type=int, default=500)
    parser.add_argument("--prior-sessions", type=int, default=20)
    parser.add_argument("--snapshots", type=int, default=3)
    args = parser.parse_args()
    if not 3 <= args.symbols <= 1000:
        parser.error("symbols must be 3–1000")
    if not 5 <= args.prior_sessions <= 30 or not 1 <= args.snapshots <= 3:
        parser.error("prior sessions must be 5–30 and snapshots 1–3")
    calendar = MarketSessionService()
    day = date(2024, 6, 28)
    opened, _ = calendar.regular_session_bounds(day)
    symbols = tuple(f"L{index:04d}" for index in range(args.symbols))
    membership = MembershipTimeline(
        tuple(
            UniverseMembership(symbol=symbol, weight=float(1 + index % 10))
            for index, symbol in enumerate(symbols)
        )
    )
    store = CausalDataStore(max_observations=args.symbols * (4 * args.prior_sessions + 6))

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

    prior_days = []
    cursor = day
    for _ in range(args.prior_sessions):
        cursor = calendar.previous_trading_day(cursor)
        prior_days.append(cursor)
    for prior in reversed(prior_days):
        prior_open, prior_close = calendar.regular_session_bounds(prior)
        for index, symbol in enumerate(symbols):
            for minute in (1, 2, 3):
                old = 100.0
                new = 100.0 + 0.01 * (index % 5) if minute == 3 else old
                add_bar(
                    symbol,
                    prior_open + (minute - 1) * MINUTE,
                    prior_open + minute * MINUTE,
                    old,
                    new,
                    float(100 + index % 10),
                )
            add_bar(symbol, prior_close - MINUTE, prior_close, 100, 100, 100)
    for index, symbol in enumerate(symbols):
        base = 100.0 * (1 - 0.0001 * (index % 5))
        for minute in (1, 2, 3):
            old = base
            new = base * (1 + 0.0001 * (index % 7)) if minute == 3 else old
            add_bar(
                symbol,
                opened + (minute - 1) * MINUTE,
                opened + minute * MINUTE,
                old,
                new,
                float(100 + index % 10),
            )
        for minute in (1, 2, 3):
            as_of = opened + minute * MINUTE
            store.add(
                QuoteObservation(
                    symbol=symbol,
                    source="synthetic",
                    event_timestamp_ns=as_of,
                    effective_timestamp_ns=as_of,
                    bid=base - 0.01,
                    ask=base + 0.01,
                )
            )
    config = MCEConfig(
        _env_file=None,
        features=FeatureSettings(
            liquidity=LiquiditySettings(
                universe=symbols,
                historical_sessions=args.prior_sessions,
                minimum_history_samples=5,
                price_adjustment_status="adjusted",
            ),
            stress=StressSettings(minimum_history_samples=5),
        ),
    )
    engine = LiquidityStressFeatureEngine(config, membership, calendar)
    times = tuple(opened + minute * MINUTE for minute in range(1, args.snapshots + 1))
    contexts = tuple(
        {
            "volatility_context": RealizedVolatilityFeatureSet.model_construct(
                as_of_ns=as_of, volatility_score=0.5
            ),
            "correlation_context": CorrelationFeatureSet.model_construct(
                as_of_ns=as_of, systemic_co_movement_score=0.5
            ),
            "breadth_context": BreadthFeatureSet.model_construct(as_of_ns=as_of, breadth_score=0),
            "dispersion_context": DispersionFeatureSet.model_construct(
                as_of_ns=as_of, dispersion_score=0.5
            ),
            "sector_context": SectorFeatureSet.model_construct(
                as_of_ns=as_of,
                sectors={
                    "a": SectorContext.model_construct(absolute_returns={"1d": -0.01}),
                    "b": SectorContext.model_construct(absolute_returns={"1d": 0.01}),
                },
            ),
        }
        for as_of in times
    )
    began = perf_counter()
    results = [
        engine.compute(store, as_of, **context)
        for as_of, context in zip(times, contexts, strict=True)
    ]
    elapsed = perf_counter() - began
    print(f"Python {platform.python_version()} on {platform.platform()}")
    print(
        f"members={len(symbols)} observations={len(store)} "
        f"sessions={args.prior_sessions + 1} snapshots={len(results)}"
    )
    print(
        f"coverage={results[-1].liquidity.member_coverage:.3f} "
        f"liquidity_score={results[-1].liquidity.liquidity_score} "
        f"stress_score={results[-1].stress.stress_score}"
    )
    print(f"total_seconds={elapsed:.3f} seconds_per_snapshot={elapsed / len(results):.3f}")


if __name__ == "__main__":
    main()
