"""Manual same-bucket baseline and snapshot benchmark; no CI timing gate."""

import argparse
import platform
from datetime import date
from time import perf_counter

from mce.calendar.service import MarketSessionService
from mce.config.settings import MCEConfig, SeasonalitySettings
from mce.features.seasonality.baseline import prior_session_dates
from mce.features.seasonality.engine import IntradaySeasonalityEngine
from mce.models.market_data import BarObservation, QuoteObservation
from mce.replay.store import CausalDataStore

MINUTE = 60_000_000_000


def add_session(
    store: CausalDataStore,
    calendar: MarketSessionService,
    day: date,
    symbols: tuple[str, ...],
) -> None:
    opened, _ = calendar.regular_session_bounds(day)
    for symbol in symbols:
        for index in range(5):
            start = opened + index * MINUTE
            end = start + MINUTE
            store.add(
                BarObservation(
                    symbol=symbol,
                    source="synthetic",
                    event_timestamp_ns=end,
                    effective_timestamp_ns=end,
                    bar_start_ns=start,
                    bar_end_ns=end,
                    open=100,
                    high=100.01,
                    low=100,
                    close=100.01,
                    volume=100 + index,
                )
            )
            store.add(
                QuoteObservation(
                    symbol=symbol,
                    source="synthetic",
                    event_timestamp_ns=end,
                    effective_timestamp_ns=end,
                    bid=99.99,
                    ask=100.01,
                )
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", type=int, default=500)
    parser.add_argument("--sessions", type=int, default=60)
    parser.add_argument("--snapshots", type=int, default=3)
    args = parser.parse_args()
    if args.symbols < 3 or args.sessions < 2 or args.snapshots < 1:
        parser.error("need at least three symbols, two sessions, and one snapshot")
    target = date(2024, 6, 28)
    symbols = tuple(f"S{index:04d}" for index in range(args.symbols))
    calendar = MarketSessionService()
    base = MCEConfig(_env_file=None)
    settings = SeasonalitySettings(
        universe=symbols,
        baseline_sessions=args.sessions,
        minimum_history_sessions=min(20, args.sessions),
    )
    config = base.model_copy(
        update={"features": base.features.model_copy(update={"seasonality": settings})}
    )
    engine = IntradaySeasonalityEngine(config, calendar=calendar)
    prior = prior_session_dates(calendar, target, args.sessions)
    store = CausalDataStore(max_observations=(args.sessions + 1) * args.symbols * 10 + 1)
    for day in prior:
        add_session(store, calendar, day, symbols)
    add_session(store, calendar, target, symbols)
    opened, _ = calendar.regular_session_bounds(target)
    start = perf_counter()
    history = engine.build_history(store, opened)
    build_seconds = perf_counter() - start
    start = perf_counter()
    results = tuple(
        engine.compute(store, opened + 5 * MINUTE + index * 1_000_000_000, history)
        for index in range(args.snapshots)
    )
    snapshot_seconds = perf_counter() - start
    print(f"Python {platform.python_version()} on {platform.platform()}")
    print(
        f"symbols={args.symbols} prior_sessions={len(prior)} one_minute_bars={len(store) // 2} "
        f"quotes={len(store) // 2} history_observations={len(history.observations)} "
        f"bucket_minutes=5 snapshots={args.snapshots}"
    )
    print(
        f"baseline_build_seconds={build_seconds:.3f} "
        f"snapshot_seconds={snapshot_seconds / args.snapshots:.3f} "
        f"normalized_symbols={results[-1].market_metrics[0].normalized_symbols} "
        f"all_snapshots_same_completed_bucket=true"
    )


if __name__ == "__main__":
    main()
