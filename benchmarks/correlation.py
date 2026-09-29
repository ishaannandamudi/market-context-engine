"""Manual causal correlation benchmark; data construction is excluded from timing."""

import argparse
import platform
from datetime import date
from math import exp, sin
from time import perf_counter

from mce.calendar.service import MarketSessionService
from mce.config.settings import CorrelationSettings, FeatureSettings, MCEConfig
from mce.features.correlation.engine import CorrelationFeatureEngine
from mce.models.market_data import BarObservation
from mce.replay.store import CausalDataStore

MINUTE = 60_000_000_000


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", type=int, default=100)
    parser.add_argument("--snapshots", type=int, default=5)
    parser.add_argument("--prior-sessions", type=int, default=2)
    args = parser.parse_args()
    if (
        not 2 <= args.symbols <= 500
        or not 1 <= args.snapshots <= 10
        or not 0 <= args.prior_sessions <= 5
    ):
        parser.error("symbols must be 2–500, snapshots 1–10, prior sessions 0–5")
    calendar = MarketSessionService()
    day = date(2024, 6, 28)
    opened, _ = calendar.regular_session_bounds(day)
    symbols = tuple(f"C{index:04d}" for index in range(args.symbols))
    store = CausalDataStore(max_observations=args.symbols * 135 * (args.prior_sessions + 1))
    prices = dict.fromkeys(symbols, 100.0)
    days = [day]
    for _ in range(args.prior_sessions):
        days.append(calendar.previous_trading_day(days[-1]))
    for session_index, session_day in enumerate(reversed(days)):
        session_open, session_close = calendar.regular_session_bounds(session_day)
        first = session_open if session_day == day else session_close - 130 * MINUTE
        for minute in range(1, 131):
            end = first + minute * MINUTE
            common = 0.001 * sin((minute + session_index * 7) * 0.31)
            for index, symbol in enumerate(symbols):
                previous = prices[symbol]
                change = common + 0.0005 * sin(minute * (0.17 + index % 19 * 0.03) + index)
                current = previous * exp(change)
                store.add(
                    BarObservation(
                        symbol=symbol,
                        source="synthetic",
                        event_timestamp_ns=end,
                        effective_timestamp_ns=end,
                        bar_start_ns=end - MINUTE,
                        bar_end_ns=end,
                        open=previous,
                        high=max(previous, current),
                        low=min(previous, current),
                        close=current,
                        volume=1,
                    )
                )
                prices[symbol] = current
    config = MCEConfig(
        _env_file=None,
        features=FeatureSettings(
            correlation=CorrelationSettings(
                universe=symbols,
                historical_baseline_sessions=args.prior_sessions,
                minimum_baseline_samples=max(1, args.prior_sessions),
            )
        ),
    )
    engine = CorrelationFeatureEngine(config, calendar=calendar)
    times = tuple(opened + (130 - index) * MINUTE for index in range(args.snapshots))
    began = perf_counter()
    results = [engine.compute(store, asof) for asof in times]
    elapsed = perf_counter() - began
    print(f"Python {platform.python_version()} on {platform.platform()}")
    print(
        f"symbols={len(symbols)} observations={len(store)} "
        f"sessions={len(days)} snapshots={len(results)}"
    )
    print(
        f"matrix_dimension={results[0].matrix_dimension} "
        f"aligned_samples={results[0].aligned_sample_count}"
    )
    print(f"total_seconds={elapsed:.3f} seconds_per_snapshot={elapsed / len(results):.3f}")


if __name__ == "__main__":
    main()
