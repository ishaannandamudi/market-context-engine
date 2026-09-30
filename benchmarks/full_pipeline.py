"""Manual end-to-end MCE-20 profile, including replay ingest and snapshot orchestration."""

import argparse
import platform
import tracemalloc
from datetime import date
from statistics import mean
from time import perf_counter

from mce.calendar.service import MarketSessionService
from mce.config.settings import MCEConfig
from mce.features.ensemble.contracts import FilteredHMMContext
from mce.features.macro.models import MacroContextFeatureSet, MacroEventContext, MacroStressContext
from mce.features.options_context.engine import OptionsContextFeatureEngine
from mce.models.enums import DataQuality
from mce.models.market_data import BarObservation
from mce.orchestration.context import SuppliedContexts
from mce.orchestration.pipeline import MarketContextPipeline
from mce.orchestration.replay import MarketContextReplay, ReplayRequest
from mce.statistical.models import LatentStateInference
from tests.fixtures.options_scenarios import options_input
from tests.fixtures.system_scenarios import bars


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", type=int, default=500)
    parser.add_argument("--bars", type=int, default=30)
    parser.add_argument("--snapshots", type=int, default=2)
    parser.add_argument("--prior-sessions", type=int, default=0)
    parser.add_argument("--memory", action="store_true")
    args = parser.parse_args()
    if (
        not 3 <= args.symbols <= 500
        or not 3 <= args.bars <= 390
        or not 1 <= args.snapshots <= args.bars
        or not 0 <= args.prior_sessions <= 60
    ):
        parser.error("symbols 3..500, bars 3..390, snapshots 1..bars, prior sessions 0..60")
    symbols = tuple(f"S{index:04d}" for index in range(args.symbols))
    groups = {
        "Technology": symbols[: len(symbols) // 2],
        "Finance": symbols[len(symbols) // 2 :],
    }
    config = MCEConfig(
        _env_file=None,
        market={"universe": {"broad_market": ("SPY",), "breadth": symbols}},
        features={
            "volatility": {
                "session_horizons": (3, 5, 10, 20),
                "short_window_sessions": 3,
                "medium_window_sessions": 10,
                "long_window_sessions": 20,
                "baseline_sessions": 20,
                "minimum_baseline_samples": 10,
            },
            "sectors": {"groups": groups},
            "correlation": {
                "universe": symbols,
                "short_window_bars": 8,
                "medium_window_bars": 15,
                "long_window_bars": 30,
                "historical_baseline_sessions": 0,
            },
            "dispersion": {"universe": symbols},
            "factor": {
                "universe": symbols,
                "window_sessions": 20,
                "minimum_samples": 10,
                "minimum_time_coverage": 0.5,
            },
            "seasonality": {
                "universe": symbols,
                "baseline_sessions": 20,
                "minimum_history_sessions": 10,
            },
            "liquidity": {"universe": symbols},
        },
        statistical={"enabled": True},
    )
    current = bars(config, symbols=("SPY", *symbols), count=args.bars)
    prior: list[BarObservation] = []
    calendar = MarketSessionService(config.market)
    historical_days: list[date] = []
    day = date(2025, 1, 2)
    for _ in range(args.prior_sessions):
        day = calendar.previous_trading_day(day)
        historical_days.append(day)
    for day_index, historical_day in enumerate(reversed(historical_days)):
        opened, closed = calendar.regular_session_bounds(historical_day)
        for symbol_index, symbol in enumerate(("SPY", *symbols)):
            price = 100 + symbol_index * 0.1 + day_index * (0.04 + symbol_index * 0.0001)
            if symbol == "SPY":
                prior.append(
                    BarObservation(
                        symbol=symbol,
                        event_timestamp_ns=closed,
                        effective_timestamp_ns=closed,
                        source=config.provider.source,
                        bar_start_ns=opened,
                        bar_end_ns=closed,
                        open=price - 0.02,
                        high=price + 0.05,
                        low=price - 0.05,
                        close=price,
                        volume=390_000,
                    )
                )
            for end in (opened + 30 * 60_000_000_000, closed):
                prior.append(
                    BarObservation(
                        symbol=symbol,
                        event_timestamp_ns=end,
                        effective_timestamp_ns=end,
                        source=config.provider.source,
                        bar_start_ns=end - 60_000_000_000,
                        bar_end_ns=end,
                        open=price - 0.02,
                        high=price + 0.05,
                        low=price - 0.05,
                        close=price,
                        volume=1000 + day_index * 10,
                    )
                )
    observations = (*prior, *current)
    end = current[-1].effective_timestamp_ns
    minute = 60_000_000_000
    times = tuple(end - minute * offset for offset in reversed(range(args.snapshots)))
    options_engine = OptionsContextFeatureEngine(config)
    supplied = {}
    for timestamp in times:
        supplied[timestamp] = SuppliedContexts(
            options_context=options_engine.compute((options_input(clock=timestamp),), timestamp),
            macro_context=MacroContextFeatureSet(
                as_of_ns=timestamp,
                config_fingerprint="a" * 64,
                event=MacroEventContext(proximity_score=0.3, upcoming_count_24h=1),
                stress=MacroStressContext(score=0.2),
                data_quality=DataQuality.GOOD,
            ),
            hmm=FilteredHMMContext(
                inference=LatentStateInference(
                    as_of_ns=timestamp,
                    state_id=0,
                    state_probabilities=(0.8, 0.2),
                    mode="filtered",
                    retrospective=False,
                    training_end_ns=timestamp - 1,
                    model_version="precomputed-benchmark/1",
                    feature_schema_version="1.0",
                )
            ),
        )
    request = ReplayRequest(
        start_ns=times[0],
        end_ns=times[-1],
        cadence="timestamps",
        timestamps=times,
        max_snapshots=args.snapshots,
    )
    if args.memory:
        tracemalloc.start()
    started = perf_counter()
    result = MarketContextReplay(MarketContextPipeline(config)).run(
        observations, request, supplied.__getitem__
    )
    elapsed = perf_counter() - started
    peak = tracemalloc.get_traced_memory()[1] if args.memory else None
    if args.memory:
        tracemalloc.stop()
    by_stage: dict[str, list[float]] = {}
    for item in result.results:
        for stage in item.diagnostics.stage_results:
            by_stage.setdefault(stage.stage_id, []).append(stage.duration_ns / 1e6)
    ranked = sorted(
        ((stage, mean(durations)) for stage, durations in by_stage.items()),
        key=lambda item: item[1],
        reverse=True,
    )
    print(f"Python {platform.python_version()} on {platform.platform()}")
    print(
        f"symbols={args.symbols} current_bars_per_symbol={args.bars} "
        f"prior_sessions={args.prior_sessions} observations={len(observations)} "
        f"snapshots={len(result.results)}"
    )
    per_snapshot_ms = elapsed / len(result.results) * 1e3
    pipeline_ms = mean(item.diagnostics.duration_ns / 1e6 for item in result.results)
    print(
        f"replay_and_ingest_s={elapsed:.3f} "
        f"replay_per_snapshot_ms={per_snapshot_ms:.1f} pipeline_mean_ms={pipeline_ms:.1f}"
    )
    print("stages_mean_ms=" + ", ".join(f"{stage}:{duration:.1f}" for stage, duration in ranked))
    print(
        "last_stage_status="
        + ", ".join(
            f"{stage.stage_id}:{stage.status.value}"
            for stage in result.results[-1].diagnostics.stage_results
        )
    )
    print("slowest=" + ", ".join(stage for stage, _ in ranked[:3]))
    if peak is not None:
        print(f"tracemalloc_peak_mib={peak / 1024 / 1024:.1f}")


if __name__ == "__main__":
    main()
