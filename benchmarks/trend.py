"""Manual local benchmark: 3 symbols, 390 minute bars, 20 five-minute snapshots."""

import platform
from datetime import date
from time import perf_counter

from mce.calendar.service import MarketSessionService
from mce.config.settings import MCEConfig
from mce.features.engine import TrendFeatureEngine
from mce.models.market_data import BarObservation
from mce.replay.store import CausalDataStore

MINUTE = 60_000_000_000


def main() -> None:
    calendar = MarketSessionService()
    opened, _ = calendar.regular_session_bounds(date(2024, 1, 3))
    store = CausalDataStore()
    symbols = ("SPY", "QQQ", "IWM")
    for symbol in symbols:
        for index in range(390):
            start = opened + index * MINUTE
            close = 100 + 0.01 * index
            store.add(
                BarObservation(
                    symbol=symbol,
                    source="synthetic",
                    bar_start_ns=start,
                    bar_end_ns=start + MINUTE,
                    event_timestamp_ns=start,
                    effective_timestamp_ns=start + MINUTE,
                    open=close,
                    high=close,
                    low=close,
                    close=close,
                    volume=0.0,
                )
            )
    engine = TrendFeatureEngine(MCEConfig(_env_file=None), calendar)
    times = [opened + minute * MINUTE for minute in range(295, 391, 5)]
    began = perf_counter()
    results = [engine.compute(store, as_of) for as_of in times]
    elapsed = perf_counter() - began
    print(f"Python {platform.python_version()} on {platform.platform()}")
    print(
        f"{len(store)} bars; {len(results)} snapshots; {elapsed:.3f} s total; "
        f"{elapsed / len(results) * 1000:.1f} ms/snapshot"
    )


if __name__ == "__main__":
    main()
