"""Manual 60-session, three-symbol minute-bar volatility benchmark; no CI time gate."""

import platform
from datetime import date
from time import perf_counter

from mce.calendar.service import MarketSessionService
from mce.config.settings import MCEConfig
from mce.features.volatility.engine import RealizedVolatilityEngine
from mce.models.market_data import BarObservation
from mce.replay.store import CausalDataStore

MINUTE = 60_000_000_000


def trading_days(calendar: MarketSessionService, final_day: date, count: int) -> tuple[date, ...]:
    days = [final_day]
    for _ in range(count - 1):
        days.append(calendar.previous_trading_day(days[-1]))
    return tuple(reversed(days))


def main() -> None:
    calendar = MarketSessionService()
    symbols = ("SPY", "QQQ", "IWM")
    days = trading_days(calendar, date(2024, 6, 28), 60)
    store = CausalDataStore(max_observations=100_000)
    for symbol_index, symbol in enumerate(symbols):
        price = 100.0
        for day_index, day in enumerate(days):
            opened, closed = calendar.regular_session_bounds(day)
            minute_count = (closed - opened) // MINUTE
            for minute in range(minute_count):
                start = opened + minute * MINUTE
                movement = (1 if (minute + day_index) % 2 else -1) * (
                    0.0003 + symbol_index * 0.0001
                )
                next_price = price * (1 + movement)
                store.add(
                    BarObservation(
                        symbol=symbol,
                        source="synthetic",
                        bar_start_ns=start,
                        bar_end_ns=start + MINUTE,
                        event_timestamp_ns=start,
                        effective_timestamp_ns=start + MINUTE,
                        open=price,
                        high=max(price, next_price) * 1.0001,
                        low=min(price, next_price) * 0.9999,
                        close=next_price,
                        volume=1.0,
                    )
                )
                price = next_price
    engine = RealizedVolatilityEngine(MCEConfig(_env_file=None), calendar)
    opened, _ = calendar.regular_session_bounds(days[-1])
    times = [opened + minute * MINUTE for minute in range(200, 391, 10)]
    began = perf_counter()
    results = [engine.compute(store, as_of) for as_of in times]
    elapsed = perf_counter() - began
    estimators = engine.config.features.volatility.enabled_estimators
    print(f"Python {platform.python_version()} on {platform.platform()}")
    print(f"{len(days)} sessions; {len(symbols)} symbols; {len(store)} bars")
    print(f"estimators: close_to_close, intraday RV/BV, {', '.join(estimators)}")
    print(
        f"{len(results)} snapshots; {elapsed:.3f} s total; "
        f"{elapsed / len(results) * 1000:.1f} ms/snapshot"
    )


if __name__ == "__main__":
    main()
