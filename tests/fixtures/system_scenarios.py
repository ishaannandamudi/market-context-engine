"""Completed regular-session bars for MCE-20 integration tests and profiling."""

from datetime import date

from mce.calendar.service import MarketSessionService
from mce.config.settings import MCEConfig
from mce.models.market_data import BarObservation


def bars(
    config: MCEConfig,
    *,
    symbols: tuple[str, ...] = ("SPY", "AAA", "BBB"),
    count: int = 30,
    day: date = date(2025, 1, 2),
    direction: float = 1.0,
) -> tuple[BarObservation, ...]:
    opened, _ = MarketSessionService(config.market).regular_session_bounds(day)
    result = []
    for minute in range(1, count + 1):
        start = opened + (minute - 1) * 60_000_000_000
        end = start + 60_000_000_000
        for index, symbol in enumerate(symbols):
            price = 100 + index * 10 + direction * minute * (0.05 + index * 0.01)
            result.append(
                BarObservation(
                    symbol=symbol,
                    event_timestamp_ns=end,
                    effective_timestamp_ns=end,
                    source=config.provider.source,
                    bar_start_ns=start,
                    bar_end_ns=end,
                    open=price - 0.02,
                    high=price + 0.05,
                    low=price - 0.05,
                    close=price,
                    volume=1000 + minute * 10,
                )
            )
    return tuple(result)
