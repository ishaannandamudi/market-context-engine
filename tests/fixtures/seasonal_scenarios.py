"""Deterministic U-shaped regular-session bars and quotes."""

from datetime import date
from math import exp

from mce.calendar.service import MarketSessionService
from mce.config.settings import MCEConfig, SeasonalitySettings
from mce.models.market_data import BarObservation, QuoteObservation
from mce.replay.store import CausalDataStore

MINUTE = 60_000_000_000
SYMBOLS = ("AAA", "BBB", "CCC")


def seasonal_config(
    *,
    symbols: tuple[str, ...] = SYMBOLS,
    baseline_sessions: int = 3,
    minimum_history_sessions: int = 2,
    bucket_mode: str = "wall_clock",
    bucket_minutes: int | None = 5,
    session_fraction_bins: int | None = None,
) -> MCEConfig:
    base = MCEConfig(_env_file=None)
    settings = SeasonalitySettings(
        universe=symbols,
        baseline_sessions=baseline_sessions,
        minimum_history_sessions=minimum_history_sessions,
        bucket_mode=bucket_mode,
        bucket_minutes=bucket_minutes,
        session_fraction_bins=session_fraction_bins,
    )
    return base.model_copy(
        update={"features": base.features.model_copy(update={"seasonality": settings})}
    )


def add_session(
    store: CausalDataStore,
    calendar: MarketSessionService,
    day: date,
    *,
    symbols: tuple[str, ...] = SYMBOLS,
    minutes: int | None = None,
    volume_scale: float = 1.0,
    spike_minutes: frozenset[int] = frozenset(),
    missing: frozenset[tuple[str, int]] = frozenset(),
    quote_until: int | None = None,
    crossed_minutes: frozenset[int] = frozenset(),
) -> None:
    opened, closed = calendar.regular_session_bounds(day)
    full = (closed - opened) // MINUTE
    count = min(minutes or full, full)
    for symbol in symbols:
        price = 100.0
        for index in range(count):
            start = opened + index * MINUTE
            end = start + MINUTE
            edge = index < 30 or index >= full - 30
            spike = index in spike_minutes
            volume = (100.0 if edge else 20.0) * volume_scale * (5 if spike else 1)
            change = (0.002 if edge else 0.0005) * (5 if spike else 1)
            close_price = price * exp(change if index % 2 == 0 else -change)
            if (symbol, index) not in missing:
                store.add(
                    BarObservation(
                        symbol=symbol,
                        source="synthetic",
                        event_timestamp_ns=end,
                        effective_timestamp_ns=end,
                        bar_start_ns=start,
                        bar_end_ns=end,
                        open=price,
                        high=max(price, close_price),
                        low=min(price, close_price),
                        close=close_price,
                        volume=volume,
                    )
                )
            if quote_until is None or index < quote_until:
                spread = (0.004 if edge else 0.001) * (5 if spike else 1)
                bid = close_price * (1 - spread / 2)
                ask = close_price * (1 + spread / 2)
                if index in crossed_minutes:
                    bid, ask = ask, bid
                store.add(
                    QuoteObservation(
                        symbol=symbol,
                        source="synthetic",
                        event_timestamp_ns=end,
                        effective_timestamp_ns=end,
                        bid=bid,
                        ask=ask,
                    )
                )
            price = close_price
