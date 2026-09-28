"""Fixed session observations; no stochastic price or market scenario generation."""

from mce.models.market_data import BarObservation, QuoteObservation, TradeObservation
from tests.fixtures.clock import utc_ns

OPEN_NS = utc_ns("2024-01-03T14:30:00Z")
MINUTE_NS = 60_000_000_000


def minute_bar(
    *,
    symbol: str = "SPY",
    start_ns: int = OPEN_NS,
    effective_delay_ns: int = 0,
    close: float = 100.5,
    volume: float = 10.0,
) -> BarObservation:
    return BarObservation(
        symbol=symbol,
        source="synthetic",
        bar_start_ns=start_ns,
        bar_end_ns=start_ns + MINUTE_NS,
        event_timestamp_ns=start_ns,
        effective_timestamp_ns=start_ns + MINUTE_NS + effective_delay_ns,
        open=100.0,
        high=101.0,
        low=99.0,
        close=close,
        volume=volume,
    )


def quote(*, symbol: str = "SPY", event_ns: int = OPEN_NS) -> QuoteObservation:
    return QuoteObservation(
        symbol=symbol,
        source="synthetic",
        event_timestamp_ns=event_ns,
        effective_timestamp_ns=event_ns + 2,
        bid=100.0,
        ask=100.1,
    )


def trade(*, symbol: str = "SPY", event_ns: int = OPEN_NS, sequence: int = 1) -> TradeObservation:
    return TradeObservation(
        symbol=symbol,
        source="synthetic",
        event_timestamp_ns=event_ns,
        effective_timestamp_ns=event_ns + 2,
        price=100.05,
        size=1.0,
        provider_sequence=sequence,
    )
