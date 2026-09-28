"""Small synthetic records with fixed times and no provider dependencies."""

from mce.models.context import MarketContextSnapshot
from mce.models.market_data import BarObservation, QuoteObservation, TradeObservation

EVENT_NS = 1_700_000_000_123_456_789
EFFECTIVE_NS = EVENT_NS + 100


def sample_bar() -> BarObservation:
    return BarObservation(
        symbol="SPY",
        event_timestamp_ns=EVENT_NS,
        effective_timestamp_ns=EFFECTIVE_NS,
        source="synthetic",
        open=450.0,
        high=451.0,
        low=449.0,
        close=450.5,
        volume=1000.0,
    )


def sample_quote() -> QuoteObservation:
    return QuoteObservation(
        symbol="SPY",
        event_timestamp_ns=EVENT_NS,
        effective_timestamp_ns=EFFECTIVE_NS,
        source="synthetic",
        bid=450.0,
        ask=450.1,
        bid_size=10.0,
        ask_size=12.0,
    )


def sample_trade() -> TradeObservation:
    return TradeObservation(
        symbol="SPY",
        event_timestamp_ns=EVENT_NS,
        effective_timestamp_ns=EFFECTIVE_NS,
        source="synthetic",
        price=450.05,
        size=3.0,
    )


def sample_snapshot() -> MarketContextSnapshot:
    return MarketContextSnapshot(as_of_ns=EFFECTIVE_NS)
