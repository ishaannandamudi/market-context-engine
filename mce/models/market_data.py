"""Provider-neutral raw observation contracts; no provider logic belongs here."""

from pydantic import Field, model_validator

from mce.models.quality import NonNegativeFinite, QualityTaggedModel, SourceId
from mce.models.time import TimestampNs


class MarketObservation(QualityTaggedModel):
    """Shared causal timestamps for market observations.

    Effective time is when the observation may enter a specific replay/live calculation.
    It cannot precede the observed event. Historical backfill ingestion may be later.
    """

    symbol: str = Field(min_length=1)
    event_timestamp_ns: TimestampNs
    effective_timestamp_ns: TimestampNs
    provider_timestamp_ns: TimestampNs | None = None
    ingestion_timestamp_ns: TimestampNs | None = None
    source: SourceId

    @model_validator(mode="after")
    def validate_causal_order(self) -> "MarketObservation":
        if self.effective_timestamp_ns < self.event_timestamp_ns:
            raise ValueError("effective_timestamp_ns cannot precede event_timestamp_ns")
        if (
            self.provider_timestamp_ns is not None
            and self.effective_timestamp_ns < self.provider_timestamp_ns
        ):
            raise ValueError("effective_timestamp_ns cannot precede provider_timestamp_ns")
        return self


class BarObservation(MarketObservation):
    open: NonNegativeFinite
    high: NonNegativeFinite
    low: NonNegativeFinite
    close: NonNegativeFinite
    volume: NonNegativeFinite
    vwap: NonNegativeFinite | None = None
    transaction_count: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_bar(self) -> "BarObservation":
        if self.low > self.high:
            raise ValueError("bar low cannot exceed high")
        if not self.low <= self.open <= self.high or not self.low <= self.close <= self.high:
            raise ValueError("open and close must lie within the bar range")
        return self


class QuoteObservation(MarketObservation):
    bid: NonNegativeFinite
    ask: NonNegativeFinite
    bid_size: NonNegativeFinite | None = None
    ask_size: NonNegativeFinite | None = None


class TradeObservation(MarketObservation):
    price: NonNegativeFinite
    size: NonNegativeFinite
