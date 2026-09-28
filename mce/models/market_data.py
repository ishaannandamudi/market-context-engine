"""Provider-neutral raw observation contracts; no provider logic belongs here."""

from typing import Annotated

from pydantic import Field, model_validator

from mce.models.quality import NonNegativeFinite, QualityTaggedModel, SourceId
from mce.models.symbol import Symbol
from mce.models.time import TimestampNs

ProviderSequence = Annotated[int, Field(ge=0, strict=True)] | Annotated[str, Field(min_length=1)]


class MarketObservation(QualityTaggedModel):
    """Shared causal timestamps for market observations.

    Effective time is when the observation may enter a specific replay/live calculation.
    It cannot precede the observed event. Historical backfill ingestion may be later.
    """

    symbol: Symbol
    provider_symbol: str | None = None
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
    bar_start_ns: TimestampNs | None = None
    bar_end_ns: TimestampNs | None = None
    open: NonNegativeFinite
    high: NonNegativeFinite
    low: NonNegativeFinite
    close: NonNegativeFinite
    volume: NonNegativeFinite
    vwap: NonNegativeFinite | None = None
    transaction_count: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_bar(self) -> "BarObservation":
        if (self.bar_start_ns is None) != (self.bar_end_ns is None):
            raise ValueError("bar_start_ns and bar_end_ns must be provided together")
        if self.bar_start_ns is not None and self.bar_end_ns is not None:
            if not self.bar_start_ns < self.bar_end_ns:
                raise ValueError("bar_end_ns must be later than bar_start_ns")
            if not self.bar_start_ns <= self.event_timestamp_ns <= self.bar_end_ns:
                raise ValueError("bar event timestamp must lie within bar bounds")
            if self.effective_timestamp_ns < self.bar_end_ns:
                raise ValueError("completed bar cannot be available before bar_end_ns")
        if self.low > self.high:
            raise ValueError("bar low cannot exceed high")
        if not self.low <= self.open <= self.high or not self.low <= self.close <= self.high:
            raise ValueError("open and close must lie within the bar range")
        return self


class QuoteObservation(MarketObservation):
    provider_sequence: ProviderSequence | None = None
    bid: NonNegativeFinite
    ask: NonNegativeFinite
    bid_size: NonNegativeFinite | None = None
    ask_size: NonNegativeFinite | None = None


class TradeObservation(MarketObservation):
    provider_sequence: ProviderSequence | None = None
    price: NonNegativeFinite
    size: NonNegativeFinite
