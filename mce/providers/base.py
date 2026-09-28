"""Provider-specific behavior stops at these adapter protocols."""

from collections.abc import AsyncIterator, Sequence
from enum import StrEnum
from typing import Protocol, runtime_checkable

from pydantic import Field, field_serializer

from mce.models._base import FrozenModel
from mce.models.market_data import BarObservation
from mce.models.quality import SourceId
from mce.models.time import TimestampNs


class ProviderCapability(StrEnum):
    HISTORICAL_BARS = "historical_bars"
    LIVE_BARS = "live_bars"
    TRADES = "trades"
    QUOTES = "quotes"
    BROAD_MARKET_DATA = "broad_market_data"
    OPTIONS_CONTEXT = "options_context"
    EVENT_DATA = "event_data"


class ProviderStatus(StrEnum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"
    UNKNOWN = "unknown"


class ProviderHealth(FrozenModel):
    status: ProviderStatus = ProviderStatus.UNKNOWN
    checked_at_ns: TimestampNs | None = None
    detail: str | None = None
    capabilities: frozenset[ProviderCapability] = Field(default_factory=frozenset)

    @field_serializer("capabilities")
    def serialize_capabilities(self, value: frozenset[ProviderCapability]) -> list[str]:
        return sorted(capability.value for capability in value)


@runtime_checkable
class MarketDataProvider(Protocol):
    @property
    def provider_name(self) -> SourceId: ...

    @property
    def capabilities(self) -> frozenset[ProviderCapability]: ...

    def health(self) -> ProviderHealth: ...


@runtime_checkable
class HistoricalMarketDataProvider(MarketDataProvider, Protocol):
    def get_bars(self, symbol: str, start_ns: int, end_ns: int) -> Sequence[BarObservation]: ...


@runtime_checkable
class LiveMarketDataProvider(MarketDataProvider, Protocol):
    def subscribe_bars(self, symbols: Sequence[str]) -> AsyncIterator[BarObservation]: ...
