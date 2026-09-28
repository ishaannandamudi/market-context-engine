"""Deterministic, credential-free historical reference provider."""

from collections import defaultdict
from collections.abc import Sequence
from typing import TypeVar

from pydantic import TypeAdapter

from mce.errors import ProviderEntitlementError, ProviderError, ProviderUnavailableError
from mce.models.market_data import (
    BarObservation,
    MarketObservation,
    QuoteObservation,
    TradeObservation,
)
from mce.models.quality import SourceId
from mce.models.symbol import normalize_symbol
from mce.providers.base import ProviderCapability, ProviderHealth, ProviderStatus
from mce.serialization.canonical import canonical_json_bytes

_SOURCE_ADAPTER: TypeAdapter[str] = TypeAdapter(SourceId)
ObservationT = TypeVar("ObservationT", bound=MarketObservation)


class InMemoryHistoricalProvider:
    """Serve fixed canonical records; optional failure/capabilities support tests."""

    def __init__(
        self,
        *,
        source: str = "synthetic",
        bars: Sequence[BarObservation] = (),
        quotes: Sequence[QuoteObservation] = (),
        trades: Sequence[TradeObservation] = (),
        capabilities: frozenset[ProviderCapability] | None = None,
        status: ProviderStatus = ProviderStatus.HEALTHY,
        failure: ProviderError | None = None,
    ) -> None:
        self._source = _SOURCE_ADAPTER.validate_python(source)
        self._capabilities = (
            capabilities
            if capabilities is not None
            else frozenset(
                {
                    ProviderCapability.HISTORICAL_BARS,
                    ProviderCapability.QUOTES,
                    ProviderCapability.TRADES,
                }
            )
        )
        self._status = status
        self._failure = failure
        for observation in (*bars, *quotes, *trades):
            if observation.source != source:
                raise ValueError("observation source differs from provider source")
        self._bars = self._group(bars)
        self._quotes = self._group(quotes)
        self._trades = self._group(trades)

    @staticmethod
    def _order(item: MarketObservation) -> tuple[int, int, bytes]:
        return (item.event_timestamp_ns, item.effective_timestamp_ns, canonical_json_bytes(item))

    @classmethod
    def _group(cls, observations: Sequence[ObservationT]) -> dict[str, tuple[ObservationT, ...]]:
        grouped: defaultdict[str, list[ObservationT]] = defaultdict(list)
        for item in observations:
            grouped[item.symbol].append(item)
        return {symbol: tuple(sorted(items, key=cls._order)) for symbol, items in grouped.items()}

    @property
    def provider_name(self) -> str:
        return self._source

    @property
    def capabilities(self) -> frozenset[ProviderCapability]:
        return self._capabilities

    def health(self) -> ProviderHealth:
        return ProviderHealth(status=self._status, capabilities=self._capabilities)

    def _check(self, capability: ProviderCapability) -> None:
        if self._status == ProviderStatus.UNAVAILABLE:
            raise ProviderUnavailableError(f"{self._source} is unavailable")
        if capability not in self._capabilities:
            raise ProviderEntitlementError(f"{capability.value} is not available")
        if self._failure is not None:
            raise self._failure

    @staticmethod
    def _select(
        observations: dict[str, tuple[ObservationT, ...]],
        symbol: str,
        start_ns: int,
        end_ns: int,
    ) -> tuple[ObservationT, ...]:
        if start_ns < 0 or end_ns < start_ns:
            raise ValueError("invalid historical range")
        normalized = normalize_symbol(symbol)
        return tuple(
            item
            for item in observations.get(normalized, ())
            if start_ns <= item.event_timestamp_ns < end_ns
        )

    def get_bars(self, symbol: str, start_ns: int, end_ns: int) -> Sequence[BarObservation]:
        self._check(ProviderCapability.HISTORICAL_BARS)
        return self._select(self._bars, symbol, start_ns, end_ns)

    def get_quotes(self, symbol: str, start_ns: int, end_ns: int) -> Sequence[QuoteObservation]:
        self._check(ProviderCapability.QUOTES)
        return self._select(self._quotes, symbol, start_ns, end_ns)

    def get_trades(self, symbol: str, start_ns: int, end_ns: int) -> Sequence[TradeObservation]:
        self._check(ProviderCapability.TRADES)
        return self._select(self._trades, symbol, start_ns, end_ns)
