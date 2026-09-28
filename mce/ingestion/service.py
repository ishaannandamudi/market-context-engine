"""Canonical-only ingestion, session annotation, and provider capability gating."""

from collections.abc import Iterable, Sequence

from pydantic import Field

from mce.calendar.service import MarketSessionService
from mce.config.settings import IngestionSettings
from mce.errors import (
    ProviderEntitlementError,
    ProviderError,
    ProviderMalformedDataError,
    ProviderUnavailableError,
)
from mce.logging import get_logger
from mce.models.enums import MarketSession, QualityFlag
from mce.models.market_data import BarObservation, QuoteObservation, TradeObservation
from mce.models.quality import QualityTaggedModel
from mce.models.symbol import normalize_symbol
from mce.models.time import TimestampNs
from mce.providers.base import (
    HistoricalMarketDataProvider,
    HistoricalQuoteProvider,
    HistoricalTradeProvider,
    MarketDataProvider,
    ProviderCapability,
    ProviderHealth,
    ProviderStatus,
)
from mce.replay.store import CausalDataStore, Observation, observation_order_key

_logger = get_logger("ingestion")


class IngestionResult(QualityTaggedModel):
    accepted_count: int = Field(ge=0)
    rejected_count: int = Field(ge=0)
    duplicate_count: int = Field(ge=0)
    symbols: tuple[str, ...] = ()
    min_effective_timestamp_ns: TimestampNs | None = None
    max_effective_timestamp_ns: TimestampNs | None = None


class IngestionService:
    def __init__(
        self,
        store: CausalDataStore,
        calendar: MarketSessionService,
        settings: IngestionSettings | None = None,
    ) -> None:
        self.store = store
        self.calendar = calendar
        self.settings = settings or IngestionSettings()

    def ingest(self, observations: Iterable[Observation]) -> IngestionResult:
        """Sort batch deterministically; reject unsafe bars and disallowed sessions."""
        accepted = rejected = duplicates = 0
        symbols: set[str] = set()
        effective_times: list[int] = []
        flags: set[QualityFlag] = set()
        for observation in sorted(observations, key=observation_order_key):
            bar_last_ns: int | None = None
            if isinstance(observation, BarObservation):
                if observation.bar_start_ns is None or observation.bar_end_ns is None:
                    rejected += 1
                    flags.update({QualityFlag.REJECTED_DATA, QualityFlag.TIMESTAMP_MISMATCH})
                    continue
                classification_ns = observation.bar_start_ns
                bar_last_ns = observation.bar_end_ns - 1
            else:
                classification_ns = observation.event_timestamp_ns
            info = self.calendar.session_for_timestamp(classification_ns)
            crosses_session = False
            end_session = info.session
            if bar_last_ns is not None:
                end_session = self.calendar.session_for_timestamp(bar_last_ns).session
                crosses_session = end_session != info.session
            if (
                info.session in (MarketSession.PRE_MARKET, MarketSession.AFTER_HOURS)
                or end_session in (MarketSession.PRE_MARKET, MarketSession.AFTER_HOURS)
            ) and not self.settings.allow_extended_hours:
                rejected += 1
                flags.update({QualityFlag.REJECTED_DATA, QualityFlag.OUTSIDE_SESSION})
                continue
            added_flags = set(observation.quality_flags)
            if crosses_session:
                added_flags.add(QualityFlag.TIMESTAMP_MISMATCH)
                flags.add(QualityFlag.TIMESTAMP_MISMATCH)
            if info.session == MarketSession.CLOSED:
                added_flags.add(QualityFlag.OUTSIDE_SESSION)
                flags.add(QualityFlag.OUTSIDE_SESSION)
            elif info.session == MarketSession.UNKNOWN:
                added_flags.add(QualityFlag.SESSION_UNKNOWN)
                flags.add(QualityFlag.SESSION_UNKNOWN)
            if added_flags != observation.quality_flags:
                observation = type(observation).model_validate(
                    {
                        **observation.model_dump(mode="python"),
                        "quality_flags": added_flags,
                    }
                )
            if self.store.add(observation):
                accepted += 1
                symbols.add(observation.symbol)
                effective_times.append(observation.effective_timestamp_ns)
            else:
                duplicates += 1
        if accepted == 0 and duplicates == 0:
            flags.add(QualityFlag.MISSING_DATA)
        result = IngestionResult(
            accepted_count=accepted,
            rejected_count=rejected,
            duplicate_count=duplicates,
            symbols=tuple(sorted(symbols)),
            min_effective_timestamp_ns=min(effective_times) if effective_times else None,
            max_effective_timestamp_ns=max(effective_times) if effective_times else None,
            quality_flags=frozenset(flags),
        )
        _logger.debug(
            "ingestion batch accepted=%d rejected=%d duplicates=%d symbols=%d",
            accepted,
            rejected,
            duplicates,
            len(symbols),
        )
        return result

    def ingest_provider(
        self,
        provider: MarketDataProvider,
        symbols: Sequence[str],
        start_ns: int,
        end_ns: int,
        requested_capabilities: frozenset[ProviderCapability] | None = None,
    ) -> IngestionResult:
        """Read only declared historical capabilities; absent ones degrade locally."""
        if start_ns < 0 or end_ns < start_ns:
            raise ValueError("invalid historical range")
        try:
            health = provider.health()
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError("provider health check failed") from exc
        if not isinstance(health, ProviderHealth):
            raise ProviderMalformedDataError("provider health result is not canonical")
        try:
            source = provider.provider_name
            capabilities = provider.capabilities
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError("provider metadata request failed") from exc
        if not isinstance(capabilities, frozenset) or any(
            not isinstance(capability, ProviderCapability) for capability in capabilities
        ):
            raise ProviderMalformedDataError("provider capability metadata is not canonical")
        if health.status == ProviderStatus.UNAVAILABLE:
            raise ProviderUnavailableError(f"{source} is unavailable")
        records: list[Observation] = []
        expected_symbols = {normalize_symbol(symbol) for symbol in symbols}
        selected_capabilities = (
            capabilities & requested_capabilities
            if requested_capabilities is not None
            else capabilities
        )
        for symbol in sorted(expected_symbols):
            try:
                if ProviderCapability.HISTORICAL_BARS in selected_capabilities:
                    if not isinstance(provider, HistoricalMarketDataProvider):
                        raise ProviderEntitlementError("historical bars declared without get_bars")
                    bars = tuple(provider.get_bars(symbol, start_ns, end_ns))
                    if any(not isinstance(item, BarObservation) for item in bars):
                        raise ProviderMalformedDataError("get_bars returned non-bar data")
                    records.extend(bars)
                if ProviderCapability.QUOTES in selected_capabilities:
                    if not isinstance(provider, HistoricalQuoteProvider):
                        raise ProviderEntitlementError("quotes declared without get_quotes")
                    quotes = tuple(provider.get_quotes(symbol, start_ns, end_ns))
                    if any(not isinstance(item, QuoteObservation) for item in quotes):
                        raise ProviderMalformedDataError("get_quotes returned non-quote data")
                    records.extend(quotes)
                if ProviderCapability.TRADES in selected_capabilities:
                    if not isinstance(provider, HistoricalTradeProvider):
                        raise ProviderEntitlementError("trades declared without get_trades")
                    trades = tuple(provider.get_trades(symbol, start_ns, end_ns))
                    if any(not isinstance(item, TradeObservation) for item in trades):
                        raise ProviderMalformedDataError("get_trades returned non-trade data")
                    records.extend(trades)
            except ProviderError:
                raise
            except Exception as exc:
                raise ProviderError(f"{source} historical request failed") from exc
        if any(record.source != source for record in records):
            raise ProviderMalformedDataError("provider returned a mismatched source identifier")
        result = self.ingest(records)
        extra_flags = set(result.quality_flags)
        if health.status != ProviderStatus.HEALTHY:
            extra_flags.add(QualityFlag.PROVIDER_DEGRADED)
        if requested_capabilities is not None and not requested_capabilities.issubset(capabilities):
            extra_flags.add(QualityFlag.PROVIDER_DEGRADED)
        if not selected_capabilities:
            extra_flags.add(QualityFlag.MISSING_DATA)
        if result.symbols and not expected_symbols.issubset(result.symbols):
            extra_flags.update({QualityFlag.PARTIAL_UNIVERSE, QualityFlag.LOW_COVERAGE})
        if extra_flags != result.quality_flags:
            result = IngestionResult.model_validate(
                {
                    **result.model_dump(mode="python"),
                    "quality_flags": extra_flags,
                }
            )
        return result
