from datetime import UTC, date, time

import pytest
from pydantic import ValidationError

from mce.calendar import MarketSessionService
from mce.config.settings import (
    FreshnessSettings,
    IngestionSettings,
    MarketSettings,
    MCEConfig,
    ReplaySettings,
    UniverseSettings,
)
from mce.errors import (
    ProviderEntitlementError,
    ProviderError,
    ProviderMalformedDataError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from mce.ingestion import IngestionService
from mce.models.enums import BarInterval, DataQuality, QualityFlag
from mce.models.market_data import BarObservation
from mce.providers import (
    HistoricalMarketDataProvider,
    InMemoryHistoricalProvider,
    ProviderCapability,
    ProviderStatus,
)
from mce.providers.base import ProviderHealth
from mce.quality import assess_quality, evaluate_freshness
from mce.replay import CausalDataStore
from mce.reproducibility import config_fingerprint
from mce.serialization import canonical_json_str
from tests.fixtures.session_samples import MINUTE_NS, OPEN_NS, minute_bar, quote, trade


def test_freshness_as_of_boundary_and_missing() -> None:
    assert evaluate_freshness("synthetic", None, 100, 10) is None
    fresh = evaluate_freshness("synthetic", 91, 100, 10)
    boundary = evaluate_freshness("synthetic", 90, 100, 10)
    stale = evaluate_freshness("synthetic", 89, 100, 10)
    assert fresh is not None and fresh.age_ns == 9 and not fresh.is_stale
    assert boundary is not None and boundary.age_ns == 10 and not boundary.is_stale
    assert stale is not None and stale.age_ns == 11 and stale.is_stale
    assert type(stale).model_validate_json(canonical_json_str(stale)) == stale
    with pytest.raises(ValueError):
        evaluate_freshness("synthetic", 101, 100, 10)
    with pytest.raises(ValueError):
        evaluate_freshness("synthetic", 1, 100, 0)


def test_quality_availability_staleness_and_degradation() -> None:
    no_expectation = assess_quality(expected_symbols=(), observed_symbols=())
    no_data = assess_quality(expected_symbols=("SPY",), observed_symbols=())
    assert no_expectation.data_quality == DataQuality.UNKNOWN
    assert no_data.data_quality == DataQuality.UNUSABLE
    assert QualityFlag.MISSING_DATA in no_data.quality_flags
    good = assess_quality(expected_symbols=("spy",), observed_symbols=("SPY",))
    assert good.data_quality == DataQuality.GOOD and not good.quality_flags
    stale = evaluate_freshness("synthetic", 0, 100, 10)
    assert stale is not None
    degraded = assess_quality(
        expected_symbols=("SPY", "QQQ"),
        observed_symbols=("SPY",),
        freshness=(stale,),
        expected_sources=("synthetic", "file"),
        provider_statuses={"synthetic": ProviderStatus.DEGRADED},
        rejected_count=1,
        insufficient_history=True,
        timestamp_mismatch=True,
    )
    assert degraded.data_quality == DataQuality.DEGRADED
    assert degraded.quality_flags >= {
        QualityFlag.PARTIAL_UNIVERSE,
        QualityFlag.STALE_DATA,
        QualityFlag.PROVIDER_DEGRADED,
        QualityFlag.REJECTED_DATA,
        QualityFlag.INSUFFICIENT_HISTORY,
        QualityFlag.TIMESTAMP_MISMATCH,
    }
    assert canonical_json_str(degraded) == canonical_json_str(
        type(degraded).model_validate_json(canonical_json_str(degraded))
    )
    source_gap = assess_quality(
        expected_symbols=("SPY",),
        observed_symbols=("SPY",),
        expected_sources=("synthetic", "file"),
        provider_statuses={"synthetic": ProviderStatus.HEALTHY},
    )
    assert source_gap.data_quality == DataQuality.DEGRADED
    assert QualityFlag.PROVIDER_DEGRADED in source_gap.quality_flags
    with pytest.raises(ValueError):
        assess_quality(expected_symbols=("SPY",), observed_symbols=("SPY",), rejected_count=-1)


def test_config_validation_and_fingerprint_extensions() -> None:
    baseline = MCEConfig(_env_file=None)
    assert baseline.market.calendar_name == "XNYS"
    assert baseline.market.universe.broad_market == ("SPY", "QQQ", "IWM")
    assert baseline.ingestion.default_bar_interval == BarInterval.ONE_MINUTE
    variants = (
        MCEConfig(_env_file=None, market=MarketSettings(calendar_end=date(2036, 1, 1))),
        MCEConfig(_env_file=None, ingestion=IngestionSettings(allow_extended_hours=False)),
        MCEConfig(_env_file=None, freshness=FreshnessSettings(bar_limit_ns=1)),
        MCEConfig(_env_file=None, replay=ReplaySettings(max_observations=2)),
    )
    assert all(config_fingerprint(item) != config_fingerprint(baseline) for item in variants)
    assert baseline.engine.log_level == "INFO"
    with pytest.raises(ValidationError):
        MarketSettings(calendar_name="FAKE")
    with pytest.raises(ValidationError):
        MarketSettings(calendar_start=date(2025, 1, 1), calendar_end=date(2024, 1, 1))
    with pytest.raises(ValidationError):
        MarketSettings(pre_market_start_local=time(21), after_hours_end_local=time(20))
    with pytest.raises(ValidationError):
        UniverseSettings(broad_market=("spy", " SPY "))
    with pytest.raises(ValidationError):
        ReplaySettings(max_observations=0)
    sectors = {"technology": " xlk "}
    universe = UniverseSettings(sectors=sectors)
    sectors["technology"] = "XLV"
    assert universe.sectors["technology"] == "XLK"
    with pytest.raises(TypeError):
        universe.sectors["technology"] = "XLV"
    with pytest.raises(ValidationError):
        MarketSettings(pre_market_start_local=time(4, tzinfo=UTC))
    assert config_fingerprint(baseline) == config_fingerprint(MCEConfig(_env_file=None))


def test_nested_universe_environment_and_mapping_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MCE_MARKET__UNIVERSE__BROAD_MARKET", '["dia", " spy "]')
    monkeypatch.setenv("MCE_INGESTION__DEFAULT_BAR_INTERVAL", "5m")
    configured = MCEConfig(_env_file=None)
    assert configured.market.universe.broad_market == ("DIA", "SPY")
    assert configured.ingestion.default_bar_interval == BarInterval.FIVE_MINUTES
    first = MCEConfig(
        _env_file=None,
        market=MarketSettings(
            universe=UniverseSettings(sectors={"technology": "XLK", "health": "XLV"})
        ),
    )
    second = MCEConfig(
        _env_file=None,
        market=MarketSettings(
            universe=UniverseSettings(sectors={"health": "XLV", "technology": "XLK"})
        ),
    )
    assert config_fingerprint(first) == config_fingerprint(second)


def test_in_memory_provider_capabilities_missing_and_failures(
    calendar: MarketSessionService,
) -> None:
    bar = minute_bar()
    provider = InMemoryHistoricalProvider(bars=(bar,), quotes=(quote(),), trades=(trade(),))
    assert isinstance(provider, HistoricalMarketDataProvider)
    assert provider.get_bars(" spy ", OPEN_NS, OPEN_NS + 1) == (bar,)
    service = IngestionService(CausalDataStore(), calendar)
    result = service.ingest_provider(provider, ("SPY", "IWM"), OPEN_NS, OPEN_NS + MINUTE_NS)
    assert result.accepted_count == 3
    assert result.symbols == ("SPY",)
    assert QualityFlag.PARTIAL_UNIVERSE in result.quality_flags
    bars_only = InMemoryHistoricalProvider(
        bars=(bar,), capabilities=frozenset({ProviderCapability.HISTORICAL_BARS})
    )
    assert (
        IngestionService(CausalDataStore(), calendar)
        .ingest_provider(bars_only, ("SPY",), OPEN_NS, OPEN_NS + 1)
        .accepted_count
        == 1
    )
    missing_requested = IngestionService(CausalDataStore(), calendar).ingest_provider(
        bars_only,
        ("SPY",),
        OPEN_NS,
        OPEN_NS + 1,
        requested_capabilities=frozenset(
            {ProviderCapability.HISTORICAL_BARS, ProviderCapability.QUOTES}
        ),
    )
    assert QualityFlag.PROVIDER_DEGRADED in missing_requested.quality_flags
    with pytest.raises(ProviderEntitlementError):
        bars_only.get_quotes("SPY", OPEN_NS, OPEN_NS + 1)
    empty_capabilities = InMemoryHistoricalProvider(capabilities=frozenset())
    empty = IngestionService(CausalDataStore(), calendar).ingest_provider(
        empty_capabilities, ("SPY",), OPEN_NS, OPEN_NS + 1
    )
    assert empty.accepted_count == 0
    assert QualityFlag.MISSING_DATA in empty.quality_flags
    unavailable = InMemoryHistoricalProvider(status=ProviderStatus.UNAVAILABLE)
    with pytest.raises(ProviderUnavailableError):
        service.ingest_provider(unavailable, ("SPY",), OPEN_NS, OPEN_NS + 1)
    with pytest.raises(ProviderUnavailableError):
        unavailable.get_bars("SPY", OPEN_NS, OPEN_NS + 1)
    failing = InMemoryHistoricalProvider(failure=ProviderTimeoutError("timeout"))
    with pytest.raises(ProviderTimeoutError):
        service.ingest_provider(failing, ("SPY",), OPEN_NS, OPEN_NS + 1)


def test_provider_degraded_and_source_mismatch(calendar: MarketSessionService) -> None:
    provider = InMemoryHistoricalProvider(
        bars=(minute_bar(),),
        status=ProviderStatus.DEGRADED,
    )
    result = IngestionService(CausalDataStore(), calendar).ingest_provider(
        provider, ("SPY",), OPEN_NS, OPEN_NS + 1
    )
    assert QualityFlag.PROVIDER_DEGRADED in result.quality_flags
    with pytest.raises(ValueError, match="source differs"):
        InMemoryHistoricalProvider(source="file", bars=(minute_bar(),))


class _IncompleteProvider:
    provider_name = "synthetic"
    capabilities = frozenset({ProviderCapability.HISTORICAL_BARS})

    def health(self) -> ProviderHealth:
        return ProviderHealth(status=ProviderStatus.HEALTHY)


class _BrokenProvider(_IncompleteProvider):
    def get_bars(self, symbol: str, start_ns: int, end_ns: int) -> tuple[()]:
        raise RuntimeError("vendor failure")


class _BrokenHealthProvider(_IncompleteProvider):
    def health(self) -> ProviderHealth:
        raise RuntimeError("vendor health failure")


class _WrongBarsProvider(_IncompleteProvider):
    def get_bars(self, symbol: str, start_ns: int, end_ns: int) -> tuple[object, ...]:
        return (quote(),)


class _WrongSourceProvider(_IncompleteProvider):
    def get_bars(self, symbol: str, start_ns: int, end_ns: int) -> tuple[BarObservation, ...]:
        record = BarObservation.model_validate({**minute_bar().model_dump(), "source": "file"})
        return (record,)


class _BadHealthProvider(_IncompleteProvider):
    def health(self) -> object:
        return None


class _TypedHealthFailure(_IncompleteProvider):
    def health(self) -> ProviderHealth:
        raise ProviderTimeoutError("health timeout")


class _MetadataFailure(_IncompleteProvider):
    @property
    def provider_name(self) -> str:
        raise RuntimeError("metadata failure")


class _TypedMetadataFailure(_IncompleteProvider):
    @property
    def provider_name(self) -> str:
        raise ProviderTimeoutError("metadata timeout")


class _MalformedCapabilities(_IncompleteProvider):
    capabilities = frozenset({"historical_bars"})


class _MissingQuoteProvider(_IncompleteProvider):
    capabilities = frozenset({ProviderCapability.QUOTES})


class _WrongQuoteProvider(_MissingQuoteProvider):
    def get_quotes(self, symbol: str, start_ns: int, end_ns: int) -> tuple[object, ...]:
        return (minute_bar(),)


class _MissingTradeProvider(_IncompleteProvider):
    capabilities = frozenset({ProviderCapability.TRADES})


class _WrongTradeProvider(_MissingTradeProvider):
    def get_trades(self, symbol: str, start_ns: int, end_ns: int) -> tuple[object, ...]:
        return (quote(),)


def test_provider_boundary_wraps_errors_and_claimed_capability(
    calendar: MarketSessionService,
) -> None:
    service = IngestionService(CausalDataStore(), calendar)
    with pytest.raises(ProviderEntitlementError):
        service.ingest_provider(_IncompleteProvider(), ("SPY",), OPEN_NS, OPEN_NS + 1)
    with pytest.raises(ProviderError, match="historical request failed"):
        service.ingest_provider(_BrokenProvider(), ("SPY",), OPEN_NS, OPEN_NS + 1)
    with pytest.raises(ProviderError, match="health check failed"):
        service.ingest_provider(_BrokenHealthProvider(), ("SPY",), OPEN_NS, OPEN_NS + 1)
    with pytest.raises(ProviderMalformedDataError, match="non-bar"):
        service.ingest_provider(_WrongBarsProvider(), ("SPY",), OPEN_NS, OPEN_NS + 1)
    with pytest.raises(ProviderMalformedDataError, match="mismatched source"):
        service.ingest_provider(_WrongSourceProvider(), ("SPY",), OPEN_NS, OPEN_NS + 1)
    with pytest.raises(ProviderMalformedDataError, match="health result"):
        service.ingest_provider(_BadHealthProvider(), ("SPY",), OPEN_NS, OPEN_NS + 1)


@pytest.mark.parametrize(
    "provider,error,match",
    [
        (_TypedHealthFailure(), ProviderTimeoutError, "health timeout"),
        (_MetadataFailure(), ProviderError, "metadata request failed"),
        (_TypedMetadataFailure(), ProviderTimeoutError, "metadata timeout"),
        (_MalformedCapabilities(), ProviderMalformedDataError, "capability metadata"),
        (_MissingQuoteProvider(), ProviderEntitlementError, "get_quotes"),
        (_WrongQuoteProvider(), ProviderMalformedDataError, "non-quote"),
        (_MissingTradeProvider(), ProviderEntitlementError, "get_trades"),
        (_WrongTradeProvider(), ProviderMalformedDataError, "non-trade"),
    ],
)
def test_provider_metadata_and_optional_capabilities_fail_clearly(
    calendar: MarketSessionService,
    provider: object,
    error: type[Exception],
    match: str,
) -> None:
    service = IngestionService(CausalDataStore(), calendar)
    with pytest.raises(error, match=match):
        service.ingest_provider(provider, ("SPY",), OPEN_NS, OPEN_NS + 1)


def test_provider_request_rejects_invalid_range(calendar: MarketSessionService) -> None:
    service = IngestionService(CausalDataStore(), calendar)
    with pytest.raises(ValueError, match="invalid historical range"):
        service.ingest_provider(InMemoryHistoricalProvider(), ("SPY",), 2, 1)
