from collections.abc import Sequence

from mce.models.market_data import BarObservation
from mce.providers import (
    HistoricalMarketDataProvider,
    MarketDataProvider,
    ProviderCapability,
    ProviderHealth,
    ProviderStatus,
)
from mce.serialization import canonical_json_str
from tests.fixtures.samples import sample_bar


class FakeHistoricalProvider:
    @property
    def provider_name(self) -> str:
        return "synthetic"

    @property
    def capabilities(self) -> frozenset[ProviderCapability]:
        return frozenset({ProviderCapability.HISTORICAL_BARS})

    def health(self) -> ProviderHealth:
        return ProviderHealth(
            status=ProviderStatus.HEALTHY,
            capabilities=self.capabilities,
            checked_at_ns=1,
        )

    def get_bars(self, symbol: str, start_ns: int, end_ns: int) -> Sequence[BarObservation]:
        bar = sample_bar()
        if bar.symbol == symbol and start_ns <= bar.event_timestamp_ns < end_ns:
            return (bar,)
        return ()


def test_fake_provider_satisfies_contract() -> None:
    provider = FakeHistoricalProvider()
    assert isinstance(provider, MarketDataProvider)
    assert isinstance(provider, HistoricalMarketDataProvider)
    assert len(provider.get_bars("SPY", 0, 2_000_000_000_000_000_000)) == 1
    assert provider.health().status == ProviderStatus.HEALTHY


def test_capability_metadata_serializes() -> None:
    health = ProviderHealth(
        capabilities=frozenset(
            {
                ProviderCapability.TRADES,
                ProviderCapability.HISTORICAL_BARS,
            }
        )
    )
    assert canonical_json_str(health) == (
        '{"capabilities":["historical_bars","trades"],"checked_at_ns":null,'
        '"detail":null,"status":"unknown"}'
    )
