import pytest

from mce.calendar import MarketSessionService


@pytest.fixture(scope="session")
def calendar() -> MarketSessionService:
    return MarketSessionService()
