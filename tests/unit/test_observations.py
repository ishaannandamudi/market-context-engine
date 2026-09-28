import pytest
from pydantic import ValidationError

from mce.models.market_data import BarObservation
from mce.serialization import canonical_json_str
from tests.fixtures.samples import sample_bar, sample_quote, sample_trade


def test_synthetic_observations_are_valid_and_serializable() -> None:
    for observation in (sample_bar(), sample_quote(), sample_trade()):
        assert observation.symbol == "SPY"
        assert observation.event_timestamp_ns <= observation.effective_timestamp_ns
        assert '"source":"synthetic"' in canonical_json_str(observation)


@pytest.mark.parametrize(
    "changes",
    [
        {"open": -1.0},
        {"high": float("nan")},
        {"close": float("inf")},
        {"volume": -1.0},
        {"event_timestamp_ns": -1},
        {"effective_timestamp_ns": 1},
        {"provider_timestamp_ns": 1_800_000_000_000_000_000},
        {"low": 452.0},
        {"open": 452.0},
        {"transaction_count": -1},
    ],
)
def test_invalid_bars_rejected(changes: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        BarObservation.model_validate({**sample_bar().model_dump(), **changes})


def test_backfill_ingestion_may_follow_effective_time() -> None:
    bar = BarObservation.model_validate(
        {
            **sample_bar().model_dump(),
            "ingestion_timestamp_ns": 1_800_000_000_000_000_000,
        }
    )
    assert bar.ingestion_timestamp_ns is not None
    assert bar.ingestion_timestamp_ns > bar.effective_timestamp_ns


def test_observations_are_frozen() -> None:
    with pytest.raises(ValidationError):
        sample_bar().close = 1.0
