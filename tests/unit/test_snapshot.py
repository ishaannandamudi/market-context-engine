from datetime import date

import pytest
from pydantic import ValidationError

from mce import __version__
from mce.models.context import (
    ContextContributor,
    EventRisk,
    ImpliedVolatilityContext,
    MarketContextSnapshot,
    SourceFreshness,
    TermStructurePoint,
    VolatilitySkewContext,
    VolatilityTermStructureContext,
)
from mce.models.enums import (
    DataQuality,
    EventRiskStatus,
    MarketRegime,
    MarketSession,
    QualityFlag,
)
from mce.serialization import canonical_json_str
from mce.version import SCHEMA_VERSION
from tests.fixtures.samples import EFFECTIVE_NS, sample_snapshot


def test_versioning_and_missing_analytics() -> None:
    snapshot = sample_snapshot()
    assert __version__ == "0.15.0"
    assert SCHEMA_VERSION == "1.0"
    assert snapshot.engine_version == __version__
    assert snapshot.schema_version == SCHEMA_VERSION
    assert snapshot.market_regime == MarketRegime.UNKNOWN
    assert snapshot.market_direction_score is None
    assert snapshot.realized_volatility is None
    assert snapshot.index_scores is None
    assert snapshot.confidence is None


def test_complete_representative_snapshot_round_trip() -> None:
    snapshot = MarketContextSnapshot(
        as_of_ns=EFFECTIVE_NS,
        market_session=MarketSession.REGULAR,
        session_trading_date=date(2023, 11, 14),
        session_identifier="XNYS-regular",
        market_regime=MarketRegime.MIXED,
        market_direction_score=0.25,
        trend_score=-0.5,
        breadth_score=0.0,
        dispersion_score=0.6,
        correlation_score=0.5,
        stress_score=0.2,
        liquidity_score=0.8,
        volatility_score=0.3,
        risk_on_off_score=0.1,
        realized_volatility=0.17,
        index_scores={"SPY": 0.3, "QQQ": 0.2},
        sector_scores={"XLK": 0.4},
        implied_volatility_context=ImpliedVolatilityContext(
            reference_symbol="SPY",
            implied_volatility=0.2,
            source="synthetic",
            as_of_ns=EFFECTIVE_NS,
        ),
        volatility_term_structure=VolatilityTermStructureContext(
            reference_symbol="SPY",
            source="synthetic",
            as_of_ns=EFFECTIVE_NS,
            points=(TermStructurePoint(tenor_days=30, implied_volatility=0.2),),
        ),
        volatility_skew_context=VolatilitySkewContext(
            reference_symbol="SPY",
            source="synthetic",
            as_of_ns=EFFECTIVE_NS,
            put_call_skew=0.01,
        ),
        event_risk=EventRisk(status=EventRiskStatus.UNKNOWN),
        confidence=0.75,
        data_quality=DataQuality.DEGRADED,
        quality_flags=frozenset({QualityFlag.PARTIAL_UNIVERSE, QualityFlag.STALE_DATA}),
        source_freshness=(
            SourceFreshness(
                source="synthetic",
                latest_effective_timestamp_ns=EFFECTIVE_NS - 10,
                age_ns=10,
                is_stale=False,
            ),
        ),
        contributors=(ContextContributor(name="sample", value=0.2),),
        feature_values={"available": True, "count": 3, "label": "sample", "rate": 0.2},
    )
    payload = canonical_json_str(snapshot)
    assert '"engine_version":"0.15.0"' in payload
    assert '"schema_version":"1.0"' in payload
    assert canonical_json_str(MarketContextSnapshot.model_validate_json(payload)) == payload


@pytest.mark.parametrize(
    "field,value",
    [
        ("market_direction_score", -1.01),
        ("trend_score", 1.01),
        ("stress_score", -0.01),
        ("confidence", 1.01),
        ("realized_volatility", float("nan")),
        ("feature_values", {"bad": float("inf")}),
    ],
)
def test_invalid_snapshot_scores_or_values_rejected(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        MarketContextSnapshot(as_of_ns=1, **{field: value})


def test_score_boundaries_and_quality_flag_order() -> None:
    snapshot = MarketContextSnapshot(
        as_of_ns=1,
        market_direction_score=-1.0,
        trend_score=1.0,
        confidence=0.0,
        stress_score=1.0,
        quality_flags=[
            QualityFlag.STALE_DATA,
            QualityFlag.PARTIAL_UNIVERSE,
            QualityFlag.STALE_DATA,
        ],
    )
    assert snapshot.quality_flags == {QualityFlag.PARTIAL_UNIVERSE, QualityFlag.STALE_DATA}
    assert snapshot.model_dump(mode="json")["quality_flags"] == ["partial_universe", "stale_data"]
    assert canonical_json_str(snapshot) == canonical_json_str(snapshot)


def test_snapshot_is_frozen_and_versions_validated() -> None:
    with pytest.raises(ValidationError):
        sample_snapshot().as_of_ns = 2
    with pytest.raises(ValidationError):
        MarketContextSnapshot(as_of_ns=1, schema_version="2.0")
    with pytest.raises(ValidationError):
        MarketContextSnapshot(as_of_ns=1, engine_version="latest")


def test_previous_engine_snapshot_remains_readable() -> None:
    previous = MarketContextSnapshot(as_of_ns=1, engine_version="0.1.0")
    loaded = MarketContextSnapshot.model_validate_json(canonical_json_str(previous))
    assert loaded.engine_version == "0.1.0"
    assert loaded.schema_version == SCHEMA_VERSION
    mce_two = MarketContextSnapshot(as_of_ns=1, engine_version="0.2.0")
    assert (
        MarketContextSnapshot.model_validate_json(canonical_json_str(mce_two)).engine_version
        == "0.2.0"
    )


def test_snapshot_scalar_maps_are_copied_and_read_only() -> None:
    scores = {"SPY": 0.5}
    snapshot = MarketContextSnapshot(as_of_ns=1, index_scores=scores)
    scores["SPY"] = 0.9
    assert snapshot.index_scores is not None
    assert snapshot.index_scores["SPY"] == 0.5
    with pytest.raises(TypeError):
        snapshot.index_scores["SPY"] = 0.1


def test_snapshot_rejects_future_inputs_and_bad_freshness() -> None:
    with pytest.raises(ValidationError, match="future data"):
        MarketContextSnapshot(
            as_of_ns=10,
            source_freshness=(
                SourceFreshness(
                    source="synthetic",
                    latest_effective_timestamp_ns=11,
                    age_ns=0,
                    is_stale=False,
                ),
            ),
        )
    with pytest.raises(ValidationError, match="age_ns"):
        MarketContextSnapshot(
            as_of_ns=10,
            source_freshness=(
                SourceFreshness(
                    source="synthetic",
                    latest_effective_timestamp_ns=9,
                    age_ns=0,
                    is_stale=False,
                ),
            ),
        )
    with pytest.raises(ValidationError, match="newer than snapshot"):
        MarketContextSnapshot(
            as_of_ns=10,
            implied_volatility_context=ImpliedVolatilityContext(
                reference_symbol="SPY",
                implied_volatility=0.2,
                source="synthetic",
                as_of_ns=11,
            ),
        )


def test_source_freshness_has_stable_order_and_unique_sources() -> None:
    entries = (
        SourceFreshness(source="zeta", latest_effective_timestamp_ns=9, age_ns=1, is_stale=False),
        SourceFreshness(source="alpha", latest_effective_timestamp_ns=8, age_ns=2, is_stale=False),
    )
    snapshot = MarketContextSnapshot(as_of_ns=10, source_freshness=entries)
    assert snapshot.source_freshness is not None
    assert [entry.source for entry in snapshot.source_freshness] == ["alpha", "zeta"]
    assert canonical_json_str(snapshot) == canonical_json_str(
        MarketContextSnapshot(as_of_ns=10, source_freshness=entries[::-1])
    )
    with pytest.raises(ValidationError, match="unique"):
        MarketContextSnapshot(as_of_ns=10, source_freshness=(entries[0], entries[0]))
