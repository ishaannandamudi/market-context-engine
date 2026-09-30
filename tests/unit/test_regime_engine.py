"""Reference vectors for deterministic MCE-10 regime decisions."""

import pytest

from mce.calendar.service import MarketSessionService
from mce.config.settings import (
    REGIME_RULE_IDS,
    BreadthSettings,
    FeatureSettings,
    MarketSettings,
    MCEConfig,
    RegimeSettings,
    UniverseSettings,
)
from mce.features.breadth.engine import BreadthFeatureEngine
from mce.features.breadth.membership import MembershipTimeline, UniverseMembership
from mce.features.breadth.models import BreadthFeatureSet
from mce.features.correlation.models import CorrelationFeatureSet
from mce.features.dispersion.models import DispersionFeatureSet
from mce.features.models import TrendFeatureSet
from mce.features.regime import (
    MarketRegimeEngine,
    RegimeEvidence,
    apply_regime_to_snapshot,
    evidence_from_features,
    evidence_from_snapshot,
)
from mce.features.regime.taxonomy import (
    BreadthState,
    DirectionState,
    HeadlineRegime,
    SectorRotationState,
    StressState,
)
from mce.features.sectors.models import SectorContext, SectorFeatureSet
from mce.features.stress.models import LiquidityContext, LiquidityStressFeatureSet, StressContext
from mce.features.volatility.models import RealizedVolatilityFeatureSet
from mce.models.context import MarketContextSnapshot
from mce.models.enums import (
    CorrelationRegime,
    DataQuality,
    DispersionRegime,
    LiquidityRegime,
    MarketRegime,
    QualityFlag,
    TrendRegime,
    VolatilityRegime,
)
from mce.models.market_data import BarObservation
from mce.reproducibility import config_fingerprint
from mce.serialization import canonical_json_str
from tests.fixtures.breadth_scenarios import MINUTE, breadth_store


def _evidence(as_of_ns: int = 100, **updates: object) -> RegimeEvidence:
    values: dict[str, object] = {
        "as_of_ns": as_of_ns,
        "market_direction_score": 0.8,
        "trend_score": 0.8,
        "breadth_score": 0.7,
        "breadth_participation_score": 0.6,
        "breadth_coverage": 1,
        "volatility_score": 0.3,
        "liquidity_score": 0.8,
        "correlation_score": 0.4,
        "dispersion_score": 0.4,
        "stress_score": 0.2,
        "stressed_member_fraction": 0.2,
        "sector_rotation_velocity": 0.2,
        "sector_leadership_spread": 0.01,
        "sector_breadth_score": 0.4,
        "sector_coverage": 1,
        "source_coverage": 1,
        "data_quality": DataQuality.GOOD,
    }
    values.update(updates)
    return RegimeEvidence.model_validate(values)


def _engine(settings: RegimeSettings | None = None) -> MarketRegimeEngine:
    return MarketRegimeEngine(
        MCEConfig(_env_file=None, features=FeatureSettings(regime=settings or RegimeSettings()))
    )


@pytest.mark.parametrize(
    ("updates", "headline"),
    [
        ({}, HeadlineRegime.RISK_ON),
        (
            {
                "market_direction_score": -0.8,
                "trend_score": -0.8,
                "breadth_score": -0.7,
                "breadth_participation_score": -0.6,
                "stress_score": 0.5,
            },
            HeadlineRegime.RISK_OFF,
        ),
        (
            {
                "market_direction_score": 0,
                "trend_score": 0,
                "breadth_score": 0,
                "stress_score": 0.1,
                "volatility_score": 0.2,
            },
            HeadlineRegime.RANGE_BOUND,
        ),
        (
            {
                "market_direction_score": -0.8,
                "trend_score": -0.8,
                "breadth_score": -0.7,
                "stress_score": 0.9,
                "correlation_score": 0.85,
                "stressed_member_fraction": 0.8,
            },
            HeadlineRegime.SYSTEMIC_STRESS,
        ),
        ({"liquidity_score": 0.1, "stress_score": 0.75}, HeadlineRegime.LIQUIDITY_STRESS),
        (
            {
                "market_direction_score": 0,
                "trend_score": 0,
                "breadth_score": 0,
                "sector_rotation_velocity": 0.9,
                "dispersion_score": 0.9,
                "correlation_score": 0.2,
                "stress_score": 0.3,
            },
            HeadlineRegime.ROTATION,
        ),
        (
            {"market_direction_score": 0.8, "trend_score": 0.8, "breadth_score": -0.8},
            HeadlineRegime.TRANSITIONAL,
        ),
        (
            {
                "market_direction_score": 0.4,
                "trend_score": -0.4,
                "breadth_score": 0,
                "stress_score": 0.4,
            },
            HeadlineRegime.MIXED,
        ),
    ],
)
def test_headline_reference_scenarios(updates: dict[str, object], headline: HeadlineRegime) -> None:
    state = _engine().compute(_evidence(**updates))
    assert state.headline_regime == headline
    assert state.candidate_regime == headline
    first = next(item for item in state.rule_trace if item.condition_met)
    assert first.rule_id == state.candidate_rule_id == state.selected_rule_id
    assert state.evidence_strength > 0
    assert 0 <= state.evidence_strength <= 1
    assert tuple(item.precedence for item in state.rule_trace) == tuple(range(8))


def test_dimension_boundaries_missing_and_orientation() -> None:
    engine = _engine()
    low = engine.compute(
        _evidence(
            volatility_score=0.249,
            liquidity_score=0.199,
            correlation_score=0.249,
            dispersion_score=0.249,
            stress_score=0.299,
        )
    )
    assert low.volatility_regime == VolatilityRegime.LOW
    assert low.liquidity_regime == LiquidityRegime.STRESSED
    assert low.correlation_regime == CorrelationRegime.LOW
    assert low.dispersion_regime == DispersionRegime.LOW
    assert low.stress_regime == StressState.CALM
    boundary = engine.compute(
        _evidence(
            volatility_score=0.7,
            liquidity_score=0.2,
            correlation_score=0.7,
            dispersion_score=0.7,
            stress_score=0.65,
            breadth_score=0.45,
            breadth_participation_score=0.15,
        )
    )
    assert boundary.volatility_regime == VolatilityRegime.HIGH
    assert boundary.liquidity_regime == LiquidityRegime.THIN
    assert boundary.correlation_regime == CorrelationRegime.HIGH
    assert boundary.dispersion_regime == DispersionRegime.HIGH
    assert boundary.stress_regime == StressState.HIGH
    assert boundary.breadth_regime == BreadthState.BROAD_POSITIVE
    extreme = engine.compute(
        _evidence(
            volatility_score=0.9,
            liquidity_score=0.75,
            correlation_score=0.9,
            dispersion_score=0.9,
            stress_score=0.85,
            stressed_member_fraction=0.6,
        )
    )
    assert extreme.volatility_regime == VolatilityRegime.EXTREME
    assert extreme.liquidity_regime == LiquidityRegime.HEALTHY
    assert extreme.correlation_regime == CorrelationRegime.EXTREME
    assert extreme.dispersion_regime == DispersionRegime.EXTREME
    assert extreme.stress_regime == StressState.SYSTEMIC
    missing = engine.compute(
        _evidence(
            volatility_score=None,
            liquidity_score=None,
            correlation_score=None,
            dispersion_score=None,
            sector_rotation_velocity=None,
        )
    )
    assert missing.volatility_regime == VolatilityRegime.UNKNOWN
    assert missing.liquidity_regime == LiquidityRegime.UNKNOWN
    assert missing.headline_regime == HeadlineRegime.UNKNOWN


def test_direction_breadth_sector_and_quality_gates() -> None:
    engine = _engine()
    negative = engine.compute(
        _evidence(
            market_direction_score=-0.8,
            trend_score=-0.8,
            breadth_score=-0.7,
            breadth_participation_score=-0.6,
        )
    )
    assert negative.direction_regime == DirectionState.NEGATIVE
    assert negative.breadth_regime == BreadthState.BROAD_NEGATIVE
    narrow = engine.compute(
        _evidence(
            breadth_score=0.3,
            breadth_participation_score=0.1,
            sector_rotation_velocity=0.8,
            sector_coverage=0.2,
        )
    )
    assert narrow.breadth_regime == BreadthState.NARROW_POSITIVE
    assert narrow.sector_rotation_state == SectorRotationState.FRAGMENTED
    partial = engine.compute(_evidence(breadth_score=None))
    assert partial.headline_regime == HeadlineRegime.UNKNOWN
    assert QualityFlag.MISSING_DATA in partial.quality_flags
    poor = engine.compute(_evidence(data_quality=DataQuality.POOR))
    assert poor.headline_regime == HeadlineRegime.UNKNOWN
    assert poor.direction_regime == DirectionState.UNKNOWN
    assert poor.evidence_strength == 0
    assert poor.evidence_strength < engine.compute(_evidence()).evidence_strength
    assert QualityFlag.LOW_COVERAGE in poor.quality_flags
    weak_coverage = engine.compute(_evidence(source_coverage=0.4))
    assert weak_coverage.headline_regime == HeadlineRegime.UNKNOWN
    weak_breadth = engine.compute(_evidence(breadth_coverage=0.4))
    assert weak_breadth.breadth_regime == BreadthState.UNKNOWN
    assert weak_breadth.headline_regime == HeadlineRegime.UNKNOWN
    assert QualityFlag.MISSING_DATA in weak_breadth.quality_flags


def test_conflicts_precedence_and_strength() -> None:
    engine = _engine()
    contradictory = engine.compute(_evidence(breadth_score=-0.8))
    assert contradictory.headline_regime == HeadlineRegime.TRANSITIONAL
    assert contradictory.conflicts[0].conflict_id == "regime.conflict.trend_up_breadth_down"
    assert contradictory.evidence_strength < engine.compute(_evidence()).evidence_strength
    systemic = engine.compute(
        _evidence(breadth_score=-0.8, stress_score=0.9, correlation_score=0.85, liquidity_score=0.1)
    )
    assert systemic.headline_regime == HeadlineRegime.SYSTEMIC_STRESS
    assert systemic.conflicts
    assert systemic.selected_rule_id == "regime.systemic_stress"
    assert dict(systemic.rule_trace[0].thresholds) == {
        "stress_systemic": 0.85,
        "breadth_confirmation": 0.15,
        "correlation_high": 0.7,
        "liquidity_stressed": 0.2,
    }
    alternate = _engine(
        RegimeSettings(
            precedence=("regime.transitional",)
            + tuple(rule for rule in REGIME_RULE_IDS if rule != "regime.transitional")
        )
    ).compute(_evidence(breadth_score=-0.8, stress_score=0.9, correlation_score=0.85))
    assert alternate.headline_regime == HeadlineRegime.TRANSITIONAL
    assert alternate.selected_rule_id == "regime.transitional"
    severe_liquidity = engine.compute(_evidence(liquidity_score=0.1, stress_score=0.75))
    assert severe_liquidity.headline_regime == HeadlineRegime.LIQUIDITY_STRESS
    mixed = engine.compute(
        _evidence(
            correlation_score=0.8,
            dispersion_score=0.8,
            stress_score=0.8,
            liquidity_score=0.8,
            sector_breadth_score=-0.8,
            sector_rotation_velocity=0.8,
        )
    )
    ids = {item.conflict_id for item in mixed.conflicts}
    assert {
        "regime.conflict.correlation_dispersion",
        "regime.conflict.stress_healthy_liquidity",
        "regime.conflict.trend_sector_weakness",
        "regime.conflict.trend_rotation",
    } <= ids


def test_hysteresis_dimension_and_headline_entry_hold_exit() -> None:
    engine = _engine()
    entry = engine.compute(
        _evidence(
            market_direction_score=0.35,
            trend_score=0.3,
            breadth_score=0.15,
            stress_score=0.2,
            volatility_score=0.7,
        )
    )
    assert entry.headline_regime == HeadlineRegime.RISK_ON
    hold = engine.compute(
        _evidence(
            101,
            market_direction_score=0.3,
            trend_score=0.3,
            breadth_score=0.15,
            volatility_score=0.66,
        ),
        history=(entry,),
    )
    assert hold.candidate_regime == HeadlineRegime.MIXED
    assert hold.headline_regime == HeadlineRegime.RISK_ON
    assert hold.hysteresis_held and "volatility" in hold.held_dimensions
    assert hold.volatility_regime == VolatilityRegime.HIGH
    assert hold.transition is None
    exit_state = engine.compute(
        _evidence(
            102,
            market_direction_score=0.19,
            trend_score=0.3,
            breadth_score=0.15,
            volatility_score=0.61,
        ),
        history=(entry, hold),
    )
    assert exit_state.headline_regime == HeadlineRegime.MIXED
    assert exit_state.volatility_regime == VolatilityRegime.NORMAL
    assert exit_state.transition is not None
    assert exit_state.transition.from_regime == HeadlineRegime.RISK_ON
    assert exit_state.transition.to_regime == HeadlineRegime.MIXED
    reset = engine.compute(
        _evidence(
            101,
            market_direction_score=0.3,
            trend_score=0.3,
            breadth_score=0.15,
            volatility_score=0.66,
        )
    )
    assert reset.headline_regime == HeadlineRegime.MIXED
    assert not reset.hysteresis_held


@pytest.mark.parametrize(
    ("field", "first", "second", "dimension", "expected"),
    [
        ("volatility_score", 0.95, 0.85, "volatility", "extreme"),
        ("volatility_score", 0.2, 0.3, "volatility", "low"),
        ("liquidity_score", 0.1, 0.25, "liquidity", "stressed"),
        ("liquidity_score", 0.4, 0.5, "liquidity", "thin"),
        ("liquidity_score", 0.8, 0.7, "liquidity", "healthy"),
        ("correlation_score", 0.95, 0.85, "correlation", "extreme"),
        ("correlation_score", 0.75, 0.66, "correlation", "high"),
        ("correlation_score", 0.2, 0.3, "correlation", "low"),
        ("dispersion_score", 0.95, 0.85, "dispersion", "extreme"),
        ("dispersion_score", 0.75, 0.66, "dispersion", "high"),
        ("dispersion_score", 0.2, 0.3, "dispersion", "low"),
        ("stress_score", 0.9, 0.8, "stress", "systemic"),
        ("stress_score", 0.7, 0.6, "stress", "high"),
        ("stress_score", 0.35, 0.28, "stress", "elevated"),
        ("sector_rotation_velocity", 0.8, 0.65, "sector", "active_rotation"),
    ],
)
def test_dimension_hysteresis_bands(
    field: str, first: float, second: float, dimension: str, expected: str
) -> None:
    engine = _engine()
    extra = {"stressed_member_fraction": 0.8} if field == "stress_score" else {}
    prior = engine.compute(_evidence(100, **{field: first}, **extra))
    current = engine.compute(_evidence(101, **{field: second}, **extra), history=(prior,))
    name = "sector_rotation_state" if dimension == "sector" else f"{dimension}_regime"
    assert getattr(current, name).value == expected
    assert dimension in current.held_dimensions


def test_headline_exit_bands_and_acute_overrides() -> None:
    engine = _engine()
    risk_on = engine.compute(_evidence(100))
    acute = engine.compute(
        _evidence(101, breadth_score=-0.7, stress_score=0.9, correlation_score=0.8),
        history=(risk_on,),
    )
    assert acute.headline_regime == HeadlineRegime.SYSTEMIC_STRESS
    assert not acute.hysteresis_held
    systemic_soft = engine.compute(
        _evidence(
            102,
            market_direction_score=-0.8,
            trend_score=-0.8,
            breadth_score=-0.7,
            stress_score=0.8,
            correlation_score=0.8,
        ),
        history=(acute,),
    )
    assert systemic_soft.candidate_regime == HeadlineRegime.RISK_OFF
    assert systemic_soft.headline_regime == HeadlineRegime.SYSTEMIC_STRESS
    liquidity = engine.compute(_evidence(100, liquidity_score=0.1, stress_score=0.75))
    liquidity_soft = engine.compute(
        _evidence(101, liquidity_score=0.25, stress_score=0.62), history=(liquidity,)
    )
    assert liquidity_soft.candidate_regime == HeadlineRegime.RISK_ON
    assert liquidity_soft.headline_regime == HeadlineRegime.LIQUIDITY_STRESS
    liquidity_exit = engine.compute(
        _evidence(102, liquidity_score=0.35, stress_score=0.5), history=(liquidity, liquidity_soft)
    )
    assert liquidity_exit.headline_regime == HeadlineRegime.RISK_ON
    assert liquidity_exit.transition is not None
    assert liquidity_exit.transition.triggering_rule_id == "regime.risk_on"
    risk_off = engine.compute(
        _evidence(
            100, market_direction_score=-0.8, trend_score=-0.8, breadth_score=-0.7, stress_score=0.5
        )
    )
    risk_off_soft = engine.compute(
        _evidence(
            101, market_direction_score=-0.3, trend_score=-0.8, breadth_score=-0.7, stress_score=0.5
        ),
        history=(risk_off,),
    )
    assert risk_off_soft.candidate_regime == HeadlineRegime.MIXED
    assert risk_off_soft.headline_regime == HeadlineRegime.RISK_OFF


def test_rotation_range_and_transitional_exit_bands() -> None:
    engine = _engine()
    rotation = engine.compute(
        _evidence(
            100,
            market_direction_score=0,
            trend_score=0,
            breadth_score=0,
            sector_rotation_velocity=0.9,
            dispersion_score=0.9,
            correlation_score=0.2,
            stress_score=0.3,
        )
    )
    rotation_soft = engine.compute(
        _evidence(
            101,
            market_direction_score=0,
            trend_score=0,
            breadth_score=0,
            sector_rotation_velocity=0.65,
            dispersion_score=0.65,
            correlation_score=0.2,
            stress_score=0.3,
        ),
        history=(rotation,),
    )
    assert rotation_soft.headline_regime == HeadlineRegime.ROTATION
    range_bound = engine.compute(
        _evidence(
            100,
            market_direction_score=0,
            trend_score=0,
            breadth_score=0,
            stress_score=0.1,
            volatility_score=0.2,
        )
    )
    range_soft = engine.compute(
        _evidence(
            101,
            market_direction_score=0.25,
            trend_score=0.35,
            breadth_score=0,
            stress_score=0.35,
            volatility_score=0.2,
        ),
        history=(range_bound,),
    )
    assert range_soft.candidate_regime == HeadlineRegime.MIXED
    assert range_soft.headline_regime == HeadlineRegime.RANGE_BOUND
    transitional = engine.compute(_evidence(100, breadth_score=-0.8))
    transitional_soft = engine.compute(
        _evidence(101, sector_breadth_score=-0.8), history=(transitional,)
    )
    assert transitional_soft.candidate_regime == HeadlineRegime.RISK_ON
    assert transitional_soft.headline_regime == HeadlineRegime.TRANSITIONAL
    assert transitional_soft.hysteresis_held
    exit_state = engine.compute(_evidence(102), history=(transitional, transitional_soft))
    assert exit_state.headline_regime == HeadlineRegime.RISK_ON


def test_transition_sequence_replay_and_future_exclusion() -> None:
    engine = _engine()
    vectors = (
        _evidence(
            100,
            market_direction_score=0,
            trend_score=0,
            breadth_score=0,
            stress_score=0.1,
            volatility_score=0.2,
        ),
        _evidence(101),
        _evidence(
            102, market_direction_score=-0.8, trend_score=-0.8, breadth_score=-0.7, stress_score=0.5
        ),
        _evidence(
            103,
            market_direction_score=-0.8,
            trend_score=-0.8,
            breadth_score=-0.7,
            stress_score=0.9,
            correlation_score=0.85,
            stressed_member_fraction=0.8,
        ),
        _evidence(
            104, market_direction_score=-0.8, trend_score=-0.8, breadth_score=-0.7, stress_score=0.7
        ),
        _evidence(
            105,
            market_direction_score=0,
            trend_score=0,
            breadth_score=0,
            stress_score=0.1,
            volatility_score=0.2,
        ),
    )
    states = []
    for vector in vectors:
        states.append(engine.compute(vector, history=tuple(states)))
    assert tuple(item.headline_regime for item in states) == (
        HeadlineRegime.RANGE_BOUND,
        HeadlineRegime.RISK_ON,
        HeadlineRegime.RISK_OFF,
        HeadlineRegime.SYSTEMIC_STRESS,
        HeadlineRegime.RISK_OFF,
        HeadlineRegime.RANGE_BOUND,
    )
    for before, after in zip(states, states[1:], strict=False):
        assert after.transition is not None
        assert after.transition.from_regime == before.headline_regime
        assert after.transition.to_regime == after.headline_regime
        assert after.transition.effective_at_ns == after.as_of_ns
        assert after.dimension_transitions
        assert after.regime_started_at_ns == after.as_of_ns
        assert after.regime_age_observations == 1
    replay = []
    for vector in vectors:
        replay.append(engine.compute(vector, history=tuple(replay)))
    assert canonical_json_str(tuple(replay)) == canonical_json_str(tuple(states))
    past = engine.compute(vectors[2], history=(states[0], states[1]))
    assert canonical_json_str(
        engine.compute(vectors[2], history=(states[0], states[1], states[-1]))
    ) == canonical_json_str(past)


def test_config_validation_fingerprint_and_history_mismatch() -> None:
    with pytest.raises(ValueError):
        RegimeSettings(volatility_low=0.8)
    with pytest.raises(ValueError):
        RegimeSettings(direction_exit=0.4)
    with pytest.raises(ValueError):
        RegimeSettings(precedence=("regime.risk_on",) * 8)
    with pytest.raises(ValueError):
        RegimeSettings(trend_strong=0.2)
    with pytest.raises(ValueError):
        RegimeSettings(breadth_confirmation=0.5)
    with pytest.raises(ValueError):
        RegimeSettings(minimum_required_families=9)
    baseline = _engine()
    changed = _engine(RegimeSettings(stress_high=0.7))
    assert baseline.fingerprint != changed.fingerprint
    prior = baseline.compute(_evidence())
    with pytest.raises(ValueError):
        changed.compute(_evidence(101), history=(prior,))
    with pytest.raises(ValueError):
        baseline.compute(_evidence(102), history=(prior, prior))


def test_snapshot_projection_and_adapters() -> None:
    as_of = 100
    state = _engine().compute(_evidence(as_of))
    snapshot = apply_regime_to_snapshot(state)
    assert snapshot.schema_version == "1.0" and snapshot.engine_version == "0.13.0"
    assert snapshot.market_regime == MarketRegime.RISK_ON
    assert snapshot.trend_regime == TrendRegime.STRONG_UP
    assert snapshot.volatility_regime == VolatilityRegime.NORMAL
    assert snapshot.liquidity_regime == LiquidityRegime.HEALTHY
    assert snapshot.breadth_regime.value == "strong_positive"
    assert snapshot.confidence is None
    assert snapshot.feature_values is not None
    assert snapshot.feature_values["regime.evidence_strength"] == state.evidence_strength
    assert canonical_json_str(
        MarketContextSnapshot.model_validate_json(canonical_json_str(snapshot))
    ) == canonical_json_str(snapshot)
    assert evidence_from_snapshot(snapshot).market_direction_score is None
    existing = MarketContextSnapshot(as_of_ns=as_of, stress_score=0.2)
    assert apply_regime_to_snapshot(state, existing).stress_score == 0.2
    with pytest.raises(ValueError):
        apply_regime_to_snapshot(state, MarketContextSnapshot(as_of_ns=as_of + 1))


def test_precomputed_feature_adapter_exact_clock() -> None:
    as_of = 100
    trend = TrendFeatureSet.model_construct(
        as_of_ns=as_of,
        market_direction_score=0.8,
        trend_score=0.8,
        coverage_ratio=1,
        data_quality=DataQuality.GOOD,
    )
    breadth = BreadthFeatureSet.model_construct(
        as_of_ns=as_of,
        breadth_score=0.7,
        participation_score=0.6,
        coverage_ratio=1,
        data_quality=DataQuality.GOOD,
    )
    volatility = RealizedVolatilityFeatureSet.model_construct(
        as_of_ns=as_of, volatility_score=0.3, data_quality=DataQuality.GOOD
    )
    correlation = CorrelationFeatureSet.model_construct(
        as_of_ns=as_of, systemic_co_movement_score=0.4, data_quality=DataQuality.GOOD
    )
    dispersion = DispersionFeatureSet.model_construct(
        as_of_ns=as_of, dispersion_score=0.4, data_quality=DataQuality.GOOD
    )
    sectors = SectorFeatureSet.model_construct(
        as_of_ns=as_of,
        sectors={"a": SectorContext.model_construct(breadth_score=0.4)},
        rotation_velocity=0.2,
        market_coverage=1,
        data_quality=DataQuality.GOOD,
    )
    liquidity_stress = LiquidityStressFeatureSet.model_construct(
        as_of_ns=as_of,
        liquidity=LiquidityContext.model_construct(liquidity_score=0.8),
        stress=StressContext.model_construct(
            stress_score=0.2, stressed_member_fraction=0.2, evidence_quality=1
        ),
        data_quality=DataQuality.GOOD,
    )
    result = evidence_from_features(
        as_of,
        trend=trend,
        breadth=breadth,
        volatility=volatility,
        correlation=correlation,
        dispersion=dispersion,
        sectors=sectors,
        liquidity_stress=liquidity_stress,
    )
    assert result.trend_score == 0.8 and result.sector_breadth_score == 0.4
    assert _engine().compute(result).headline_regime == HeadlineRegime.RISK_ON
    with pytest.raises(ValueError):
        evidence_from_features(as_of, trend=trend.model_copy(update={"as_of_ns": as_of + 1}))
    with pytest.raises(ValueError):
        evidence_from_features(as_of, breadth=breadth.model_copy(update={"as_of_ns": as_of + 1}))
    with pytest.raises(ValueError):
        evidence_from_features(as_of, sectors=sectors.model_copy(update={"as_of_ns": as_of + 1}))
    with pytest.raises(ValueError):
        evidence_from_features(
            as_of,
            liquidity_stress=liquidity_stress.model_copy(update={"as_of_ns": as_of + 1}),
        )
    assert config_fingerprint(_engine().config) == _engine().fingerprint


def test_future_membership_weight_and_bar_cannot_change_past_regime(
    calendar: MarketSessionService,
) -> None:
    symbols = ("AAA", "BBB", "CCC", "DDD", "EEE")
    store, as_of = breadth_store(calendar, symbols, (0.02,) * len(symbols))
    future = as_of + MINUTE
    config = MCEConfig(
        _env_file=None,
        market=MarketSettings(universe=UniverseSettings(breadth=symbols)),
        features=FeatureSettings(breadth=BreadthSettings(price_adjustment_status="adjusted")),
    )
    baseline_members = MembershipTimeline(
        tuple(UniverseMembership(symbol=symbol, weight=1) for symbol in symbols)
    )
    revised_members = MembershipTimeline(
        (
            UniverseMembership(symbol="AAA", weight=1, effective_to_ns=future),
            UniverseMembership(symbol="AAA", weight=9, effective_from_ns=future),
            *(UniverseMembership(symbol=symbol, weight=1) for symbol in symbols[1:]),
            UniverseMembership(symbol="NEW", weight=2, effective_from_ns=future),
        )
    )
    trend = TrendFeatureSet.model_construct(
        as_of_ns=as_of,
        market_direction_score=0.8,
        trend_score=0.8,
        coverage_ratio=1,
        data_quality=DataQuality.GOOD,
    )
    liquidity_stress = LiquidityStressFeatureSet.model_construct(
        as_of_ns=as_of,
        liquidity=LiquidityContext.model_construct(liquidity_score=0.8),
        stress=StressContext.model_construct(
            stress_score=0.2, stressed_member_fraction=0.2, evidence_quality=1
        ),
        data_quality=DataQuality.GOOD,
    )
    classifier = MarketRegimeEngine(config)

    def regime(members: MembershipTimeline) -> str:
        breadth = BreadthFeatureEngine(config, members, calendar).compute(store, as_of)
        evidence = evidence_from_features(
            as_of, trend=trend, breadth=breadth, liquidity_stress=liquidity_stress
        )
        return canonical_json_str(classifier.compute(evidence))

    before = regime(baseline_members)
    assert regime(revised_members) == before
    store.add(
        BarObservation(
            symbol="AAA",
            source="synthetic",
            event_timestamp_ns=as_of,
            effective_timestamp_ns=future,
            bar_start_ns=as_of,
            bar_end_ns=future,
            open=100,
            high=200,
            low=100,
            close=200,
            volume=1,
        )
    )
    assert regime(revised_members) == before
