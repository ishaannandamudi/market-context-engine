"""Immutable evidence, decisions, explanations, and causal transition records."""

from pydantic import Field, model_validator

from mce.features.regime.taxonomy import (
    BreadthState,
    DirectionState,
    HeadlineRegime,
    SectorRotationState,
    StressState,
)
from mce.models.enums import (
    CorrelationRegime,
    DataQuality,
    DispersionRegime,
    LiquidityRegime,
    TrendRegime,
    VolatilityRegime,
)
from mce.models.quality import (
    DirectionalScore,
    FiniteNumber,
    NonNegativeFinite,
    QualityTaggedModel,
    UnitScore,
)
from mce.models.time import TimestampNs


class RegimeEvidence(QualityTaggedModel):
    as_of_ns: TimestampNs
    market_direction_score: DirectionalScore | None = None
    trend_score: DirectionalScore | None = None
    breadth_score: DirectionalScore | None = None
    breadth_participation_score: DirectionalScore | None = None
    breadth_coverage: UnitScore | None = None
    index_proxy_divergence: FiniteNumber | None = None
    volatility_score: UnitScore | None = None
    liquidity_score: UnitScore | None = None
    correlation_score: UnitScore | None = None
    average_correlation: DirectionalScore | None = None
    dispersion_score: UnitScore | None = None
    differentiation_score: UnitScore | None = None
    stress_score: UnitScore | None = None
    stressed_member_fraction: UnitScore | None = None
    sector_rotation_velocity: UnitScore | None = None
    sector_leadership_spread: NonNegativeFinite | None = None
    sector_breadth_score: DirectionalScore | None = None
    sector_coverage: UnitScore | None = None
    source_coverage: UnitScore = 1.0
    data_quality: DataQuality = DataQuality.GOOD


class RegimeConflict(QualityTaggedModel):
    conflict_id: str = Field(min_length=1)
    severity: UnitScore
    source_features: tuple[str, ...]
    description: str = Field(min_length=1)


class RuleEvaluation(QualityTaggedModel):
    rule_id: str = Field(min_length=1)
    precedence: int = Field(ge=0)
    condition_met: bool
    source_features: tuple[str, ...]
    thresholds: tuple[tuple[str, FiniteNumber], ...] = ()
    score: UnitScore | None = None
    reason: str = Field(min_length=1)


class RegimeScorecard(QualityTaggedModel):
    evidence: RegimeEvidence
    active_families: tuple[str, ...]
    quality_factor: UnitScore
    family_factor: UnitScore
    threshold_margin: UnitScore
    agreement_factor: UnitScore


class RegimeTransition(QualityTaggedModel):
    from_regime: HeadlineRegime
    to_regime: HeadlineRegime
    effective_at_ns: TimestampNs
    triggering_rule_id: str = Field(min_length=1)
    triggering_evidence: tuple[str, ...]
    evidence_strength_before: UnitScore
    evidence_strength_after: UnitScore
    conflict_ids: tuple[str, ...] = ()


class DimensionTransition(QualityTaggedModel):
    dimension: str = Field(min_length=1)
    from_state: str = Field(min_length=1)
    to_state: str = Field(min_length=1)
    effective_at_ns: TimestampNs


class MarketRegimeState(QualityTaggedModel):
    as_of_ns: TimestampNs
    available_at_ns: TimestampNs
    config_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    headline_regime: HeadlineRegime
    direction_regime: DirectionState
    trend_regime: TrendRegime
    volatility_regime: VolatilityRegime
    liquidity_regime: LiquidityRegime
    correlation_regime: CorrelationRegime
    breadth_regime: BreadthState
    dispersion_regime: DispersionRegime
    stress_regime: StressState
    sector_rotation_state: SectorRotationState
    evidence_strength: UnitScore
    scorecard: RegimeScorecard
    conflicts: tuple[RegimeConflict, ...] = ()
    rule_trace: tuple[RuleEvaluation, ...]
    candidate_regime: HeadlineRegime
    candidate_rule_id: str | None = None
    selected_rule_id: str | None = None
    hysteresis_held: bool = False
    held_dimensions: tuple[str, ...] = ()
    transition: RegimeTransition | None = None
    dimension_transitions: tuple[DimensionTransition, ...] = ()
    regime_started_at_ns: TimestampNs | None = None
    regime_age_observations: int | None = Field(default=None, ge=1)
    data_quality: DataQuality

    @model_validator(mode="after")
    def validate_state(self) -> "MarketRegimeState":
        if self.available_at_ns < self.as_of_ns:
            raise ValueError("regime state cannot be available before observation")
        if self.scorecard.evidence.as_of_ns != self.as_of_ns:
            raise ValueError("regime evidence and state clock must match")
        if self.regime_started_at_ns is not None and self.regime_started_at_ns > self.as_of_ns:
            raise ValueError("regime start cannot be in the future")
        if self.transition is not None and self.transition.effective_at_ns != self.as_of_ns:
            raise ValueError("transition must occur at state clock")
        if any(item.effective_at_ns != self.as_of_ns for item in self.dimension_transitions):
            raise ValueError("dimension transitions must occur at state clock")
        return self
