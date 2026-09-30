"""Immutable ensemble scorecard, weighting, conflicts, and rule explanations."""

from enum import StrEnum
from hashlib import sha256
from typing import Literal

from pydantic import Field, model_validator

from mce.models.enums import DataQuality
from mce.models.quality import DirectionalScore, FiniteNumber, QualityTaggedModel, UnitScore
from mce.models.time import TimestampNs
from mce.serialization.canonical import canonical_json_bytes

GroupName = Literal[
    "direction",
    "participation",
    "volatility_stress",
    "liquidity_health",
    "systemic_structure",
    "external_risk",
]


class EnsembleHeadline(StrEnum):
    BROAD_RISK_ON = "broad_risk_on"
    BROAD_RISK_OFF = "broad_risk_off"
    CALM_RANGE = "calm_range"
    ROTATION = "rotation"
    SYSTEMIC_STRESS = "systemic_stress"
    LIQUIDITY_STRESS = "liquidity_stress"
    VOLATILITY_STRESS = "volatility_stress"
    EXTERNAL_EVENT_RISK = "external_event_risk"
    TRANSITIONAL = "transitional"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class EnsembleEvidence(QualityTaggedModel):
    family: str = Field(min_length=1)
    group: GroupName
    value: DirectionalScore | UnitScore
    source_feature: str = Field(min_length=1)
    source_as_of_ns: TimestampNs
    source_quality: DataQuality
    coverage: UnitScore
    reliability: UnitScore
    configured_weight: UnitScore
    adjusted_weight: UnitScore
    normalized_active_weight: UnitScore = 0.0


class EnsembleGroup(QualityTaggedModel):
    name: GroupName
    score: DirectionalScore | UnitScore | None = None
    reliability: UnitScore = 0.0
    active_families: tuple[str, ...] = ()
    configured_weight: UnitScore
    adjusted_weight: UnitScore = 0.0
    normalized_active_weight: UnitScore = 0.0


class EnsembleConflict(QualityTaggedModel):
    conflict_id: str = Field(min_length=1)
    family_a: str = Field(min_length=1)
    family_b: str = Field(min_length=1)
    evidence_a: FiniteNumber
    evidence_b: FiniteNumber
    severity: UnitScore
    description: str = Field(min_length=1)


class EnsembleContextReliability(QualityTaggedModel):
    family: str = Field(min_length=1)
    reliability: UnitScore
    reason: str = Field(min_length=1)


class EnsembleContribution(QualityTaggedModel):
    group: GroupName
    oriented_value: DirectionalScore
    normalized_weight: UnitScore
    contribution: DirectionalScore


class EnsembleRuleTrace(QualityTaggedModel):
    rule_id: str = Field(min_length=1)
    precedence: int = Field(ge=0)
    condition: str = Field(min_length=1)
    matched: bool
    inputs: tuple[tuple[str, FiniteNumber | None], ...]
    thresholds: tuple[tuple[str, FiniteNumber], ...]


class EnsembleScorecard(QualityTaggedModel):
    direction: DirectionalScore | None = None
    participation: DirectionalScore | None = None
    volatility_stress: UnitScore | None = None
    liquidity_health: UnitScore | None = None
    systemic_structure: UnitScore | None = None
    external_risk: UnitScore | None = None
    core_market_score: DirectionalScore | None = None
    disagreement: UnitScore | None = None
    consensus: UnitScore | None = None
    uncertainty: UnitScore | None = None
    evidence_strength: UnitScore | None = None
    temporal_stability: UnitScore | None = None


class EnsembleContextFeatureSet(QualityTaggedModel):
    as_of_ns: TimestampNs
    config_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    headline: EnsembleHeadline
    scorecard: EnsembleScorecard
    evidence: tuple[EnsembleEvidence, ...] = ()
    groups: tuple[EnsembleGroup, ...] = ()
    conflicts: tuple[EnsembleConflict, ...] = ()
    context_reliability: tuple[EnsembleContextReliability, ...] = ()
    contributors: tuple[EnsembleContribution, ...] = ()
    rule_trace: tuple[EnsembleRuleTrace, ...] = ()
    deterministic_regime: str | None = None
    hmm_state_id: int | None = None
    data_quality: DataQuality

    @model_validator(mode="after")
    def validate_explanation(self) -> "EnsembleContextFeatureSet":
        score = self.scorecard.core_market_score
        if (
            score is not None
            and abs(sum(item.contribution for item in self.contributors) - score) > 1e-9
        ):
            raise ValueError("ensemble contributors must reconcile to the core score")
        active = [item for item in self.groups if item.normalized_active_weight > 0]
        if active and abs(sum(item.normalized_active_weight for item in active) - 1) > 1e-9:
            raise ValueError("ensemble core active weights must sum to one")
        return self

    @property
    def fingerprint(self) -> str:
        return sha256(canonical_json_bytes(self)).hexdigest()
