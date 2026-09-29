"""Immutable, finite liquidity and market-stress evidence."""

from collections.abc import Mapping
from types import MappingProxyType

from pydantic import Field, field_serializer, model_validator

from mce.models.context import ContextContributor
from mce.models.enums import DataQuality
from mce.models.quality import FiniteNumber, NonNegativeFinite, QualityTaggedModel, UnitScore
from mce.models.symbol import Symbol
from mce.models.time import TimestampNs


class LiquidityMember(QualityTaggedModel):
    symbol: Symbol
    weight: NonNegativeFinite | None = None
    session_volume: NonNegativeFinite | None = None
    session_dollar_volume: NonNegativeFinite | None = None
    session_return: FiniteNumber | None = None
    relative_volume: NonNegativeFinite | None = None
    illiquidity_proxy: NonNegativeFinite | None = None
    illiquidity_percentile: UnitScore | None = None
    quoted_spread: NonNegativeFinite | None = None
    relative_quoted_spread: NonNegativeFinite | None = None
    gap_return: FiniteNumber | None = None
    stale_bar: bool = False
    stale_quote: bool = False
    explicit_zero_volume: bool = False


class LiquidityContext(QualityTaggedModel):
    per_member: Mapping[str, LiquidityMember]
    expected_member_count: int = Field(ge=0)
    valid_member_count: int = Field(ge=0)
    stale_bar_count: int = Field(ge=0)
    stale_quote_count: int = Field(ge=0)
    member_coverage: UnitScore
    quote_coverage: UnitScore
    weight_coverage: UnitScore | None = None
    session_volume: NonNegativeFinite | None = None
    session_dollar_volume: NonNegativeFinite | None = None
    turnover: None = None
    relative_volume: NonNegativeFinite | None = None
    relative_volume_5d: NonNegativeFinite | None = None
    relative_volume_20d: NonNegativeFinite | None = None
    relative_volume_dispersion: NonNegativeFinite | None = None
    volume_percentile: UnitScore | None = None
    volume_robust_z: FiniteNumber | None = None
    volume_breadth: UnitScore | None = None
    large_move_weak_volume_fraction: UnitScore | None = None
    large_move_high_volume_fraction: UnitScore | None = None
    illiquidity_proxy: NonNegativeFinite | None = None
    weighted_illiquidity_proxy: NonNegativeFinite | None = None
    illiquidity_dispersion: NonNegativeFinite | None = None
    illiquidity_percentile: UnitScore | None = None
    illiquidity_robust_z: FiniteNumber | None = None
    price_impact_proxy: NonNegativeFinite | None = None
    quoted_spread: NonNegativeFinite | None = None
    relative_quoted_spread: NonNegativeFinite | None = None
    spread_dispersion: NonNegativeFinite | None = None
    dollar_volume_hhi: UnitScore | None = None
    effective_active_names: NonNegativeFinite | None = None
    liquidity_change: FiniteNumber | None = None
    liquidity_score: UnitScore | None = None
    active_weights: tuple[tuple[str, UnitScore], ...] = ()
    evidence_quality: UnitScore
    data_quality: DataQuality
    contributors: tuple[ContextContributor, ...] = ()

    @field_serializer("per_member")
    def serialize_members(
        self, values: Mapping[str, LiquidityMember]
    ) -> dict[str, LiquidityMember]:
        return dict(values)

    @model_validator(mode="after")
    def validate_members(self) -> "LiquidityContext":
        if len(self.per_member) != self.expected_member_count:
            raise ValueError("liquidity members must match expected count")
        if self.valid_member_count > self.expected_member_count:
            raise ValueError("valid members exceed expected count")
        if self.member_coverage != (
            self.valid_member_count / self.expected_member_count
            if self.expected_member_count
            else 0.0
        ):
            raise ValueError("liquidity member coverage does not match counts")
        if any(symbol != member.symbol for symbol, member in self.per_member.items()):
            raise ValueError("liquidity member keys must match symbols")
        object.__setattr__(
            self, "per_member", MappingProxyType(dict(sorted(self.per_member.items())))
        )
        return self


class StressHistoryPoint(QualityTaggedModel):
    as_of_ns: TimestampNs
    available_at_ns: TimestampNs
    stress_score: UnitScore

    @model_validator(mode="after")
    def validate_clock(self) -> "StressHistoryPoint":
        if self.available_at_ns < self.as_of_ns:
            raise ValueError("stress history cannot be available before its observation")
        return self


class StressContext(QualityTaggedModel):
    gap_up_fraction: UnitScore | None = None
    gap_down_fraction: UnitScore | None = None
    gap_magnitude: NonNegativeFinite | None = None
    gap_dispersion: NonNegativeFinite | None = None
    gap_percentile: UnitScore | None = None
    gap_coverage: UnitScore = 0.0
    volatility_stress: UnitScore | None = None
    correlation_stress: UnitScore | None = None
    breadth_stress: UnitScore | None = None
    liquidity_stress: UnitScore | None = None
    gap_stress: UnitScore | None = None
    dispersion_context_stress: UnitScore | None = None
    sector_stress: UnitScore | None = None
    stress_score: UnitScore | None = None
    systemic_stress: UnitScore | None = None
    localized_stress: UnitScore | None = None
    stressed_member_fraction: UnitScore | None = None
    stress_percentile: UnitScore | None = None
    stress_change: FiniteNumber | None = None
    stress_acceleration: FiniteNumber | None = None
    stress_persistence: UnitScore | None = None
    stress_shock_score: UnitScore | None = None
    historical_sample_count: int = Field(ge=0)
    active_weights: tuple[tuple[str, UnitScore], ...] = ()
    evidence_quality: UnitScore
    data_quality: DataQuality
    contributors: tuple[ContextContributor, ...] = ()


class LiquidityStressFeatureSet(QualityTaggedModel):
    as_of_ns: TimestampNs
    liquidity: LiquidityContext
    stress: StressContext
    data_quality: DataQuality
