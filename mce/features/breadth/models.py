"""Immutable, typed breadth evidence; missing values remain absent."""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Literal

from pydantic import Field, field_serializer, model_validator

from mce.models.context import ContextContributor
from mce.models.enums import DataQuality
from mce.models.quality import (
    DirectionalScore,
    FiniteNumber,
    NonNegativeFinite,
    QualityTaggedModel,
    UnitScore,
)
from mce.models.symbol import Symbol
from mce.models.time import TimestampNs


class MemberBreadth(QualityTaggedModel):
    symbol: Symbol
    weight: NonNegativeFinite | None = None
    current_price: NonNegativeFinite | None = None
    previous_close: NonNegativeFinite | None = None
    simple_return: FiniteNumber | None = None
    stale: bool = False
    above_short_reference: bool | None = None
    above_medium_reference: bool | None = None
    above_long_reference: bool | None = None
    new_high: bool | None = None
    new_low: bool | None = None


class ReferenceParticipation(QualityTaggedModel):
    sessions: int = Field(gt=0)
    above_count: int = Field(ge=0)
    comparable_count: int = Field(ge=0)
    fraction: UnitScore | None = None

    @model_validator(mode="after")
    def validate_counts(self) -> "ReferenceParticipation":
        if self.above_count > self.comparable_count:
            raise ValueError("above count cannot exceed comparable count")
        expected = self.above_count / self.comparable_count if self.comparable_count else None
        if self.fraction != expected:
            raise ValueError("reference fraction must match counts")
        return self


class BreadthFeatureSet(QualityTaggedModel):
    as_of_ns: TimestampNs
    per_member: Mapping[str, MemberBreadth]
    expected_member_count: int = Field(ge=0)
    available_member_count: int = Field(ge=0)
    stale_member_count: int = Field(ge=0)
    coverage_ratio: UnitScore
    weight_coverage: UnitScore | None = None
    advancers: int = Field(ge=0)
    decliners: int = Field(ge=0)
    unchanged: int = Field(ge=0)
    advance_fraction: UnitScore | None = None
    decline_fraction: UnitScore | None = None
    net_advance_fraction: DirectionalScore | None = None
    advance_decline_ratio: NonNegativeFinite | None = None
    short_reference: ReferenceParticipation
    medium_reference: ReferenceParticipation
    long_reference: ReferenceParticipation
    new_high_low_window_sessions: int = Field(gt=0)
    new_high_fraction: UnitScore | None = None
    new_low_fraction: UnitScore | None = None
    net_new_high_fraction: DirectionalScore | None = None
    equal_weight_return: FiniteNumber | None = None
    median_member_return: FiniteNumber | None = None
    weighted_return: FiniteNumber | None = None
    equal_weight_divergence: FiniteNumber | None = None
    index_proxy_return: FiniteNumber | None = None
    index_proxy_divergence: FiniteNumber | None = None
    directional_index_divergence: FiniteNumber | None = None
    contribution_hhi: UnitScore | None = None
    effective_contributors: NonNegativeFinite | None = None
    contribution_basis: Literal["equal", "weighted"] | None = None
    participation_score: DirectionalScore | None = None
    breadth_score: DirectionalScore | None = None
    breadth_momentum: FiniteNumber | None = None
    breadth_acceleration: FiniteNumber | None = None
    breadth_thrust: FiniteNumber | None = None
    participation_persistence: UnitScore | None = None
    evidence_quality: UnitScore
    data_quality: DataQuality
    contributors: tuple[ContextContributor, ...] = ()

    @field_serializer("per_member")
    def serialize_members(self, values: Mapping[str, MemberBreadth]) -> dict[str, MemberBreadth]:
        return dict(values)

    @model_validator(mode="after")
    def validate_result(self) -> "BreadthFeatureSet":
        if len(self.per_member) != self.expected_member_count:
            raise ValueError("per_member must contain each expected member")
        if self.advancers + self.decliners + self.unchanged != self.available_member_count:
            raise ValueError("direction counts must match available members")
        if self.coverage_ratio != (
            self.available_member_count / self.expected_member_count
            if self.expected_member_count
            else 0.0
        ):
            raise ValueError("coverage must match member counts")
        if any(key != item.symbol for key, item in self.per_member.items()):
            raise ValueError("per_member keys must match symbols")
        object.__setattr__(self, "per_member", MappingProxyType(dict(self.per_member)))
        return self
