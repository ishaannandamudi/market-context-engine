"""Immutable public sector strength, breadth, volatility, and rotation evidence."""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Literal

from pydantic import Field, field_serializer, model_validator

from mce.features.breadth.models import ReferenceParticipation
from mce.models.context import ContextContributor
from mce.models.enums import DataQuality
from mce.models.quality import (
    DirectionalScore,
    FiniteNumber,
    NonNegativeFinite,
    QualityTaggedModel,
    UnitScore,
)
from mce.models.sector import SectorName
from mce.models.symbol import Symbol
from mce.models.time import TimestampNs


class SectorMemberFeatures(QualityTaggedModel):
    symbol: Symbol
    weight: NonNegativeFinite | None = None
    current_price: NonNegativeFinite | None = None
    previous_close: NonNegativeFinite | None = None
    session_return: FiniteNumber | None = None
    stale: bool = False


class SectorContext(QualityTaggedModel):
    sector: SectorName
    as_of_ns: TimestampNs
    members: Mapping[str, SectorMemberFeatures]
    expected_member_count: int = Field(ge=0)
    valid_member_count: int = Field(ge=0)
    stale_member_count: int = Field(ge=0)
    member_coverage: UnitScore
    weight_coverage: UnitScore | None = None
    advancers: int = Field(ge=0)
    decliners: int = Field(ge=0)
    unchanged: int = Field(ge=0)
    net_advance_fraction: DirectionalScore | None = None
    breadth_references: Mapping[str, ReferenceParticipation]
    breadth_score: DirectionalScore | None = None
    absolute_returns: Mapping[str, FiniteNumber | None]
    horizon_coverage: Mapping[str, UnitScore]
    weighted_returns: Mapping[str, FiniteNumber | None]
    weight_coverage_by_horizon: Mapping[str, UnitScore | None]
    proxy_returns: Mapping[str, FiniteNumber | None]
    relative_returns: Mapping[str, FiniteNumber | None]
    proxy_divergence: Mapping[str, FiniteNumber | None]
    breadth_proxy_divergence: FiniteNumber | None = None
    cross_horizon_agreement: Literal["positive", "negative", "mixed", "unavailable"]
    absolute_strength_score: DirectionalScore | None = None
    relative_strength_score: DirectionalScore | None = None
    sector_momentum: FiniteNumber | None = None
    sector_acceleration: FiniteNumber | None = None
    realized_volatility: NonNegativeFinite | None = None
    volatility_short: NonNegativeFinite | None = None
    volatility_long: NonNegativeFinite | None = None
    volatility_percentile: UnitScore | None = None
    market_volatility: NonNegativeFinite | None = None
    relative_volatility: NonNegativeFinite | None = None
    rank: int | None = Field(default=None, gt=0)
    percentile: UnitScore | None = None
    leadership_persistence: UnitScore | None = None
    rank_change: int | None = None
    relative_strength_change: FiniteNumber | None = None
    evidence_quality: UnitScore
    data_quality: DataQuality
    contributors: tuple[ContextContributor, ...] = ()

    @field_serializer(
        "members",
        "breadth_references",
        "absolute_returns",
        "horizon_coverage",
        "weighted_returns",
        "weight_coverage_by_horizon",
        "proxy_returns",
        "relative_returns",
        "proxy_divergence",
    )
    def serialize_mapping(self, values: Mapping[str, object]) -> dict[str, object]:
        return dict(values)

    @model_validator(mode="after")
    def validate_context(self) -> "SectorContext":
        if len(self.members) != self.expected_member_count:
            raise ValueError("sector members must match expected count")
        if self.advancers + self.decliners + self.unchanged != self.valid_member_count:
            raise ValueError("sector directional counts must match valid members")
        if self.member_coverage != (
            self.valid_member_count / self.expected_member_count
            if self.expected_member_count
            else 0.0
        ):
            raise ValueError("sector member coverage must match counts")
        for field in (
            "members",
            "breadth_references",
            "absolute_returns",
            "horizon_coverage",
            "weighted_returns",
            "weight_coverage_by_horizon",
            "proxy_returns",
            "relative_returns",
            "proxy_divergence",
        ):
            object.__setattr__(self, field, MappingProxyType(dict(getattr(self, field))))
        return self


class SectorFeatureSet(QualityTaggedModel):
    as_of_ns: TimestampNs
    sectors: Mapping[str, SectorContext]
    leader_sector: SectorName | None = None
    laggard_sector: SectorName | None = None
    previous_leader_sector: SectorName | None = None
    leader_changed: bool | None = None
    leadership_spread: NonNegativeFinite | None = None
    leadership_concentration: UnitScore | None = None
    sector_score_dispersion: NonNegativeFinite | None = None
    rotation_velocity: UnitScore | None = None
    market_coverage: UnitScore
    evidence_quality: UnitScore
    data_quality: DataQuality
    contributors: tuple[ContextContributor, ...] = ()

    @field_serializer("sectors")
    def serialize_sectors(self, values: Mapping[str, SectorContext]) -> dict[str, SectorContext]:
        return dict(values)

    @model_validator(mode="after")
    def validate_result(self) -> "SectorFeatureSet":
        if any(
            name != item.sector or item.as_of_ns != self.as_of_ns
            for name, item in self.sectors.items()
        ):
            raise ValueError("sector context keys and clocks must match")
        object.__setattr__(self, "sectors", MappingProxyType(dict(sorted(self.sectors.items()))))
        return self
