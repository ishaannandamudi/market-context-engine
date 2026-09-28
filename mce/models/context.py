"""Versioned public market-context snapshot and small future-facing contracts."""

import re
from collections.abc import Mapping
from datetime import date
from math import isfinite
from types import MappingProxyType
from typing import TypeAlias

from pydantic import Field, field_serializer, field_validator, model_validator

from mce.models._base import FrozenModel
from mce.models.enums import (
    BreadthRegime,
    CorrelationRegime,
    DataQuality,
    DispersionRegime,
    EventRiskStatus,
    LiquidityRegime,
    MarketRegime,
    MarketSession,
    TrendRegime,
    VolatilityRegime,
)
from mce.models.quality import (
    ConfidenceScore,
    DirectionalScore,
    FiniteNumber,
    NonNegativeFinite,
    QualityTaggedModel,
    SourceId,
    UnitScore,
)
from mce.models.time import TimestampNs, ns_to_iso
from mce.version import SCHEMA_VERSION, __version__

FeatureScalar: TypeAlias = str | bool | int | float


class SourceFreshness(FrozenModel):
    source: SourceId
    latest_effective_timestamp_ns: TimestampNs
    age_ns: TimestampNs
    is_stale: bool
    freshness_limit_ns: TimestampNs | None = None


class ContextContributor(FrozenModel):
    name: str = Field(min_length=1)
    value: FiniteNumber
    normalized_value: DirectionalScore | None = None
    contribution: FiniteNumber | None = None
    description: str | None = None
    source_feature: str | None = None


class EventRisk(FrozenModel):
    status: EventRiskStatus = EventRiskStatus.UNKNOWN
    event_type: str | None = None
    event_name: str | None = None
    scheduled_timestamp_ns: TimestampNs | None = None
    seconds_to_event: int | None = None
    importance: UnitScore | None = None
    risk_score: UnitScore | None = None


class ImpliedVolatilityContext(FrozenModel):
    reference_symbol: str = Field(min_length=1)
    implied_volatility: NonNegativeFinite
    percentile: UnitScore | None = None
    source: SourceId
    as_of_ns: TimestampNs


class TermStructurePoint(FrozenModel):
    tenor_days: int = Field(gt=0)
    implied_volatility: NonNegativeFinite


class VolatilityTermStructureContext(FrozenModel):
    reference_symbol: str = Field(min_length=1)
    points: tuple[TermStructurePoint, ...]
    source: SourceId
    as_of_ns: TimestampNs


class VolatilitySkewContext(FrozenModel):
    reference_symbol: str = Field(min_length=1)
    put_call_skew: FiniteNumber | None = None
    source: SourceId
    as_of_ns: TimestampNs


class MarketContextSnapshot(QualityTaggedModel):
    """As-of public output. Every populated input must be effectively available by as_of_ns.

    Analytics not produced yet remain None. A zero is a real observed/calculated value.
    """

    schema_version: str = SCHEMA_VERSION
    engine_version: str = __version__
    as_of_ns: TimestampNs
    as_of_iso: str = ""

    market_session: MarketSession = MarketSession.UNKNOWN
    session_trading_date: date | None = None
    session_identifier: str | None = None

    market_regime: MarketRegime = MarketRegime.UNKNOWN
    trend_regime: TrendRegime = TrendRegime.UNKNOWN
    volatility_regime: VolatilityRegime = VolatilityRegime.UNKNOWN
    correlation_regime: CorrelationRegime = CorrelationRegime.UNKNOWN
    liquidity_regime: LiquidityRegime = LiquidityRegime.UNKNOWN
    dispersion_regime: DispersionRegime = DispersionRegime.UNKNOWN
    breadth_regime: BreadthRegime = BreadthRegime.UNKNOWN

    market_direction_score: DirectionalScore | None = None
    trend_score: DirectionalScore | None = None
    breadth_score: DirectionalScore | None = None
    dispersion_score: UnitScore | None = None
    correlation_score: UnitScore | None = None
    stress_score: UnitScore | None = None
    liquidity_score: UnitScore | None = None
    volatility_score: UnitScore | None = None
    risk_on_off_score: DirectionalScore | None = None

    realized_volatility: NonNegativeFinite | None = None
    index_scores: Mapping[str, DirectionalScore] | None = None
    sector_scores: Mapping[str, DirectionalScore] | None = None

    implied_volatility_context: ImpliedVolatilityContext | None = None
    volatility_term_structure: VolatilityTermStructureContext | None = None
    volatility_skew_context: VolatilitySkewContext | None = None
    event_risk: EventRisk | None = None

    confidence: ConfidenceScore | None = None
    data_quality: DataQuality = DataQuality.UNKNOWN
    source_freshness: tuple[SourceFreshness, ...] | None = None
    contributors: tuple[ContextContributor, ...] | None = None
    feature_values: Mapping[str, FeatureScalar] | None = None

    @field_serializer("index_scores", "sector_scores", "feature_values")
    def serialize_mapping(self, value: Mapping[str, object] | None) -> dict[str, object] | None:
        return dict(value) if value is not None else None

    @field_validator("feature_values")
    @classmethod
    def validate_features(
        cls, values: Mapping[str, FeatureScalar] | None
    ) -> Mapping[str, FeatureScalar] | None:
        if values is not None:
            for name, value in values.items():
                if not name:
                    raise ValueError("feature names must be nonempty")
                if isinstance(value, float) and not isfinite(value):
                    raise ValueError("feature values must be finite")
        return values

    @model_validator(mode="after")
    def validate_contract(self) -> "MarketContextSnapshot":
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version: {self.schema_version}")
        if re.fullmatch(r"\d+\.\d+\.\d+", self.engine_version) is None:
            raise ValueError("engine_version must be a semantic version")
        expected_iso = ns_to_iso(self.as_of_ns)
        if self.as_of_iso and self.as_of_iso != expected_iso:
            raise ValueError("as_of_iso does not match as_of_ns")
        object.__setattr__(self, "as_of_iso", expected_iso)
        if self.source_freshness is not None:
            sources = [entry.source for entry in self.source_freshness]
            if len(sources) != len(set(sources)):
                raise ValueError("source_freshness source identifiers must be unique")
            for entry in self.source_freshness:
                if entry.latest_effective_timestamp_ns > self.as_of_ns:
                    raise ValueError("source freshness cannot describe future data")
                if entry.age_ns != self.as_of_ns - entry.latest_effective_timestamp_ns:
                    raise ValueError("source freshness age_ns must match as_of_ns")
            object.__setattr__(
                self,
                "source_freshness",
                tuple(sorted(self.source_freshness, key=lambda item: item.source)),
            )
        for context in (
            self.implied_volatility_context,
            self.volatility_term_structure,
            self.volatility_skew_context,
        ):
            if context is not None and context.as_of_ns > self.as_of_ns:
                raise ValueError("options context cannot be newer than snapshot")
        for field_name in ("index_scores", "sector_scores", "feature_values"):
            mapping = getattr(self, field_name)
            if mapping is not None:
                object.__setattr__(self, field_name, MappingProxyType(dict(mapping)))
        return self
