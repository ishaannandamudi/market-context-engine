"""Typed descriptive macro feature results with preserved source evidence."""

from collections.abc import Mapping
from hashlib import sha256
from types import MappingProxyType
from typing import Literal

from pydantic import Field, field_serializer, model_validator

from mce.features.macro.contracts import (
    MacroCategory,
    MacroContextInput,
    MacroObservation,
    MacroRelease,
    MacroUnit,
    ScheduledMacroEvent,
    SeriesCadence,
)
from mce.models.enums import DataQuality
from mce.models.quality import FiniteNumber, NonNegativeFinite, QualityTaggedModel, UnitScore
from mce.models.time import TimestampNs
from mce.serialization.canonical import canonical_json_bytes


class MacroSourceMetadata(QualityTaggedModel):
    source: str
    source_version: str
    source_fingerprint: str | None = None
    as_of_ns: TimestampNs
    available_at_ns: TimestampNs
    quality: DataQuality

    @classmethod
    def from_input(cls, item: MacroContextInput) -> "MacroSourceMetadata":
        return cls(
            source=item.source,
            source_version=item.source_version,
            source_fingerprint=item.source_fingerprint,
            as_of_ns=item.as_of_ns,
            available_at_ns=item.available_at_ns,
            quality=item.quality,
        )


class MacroComparison(QualityTaggedModel):
    sample_count: int = Field(ge=0)
    percentile: UnitScore | None = None
    robust_z: FiniteNumber | None = None
    median: FiniteNumber | None = None


class MacroSurpriseContext(QualityTaggedModel):
    raw: FiniteNumber | None = None
    normalized_z: FiniteNumber | None = None
    historical_scale: NonNegativeFinite | None = None
    historical_sample_count: int = Field(default=0, ge=0)
    orientation_axis: Literal[
        "inflation_pressure", "growth_strength", "labor_strength", "policy_tightening", "none"
    ] = "none"
    oriented_z: FiniteNumber | None = None


class MacroSeriesContext(QualityTaggedModel):
    source: str
    series_id: str
    category: MacroCategory
    unit: MacroUnit
    cadence: SeriesCadence
    observation: MacroObservation | None = None
    previous_value: FiniteNumber | None = None
    change: FiniteNumber | None = None
    history: MacroComparison | None = None
    release: MacroRelease | None = None
    surprise: MacroSurpriseContext | None = None
    stale: bool = False
    release_stale: bool = False
    data_quality: DataQuality


class MacroDimension(QualityTaggedModel):
    name: Literal["inflation", "growth", "labor"]
    primary_series_id: str | None = None
    latest_value: FiniteNumber | None = None
    unit: MacroUnit | None = None
    latest_surprise: FiniteNumber | None = None
    latest_surprise_z: FiniteNumber | None = None
    normalized_signal: FiniteNumber | None = None
    contributing_series: int = Field(ge=0)


class MacroEventContext(QualityTaggedModel):
    next_event: ScheduledMacroEvent | None = None
    next_high_importance: ScheduledMacroEvent | None = None
    recent_release: MacroRelease | None = None
    seconds_to_next: int | None = None
    hours_to_next: float | None = None
    hours_to_next_high: float | None = None
    seconds_since_recent_release: int | None = None
    upcoming_count_24h: int | None = Field(default=None, ge=0)
    high_importance_count_24h: int | None = Field(default=None, ge=0)
    weighted_count_24h: NonNegativeFinite | None = None
    event_count_window: int | None = Field(default=None, ge=0)
    weighted_count_window: NonNegativeFinite | None = None
    proximity_score: UnitScore | None = None


class PolicyRateContext(QualityTaggedModel):
    series_id: str | None = None
    current_rate_percent: FiniteNumber | None = None
    latest_change_pp: FiniteNumber | None = None
    trailing_change_count: int | None = Field(default=None, ge=0)
    cumulative_change_pp: FiniteNumber | None = None
    seconds_since_last_change: int | None = Field(default=None, ge=0)


class YieldCurveContext(QualityTaggedModel):
    yields_percent: Mapping[str, float] = Field(default_factory=dict)
    observed_period_end_ns: TimestampNs | None = None
    spread_2s10s_pp: FiniteNumber | None = None
    spread_3m10y_pp: FiniteNumber | None = None
    spread_30y5y_pp: FiniteNumber | None = None
    level_10y_percent: FiniteNumber | None = None
    mean_level_percent: FiniteNumber | None = None
    curvature_2y5y10y_pp: FiniteNumber | None = None
    inversion_magnitude_pp: NonNegativeFinite | None = None
    inverted: bool | None = None
    change_10y_1d_pp: FiniteNumber | None = None
    change_10y_5d_pp: FiniteNumber | None = None
    change_10y_20d_pp: FiniteNumber | None = None
    parallel_shift_5d_pp: FiniteNumber | None = None
    steepening_2s10s_5d_pp: FiniteNumber | None = None

    @field_serializer("yields_percent")
    def serialize_yields(self, value: Mapping[str, float]) -> dict[str, float]:
        return dict(value)

    @model_validator(mode="after")
    def freeze_yields(self) -> "YieldCurveContext":
        object.__setattr__(self, "yields_percent", MappingProxyType(dict(self.yields_percent)))
        return self


class MacroStressContext(QualityTaggedModel):
    rates: UnitScore | None = None
    curve: UnitScore | None = None
    surprise: UnitScore | None = None
    score: UnitScore | None = None
    active_weights: tuple[tuple[str, float], ...] = ()


class MacroContextFeatureSet(QualityTaggedModel):
    as_of_ns: TimestampNs
    config_fingerprint: str
    sources: tuple[MacroSourceMetadata, ...] = ()
    event: MacroEventContext | None = None
    per_series: tuple[MacroSeriesContext, ...] = ()
    dimensions: tuple[MacroDimension, ...] = ()
    policy: PolicyRateContext | None = None
    curve: YieldCurveContext | None = None
    stress: MacroStressContext | None = None
    data_quality: DataQuality

    @property
    def fingerprint(self) -> str:
        return sha256(canonical_json_bytes(self)).hexdigest()
