"""Versioned provider-neutral macro, release, and event-calendar inputs."""

from enum import StrEnum
from math import isfinite
from typing import Literal

from pydantic import ConfigDict, Field, model_validator

from mce.models._base import FrozenModel
from mce.models.enums import DataQuality
from mce.models.quality import FiniteNumber, SourceId
from mce.models.time import TimestampNs

MACRO_CONTEXT_SCHEMA_VERSION = "1.0"
DAY_NS = 86_400_000_000_000


class MacroCategory(StrEnum):
    POLICY_RATE = "policy_rate"
    TREASURY_YIELD = "treasury_yield"
    INFLATION = "inflation"
    EMPLOYMENT = "employment"
    UNEMPLOYMENT = "unemployment"
    WAGES = "wages"
    GROWTH = "growth"
    ACTIVITY = "activity"
    CONSUMER = "consumer"
    HOUSING = "housing"
    CREDIT = "credit"
    LIQUIDITY = "liquidity"
    OTHER = "other"


class MacroUnit(StrEnum):
    PERCENT = "percent"
    PERCENTAGE_POINTS = "percentage_points"
    BASIS_POINTS = "basis_points"
    DECIMAL_RATE = "decimal_rate"
    INDEX_LEVEL = "index_level"
    ANNUALIZED_RATE = "annualized_rate"
    MILLIONS = "millions"
    THOUSANDS = "thousands"
    RATIO = "ratio"


class SeriesCadence(StrEnum):
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    EVENT_DRIVEN = "event_driven"


class RevisionStatus(StrEnum):
    PRELIMINARY = "preliminary"
    FINAL = "final"
    REVISED = "revised"


class MacroEventType(StrEnum):
    FOMC_RATE_DECISION = "fomc_rate_decision"
    FOMC_MINUTES = "fomc_minutes"
    CPI = "cpi"
    CORE_CPI = "core_cpi"
    PCE = "pce"
    CORE_PCE = "core_pce"
    NONFARM_PAYROLLS = "nonfarm_payrolls"
    UNEMPLOYMENT_RATE = "unemployment_rate"
    INITIAL_CLAIMS = "initial_claims"
    GDP = "gdp"
    RETAIL_SALES = "retail_sales"
    ISM_MANUFACTURING = "ism_manufacturing"
    ISM_SERVICES = "ism_services"
    CONSUMER_SENTIMENT = "consumer_sentiment"
    TREASURY_AUCTION = "treasury_auction"
    OTHER = "other"


class EventImportance(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class MacroObservation(FrozenModel):
    source: SourceId
    series_id: str = Field(min_length=1)
    category: MacroCategory
    cadence: SeriesCadence
    value: FiniteNumber
    unit: MacroUnit
    period_end_ns: TimestampNs
    release_at_ns: TimestampNs
    available_at_ns: TimestampNs
    vintage_at_ns: TimestampNs
    revision_status: RevisionStatus = RevisionStatus.FINAL
    quality: DataQuality = DataQuality.GOOD
    tenor_months: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_observation(self) -> "MacroObservation":
        if (
            not self.period_end_ns
            <= self.release_at_ns
            <= self.vintage_at_ns
            <= self.available_at_ns
        ):
            raise ValueError(
                "macro observation period, release, vintage, or availability is inconsistent"
            )
        rate = self.category in (MacroCategory.POLICY_RATE, MacroCategory.TREASURY_YIELD)
        if rate and self.unit not in (
            MacroUnit.PERCENT,
            MacroUnit.DECIMAL_RATE,
            MacroUnit.BASIS_POINTS,
        ):
            raise ValueError("rate observations require an explicit convertible rate unit")
        if rate and self.unit == MacroUnit.DECIMAL_RATE and not isfinite(self.value * 100):
            raise ValueError("decimal rate cannot be represented in percent units")
        if (self.category == MacroCategory.TREASURY_YIELD) != (self.tenor_months is not None):
            raise ValueError("Treasury yields require tenor months; other series cannot set tenor")
        return self


class MacroRelease(FrozenModel):
    source: SourceId
    event_id: str = Field(min_length=1)
    series_id: str = Field(min_length=1)
    category: MacroCategory
    cadence: SeriesCadence
    actual: FiniteNumber
    consensus: FiniteNumber | None = None
    consensus_available_at_ns: TimestampNs | None = None
    prior: FiniteNumber | None = None
    revised_prior: FiniteNumber | None = None
    unit: MacroUnit
    period_end_ns: TimestampNs
    release_at_ns: TimestampNs
    available_at_ns: TimestampNs
    vintage_at_ns: TimestampNs
    revision_status: RevisionStatus = RevisionStatus.FINAL
    quality: DataQuality = DataQuality.GOOD

    @model_validator(mode="after")
    def validate_release(self) -> "MacroRelease":
        if (
            not self.period_end_ns
            <= self.release_at_ns
            <= self.vintage_at_ns
            <= self.available_at_ns
        ):
            raise ValueError("macro release clocks are inconsistent")
        if self.category in (
            MacroCategory.POLICY_RATE,
            MacroCategory.TREASURY_YIELD,
        ) and self.unit not in (MacroUnit.PERCENT, MacroUnit.DECIMAL_RATE, MacroUnit.BASIS_POINTS):
            raise ValueError("rate releases require an explicit convertible rate unit")
        if (
            self.category in (MacroCategory.POLICY_RATE, MacroCategory.TREASURY_YIELD)
            and self.unit == MacroUnit.DECIMAL_RATE
            and not isfinite(self.actual * 100)
        ):
            raise ValueError("decimal rate cannot be represented in percent units")
        if (self.consensus is None) != (self.consensus_available_at_ns is None):
            raise ValueError("consensus and its availability must be supplied together")
        if (
            self.consensus_available_at_ns is not None
            and self.consensus_available_at_ns > self.release_at_ns
        ):
            raise ValueError("consensus must be known no later than the release")
        if self.revised_prior is not None and self.prior is None:
            raise ValueError("revised prior requires an original prior")
        return self


class ScheduledMacroEvent(FrozenModel):
    source: SourceId
    event_id: str = Field(min_length=1)
    event_type: MacroEventType
    scheduled_at_ns: TimestampNs
    calendar_available_at_ns: TimestampNs
    importance: EventImportance
    affected_series: tuple[str, ...] = ()
    expected_release_window_ns: int | None = Field(default=None, gt=0)
    previous_scheduled_at_ns: TimestampNs | None = None
    status: Literal["scheduled", "cancelled"] = "scheduled"
    quality: DataQuality = DataQuality.GOOD

    @model_validator(mode="after")
    def validate_event(self) -> "ScheduledMacroEvent":
        if self.previous_scheduled_at_ns == self.scheduled_at_ns:
            raise ValueError("schedule revision must change event time")
        if len(set(self.affected_series)) != len(self.affected_series):
            raise ValueError("affected series must be unique")
        return self


class MacroContextInput(FrozenModel):
    """A source batch; children were known by the batch calculation clock."""

    model_config = ConfigDict(frozen=True, extra="ignore", strict=True, validate_default=True)

    macro_context_schema_version: Literal["1.0"]
    source: SourceId
    source_version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    source_fingerprint: str | None = None
    as_of_ns: TimestampNs
    available_at_ns: TimestampNs
    quality: DataQuality = DataQuality.GOOD
    observations: tuple[MacroObservation, ...] = ()
    releases: tuple[MacroRelease, ...] = ()
    scheduled_events: tuple[ScheduledMacroEvent, ...] = ()
    calendar_coverage_declared: bool = False

    @model_validator(mode="after")
    def validate_batch(self) -> "MacroContextInput":
        if self.as_of_ns > self.available_at_ns:
            raise ValueError("macro source availability precedes calculation")
        for item in self.observations:
            if item.source != self.source or item.available_at_ns > self.as_of_ns:
                raise ValueError("macro child source or availability exceeds batch clock")
        for release in self.releases:
            if release.source != self.source or release.available_at_ns > self.as_of_ns:
                raise ValueError("macro child source or availability exceeds batch clock")
        for event in self.scheduled_events:
            if event.source != self.source or event.calendar_available_at_ns > self.as_of_ns:
                raise ValueError("event source or announcement exceeds batch clock")
        return self
