"""Immutable descriptive transition, dwell, occupancy, and stability contracts."""

from pydantic import Field, model_validator

from mce.features.regime.taxonomy import HeadlineRegime
from mce.models.enums import DataQuality
from mce.models.quality import NonNegativeFinite, QualityTaggedModel, UnitScore
from mce.models.time import TimestampNs


class ObservedTransitionTarget(QualityTaggedModel):
    to_regime: HeadlineRegime
    count: int = Field(ge=1)
    observed_frequency: UnitScore


class ObservedTransitionRow(QualityTaggedModel):
    from_regime: HeadlineRegime
    observed_exits: int = Field(ge=1)
    targets: tuple[ObservedTransitionTarget, ...]

    @model_validator(mode="after")
    def validate_row(self) -> "ObservedTransitionRow":
        if sum(item.count for item in self.targets) != self.observed_exits:
            raise ValueError("observed transition counts must reconcile with exits")
        if abs(sum(item.observed_frequency for item in self.targets) - 1) > 1e-9:
            raise ValueError("observed transition frequencies must sum to one")
        return self


class RegimeDurationSummary(QualityTaggedModel):
    regime: HeadlineRegime
    completed_episodes: int = Field(ge=1)
    mean_sessions: NonNegativeFinite | None = None
    median_sessions: NonNegativeFinite | None = None
    minimum_sessions: int | None = Field(default=None, ge=0)
    maximum_sessions: int | None = Field(default=None, ge=0)
    mean_trading_ns: NonNegativeFinite | None = None


class RegimeOccupancy(QualityTaggedModel):
    regime: HeadlineRegime
    observed_trading_ns: int = Field(ge=1)
    fraction: UnitScore


class DominantTransitionPair(QualityTaggedModel):
    first: HeadlineRegime
    second: HeadlineRegime
    count: int = Field(ge=1)


class TransitionSequenceEntry(QualityTaggedModel):
    at_ns: TimestampNs
    from_regime: HeadlineRegime
    to_regime: HeadlineRegime
    changed_dimensions: tuple[str, ...]
    changed_dimension_fraction: UnitScore


class DirectedTransitionIntensity(QualityTaggedModel):
    from_regime: HeadlineRegime
    to_regime: HeadlineRegime
    observed_count: int = Field(ge=1)
    mean_changed_dimension_fraction: UnitScore


class TransitionWindowSummary(QualityTaggedModel):
    sessions: int = Field(ge=1)
    start_ns: TimestampNs
    observations: int = Field(ge=0)
    observed_sessions: int = Field(ge=0)
    headline_transitions: int = Field(ge=0)
    transition_rate_per_session: NonNegativeFinite | None = None
    headline_churn: UnitScore | None = None
    unique_regimes: int = Field(ge=0)
    reversal_count: int = Field(ge=0)
    reversal_fraction: UnitScore | None = None
    oscillation_score: UnitScore | None = None
    time_coverage: UnitScore | None = None
    unknown_fraction: UnitScore | None = None
    usable_quality_fraction: UnitScore | None = None
    occupancy: tuple[RegimeOccupancy, ...] = ()
    dominant_transition_pair: DominantTransitionPair | None = None


class DimensionTransitionSummary(QualityTaggedModel):
    dimension: str = Field(min_length=1)
    transition_count: int = Field(ge=0)
    transition_rate_per_session: NonNegativeFinite | None = None
    current_age_sessions: int | None = Field(default=None, ge=1)
    stability_score: UnitScore | None = None
    synchronized_headline_count: int = Field(ge=0)
    synchronized_headline_fraction: UnitScore | None = None


class StabilityContributor(QualityTaggedModel):
    name: str = Field(min_length=1)
    value: UnitScore
    weight: UnitScore
    contribution: UnitScore


class RegimeTransitionFeatureSet(QualityTaggedModel):
    as_of_ns: TimestampNs
    config_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    classifier_fingerprint: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    current_regime: HeadlineRegime | None = None
    current_regime_started_at_ns: TimestampNs | None = None
    current_age_ns: TimestampNs | None = None
    current_age_trading_ns: TimestampNs | None = None
    current_age_sessions: int | None = Field(default=None, ge=1)
    current_age_observations: int | None = Field(default=None, ge=1)
    primary_window_sessions: int = Field(ge=1)
    windows: tuple[TransitionWindowSummary, ...]
    observed_transition_frequency: tuple[ObservedTransitionRow, ...] = ()
    historical_durations: tuple[RegimeDurationSummary, ...] = ()
    recent_transition_sequence: tuple[TransitionSequenceEntry, ...] = ()
    directed_transition_intensity: tuple[DirectedTransitionIntensity, ...] = ()
    dimension_stability: tuple[DimensionTransitionSummary, ...] = ()
    synchronized_headline_fraction: UnitScore | None = None
    co_transition_count: int = Field(default=0, ge=0)
    dominant_unstable_dimension: str | None = None
    stability_score: UnitScore | None = None
    instability_score: UnitScore | None = None
    oscillation_score: UnitScore | None = None
    transition_intensity: UnitScore | None = None
    contributors: tuple[StabilityContributor, ...] = ()
    data_quality: DataQuality = DataQuality.UNKNOWN

    @model_validator(mode="after")
    def validate_result(self) -> "RegimeTransitionFeatureSet":
        if (
            self.current_regime_started_at_ns is not None
            and self.current_regime_started_at_ns > self.as_of_ns
        ):
            raise ValueError("current regime start cannot follow the result clock")
        if (
            self.stability_score is not None
            and self.instability_score is not None
            and abs(self.stability_score + self.instability_score - 1) > 1e-9
        ):
            raise ValueError("stability and instability must be complements")
        if (
            self.contributors
            and self.stability_score is not None
            and abs(sum(item.contribution for item in self.contributors) - self.stability_score)
            > 1e-9
        ):
            raise ValueError("stability contributors must reconcile")
        return self
