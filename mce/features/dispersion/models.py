"""Immutable, JSON-safe cross-sectional differentiation evidence."""

from pydantic import Field, model_validator

from mce.models.context import ContextContributor
from mce.models.enums import DataQuality
from mce.models.quality import FiniteNumber, NonNegativeFinite, QualityTaggedModel, UnitScore
from mce.models.symbol import Symbol
from mce.models.time import TimestampNs


class HorizonDispersion(QualityTaggedModel):
    name: str
    valid_member_count: int = Field(ge=0)
    member_coverage: UnitScore
    return_std: NonNegativeFinite | None = None
    return_mad: NonNegativeFinite | None = None


class SectorReturnDispersion(QualityTaggedModel):
    sector: str
    member_count: int = Field(gt=0)
    mean_return: FiniteNumber
    return_std: NonNegativeFinite | None = None


class CorrelationDispersionContext(QualityTaggedModel):
    average_correlation: FiniteNumber | None = None
    systemic_correlation_score: UnitScore | None = None
    correlation_percentile: UnitScore | None = None
    return_dispersion: NonNegativeFinite | None = None
    dispersion_percentile: UnitScore | None = None
    differentiation_score: UnitScore | None = None
    dispersion_minus_correlation: FiniteNumber | None = None


class DispersionFeatureSet(QualityTaggedModel):
    as_of_ns: TimestampNs
    expected_symbols: tuple[Symbol, ...]
    valid_symbols: tuple[Symbol, ...]
    stale_symbols: tuple[Symbol, ...] = ()
    missing_symbols: tuple[Symbol, ...] = ()
    expected_member_count: int = Field(ge=0)
    valid_member_count: int = Field(ge=0)
    member_coverage: UnitScore
    weight_coverage: UnitScore | None = None
    return_dispersion: NonNegativeFinite | None = None
    robust_return_dispersion: NonNegativeFinite | None = None
    return_mad: NonNegativeFinite | None = None
    return_iqr: NonNegativeFinite | None = None
    return_tail_spread: NonNegativeFinite | None = None
    return_range: NonNegativeFinite | None = None
    weighted_return_dispersion: NonNegativeFinite | None = None
    upside_dispersion: NonNegativeFinite | None = None
    downside_dispersion: NonNegativeFinite | None = None
    downside_upside_spread: FiniteNumber | None = None
    cross_sectional_skewness: FiniteNumber | None = None
    cross_sectional_excess_kurtosis: FiniteNumber | None = None
    horizons: tuple[HorizonDispersion, ...] = ()
    short_dispersion: NonNegativeFinite | None = None
    medium_dispersion: NonNegativeFinite | None = None
    long_dispersion: NonNegativeFinite | None = None
    short_long_ratio: NonNegativeFinite | None = None
    dispersion_acceleration: FiniteNumber | None = None
    dispersion_change: FiniteNumber | None = None
    session_change: FiniteNumber | None = None
    five_session_change: FiniteNumber | None = None
    baseline_sample_count: int = Field(ge=0)
    dispersion_percentile: UnitScore | None = None
    dispersion_robust_z: FiniteNumber | None = None
    dispersion_shock_score: UnitScore | None = None
    dispersion_persistence: UnitScore | None = None
    sector_dispersion: NonNegativeFinite | None = None
    sector_relative_dispersion: NonNegativeFinite | None = None
    sector_relative_mad: NonNegativeFinite | None = None
    sector_spread: NonNegativeFinite | None = None
    sector_valid_count: int = Field(ge=0)
    sector_return_dispersion: tuple[SectorReturnDispersion, ...] = ()
    decomposition_member_count: int = Field(ge=0)
    decomposition_coverage: UnitScore | None = None
    decomposition_total_variance: NonNegativeFinite | None = None
    within_sector_variance: NonNegativeFinite | None = None
    between_sector_variance: NonNegativeFinite | None = None
    within_sector_share: UnitScore | None = None
    between_sector_share: UnitScore | None = None
    volatility_dispersion: NonNegativeFinite | None = None
    robust_volatility_dispersion: NonNegativeFinite | None = None
    relative_volatility_dispersion: NonNegativeFinite | None = None
    volatility_member_count: int = Field(ge=0)
    volatility_coverage: UnitScore | None = None
    movement_hhi: UnitScore | None = None
    effective_movers: NonNegativeFinite | None = None
    weighted_movement_hhi: UnitScore | None = None
    weighted_effective_movers: NonNegativeFinite | None = None
    correlation_context: CorrelationDispersionContext | None = None
    dispersion_score: UnitScore | None = None
    evidence_quality: UnitScore
    data_quality: DataQuality
    contributors: tuple[ContextContributor, ...] = ()

    @model_validator(mode="after")
    def validate_counts(self) -> "DispersionFeatureSet":
        if self.expected_member_count != len(self.expected_symbols):
            raise ValueError("expected member count does not match symbols")
        if self.valid_member_count != len(self.valid_symbols):
            raise ValueError("valid member count does not match symbols")
        if self.member_coverage != (
            self.valid_member_count / self.expected_member_count
            if self.expected_member_count
            else 0.0
        ):
            raise ValueError("member coverage does not match counts")
        if (
            self.within_sector_share is not None
            and self.between_sector_share is not None
            and abs(self.within_sector_share + self.between_sector_share - 1) > 1e-9
        ):
            raise ValueError("sector variance shares must sum to one")
        return self
