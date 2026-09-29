"""Immutable, serializable correlation evidence with deterministic matrix order."""

from pydantic import Field, model_validator

from mce.models._base import FrozenModel
from mce.models.context import ContextContributor
from mce.models.enums import DataQuality
from mce.models.quality import FiniteNumber, QualityTaggedModel, UnitScore
from mce.models.symbol import Symbol
from mce.models.time import TimestampNs


class CorrelationMatrix(FrozenModel):
    symbols: tuple[Symbol, ...]
    values: tuple[tuple[FiniteNumber, ...], ...]

    @model_validator(mode="after")
    def validate_matrix(self) -> "CorrelationMatrix":
        size = len(self.symbols)
        if size < 2 or self.symbols != tuple(sorted(set(self.symbols))):
            raise ValueError("matrix symbols must be unique, sorted and at least two")
        if len(self.values) != size or any(len(row) != size for row in self.values):
            raise ValueError("matrix dimension and symbol count differ")
        for index, row in enumerate(self.values):
            if abs(row[index] - 1) > 1e-9:
                raise ValueError("correlation matrix diagonal must be one")
            if any(abs(value) > 1 + 1e-9 for value in row):
                raise ValueError("correlation values must be in [-1,1]")
            if any(abs(row[j] - self.values[j][index]) > 1e-9 for j in range(size)):
                raise ValueError("correlation matrix must be symmetric")
        return self


class SectorPairCorrelation(FrozenModel):
    first: str
    second: str
    pair_count: int = Field(gt=0)
    average: FiniteNumber


class HorizonCorrelation(FrozenModel):
    name: str
    expected_observations: int = Field(ge=0)
    aligned_sample_count: int = Field(ge=0)
    time_coverage: UnitScore
    symbol_coverage: UnitScore
    average: FiniteNumber | None = None
    matrix: CorrelationMatrix | None = None


class CorrelationFeatureSet(QualityTaggedModel):
    as_of_ns: TimestampNs
    expected_symbols: tuple[Symbol, ...]
    valid_symbols: tuple[Symbol, ...]
    stale_symbols: tuple[Symbol, ...] = ()
    missing_symbols: tuple[Symbol, ...] = ()
    constant_symbols: tuple[Symbol, ...] = ()
    expected_symbol_count: int = Field(ge=0)
    valid_symbol_count: int = Field(ge=0)
    symbol_coverage: UnitScore
    expected_observations: int = Field(ge=0)
    aligned_sample_count: int = Field(ge=0)
    time_coverage: UnitScore
    matrix_dimension: int = Field(ge=0)
    correlation_matrix: CorrelationMatrix | None = None
    spearman_matrix: CorrelationMatrix | None = None
    shrunk_matrix: CorrelationMatrix | None = None
    average_pairwise_correlation: FiniteNumber | None = None
    median_pairwise_correlation: FiniteNumber | None = None
    weighted_average_correlation: FiniteNumber | None = None
    correlation_dispersion: FiniteNumber | None = None
    spearman_average: FiniteNumber | None = None
    short_correlation: FiniteNumber | None = None
    medium_correlation: FiniteNumber | None = None
    long_correlation: FiniteNumber | None = None
    daily_horizons: tuple[HorizonCorrelation, ...] = ()
    correlation_acceleration: FiniteNumber | None = None
    correlation_change: FiniteNumber | None = None
    session_change: FiniteNumber | None = None
    five_session_change: FiniteNumber | None = None
    correlation_percentile: UnitScore | None = None
    correlation_robust_z: FiniteNumber | None = None
    largest_eigenvalue: FiniteNumber | None = None
    first_eigenvalue_share: UnitScore | None = None
    effective_dimension: FiniteNumber | None = None
    market_mode_strength: UnitScore | None = None
    systemic_co_movement_score: UnitScore | None = None
    cluster_groups: tuple[tuple[Symbol, ...], ...] = ()
    sector_pairs: tuple[SectorPairCorrelation, ...] = ()
    within_sector_correlation: FiniteNumber | None = None
    cross_sector_correlation: FiniteNumber | None = None
    within_cross_spread: FiniteNumber | None = None
    weight_coverage: UnitScore | None = None
    evidence_quality: UnitScore
    data_quality: DataQuality
    contributors: tuple[ContextContributor, ...] = ()

    @model_validator(mode="after")
    def validate_counts(self) -> "CorrelationFeatureSet":
        if self.expected_symbol_count != len(self.expected_symbols):
            raise ValueError("expected symbol count mismatch")
        if self.valid_symbol_count != len(self.valid_symbols):
            raise ValueError("valid symbol count mismatch")
        if self.matrix_dimension != (
            len(self.correlation_matrix.symbols) if self.correlation_matrix else 0
        ):
            raise ValueError("matrix dimension mismatch")
        return self
