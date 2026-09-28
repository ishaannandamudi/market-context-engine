"""Immutable public realized-volatility feature records; annualized values are decimals."""

from collections.abc import Mapping
from types import MappingProxyType

from pydantic import Field, field_serializer, model_validator

from mce.models.context import ContextContributor
from mce.models.enums import DataQuality
from mce.models.quality import FiniteNumber, NonNegativeFinite, QualityTaggedModel, UnitScore
from mce.models.symbol import Symbol
from mce.models.time import TimestampNs


class EstimatorValue(QualityTaggedModel):
    annualized_volatility: NonNegativeFinite
    variance_per_session: NonNegativeFinite
    sample_count: int = Field(gt=0)
    sum_squared_returns: NonNegativeFinite | None = None


class IntradayVolatilityWindow(QualityTaggedModel):
    name: str
    annualized_volatility: NonNegativeFinite | None = None
    realized_variance: NonNegativeFinite | None = None
    bipower_variation: NonNegativeFinite | None = None
    jump_variation: NonNegativeFinite | None = None
    jump_share: UnitScore | None = None
    downside_semivariance: NonNegativeFinite | None = None
    upside_semivariance: NonNegativeFinite | None = None
    downside_share: UnitScore | None = None
    observed_return_count: int = Field(ge=0)
    expected_return_count: int = Field(ge=0)
    coverage_ratio: UnitScore


class DailyVolatilityWindow(QualityTaggedModel):
    name: str
    sessions: int = Field(ge=2)
    close_to_close: EstimatorValue | None = None
    parkinson: EstimatorValue | None = None
    garman_klass: EstimatorValue | None = None
    rogers_satchell: EstimatorValue | None = None
    yang_zhang: EstimatorValue | None = None
    overnight_variance_share: UnitScore | None = None
    intraday_variance_share: UnitScore | None = None
    latest_gap_magnitude: NonNegativeFinite | None = None
    estimator_median: NonNegativeFinite | None = None
    estimator_range: NonNegativeFinite | None = None
    estimator_std: NonNegativeFinite | None = None
    estimator_agreement: UnitScore | None = None
    observed_sessions: int = Field(ge=0)
    expected_sessions: int = Field(gt=0)
    coverage_ratio: UnitScore


class SymbolVolatilityFeatures(QualityTaggedModel):
    symbol: Symbol
    as_of_ns: TimestampNs
    intraday: Mapping[str, IntradayVolatilityWindow]
    daily: Mapping[str, DailyVolatilityWindow]
    rv_short: NonNegativeFinite | None = None
    rv_medium: NonNegativeFinite | None = None
    rv_long: NonNegativeFinite | None = None
    short_long_ratio: NonNegativeFinite | None = None
    medium_long_ratio: NonNegativeFinite | None = None
    volatility_acceleration: FiniteNumber | None = None
    volatility_change_1d: FiniteNumber | None = None
    volatility_percentile: UnitScore | None = None
    robust_z: FiniteNumber | None = None
    volatility_shock_score: UnitScore | None = None
    baseline_sample_count: int = Field(ge=0)
    sample_count: int = Field(ge=0)
    expected_count: int = Field(ge=0)
    coverage_ratio: UnitScore
    contributors: tuple[ContextContributor, ...] = ()

    @field_serializer("intraday", "daily")
    def serialize_windows(self, values: Mapping[str, object]) -> dict[str, object]:
        return dict(values)

    @model_validator(mode="after")
    def freeze_windows(self) -> "SymbolVolatilityFeatures":
        object.__setattr__(self, "intraday", MappingProxyType(dict(self.intraday)))
        object.__setattr__(self, "daily", MappingProxyType(dict(self.daily)))
        return self


class RealizedVolatilityFeatureSet(QualityTaggedModel):
    as_of_ns: TimestampNs
    per_symbol: Mapping[str, SymbolVolatilityFeatures]
    market_realized_volatility: NonNegativeFinite | None = None
    volatility_score: UnitScore | None = None
    index_percentiles: Mapping[str, UnitScore]
    cross_index_volatility_range: NonNegativeFinite | None = None
    cross_index_volatility_ratio: NonNegativeFinite | None = None
    available_symbol_count: int = Field(ge=0)
    expected_symbol_count: int = Field(ge=0)
    coverage_ratio: UnitScore
    data_quality: DataQuality
    contributors: tuple[ContextContributor, ...] = ()

    @field_serializer("per_symbol", "index_percentiles")
    def serialize_maps(self, values: Mapping[str, object]) -> dict[str, object]:
        return dict(values)

    @model_validator(mode="after")
    def freeze_maps(self) -> "RealizedVolatilityFeatureSet":
        if self.available_symbol_count > self.expected_symbol_count:
            raise ValueError("available symbols cannot exceed expected symbols")
        if self.coverage_ratio != (
            self.available_symbol_count / self.expected_symbol_count
            if self.expected_symbol_count
            else 0.0
        ):
            raise ValueError("coverage_ratio must match counts")
        if any(
            symbol != item.symbol or item.as_of_ns != self.as_of_ns
            for symbol, item in self.per_symbol.items()
        ):
            raise ValueError("per-symbol records must match keys and as_of_ns")
        object.__setattr__(self, "per_symbol", MappingProxyType(dict(self.per_symbol)))
        object.__setattr__(
            self, "index_percentiles", MappingProxyType(dict(self.index_percentiles))
        )
        return self
