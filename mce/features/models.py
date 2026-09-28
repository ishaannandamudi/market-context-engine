"""Immutable, finite feature results with stable serialization."""

from collections.abc import Mapping
from types import MappingProxyType

from pydantic import Field, field_serializer, model_validator

from mce.models.context import ContextContributor
from mce.models.enums import DataQuality
from mce.models.quality import (
    DirectionalScore,
    FiniteNumber,
    QualityTaggedModel,
    UnitScore,
)
from mce.models.symbol import Symbol
from mce.models.time import TimestampNs


class ReturnFeature(QualityTaggedModel):
    name: str
    value: FiniteNumber | None = None
    reference_price: FiniteNumber | None = None
    current_price: FiniteNumber | None = None
    sample_count: int = Field(ge=0)
    expected_count: int | None = Field(default=None, gt=0)
    coverage_ratio: UnitScore | None = None


class RegressionFeature(QualityTaggedModel):
    slope_per_hour: FiniteNumber | None = None
    intercept: FiniteNumber | None = None
    r_squared: UnitScore | None = None
    sample_count: int = Field(ge=0)
    expected_count: int | None = Field(default=None, gt=0)
    coverage_ratio: UnitScore | None = None


class SymbolTrendFeatures(QualityTaggedModel):
    symbol: Symbol
    as_of_ns: TimestampNs
    returns: Mapping[str, ReturnFeature]
    slope_short: RegressionFeature
    slope_long: RegressionFeature
    persistence: UnitScore | None = None
    path_efficiency: UnitScore | None = None
    directional_agreement: UnitScore | None = None
    acceleration_per_hour: FiniteNumber | None = None
    distance_from_sma: FiniteNumber | None = None
    trend_score: DirectionalScore | None = None
    sample_count: int = Field(ge=0)
    expected_count: int | None = Field(default=None, gt=0)
    coverage_ratio: UnitScore | None = None
    contributors: tuple[ContextContributor, ...] = ()

    @field_serializer("returns")
    def serialize_returns(self, values: Mapping[str, ReturnFeature]) -> dict[str, ReturnFeature]:
        return dict(values)

    @model_validator(mode="after")
    def freeze_returns(self) -> "SymbolTrendFeatures":
        object.__setattr__(self, "returns", MappingProxyType(dict(self.returns)))
        return self


class TrendFeatureSet(QualityTaggedModel):
    as_of_ns: TimestampNs
    symbols: Mapping[str, SymbolTrendFeatures]
    index_scores: Mapping[str, DirectionalScore]
    market_direction_score: DirectionalScore | None = None
    trend_score: DirectionalScore | None = None
    index_agreement: UnitScore | None = None
    available_symbol_count: int = Field(ge=0)
    expected_symbol_count: int = Field(ge=0)
    coverage_ratio: UnitScore
    data_quality: DataQuality
    contributors: tuple[ContextContributor, ...] = ()

    @field_serializer("symbols", "index_scores")
    def serialize_maps(self, values: Mapping[str, object]) -> dict[str, object]:
        return dict(values)

    @model_validator(mode="after")
    def freeze_maps(self) -> "TrendFeatureSet":
        if self.available_symbol_count != len(self.index_scores):
            raise ValueError("available_symbol_count must match index_scores")
        if self.available_symbol_count > self.expected_symbol_count:
            raise ValueError("available symbols cannot exceed expected symbols")
        expected_coverage = (
            self.available_symbol_count / self.expected_symbol_count
            if self.expected_symbol_count
            else 0.0
        )
        if self.coverage_ratio != expected_coverage:
            raise ValueError("coverage_ratio must match symbol counts")
        if not set(self.index_scores).issubset(self.symbols):
            raise ValueError("index_scores symbols must have feature records")
        if any(
            symbol != item.symbol or item.as_of_ns != self.as_of_ns
            for symbol, item in self.symbols.items()
        ):
            raise ValueError("symbol features must match keys and as_of_ns")
        object.__setattr__(self, "symbols", MappingProxyType(dict(self.symbols)))
        object.__setattr__(self, "index_scores", MappingProxyType(dict(self.index_scores)))
        return self
