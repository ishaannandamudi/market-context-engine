"""Versioned, provider-neutral observations supplied by an external options engine."""

from typing import Literal

from pydantic import ConfigDict, Field, model_validator

from mce.models._base import FrozenModel
from mce.models.enums import DataQuality
from mce.models.quality import FiniteNumber, NonNegativeFinite, SourceId, UnitScore
from mce.models.symbol import Symbol
from mce.models.time import TimestampNs

OPTIONS_CONTEXT_SCHEMA_VERSION = "1.0"
DAY_NS = 86_400_000_000_000
DataMode = Literal["live", "delayed", "eod", "synthetic"]
CoordinateSystem = Literal["delta", "log_moneyness", "relative_moneyness"]
EvidenceOrigin = Literal["independent_neighbors", "own_contract_fallback", "sparse_surface"]


class ATMIVPoint(FrozenModel):
    dte_days: float = Field(gt=0, allow_inf_nan=False)
    expiration_ns: TimestampNs
    implied_volatility: NonNegativeFinite
    combination: Literal["combined", "call", "put"]
    strike: NonNegativeFinite | None = None
    source_method: str = Field(min_length=1)
    evidence_origin: EvidenceOrigin = "independent_neighbors"
    data_as_of_ns: TimestampNs
    available_at_ns: TimestampNs
    quality: DataQuality = DataQuality.GOOD

    @model_validator(mode="after")
    def validate_point(self) -> "ATMIVPoint":
        if self.expiration_ns <= self.data_as_of_ns or self.available_at_ns < self.data_as_of_ns:
            raise ValueError("IV expiration or availability is inconsistent")
        actual = (self.expiration_ns - self.data_as_of_ns) / DAY_NS
        if abs(actual - self.dte_days) > 1.5:
            raise ValueError("IV DTE differs from expiration timestamp")
        if self.strike is not None and self.strike <= 0:
            raise ValueError("ATM strike must be positive")
        return self


class SurfaceIVPoint(FrozenModel):
    dte_days: float = Field(gt=0, allow_inf_nan=False)
    expiration_ns: TimestampNs
    side: Literal["put", "call"]
    coordinate_system: CoordinateSystem
    coordinate: FiniteNumber
    implied_volatility: NonNegativeFinite
    source_method: str = Field(min_length=1)
    evidence_origin: EvidenceOrigin = "independent_neighbors"
    data_as_of_ns: TimestampNs
    available_at_ns: TimestampNs
    quality: DataQuality = DataQuality.GOOD

    @model_validator(mode="after")
    def validate_point(self) -> "SurfaceIVPoint":
        if self.expiration_ns <= self.data_as_of_ns or self.available_at_ns < self.data_as_of_ns:
            raise ValueError("surface expiration or availability is inconsistent")
        if abs((self.expiration_ns - self.data_as_of_ns) / DAY_NS - self.dte_days) > 1.5:
            raise ValueError("surface DTE differs from expiration timestamp")
        if (self.side == "put" and self.coordinate >= 0) or (
            self.side == "call" and self.coordinate <= 0
        ):
            raise ValueError("wing coordinate sign must match put/call side")
        if self.coordinate_system == "delta" and abs(self.coordinate) >= 1:
            raise ValueError("delta wing coordinate must be inside (-1,1)")
        return self


class SurfaceQuality(FrozenModel):
    tenor_coverage: UnitScore
    strike_coverage: UnitScore
    stale_contract_count: int = Field(default=0, ge=0)
    crossed_quote_count: int = Field(default=0, ge=0)
    interpolated: bool = False
    extrapolated: bool = False
    own_contract_fallback: bool = False
    independent_neighbor_count: int = Field(default=0, ge=0)


class OptionsMarketContextInput(FrozenModel):
    """Top-level version is strict; unknown additive source fields are ignored."""

    model_config = ConfigDict(frozen=True, extra="ignore", strict=True, validate_default=True)

    options_context_schema_version: Literal["1.0"]
    iv_units: Literal["annualized_decimal"]
    underlying_symbol: Symbol
    scope: Literal["broad", "sector"] = "broad"
    source_engine: SourceId
    source_engine_version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    source_fingerprint: str | None = None
    calculation_mode: str = Field(min_length=1)
    market_data_mode: DataMode
    as_of_ns: TimestampNs
    data_as_of_ns: TimestampNs
    available_at_ns: TimestampNs
    underlying_price: NonNegativeFinite | None = None
    atm_points: tuple[ATMIVPoint, ...] = ()
    surface_points: tuple[SurfaceIVPoint, ...] = ()
    surface_quality: SurfaceQuality
    quality: DataQuality = DataQuality.UNKNOWN

    @model_validator(mode="after")
    def validate_context(self) -> "OptionsMarketContextInput":
        if not self.data_as_of_ns <= self.as_of_ns <= self.available_at_ns:
            raise ValueError("options source timestamps are inconsistent")
        if self.underlying_price is not None and self.underlying_price <= 0:
            raise ValueError("underlying price must be positive")
        keys: set[tuple[float, str]] = set()
        expiry_keys: set[tuple[int, str]] = set()
        for atm_point in self.atm_points:
            atm_key = (atm_point.dte_days, atm_point.combination)
            expiry_key = (atm_point.expiration_ns, atm_point.combination)
            if atm_key in keys or expiry_key in expiry_keys:
                raise ValueError("duplicate ATM tenor/combination")
            keys.add(atm_key)
            expiry_keys.add(expiry_key)
            if (
                atm_point.data_as_of_ns > self.data_as_of_ns
                or atm_point.available_at_ns > self.available_at_ns
            ):
                raise ValueError("future ATM point exceeds source clock")
        surface_keys: set[tuple[float, str, str, float]] = set()
        surface_expiry_keys: set[tuple[int, str, str, float]] = set()
        for surface_point in self.surface_points:
            surface_key = (
                surface_point.dte_days,
                surface_point.side,
                surface_point.coordinate_system,
                surface_point.coordinate,
            )
            surface_expiry_key = (
                surface_point.expiration_ns,
                surface_point.side,
                surface_point.coordinate_system,
                surface_point.coordinate,
            )
            if surface_key in surface_keys or surface_expiry_key in surface_expiry_keys:
                raise ValueError("duplicate surface coordinate")
            surface_keys.add(surface_key)
            surface_expiry_keys.add(surface_expiry_key)
            if (
                surface_point.data_as_of_ns > self.data_as_of_ns
                or surface_point.available_at_ns > self.available_at_ns
            ):
                raise ValueError("future surface point exceeds source clock")
        return self
