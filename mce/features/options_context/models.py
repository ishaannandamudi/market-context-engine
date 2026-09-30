"""Typed descriptive options context, explicit history, and quality diagnostics."""

from hashlib import sha256

from pydantic import Field, model_validator

from mce.features.options_context.contracts import OptionsMarketContextInput
from mce.models.enums import DataQuality
from mce.models.quality import FiniteNumber, NonNegativeFinite, QualityTaggedModel, UnitScore
from mce.models.symbol import Symbol
from mce.models.time import TimestampNs
from mce.serialization.canonical import canonical_json_bytes


class TenorIV(QualityTaggedModel):
    dte_days: float = Field(gt=0, allow_inf_nan=False)
    implied_volatility: NonNegativeFinite


class TermStructureContext(QualityTaggedModel):
    tenors: tuple[TenorIV, ...]
    short: TenorIV | None = None
    medium: TenorIV | None = None
    long: TenorIV | None = None
    short_long_spread: FiniteNumber | None = None
    short_long_ratio: NonNegativeFinite | None = None
    slope_per_day: FiniteNumber | None = None
    curvature: FiniteNumber | None = None


class SkewContext(QualityTaggedModel):
    coordinate_system: str
    put_coordinate: FiniteNumber | None = None
    call_coordinate: FiniteNumber | None = None
    put_iv: NonNegativeFinite | None = None
    call_iv: NonNegativeFinite | None = None
    downside: FiniteNumber | None = None
    upside: FiniteNumber | None = None
    risk_reversal_call_minus_put: FiniteNumber | None = None
    butterfly: FiniteNumber | None = None
    atm_call_minus_put: FiniteNumber | None = None


class IVRVContext(QualityTaggedModel):
    iv_tenor_days: float = Field(gt=0, allow_inf_nan=False)
    rv_horizon_sessions: int | None = None
    rv_equivalent_calendar_days: float | None = None
    horizon_mismatch_days: float | None = None
    exact_match: bool = False
    comparable: bool = False
    trailing_realized_volatility: NonNegativeFinite | None = None
    spread: FiniteNumber | None = None
    ratio: NonNegativeFinite | None = None
    vrp_proxy: FiniteNumber | None = None


class OptionsHistoricalPoint(QualityTaggedModel):
    underlying_symbol: Symbol
    as_of_ns: TimestampNs
    available_at_ns: TimestampNs
    atm_iv: NonNegativeFinite | None = None
    downside_skew: FiniteNumber | None = None
    term_short_long_spread: FiniteNumber | None = None
    iv_rv_spread: FiniteNumber | None = None
    source_fingerprint: str | None = None

    @model_validator(mode="after")
    def validate_availability(self) -> "OptionsHistoricalPoint":
        if self.available_at_ns < self.as_of_ns:
            raise ValueError("historical options availability precedes observation")
        return self


class OptionsContextHistory(QualityTaggedModel):
    points: tuple[OptionsHistoricalPoint, ...] = ()

    @model_validator(mode="after")
    def validate_history(self) -> "OptionsContextHistory":
        ordered = tuple(
            sorted(self.points, key=lambda point: (point.as_of_ns, point.underlying_symbol))
        )
        if len({(point.underlying_symbol, point.as_of_ns) for point in ordered}) != len(ordered):
            raise ValueError("duplicate options history observation or source revision")
        object.__setattr__(self, "points", ordered)
        return self

    def prior(self, symbol: str, as_of_ns: int, limit: int) -> tuple[OptionsHistoricalPoint, ...]:
        return tuple(
            point
            for point in self.points
            if point.underlying_symbol == symbol
            and point.as_of_ns < as_of_ns
            and point.available_at_ns <= as_of_ns
        )[-limit:]


class HistoricalComparison(QualityTaggedModel):
    sample_count: int = Field(ge=0)
    percentile: UnitScore | None = None
    robust_z: FiniteNumber | None = None
    median: FiniteNumber | None = None


class OptionsStressContext(QualityTaggedModel):
    iv: UnitScore | None = None
    skew: UnitScore | None = None
    term: UnitScore | None = None
    dislocation: UnitScore | None = None
    score: UnitScore | None = None
    active_weights: tuple[tuple[str, float], ...] = ()


class UnderlyingOptionsContext(QualityTaggedModel):
    source: OptionsMarketContextInput
    atm_iv: NonNegativeFinite | None = None
    atm_tenor_days: float | None = None
    iv_history: HistoricalComparison | None = None
    skew_history: HistoricalComparison | None = None
    term_history: HistoricalComparison | None = None
    dislocation_history: HistoricalComparison | None = None
    term: TermStructureContext | None = None
    skew: SkewContext | None = None
    iv_rv: IVRVContext | None = None
    stress: OptionsStressContext | None = None
    relative_to_broad_iv: FiniteNumber | None = None
    relative_to_broad_skew: FiniteNumber | None = None
    relative_to_broad_term: FiniteNumber | None = None
    market_data_age_ns: TimestampNs
    headline_eligible: bool
    data_quality: DataQuality


class OptionsContextFeatureSet(QualityTaggedModel):
    as_of_ns: TimestampNs
    config_fingerprint: str
    expected_underlyings: tuple[Symbol, ...]
    available_underlyings: tuple[Symbol, ...]
    coverage: UnitScore
    per_underlying: tuple[UnderlyingOptionsContext, ...]
    primary_symbol: Symbol | None = None
    atm_iv_dispersion: NonNegativeFinite | None = None
    options_stress_score: UnitScore | None = None
    data_quality: DataQuality

    @property
    def fingerprint(self) -> str:
        return sha256(canonical_json_bytes(self)).hexdigest()
