"""Typed, credential-free configuration for historical market-data infrastructure."""

from collections.abc import Mapping
from datetime import date, time
from math import isfinite
from types import MappingProxyType
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_serializer, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from mce.features.windows import WindowSpec, WindowUnit
from mce.models.enums import BarInterval
from mce.models.quality import SourceId
from mce.models.symbol import Symbol, normalize_symbol


class SettingsSection(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class EngineSettings(SettingsSection):
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"


class UniverseSettings(SettingsSection):
    broad_market: tuple[Symbol, ...] = ("SPY", "QQQ", "IWM")
    sectors: Mapping[str, Symbol] = Field(default_factory=dict)
    volatility: tuple[Symbol, ...] = ()
    breadth: tuple[Symbol, ...] = ()

    @field_serializer("sectors")
    def serialize_sectors(self, sectors: Mapping[str, str]) -> dict[str, str]:
        return dict(sectors)

    @model_validator(mode="after")
    def validate_universe(self) -> "UniverseSettings":
        for group in (self.broad_market, self.volatility, self.breadth):
            if len(group) != len(set(group)):
                raise ValueError("universe symbols must be unique within each group")
        if any(not name.strip() for name in self.sectors):
            raise ValueError("sector names must be nonempty")
        object.__setattr__(self, "sectors", MappingProxyType(dict(self.sectors)))
        return self


class MarketSettings(SettingsSection):
    region: Literal["US"] = "US"
    exchange_timezone: Literal["America/New_York"] = "America/New_York"
    calendar_name: Literal["XNYS"] = "XNYS"
    calendar_start: date = date(2000, 1, 1)
    calendar_end: date = date(2035, 12, 31)
    pre_market_start_local: time = time(4, 0)
    after_hours_end_local: time = time(20, 0)
    universe: UniverseSettings = Field(default_factory=UniverseSettings)

    @model_validator(mode="after")
    def validate_calendar_range(self) -> "MarketSettings":
        if self.calendar_start >= self.calendar_end:
            raise ValueError("calendar_start must precede calendar_end")
        if self.pre_market_start_local.tzinfo or self.after_hours_end_local.tzinfo:
            raise ValueError("extended-hours local times must be timezone-naive")
        if self.pre_market_start_local >= self.after_hours_end_local:
            raise ValueError("pre-market start must precede after-hours end")
        return self


class ProviderSettings(SettingsSection):
    source: SourceId = "synthetic"


class SnapshotSettings(SettingsSection):
    cadence_ns: int = Field(default=60_000_000_000, gt=0)


class IngestionSettings(SettingsSection):
    default_bar_interval: BarInterval = BarInterval.ONE_MINUTE
    allow_extended_hours: bool = True


class FreshnessSettings(SettingsSection):
    bar_limit_ns: int = Field(default=120_000_000_000, gt=0)
    quote_limit_ns: int = Field(default=10_000_000_000, gt=0)
    trade_limit_ns: int = Field(default=30_000_000_000, gt=0)


class ReplaySettings(SettingsSection):
    max_observations: int = Field(default=100_000, gt=0)


def _default_horizons() -> tuple[WindowSpec, ...]:
    return (
        WindowSpec(name="1m", unit=WindowUnit.ELAPSED, length=60_000_000_000),
        WindowSpec(name="5m", unit=WindowUnit.ELAPSED, length=300_000_000_000),
        WindowSpec(name="15m", unit=WindowUnit.ELAPSED, length=900_000_000_000),
        WindowSpec(name="30m", unit=WindowUnit.ELAPSED, length=1_800_000_000_000),
        WindowSpec(name="1h", unit=WindowUnit.ELAPSED, length=3_600_000_000_000),
        WindowSpec(name="session", unit=WindowUnit.SESSION, length=1),
        WindowSpec(name="1d", unit=WindowUnit.TRADING_DAYS, length=1),
        WindowSpec(name="5d", unit=WindowUnit.TRADING_DAYS, length=5),
    )


class TrendScoreWeights(SettingsSection):
    returns: float = Field(default=0.40, ge=0, le=1, allow_inf_nan=False)
    slope: float = Field(default=0.25, ge=0, le=1, allow_inf_nan=False)
    persistence: float = Field(default=0.15, ge=0, le=1, allow_inf_nan=False)
    acceleration: float = Field(default=0.10, ge=0, le=1, allow_inf_nan=False)
    distance: float = Field(default=0.10, ge=0, le=1, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_sum(self) -> "TrendScoreWeights":
        if (
            abs(
                sum((self.returns, self.slope, self.persistence, self.acceleration, self.distance))
                - 1
            )
            > 1e-9
        ):
            raise ValueError("trend score weights must sum to 1")
        return self


class TrendSettings(SettingsSection):
    enabled: bool = True
    horizons: tuple[WindowSpec, ...] = Field(default_factory=_default_horizons)
    bar_cadence_ns: int = Field(default=60_000_000_000, gt=0)
    short_slope_window_ns: int = Field(default=300_000_000_000, gt=0)
    long_slope_window_ns: int = Field(default=1_800_000_000_000, gt=0)
    min_regression_samples: int = Field(default=3, ge=3)
    minimum_window_coverage: float = Field(default=0.8, gt=0, le=1, allow_inf_nan=False)
    minimum_market_coverage: float = Field(default=0.67, gt=0, le=1, allow_inf_nan=False)
    return_scale: float = Field(default=0.01, gt=0, allow_inf_nan=False)
    slope_scale_per_hour: float = Field(default=0.01, gt=0, allow_inf_nan=False)
    acceleration_scale_per_hour: float = Field(default=0.01, gt=0, allow_inf_nan=False)
    score_weights: TrendScoreWeights = Field(default_factory=TrendScoreWeights)
    index_weights: Mapping[str, float] = Field(default_factory=dict)

    @field_serializer("index_weights")
    def serialize_index_weights(self, weights: Mapping[str, float]) -> dict[str, float]:
        return dict(weights)

    @model_validator(mode="after")
    def validate_trend(self) -> "TrendSettings":
        names = [h.name for h in self.horizons]
        if not names or len(names) != len(set(names)):
            raise ValueError("trend horizon names must be nonempty and unique")
        if self.short_slope_window_ns >= self.long_slope_window_ns:
            raise ValueError("short slope window must be shorter than long slope window")
        normalized_weights: dict[str, float] = {}
        for name, weight in self.index_weights.items():
            symbol = normalize_symbol(name)
            if weight <= 0 or not isfinite(weight):
                raise ValueError("index weights must be positive finite values")
            if symbol in normalized_weights:
                raise ValueError("index weight symbols must be unique")
            normalized_weights[symbol] = weight
        object.__setattr__(self, "index_weights", MappingProxyType(normalized_weights))
        return self


class FeatureSettings(SettingsSection):
    trend: TrendSettings = Field(default_factory=TrendSettings)


class ReproducibilitySettings(SettingsSection):
    seed: int | None = Field(default=None, ge=0)


class MCEConfig(BaseSettings):
    """Environment variables use MCE_<SECTION>__<FIELD> names."""

    model_config = SettingsConfigDict(
        env_prefix="MCE_", env_nested_delimiter="__", extra="forbid", frozen=True
    )

    engine: EngineSettings = Field(default_factory=EngineSettings)
    market: MarketSettings = Field(default_factory=MarketSettings)
    provider: ProviderSettings = Field(default_factory=ProviderSettings)
    snapshot: SnapshotSettings = Field(default_factory=SnapshotSettings)
    ingestion: IngestionSettings = Field(default_factory=IngestionSettings)
    freshness: FreshnessSettings = Field(default_factory=FreshnessSettings)
    replay: ReplaySettings = Field(default_factory=ReplaySettings)
    features: FeatureSettings = Field(default_factory=FeatureSettings)
    reproducibility: ReproducibilitySettings = Field(default_factory=ReproducibilitySettings)
