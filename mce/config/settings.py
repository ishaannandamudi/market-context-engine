"""Typed, credential-free configuration for historical market-data infrastructure."""

from collections.abc import Mapping
from datetime import date, time
from types import MappingProxyType
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_serializer, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from mce.models.enums import BarInterval
from mce.models.quality import SourceId
from mce.models.symbol import Symbol


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
    reproducibility: ReproducibilitySettings = Field(default_factory=ReproducibilitySettings)
