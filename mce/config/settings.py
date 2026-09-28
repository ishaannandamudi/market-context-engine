"""Minimal settings for the foundation phase; no credentials are defined."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from mce.models.quality import SourceId


class SettingsSection(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class EngineSettings(SettingsSection):
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"


class MarketSettings(SettingsSection):
    region: Literal["US"] = "US"
    exchange_timezone: Literal["America/New_York"] = "America/New_York"


class ProviderSettings(SettingsSection):
    source: SourceId = "synthetic"


class SnapshotSettings(SettingsSection):
    cadence_ns: int = Field(default=60_000_000_000, gt=0)


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
    reproducibility: ReproducibilitySettings = Field(default_factory=ReproducibilitySettings)
