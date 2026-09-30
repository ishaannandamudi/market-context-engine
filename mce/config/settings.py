"""Typed, credential-free configuration for historical market-data infrastructure."""

from collections.abc import Mapping
from datetime import date, time
from math import isfinite
from re import fullmatch
from types import MappingProxyType
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

from mce.features.windows import WindowSpec, WindowUnit
from mce.models.enums import BarInterval
from mce.models.quality import SourceId
from mce.models.sector import SectorName, normalize_sector
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


class VolatilitySettings(SettingsSection):
    enabled: bool = True
    intraday_horizons_ns: Mapping[str, int] = Field(
        default_factory=lambda: {
            "5m": 300_000_000_000,
            "15m": 900_000_000_000,
            "30m": 1_800_000_000_000,
            "1h": 3_600_000_000_000,
        }
    )
    session_horizons: tuple[int, ...] = (5, 10, 20, 60)
    short_window_sessions: int = Field(default=5, ge=2)
    medium_window_sessions: int = Field(default=20, ge=2)
    long_window_sessions: int = Field(default=60, ge=2)
    baseline_sessions: int = Field(default=60, gt=0)
    minimum_baseline_samples: int = Field(default=20, gt=0)
    bar_cadence_ns: int = Field(default=60_000_000_000, gt=0)
    trading_sessions_per_year: int = Field(default=252, gt=0)
    minimum_window_coverage: float = Field(default=0.8, gt=0, le=1, allow_inf_nan=False)
    minimum_market_coverage: float = Field(default=0.67, gt=0, le=1, allow_inf_nan=False)
    enabled_estimators: tuple[
        Literal["parkinson", "garman_klass", "rogers_satchell", "yang_zhang"], ...
    ] = ("parkinson", "garman_klass", "rogers_satchell", "yang_zhang")
    index_weights: Mapping[str, float] = Field(default_factory=dict)

    @field_serializer("intraday_horizons_ns", "index_weights")
    def serialize_maps(self, values: Mapping[str, object]) -> dict[str, object]:
        return dict(values)

    @model_validator(mode="after")
    def validate_volatility(self) -> "VolatilitySettings":
        if not self.short_window_sessions < self.medium_window_sessions < self.long_window_sessions:
            raise ValueError("volatility windows must satisfy short < medium < long")
        if len(set(self.session_horizons)) != len(self.session_horizons) or any(
            item < 2 for item in self.session_horizons
        ):
            raise ValueError("session horizons must be unique and at least two")
        if (
            self.medium_window_sessions not in self.session_horizons
            or self.short_window_sessions not in self.session_horizons
            or self.long_window_sessions not in self.session_horizons
        ):
            raise ValueError("short, medium and long windows must be session horizons")
        if self.minimum_baseline_samples > self.baseline_sessions:
            raise ValueError("minimum baseline samples cannot exceed baseline sessions")
        if not self.intraday_horizons_ns or any(
            name == "session" or fullmatch(r"[a-z0-9][a-z0-9_]*", name) is None or length <= 0
            for name, length in self.intraday_horizons_ns.items()
        ):
            raise ValueError("intraday horizons need unique canonical names and positive lengths")
        if len(set(self.enabled_estimators)) != len(self.enabled_estimators):
            raise ValueError("enabled estimators must be unique")
        normalized_weights: dict[str, float] = {}
        for name, weight in self.index_weights.items():
            symbol = normalize_symbol(name)
            if weight <= 0 or not isfinite(weight) or symbol in normalized_weights:
                raise ValueError(
                    "index weights must have unique symbols and positive finite values"
                )
            normalized_weights[symbol] = weight
        object.__setattr__(
            self, "intraday_horizons_ns", MappingProxyType(dict(self.intraday_horizons_ns))
        )
        object.__setattr__(self, "index_weights", MappingProxyType(normalized_weights))
        return self


class BreadthScoreWeights(SettingsSection):
    net_advance: float = Field(default=0.5, ge=0, le=1, allow_inf_nan=False)
    short_reference: float = Field(default=0.2, ge=0, le=1, allow_inf_nan=False)
    medium_reference: float = Field(default=0.2, ge=0, le=1, allow_inf_nan=False)
    long_reference: float = Field(default=0.1, ge=0, le=1, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_sum(self) -> "BreadthScoreWeights":
        if (
            abs(
                self.net_advance
                + self.short_reference
                + self.medium_reference
                + self.long_reference
                - 1
            )
            > 1e-9
        ):
            raise ValueError("breadth score weights must sum to 1")
        return self


class BreadthSettings(SettingsSection):
    enabled: bool = True
    minimum_member_coverage: float = Field(default=0.8, gt=0, le=1, allow_inf_nan=False)
    minimum_weight_coverage: float = Field(default=0.8, gt=0, le=1, allow_inf_nan=False)
    unchanged_return_tolerance: float = Field(default=1e-8, ge=0, le=1, allow_inf_nan=False)
    reference_windows_sessions: tuple[int, int, int] = (5, 20, 60)
    new_high_low_window_sessions: int = Field(default=20, gt=0)
    momentum_short_window: int = Field(default=3, gt=0)
    momentum_long_window: int = Field(default=10, gt=0)
    index_proxy_symbol: Symbol | None = "SPY"
    price_adjustment_status: Literal["adjusted", "raw", "unknown"] = "unknown"
    score_weights: BreadthScoreWeights = Field(default_factory=BreadthScoreWeights)

    @model_validator(mode="after")
    def validate_breadth(self) -> "BreadthSettings":
        short, medium, long = self.reference_windows_sessions
        if not 0 < short < medium < long:
            raise ValueError("breadth reference windows must increase strictly")
        if self.momentum_short_window >= self.momentum_long_window:
            raise ValueError("breadth momentum short window must precede long window")
        return self


class SectorScoreWeights(SettingsSection):
    short_relative: float = Field(default=0.35, ge=0, le=1, allow_inf_nan=False)
    medium_relative: float = Field(default=0.30, ge=0, le=1, allow_inf_nan=False)
    long_relative: float = Field(default=0.25, ge=0, le=1, allow_inf_nan=False)
    breadth_confirmation: float = Field(default=0.10, ge=0, le=1, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_sum(self) -> "SectorScoreWeights":
        if self.short_relative + self.medium_relative + self.long_relative <= 0:
            raise ValueError("sector relative score needs positive return weight")
        if (
            abs(
                self.short_relative
                + self.medium_relative
                + self.long_relative
                + self.breadth_confirmation
                - 1
            )
            > 1e-9
        ):
            raise ValueError("sector score weights must sum to 1")
        return self


class SectorSettings(SettingsSection):
    enabled: bool = True
    groups: Mapping[SectorName, tuple[Symbol, ...]] = Field(default_factory=dict)
    exclusive_membership: bool = True
    proxy_mapping: Mapping[SectorName, Symbol] = Field(default_factory=dict)
    reference_symbol: Symbol = "SPY"
    bar_cadence_ns: int = Field(default=60_000_000_000, gt=0)
    minimum_member_coverage: float = Field(default=0.8, gt=0, le=1, allow_inf_nan=False)
    minimum_weight_coverage: float = Field(default=0.8, gt=0, le=1, allow_inf_nan=False)
    minimum_member_count: int = Field(default=3, gt=0)
    minimum_valid_sectors: int = Field(default=2, ge=2)
    short_horizon: WindowSpec = Field(
        default_factory=lambda: WindowSpec(name="session", unit=WindowUnit.SESSION, length=1)
    )
    medium_horizon: WindowSpec = Field(
        default_factory=lambda: WindowSpec(name="1d", unit=WindowUnit.TRADING_DAYS, length=1)
    )
    long_horizon: WindowSpec = Field(
        default_factory=lambda: WindowSpec(name="5d", unit=WindowUnit.TRADING_DAYS, length=5)
    )
    additional_horizons: tuple[WindowSpec, ...] = (
        WindowSpec(name="5m", unit=WindowUnit.ELAPSED, length=300_000_000_000),
        WindowSpec(name="30m", unit=WindowUnit.ELAPSED, length=1_800_000_000_000),
        WindowSpec(name="20d", unit=WindowUnit.TRADING_DAYS, length=20),
    )
    return_scale: float = Field(default=0.01, gt=0, allow_inf_nan=False)
    score_weights: SectorScoreWeights = Field(default_factory=SectorScoreWeights)
    breadth_reference_sessions: tuple[int, int, int] = (5, 20, 60)
    volatility_windows_sessions: tuple[int, int, int] = (5, 20, 60)
    volatility_baseline_sessions: int = Field(default=20, gt=0)
    minimum_volatility_baseline_samples: int = Field(default=10, gt=0)
    persistence_window_sessions: int = Field(default=5, gt=1)
    rotation_window_sessions: int = Field(default=3, gt=0)
    price_adjustment_status: Literal["adjusted", "raw", "unknown"] = "unknown"

    @field_validator("groups", "proxy_mapping", mode="before")
    @classmethod
    def reject_duplicate_names(cls, value: object) -> object:
        if isinstance(value, Mapping):
            names = [normalize_sector(name) for name in value]
            if len(names) != len(set(names)):
                raise ValueError("sector names must be unique after normalization")
        return value

    @field_serializer("groups", "proxy_mapping")
    def serialize_maps(self, value: Mapping[str, object]) -> dict[str, object]:
        return dict(value)

    @model_validator(mode="after")
    def validate_sectors(self) -> "SectorSettings":
        names = (
            self.short_horizon.name,
            self.medium_horizon.name,
            self.long_horizon.name,
            *(item.name for item in self.additional_horizons),
        )
        if len(names) != len(set(names)):
            raise ValueError("sector horizons must have unique names")
        if (
            any(
                not 0 < a < b < c
                for a, b, c in (self.breadth_reference_sessions, self.volatility_windows_sessions)
            )
            or self.volatility_windows_sessions[0] < 2
        ):
            raise ValueError("sector reference and volatility windows must increase")
        if self.minimum_volatility_baseline_samples > self.volatility_baseline_sessions:
            raise ValueError("minimum sector volatility baseline exceeds its window")
        groups: dict[str, tuple[str, ...]] = {}
        for name, symbols in self.groups.items():
            canonical = normalize_sector(name)
            if canonical in groups or len(symbols) != len(set(symbols)):
                raise ValueError("sector groups must have unique names and symbols")
            groups[canonical] = tuple(sorted(symbols))
        if self.exclusive_membership:
            all_symbols = [symbol for symbols in groups.values() for symbol in symbols]
            if len(all_symbols) != len(set(all_symbols)):
                raise ValueError("exclusive sector groups cannot share symbols")
        proxies: dict[str, str] = {}
        for name, symbol in self.proxy_mapping.items():
            canonical = normalize_sector(name)
            if canonical in proxies:
                raise ValueError("sector proxy names must be unique")
            proxies[canonical] = symbol
        object.__setattr__(self, "groups", MappingProxyType(dict(sorted(groups.items()))))
        object.__setattr__(self, "proxy_mapping", MappingProxyType(dict(sorted(proxies.items()))))
        return self


class CorrelationScoreWeights(SettingsSection):
    average: float = Field(default=0.5, ge=0, le=1, allow_inf_nan=False)
    market_mode: float = Field(default=0.5, ge=0, le=1, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_sum(self) -> "CorrelationScoreWeights":
        if abs(self.average + self.market_mode - 1) > 1e-9:
            raise ValueError("correlation score weights must sum to 1")
        return self


class CorrelationSettings(SettingsSection):
    enabled: bool = True
    universe: tuple[Symbol, ...] = ()
    bar_cadence_ns: int = Field(default=60_000_000_000, gt=0)
    short_window_bars: int = Field(default=30, ge=3)
    medium_window_bars: int = Field(default=60, ge=3)
    long_window_bars: int = Field(default=120, ge=3)
    daily_horizons_sessions: tuple[int, ...] = ()
    minimum_symbols: int = Field(default=3, ge=2)
    minimum_aligned_samples: int = Field(default=8, ge=3)
    minimum_symbol_coverage: float = Field(default=0.8, gt=0, le=1, allow_inf_nan=False)
    minimum_time_coverage: float = Field(default=0.8, gt=0, le=1, allow_inf_nan=False)
    historical_baseline_sessions: int = Field(default=20, ge=0)
    minimum_baseline_samples: int = Field(default=10, ge=1)
    shrinkage_strength: float = Field(default=0.1, ge=0, le=1, allow_inf_nan=False)
    robust_estimator_enabled: bool = True
    clustering_enabled: bool = True
    cluster_correlation_threshold: float = Field(default=0.7, ge=-1, le=1, allow_inf_nan=False)
    score_weights: CorrelationScoreWeights = Field(default_factory=CorrelationScoreWeights)

    @model_validator(mode="after")
    def validate_correlation(self) -> "CorrelationSettings":
        if not self.short_window_bars < self.medium_window_bars < self.long_window_bars:
            raise ValueError("correlation windows must increase strictly")
        if self.minimum_aligned_samples > self.short_window_bars:
            raise ValueError("minimum aligned samples exceeds short window")
        if (
            self.minimum_baseline_samples > self.historical_baseline_sessions
            and self.historical_baseline_sessions
        ):
            raise ValueError("minimum baseline samples exceeds baseline sessions")
        if len(self.universe) != len(set(self.universe)):
            raise ValueError("correlation universe symbols must be unique")
        if len(set(self.daily_horizons_sessions)) != len(self.daily_horizons_sessions) or any(
            sessions < self.minimum_aligned_samples for sessions in self.daily_horizons_sessions
        ):
            raise ValueError("daily horizons must be unique and meet the minimum sample size")
        return self


class DispersionSettings(SettingsSection):
    enabled: bool = True
    universe: tuple[Symbol, ...] = ()
    bar_cadence_ns: int = Field(default=60_000_000_000, gt=0)
    short_horizon: WindowSpec = Field(
        default_factory=lambda: WindowSpec(
            name="5m", unit=WindowUnit.ELAPSED, length=300_000_000_000
        )
    )
    medium_horizon: WindowSpec = Field(
        default_factory=lambda: WindowSpec(
            name="30m", unit=WindowUnit.ELAPSED, length=1_800_000_000_000
        )
    )
    long_horizon: WindowSpec = Field(
        default_factory=lambda: WindowSpec(name="1d", unit=WindowUnit.TRADING_DAYS, length=1)
    )
    additional_horizons: tuple[WindowSpec, ...] = (
        WindowSpec(name="session", unit=WindowUnit.SESSION, length=1),
        WindowSpec(name="5d", unit=WindowUnit.TRADING_DAYS, length=5),
        WindowSpec(name="20d", unit=WindowUnit.TRADING_DAYS, length=20),
    )
    minimum_members: int = Field(default=3, ge=2)
    minimum_member_coverage: float = Field(default=0.8, gt=0, le=1, allow_inf_nan=False)
    minimum_weight_coverage: float = Field(default=0.8, gt=0, le=1, allow_inf_nan=False)
    minimum_sector_count: int = Field(default=2, ge=2)
    minimum_sector_coverage: float = Field(default=0.8, gt=0, le=1, allow_inf_nan=False)
    minimum_volatility_coverage: float = Field(default=0.8, gt=0, le=1, allow_inf_nan=False)
    historical_baseline_sessions: int = Field(default=20, ge=0)
    minimum_baseline_samples: int = Field(default=10, ge=1)
    persistence_observations: int = Field(default=5, ge=2)
    tail_quantiles: tuple[float, float] = (0.1, 0.9)
    price_adjustment_status: Literal["adjusted", "raw", "unknown"] = "unknown"

    @model_validator(mode="after")
    def validate_dispersion(self) -> "DispersionSettings":
        names = (
            self.short_horizon.name,
            self.medium_horizon.name,
            self.long_horizon.name,
            *(item.name for item in self.additional_horizons),
        )
        if len(names) != len(set(names)):
            raise ValueError("dispersion horizon names must be unique")
        if len(self.universe) != len(set(self.universe)):
            raise ValueError("dispersion universe symbols must be unique")
        if (
            self.historical_baseline_sessions
            and self.minimum_baseline_samples > self.historical_baseline_sessions
        ):
            raise ValueError("minimum baseline samples exceeds baseline sessions")
        low, high = self.tail_quantiles
        if not 0 < low < 0.5 < high < 1 or abs(low + high - 1) > 1e-9:
            raise ValueError("tail quantiles must be symmetric around the median")
        return self


class LiquidityScoreWeights(SettingsSection):
    illiquidity: float = Field(default=0.45, ge=0, le=1, allow_inf_nan=False)
    relative_volume: float = Field(default=0.35, ge=0, le=1, allow_inf_nan=False)
    quoted_spread: float = Field(default=0.20, ge=0, le=1, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_sum(self) -> "LiquidityScoreWeights":
        if abs(self.illiquidity + self.relative_volume + self.quoted_spread - 1) > 1e-9:
            raise ValueError("liquidity score weights must sum to one")
        return self


class LiquiditySettings(SettingsSection):
    enabled: bool = True
    universe: tuple[Symbol, ...] = ()
    bar_cadence_ns: int = Field(default=60_000_000_000, gt=0)
    historical_sessions: int = Field(default=20, ge=1)
    minimum_history_samples: int = Field(default=5, ge=1)
    minimum_members: int = Field(default=3, ge=2)
    minimum_member_coverage: float = Field(default=0.8, gt=0, le=1, allow_inf_nan=False)
    minimum_weight_coverage: float = Field(default=0.8, gt=0, le=1, allow_inf_nan=False)
    minimum_quote_coverage: float = Field(default=0.5, gt=0, le=1, allow_inf_nan=False)
    minimum_score_components: int = Field(default=2, ge=1, le=3)
    quote_spread_enabled: bool = True
    amihud_enabled: bool = True
    spread_scale: float = Field(default=0.001, gt=0, allow_inf_nan=False)
    large_move_threshold: float = Field(default=0.01, gt=0, allow_inf_nan=False)
    weak_volume_rvol_threshold: float = Field(default=0.75, ge=0, allow_inf_nan=False)
    high_volume_rvol_threshold: float = Field(default=2.0, gt=0, allow_inf_nan=False)
    price_adjustment_status: Literal["adjusted", "raw", "unknown"] = "unknown"
    score_weights: LiquidityScoreWeights = Field(default_factory=LiquidityScoreWeights)

    @model_validator(mode="after")
    def validate_liquidity(self) -> "LiquiditySettings":
        if len(self.universe) != len(set(self.universe)):
            raise ValueError("liquidity universe symbols must be unique")
        if self.minimum_history_samples > self.historical_sessions:
            raise ValueError("liquidity minimum history exceeds the baseline")
        if self.weak_volume_rvol_threshold >= self.high_volume_rvol_threshold:
            raise ValueError("weak volume threshold must be below high volume threshold")
        return self


class StressScoreWeights(SettingsSection):
    volatility: float = Field(default=0.22, ge=0, le=1, allow_inf_nan=False)
    correlation: float = Field(default=0.17, ge=0, le=1, allow_inf_nan=False)
    breadth: float = Field(default=0.20, ge=0, le=1, allow_inf_nan=False)
    liquidity: float = Field(default=0.18, ge=0, le=1, allow_inf_nan=False)
    gap: float = Field(default=0.13, ge=0, le=1, allow_inf_nan=False)
    dispersion: float = Field(default=0.06, ge=0, le=1, allow_inf_nan=False)
    sector: float = Field(default=0.04, ge=0, le=1, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_sum(self) -> "StressScoreWeights":
        if (
            abs(
                sum(
                    (
                        self.volatility,
                        self.correlation,
                        self.breadth,
                        self.liquidity,
                        self.gap,
                        self.dispersion,
                        self.sector,
                    )
                )
                - 1
            )
            > 1e-9
        ):
            raise ValueError("stress score weights must sum to one")
        return self


class StressSettings(SettingsSection):
    enabled: bool = True
    minimum_context_components: int = Field(default=2, ge=1, le=7)
    historical_baseline_observations: int = Field(default=20, ge=1)
    minimum_history_samples: int = Field(default=5, ge=1)
    persistence_observations: int = Field(default=5, ge=2)
    gap_scale: float = Field(default=0.02, gt=0, allow_inf_nan=False)
    stress_breadth_threshold: float = Field(default=0.8, gt=0, le=1, allow_inf_nan=False)
    score_weights: StressScoreWeights = Field(default_factory=StressScoreWeights)

    @model_validator(mode="after")
    def validate_stress(self) -> "StressSettings":
        if self.minimum_history_samples > self.historical_baseline_observations:
            raise ValueError("stress minimum history exceeds the baseline")
        return self


REGIME_RULE_IDS = (
    "regime.systemic_stress",
    "regime.liquidity_stress",
    "regime.transitional",
    "regime.risk_off",
    "regime.risk_on",
    "regime.rotation",
    "regime.range_bound",
    "regime.mixed",
)


class RegimeSettings(SettingsSection):
    enabled: bool = True
    minimum_required_families: int = Field(default=4, ge=3, le=8)
    minimum_evidence_quality: float = Field(default=0.5, ge=0, le=1, allow_inf_nan=False)
    minimum_evidence_strength: float = Field(default=0.2, ge=0, le=1, allow_inf_nan=False)
    direction_entry: float = Field(default=0.35, gt=0, le=1, allow_inf_nan=False)
    direction_exit: float = Field(default=0.2, ge=0, lt=1, allow_inf_nan=False)
    trend_confirmation: float = Field(default=0.3, gt=0, le=1, allow_inf_nan=False)
    trend_strong: float = Field(default=0.7, gt=0, le=1, allow_inf_nan=False)
    breadth_confirmation: float = Field(default=0.15, ge=0, le=1, allow_inf_nan=False)
    breadth_conflict: float = Field(default=0.45, gt=0, le=1, allow_inf_nan=False)
    volatility_low: float = Field(default=0.25, ge=0, le=1, allow_inf_nan=False)
    volatility_high: float = Field(default=0.7, ge=0, le=1, allow_inf_nan=False)
    volatility_extreme: float = Field(default=0.9, ge=0, le=1, allow_inf_nan=False)
    liquidity_stressed: float = Field(default=0.2, ge=0, le=1, allow_inf_nan=False)
    liquidity_degraded: float = Field(default=0.45, ge=0, le=1, allow_inf_nan=False)
    liquidity_healthy: float = Field(default=0.75, ge=0, le=1, allow_inf_nan=False)
    correlation_low: float = Field(default=0.25, ge=0, le=1, allow_inf_nan=False)
    correlation_high: float = Field(default=0.7, ge=0, le=1, allow_inf_nan=False)
    correlation_systemic: float = Field(default=0.9, ge=0, le=1, allow_inf_nan=False)
    dispersion_low: float = Field(default=0.25, ge=0, le=1, allow_inf_nan=False)
    dispersion_high: float = Field(default=0.7, ge=0, le=1, allow_inf_nan=False)
    dispersion_extreme: float = Field(default=0.9, ge=0, le=1, allow_inf_nan=False)
    stress_elevated: float = Field(default=0.3, ge=0, le=1, allow_inf_nan=False)
    stress_high: float = Field(default=0.65, ge=0, le=1, allow_inf_nan=False)
    stress_systemic: float = Field(default=0.85, ge=0, le=1, allow_inf_nan=False)
    systemic_member_fraction: float = Field(default=0.6, ge=0, le=1, allow_inf_nan=False)
    rotation_active: float = Field(default=0.7, ge=0, le=1, allow_inf_nan=False)
    rotation_stable: float = Field(default=0.4, ge=0, le=1, allow_inf_nan=False)
    conflict_override_severity: float = Field(default=0.8, ge=0, le=1, allow_inf_nan=False)
    hysteresis_band: float = Field(default=0.08, ge=0, le=0.15, allow_inf_nan=False)
    precedence: tuple[str, ...] = REGIME_RULE_IDS

    @model_validator(mode="after")
    def validate_regime(self) -> "RegimeSettings":
        ordered = (
            (self.volatility_low, self.volatility_high, self.volatility_extreme),
            (self.liquidity_stressed, self.liquidity_degraded, self.liquidity_healthy),
            (self.correlation_low, self.correlation_high, self.correlation_systemic),
            (self.dispersion_low, self.dispersion_high, self.dispersion_extreme),
            (self.stress_elevated, self.stress_high, self.stress_systemic),
            (self.rotation_stable, self.rotation_active),
        )
        if any(any(a >= b for a, b in zip(group, group[1:], strict=False)) for group in ordered):
            raise ValueError("regime dimension thresholds must increase strictly")
        if self.direction_exit >= self.direction_entry:
            raise ValueError("direction exit must be below entry")
        if self.trend_confirmation >= self.trend_strong:
            raise ValueError("strong trend threshold must exceed confirmation")
        if self.breadth_confirmation >= self.breadth_conflict:
            raise ValueError("breadth confirmation must be below conflict")
        if len(self.precedence) != len(REGIME_RULE_IDS) or set(self.precedence) != set(
            REGIME_RULE_IDS
        ):
            raise ValueError("regime precedence must contain each stable rule ID once")
        return self


class RegimeTransitionStabilityWeights(SettingsSection):
    dwell: float = Field(default=0.35, ge=0, le=1, allow_inf_nan=False)
    headline_churn: float = Field(default=0.30, ge=0, le=1, allow_inf_nan=False)
    dimension_churn: float = Field(default=0.20, ge=0, le=1, allow_inf_nan=False)
    oscillation: float = Field(default=0.15, ge=0, le=1, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_sum(self) -> "RegimeTransitionStabilityWeights":
        if (
            abs(sum((self.dwell, self.headline_churn, self.dimension_churn, self.oscillation)) - 1)
            > 1e-9
        ):
            raise ValueError("regime transition stability weights must sum to one")
        return self


class RegimeTransitionSettings(SettingsSection):
    enabled: bool = True
    recent_windows_sessions: tuple[int, ...] = (1, 5, 20, 60)
    primary_window_sessions: int = Field(default=20, ge=1)
    minimum_observations: int = Field(default=5, ge=2)
    minimum_sessions: int = Field(default=2, ge=1)
    minimum_completed_episodes: int = Field(default=2, ge=1)
    maximum_gap_trading_ns: int = Field(default=46_800_000_000_000, gt=0)
    minimum_time_coverage: float = Field(default=0.5, ge=0, le=1, allow_inf_nan=False)
    maximum_unknown_fraction: float = Field(default=0.5, ge=0, le=1, allow_inf_nan=False)
    minimum_usable_quality_fraction: float = Field(default=0.5, ge=0, le=1, allow_inf_nan=False)
    reversal_window_sessions: float = Field(default=2, gt=0, allow_inf_nan=False)
    churn_saturation_per_session: float = Field(default=2, gt=0, allow_inf_nan=False)
    max_recent_transitions: int = Field(default=12, ge=1, le=100)
    stability_weights: RegimeTransitionStabilityWeights = Field(
        default_factory=RegimeTransitionStabilityWeights
    )

    @model_validator(mode="after")
    def validate_transition(self) -> "RegimeTransitionSettings":
        windows = self.recent_windows_sessions
        if (
            not windows
            or any(value < 1 for value in windows)
            or tuple(sorted(set(windows))) != windows
        ):
            raise ValueError(
                "recent regime transition windows must be positive, sorted, and unique"
            )
        if self.primary_window_sessions not in windows:
            raise ValueError("primary regime transition window must be a recent window")
        if self.minimum_sessions > self.primary_window_sessions:
            raise ValueError("minimum sessions cannot exceed the primary window")
        return self


class FeatureSettings(SettingsSection):
    trend: TrendSettings = Field(default_factory=TrendSettings)
    volatility: VolatilitySettings = Field(default_factory=VolatilitySettings)
    breadth: BreadthSettings = Field(default_factory=BreadthSettings)
    sectors: SectorSettings = Field(default_factory=SectorSettings)
    correlation: CorrelationSettings = Field(default_factory=CorrelationSettings)
    dispersion: DispersionSettings = Field(default_factory=DispersionSettings)
    liquidity: LiquiditySettings = Field(default_factory=LiquiditySettings)
    stress: StressSettings = Field(default_factory=StressSettings)
    regime: RegimeSettings = Field(default_factory=RegimeSettings)
    regime_transition: RegimeTransitionSettings = Field(default_factory=RegimeTransitionSettings)


class ReproducibilitySettings(SettingsSection):
    seed: int | None = Field(default=None, ge=0)


class StatisticalSplitSettings(SettingsSection):
    train_sessions: int = Field(default=60, ge=1)
    validation_sessions: int = Field(default=20, ge=1)
    test_sessions: int = Field(default=20, ge=1)
    embargo_sessions: int = Field(default=1, ge=0)
    step_sessions: int = Field(default=20, ge=1)
    mode: Literal["expanding", "rolling"] = "expanding"


class HMMSettings(SettingsSection):
    state_count: int = Field(default=3, ge=2, le=6)
    max_iterations: int = Field(default=100, ge=1, le=1000)
    tolerance: float = Field(default=1e-5, gt=0, allow_inf_nan=False)
    covariance_floor: float = Field(default=1e-4, gt=0, allow_inf_nan=False)
    restarts: int = Field(default=3, ge=1, le=20)
    minimum_occupancy: float = Field(default=0.01, ge=0, lt=1, allow_inf_nan=False)


class KMeansSettings(SettingsSection):
    cluster_count: int = Field(default=3, ge=2, le=6)
    max_iterations: int = Field(default=100, ge=1, le=1000)
    restarts: int = Field(default=3, ge=1, le=20)


class ChangePointSettings(SettingsSection):
    left_window: int = Field(default=20, ge=2)
    right_window: int = Field(default=20, ge=2)
    threshold: float = Field(default=2.0, gt=0, allow_inf_nan=False)
    minimum_separation: int = Field(default=10, ge=1)


class StatisticalSettings(SettingsSection):
    enabled: bool = False
    feature_set: Literal["1.0"] = "1.0"
    normalization: Literal["schema", "none", "standard", "robust"] = "schema"
    quality_policy: Literal["reject_unusable", "reject_poor", "include"] = "reject_unusable"
    split: StatisticalSplitSettings = Field(default_factory=StatisticalSplitSettings)
    state_count: int = Field(default=3, ge=2, le=6)
    random_seed: int = Field(default=0, ge=0)
    hmm: HMMSettings = Field(default_factory=HMMSettings)
    kmeans: KMeansSettings = Field(default_factory=KMeansSettings)
    change_point: ChangePointSettings = Field(default_factory=ChangePointSettings)


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
    statistical: StatisticalSettings = Field(default_factory=StatisticalSettings)
    reproducibility: ReproducibilitySettings = Field(default_factory=ReproducibilitySettings)
