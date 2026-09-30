"""Immutable session, historical sample, baseline, and normalized result contracts."""

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from datetime import date
from hashlib import sha256
from math import isfinite
from types import MappingProxyType
from typing import Literal

from mce.models.enums import DataQuality, QualityFlag
from mce.serialization.canonical import canonical_json_bytes

MetricName = Literal[
    "volume",
    "cumulative_volume",
    "dollar_volume",
    "volatility",
    "quoted_spread",
    "spread",
    "illiquidity",
    "correlation",
]
BucketMode = Literal["wall_clock", "session_fraction"]


@dataclass(frozen=True, slots=True)
class SessionBucket:
    index: int
    start_ns: int
    end_ns: int

    def __post_init__(self) -> None:
        if self.index < 0 or self.start_ns < 0 or self.end_ns <= self.start_ns:
            raise ValueError("invalid session bucket")


@dataclass(frozen=True, slots=True)
class SessionRelativeTime:
    trading_date: date
    as_of_ns: int
    session_open_ns: int
    session_close_ns: int
    elapsed_ns: int
    remaining_ns: int
    elapsed_fraction: float
    regular_session_length_ns: int
    completed_bucket: SessionBucket | None
    is_early_close: bool
    bucket_mode: BucketMode


@dataclass(frozen=True, slots=True)
class SeasonalObservation:
    metric: MetricName
    symbol: str | None
    trading_date: date
    bucket_index: int
    bucket_end_ns: int
    available_at_ns: int
    value: float

    def __post_init__(self) -> None:
        if (
            self.bucket_index < 0
            or self.bucket_end_ns < 0
            or self.available_at_ns < self.bucket_end_ns
            or not isfinite(self.value)
        ):
            raise ValueError("invalid historical seasonal observation")
        if (self.metric == "correlation") != (self.symbol is None):
            raise ValueError("correlation is market-wide; other seasonal metrics need a symbol")
        if self.metric != "correlation" and self.value < 0:
            raise ValueError("per-symbol seasonal metric cannot be negative")


@dataclass(frozen=True, slots=True)
class SeasonalHistoryIndex:
    target_date: date
    prior_dates: tuple[date, ...]
    built_as_of_ns: int
    config_fingerprint: str
    observations: tuple[SeasonalObservation, ...]
    _by_key: Mapping[tuple[MetricName, str | None, int], tuple[SeasonalObservation, ...]] = field(
        init=False, repr=False, compare=False
    )
    _fingerprint: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if (
            self.built_as_of_ns < 0
            or len(self.config_fingerprint) != 64
            or self.prior_dates != tuple(sorted(set(self.prior_dates)))
            or any(day >= self.target_date for day in self.prior_dates)
        ):
            raise ValueError("invalid seasonal history window or configuration")
        ordered = tuple(
            sorted(
                self.observations,
                key=lambda item: (
                    item.metric,
                    item.symbol or "",
                    item.bucket_index,
                    item.trading_date,
                    item.available_at_ns,
                    item.value,
                ),
            )
        )
        object.__setattr__(self, "observations", ordered)
        keys: set[tuple[MetricName, str | None, int, date]] = set()
        by_key: dict[tuple[MetricName, str | None, int], list[SeasonalObservation]] = {}
        for observation in self.observations:
            key = (
                observation.metric,
                observation.symbol,
                observation.bucket_index,
                observation.trading_date,
            )
            if (
                observation.trading_date not in self.prior_dates
                or observation.available_at_ns > self.built_as_of_ns
                or key in keys
            ):
                raise ValueError(
                    "seasonal baseline contains future, duplicate, or out-of-window data"
                )
            keys.add(key)
            by_key.setdefault(key[:3], []).append(observation)
        object.__setattr__(
            self,
            "_by_key",
            MappingProxyType(
                {
                    key: tuple(sorted(rows, key=lambda row: row.trading_date))
                    for key, rows in by_key.items()
                }
            ),
        )
        digest = sha256(
            canonical_json_bytes(
                (self.target_date, self.prior_dates, self.built_as_of_ns, self.config_fingerprint)
            )
        )
        for observation in ordered:
            digest.update(canonical_json_bytes(asdict(observation)))
        object.__setattr__(self, "_fingerprint", digest.hexdigest())

    def samples(
        self, metric: MetricName, symbol: str | None, bucket_index: int
    ) -> tuple[SeasonalObservation, ...]:
        return self._by_key.get((metric, symbol, bucket_index), ())

    @property
    def fingerprint(self) -> str:
        return self._fingerprint


@dataclass(frozen=True, slots=True)
class SeasonalBaseline:
    metric: MetricName
    symbol: str | None
    bucket_index: int
    sample_count: int
    expected_sessions: int
    coverage: float
    median: float | None
    mean: float | None
    std: float | None
    mad: float | None
    q25: float | None
    q75: float | None


@dataclass(frozen=True, slots=True)
class SeasonallyAdjustedMetric:
    metric: MetricName
    raw_value: float | None
    baseline: SeasonalBaseline | None
    ratio: float | None
    percentile: float | None
    robust_z: float | None
    quality_flags: frozenset[QualityFlag] = frozenset()


@dataclass(frozen=True, slots=True)
class SymbolSeasonality:
    symbol: str
    metrics: tuple[SeasonallyAdjustedMetric, ...]


@dataclass(frozen=True, slots=True)
class MarketSeasonalMetric:
    metric: MetricName
    ratio: float | None
    percentile: float | None
    robust_z: float | None
    normalized_symbols: int
    coverage: float


@dataclass(frozen=True, slots=True)
class IntradaySeasonalityFeatureSet:
    as_of_ns: int
    session_time: SessionRelativeTime | None
    history_fingerprint: str | None
    config_fingerprint: str
    expected_symbols: tuple[str, ...]
    available_symbols: tuple[str, ...]
    member_coverage: float
    per_symbol: tuple[SymbolSeasonality, ...]
    market_metrics: tuple[MarketSeasonalMetric, ...]
    correlation: SeasonallyAdjustedMetric | None
    quality_flags: frozenset[QualityFlag]
    data_quality: DataQuality

    @property
    def fingerprint(self) -> str:
        return sha256(canonical_json_bytes(asdict(self))).hexdigest()
