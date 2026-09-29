"""Immutable, indexed MCE-10 observations with explicit causal availability."""

from bisect import bisect_right
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

from mce.features.regime.models import MarketRegimeState
from mce.features.regime.taxonomy import (
    BreadthState,
    DirectionState,
    HeadlineRegime,
    SectorRotationState,
    StressState,
)
from mce.models.enums import (
    CorrelationRegime,
    DataQuality,
    DispersionRegime,
    LiquidityRegime,
    QualityFlag,
    TrendRegime,
    VolatilityRegime,
)

DimensionName = Literal[
    "direction",
    "trend",
    "volatility",
    "liquidity",
    "correlation",
    "breadth",
    "dispersion",
    "stress",
    "sector_rotation",
]

DIMENSION_NAMES: tuple[DimensionName, ...] = (
    "direction",
    "trend",
    "volatility",
    "liquidity",
    "correlation",
    "breadth",
    "dispersion",
    "stress",
    "sector_rotation",
)

_DIMENSION_ENUMS = (
    DirectionState,
    TrendRegime,
    VolatilityRegime,
    LiquidityRegime,
    CorrelationRegime,
    BreadthState,
    DispersionRegime,
    StressState,
    SectorRotationState,
)


@dataclass(frozen=True, slots=True)
class RegimeHistoryPoint:
    as_of_ns: int
    available_at_ns: int
    headline_regime: HeadlineRegime
    dimensions: tuple[str, ...]
    evidence_strength: float
    data_quality: DataQuality
    config_fingerprint: str
    quality_flags: frozenset[QualityFlag] = frozenset()

    def __post_init__(self) -> None:
        if self.as_of_ns < 0 or self.available_at_ns < self.as_of_ns:
            raise ValueError("regime observation and availability clocks are invalid")
        if len(self.dimensions) != len(DIMENSION_NAMES) or any(
            not item for item in self.dimensions
        ):
            raise ValueError("regime history needs all nine dimension labels")
        if any(
            value not in {member.value for member in enum}
            for value, enum in zip(self.dimensions, _DIMENSION_ENUMS, strict=True)
        ):
            raise ValueError("regime history dimension label is invalid")
        if not isinstance(self.headline_regime, HeadlineRegime) or not isinstance(
            self.data_quality, DataQuality
        ):
            raise ValueError("regime history needs typed headline and quality labels")
        if not 0 <= self.evidence_strength <= 1:
            raise ValueError("regime evidence strength must be in [0, 1]")
        if len(self.config_fingerprint) != 64 or any(
            character not in "0123456789abcdef" for character in self.config_fingerprint
        ):
            raise ValueError("regime history needs a valid classifier fingerprint")

    @classmethod
    def from_state(cls, state: MarketRegimeState) -> "RegimeHistoryPoint":
        return cls(
            as_of_ns=state.as_of_ns,
            available_at_ns=state.available_at_ns,
            headline_regime=state.headline_regime,
            dimensions=(
                state.direction_regime.value,
                state.trend_regime.value,
                state.volatility_regime.value,
                state.liquidity_regime.value,
                state.correlation_regime.value,
                state.breadth_regime.value,
                state.dispersion_regime.value,
                state.stress_regime.value,
                state.sector_rotation_state.value,
            ),
            evidence_strength=state.evidence_strength,
            data_quality=state.data_quality,
            config_fingerprint=state.config_fingerprint,
            quality_flags=state.quality_flags,
        )


@dataclass(frozen=True, slots=True)
class HistoryStep:
    at_ns: int
    from_regime: HeadlineRegime
    to_regime: HeadlineRegime
    changed_dimensions: tuple[DimensionName, ...]

    @property
    def headline_changed(self) -> bool:
        return self.from_regime != self.to_regime


@dataclass(frozen=True, slots=True)
class RegimeHistory:
    points: tuple[RegimeHistoryPoint, ...]
    timestamps: tuple[int, ...]
    steps: tuple[HistoryStep, ...]
    step_timestamps: tuple[int, ...]
    episode_start_indices: tuple[int, ...]
    dimension_start_indices: tuple[tuple[int, ...], ...]
    first_fingerprint_change: int | None
    first_delayed: int | None

    @classmethod
    def build(
        cls, observations: Iterable[MarketRegimeState | RegimeHistoryPoint]
    ) -> "RegimeHistory":
        points = tuple(
            sorted(
                (
                    item
                    if isinstance(item, RegimeHistoryPoint)
                    else RegimeHistoryPoint.from_state(item)
                    for item in observations
                ),
                key=lambda item: item.as_of_ns,
            )
        )
        timestamps = tuple(item.as_of_ns for item in points)
        if any(left == right for left, right in zip(timestamps, timestamps[1:], strict=False)):
            raise ValueError("duplicate regime history timestamps")
        steps = tuple(
            HistoryStep(
                at_ns=current.as_of_ns,
                from_regime=previous.headline_regime,
                to_regime=current.headline_regime,
                changed_dimensions=tuple(
                    name
                    for name, before, after in zip(
                        DIMENSION_NAMES, previous.dimensions, current.dimensions, strict=True
                    )
                    if before != after
                ),
            )
            for previous, current in zip(points, points[1:], strict=False)
        )
        starts: list[int] = []
        dimension_starts: list[tuple[int, ...]] = []
        first = 0
        last_dimension_change = [0] * len(DIMENSION_NAMES)
        for index, point in enumerate(points):
            if index and point.headline_regime != points[index - 1].headline_regime:
                first = index
            if index:
                for name in steps[index - 1].changed_dimensions:
                    last_dimension_change[DIMENSION_NAMES.index(name)] = index
            starts.append(first)
            dimension_starts.append(tuple(last_dimension_change))
        fingerprint_change = (
            next(
                (
                    index
                    for index, point in enumerate(points[1:], 1)
                    if point.config_fingerprint != points[0].config_fingerprint
                ),
                None,
            )
            if points
            else None
        )
        delayed = next(
            (index for index, point in enumerate(points) if point.available_at_ns > point.as_of_ns),
            None,
        )
        return cls(
            points=points,
            timestamps=timestamps,
            steps=steps,
            step_timestamps=tuple(step.at_ns for step in steps),
            episode_start_indices=tuple(starts),
            dimension_start_indices=tuple(dimension_starts),
            first_fingerprint_change=fingerprint_change,
            first_delayed=delayed,
        )

    def causal_prefix(self, as_of_ns: int) -> tuple["RegimeHistory", int]:
        end = bisect_right(self.timestamps, as_of_ns)
        if self.first_delayed is not None and self.first_delayed < end:
            accepted = tuple(
                point for point in self.points[:end] if point.available_at_ns <= as_of_ns
            )
            filtered = RegimeHistory.build(accepted)
            if filtered.first_fingerprint_change is not None:
                raise ValueError("regime history mixes classifier fingerprints")
            return filtered, len(accepted)
        if self.first_fingerprint_change is not None and self.first_fingerprint_change < end:
            raise ValueError("regime history mixes classifier fingerprints")
        return self, end
