"""Optional same-clock sector, realized-volatility and correlation summaries."""

from dataclasses import dataclass

from mce.features.dispersion.cross_section import cross_section
from mce.features.sectors.models import SectorFeatureSet
from mce.features.volatility.models import RealizedVolatilityFeatureSet


@dataclass(frozen=True)
class SectorSummary:
    absolute_std: float | None
    relative_std: float | None
    relative_mad: float | None
    spread: float | None
    valid_count: int


def sector_summary(context: SectorFeatureSet, horizon: str, minimum: int) -> SectorSummary:
    absolute = tuple(
        value
        for item in context.sectors.values()
        if (value := item.absolute_returns.get(horizon)) is not None
    )
    relative = tuple(
        value
        for item in context.sectors.values()
        if (value := item.relative_returns.get(horizon)) is not None
    )
    raw = cross_section(absolute, minimum=minimum)
    adjusted = cross_section(relative, minimum=minimum)
    return SectorSummary(
        absolute_std=raw.std if raw else None,
        relative_std=adjusted.std if adjusted else None,
        relative_mad=adjusted.mad if adjusted else None,
        spread=adjusted.full_range if adjusted else None,
        valid_count=len(relative),
    )


@dataclass(frozen=True)
class VolatilitySummary:
    std: float | None
    mad: float | None
    relative_std: float | None
    valid_count: int
    coverage: float


def volatility_summary(
    context: RealizedVolatilityFeatureSet,
    expected_symbols: tuple[str, ...],
    minimum_coverage: float,
    minimum_members: int,
) -> VolatilitySummary:
    expected = set(expected_symbols)
    values = tuple(
        value
        for symbol, item in context.per_symbol.items()
        if symbol in expected and (value := item.rv_medium) is not None
    )
    coverage = len(values) / len(expected) if expected else 0.0
    if coverage < minimum_coverage:
        return VolatilitySummary(None, None, None, len(values), coverage)
    sample = cross_section(values, minimum=minimum_members)
    market = context.market_realized_volatility
    return VolatilitySummary(
        std=sample.std if sample else None,
        mad=sample.mad if sample else None,
        relative_std=sample.std / market if sample and market and market > 1e-12 else None,
        valid_count=len(values),
        coverage=coverage,
    )
