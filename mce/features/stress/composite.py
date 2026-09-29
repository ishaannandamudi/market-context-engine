"""Transparent bounded liquidity and stress scores with exact contributors."""

from math import isfinite

from mce.config.settings import LiquiditySettings, StressSettings
from mce.models.context import ContextContributor

ScoreTerms = tuple[tuple[str, float | None, float], ...]


def weighted_score(
    terms: ScoreTerms, minimum_components: int, prefix: str
) -> tuple[float | None, tuple[tuple[str, float], ...], tuple[ContextContributor, ...]]:
    active = tuple(
        (name, value, weight)
        for name, value, weight in terms
        if value is not None and isfinite(value) and 0 <= value <= 1 and weight > 0
    )
    if len(active) < minimum_components:
        return None, (), ()
    total = sum(weight for _, _, weight in active)
    if total <= 0:
        return None, (), ()
    normalized = tuple((name, weight / total) for name, _, weight in active)
    score = sum(value * weight / total for _, value, weight in active)
    contributors = tuple(
        ContextContributor(
            name=f"{prefix}.{name}",
            value=value,
            normalized_value=value,
            contribution=value * weight / total,
            source_feature=f"{prefix}.{name}",
        )
        for name, value, weight in active
    )
    return score, normalized, contributors


def liquidity_score(
    illiquidity_percentile: float | None,
    relative_volume: float | None,
    relative_spread: float | None,
    settings: LiquiditySettings,
) -> tuple[float | None, tuple[tuple[str, float], ...], tuple[ContextContributor, ...]]:
    weights = settings.score_weights
    terms: ScoreTerms = (
        (
            "illiquidity",
            1 - illiquidity_percentile if illiquidity_percentile is not None else None,
            weights.illiquidity,
        ),
        (
            "relative_volume",
            1 - 1 / (1 + relative_volume) if relative_volume is not None else None,
            weights.relative_volume,
        ),
        (
            "quoted_spread",
            1 / (1 + relative_spread / settings.spread_scale)
            if relative_spread is not None
            else None,
            weights.quoted_spread,
        ),
    )
    return weighted_score(terms, settings.minimum_score_components, "liquidity")


def gap_stress(
    negative_fraction: float | None,
    negative_gap_magnitude: float | None,
    gap_percentile: float | None,
    settings: StressSettings,
) -> float | None:
    if negative_fraction is None or negative_gap_magnitude is None:
        return None
    intensity = min(1.0, negative_gap_magnitude / settings.gap_scale)
    historical_factor = 0.5 + 0.5 * gap_percentile if gap_percentile is not None else 1.0
    return negative_fraction * intensity * historical_factor


def dollar_volume_concentration(values: tuple[float, ...]) -> tuple[float, float] | None:
    if not values or any(not isfinite(value) or value < 0 for value in values):
        return None
    total = sum(values)
    if total <= 0 or not isfinite(total):
        return None
    hhi = sum((value / total) ** 2 for value in values)
    return hhi, 1 / hhi
