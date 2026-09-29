"""Transparent bounded persistence and movement composites."""

from mce.config.settings import RegimeTransitionSettings
from mce.features.regime_transition.models import StabilityContributor


def normalized_churn(
    rate_per_session: float | None, config: RegimeTransitionSettings
) -> float | None:
    if rate_per_session is None:
        return None
    return min(1.0, rate_per_session / config.churn_saturation_per_session)


def temporal_stability(
    age_sessions: int,
    window_sessions: int,
    headline_churn: float,
    dimension_churn: float,
    oscillation: float | None,
    config: RegimeTransitionSettings,
) -> tuple[float, tuple[StabilityContributor, ...]]:
    weights = config.stability_weights
    factors = (
        ("dwell", min(1.0, age_sessions / window_sessions), weights.dwell),
        ("headline_churn", 1 - headline_churn, weights.headline_churn),
        ("dimension_churn", 1 - dimension_churn, weights.dimension_churn),
        ("oscillation", 1 - (oscillation or 0), weights.oscillation),
    )
    contributors = tuple(
        StabilityContributor(name=name, value=value, weight=weight, contribution=value * weight)
        for name, value, weight in factors
    )
    return sum(item.contribution for item in contributors), contributors


def transition_intensity(headline_churn: float, dimension_churn: float) -> float:
    return 0.5 * headline_churn + 0.5 * dimension_churn
