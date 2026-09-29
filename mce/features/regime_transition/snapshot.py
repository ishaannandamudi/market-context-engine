"""Add stable MCE-11 scalars without changing MCE-10 categorical snapshots."""

from mce.features.regime_transition.models import RegimeTransitionFeatureSet
from mce.models.context import MarketContextSnapshot
from mce.version import __version__


def apply_regime_transition_to_snapshot(
    result: RegimeTransitionFeatureSet,
    snapshot: MarketContextSnapshot | None = None,
) -> MarketContextSnapshot:
    base = snapshot or MarketContextSnapshot(as_of_ns=result.as_of_ns)
    if base.as_of_ns != result.as_of_ns:
        raise ValueError("transition result and snapshot clocks must match")
    values = dict(base.feature_values or {})
    values["regime_transition.primary_window_sessions"] = result.primary_window_sessions
    values["regime_transition.quality"] = result.data_quality.value
    if result.current_regime is not None:
        values["regime_transition.current_regime"] = result.current_regime.value
    if result.current_age_ns is not None:
        values["regime_transition.age_ns"] = result.current_age_ns
    if result.current_age_trading_ns is not None:
        values["regime_transition.age_trading_ns"] = result.current_age_trading_ns
    if result.current_age_sessions is not None:
        values["regime_transition.age_sessions"] = result.current_age_sessions
    if result.current_age_observations is not None:
        values["regime_transition.age_observations"] = result.current_age_observations
    for window in result.windows:
        values[f"regime_transition.count.{window.sessions}d"] = window.headline_transitions
        if window.transition_rate_per_session is not None:
            values[f"regime_transition.rate.{window.sessions}d"] = (
                window.transition_rate_per_session
            )
        if window.headline_churn is not None:
            values[f"regime_transition.churn.{window.sessions}d"] = window.headline_churn
    for name, value in (
        (
            "rate",
            next(
                (
                    item.transition_rate_per_session
                    for item in result.windows
                    if item.sessions == result.primary_window_sessions
                ),
                None,
            ),
        ),
        (
            "churn",
            next(
                (
                    item.headline_churn
                    for item in result.windows
                    if item.sessions == result.primary_window_sessions
                ),
                None,
            ),
        ),
        ("stability", result.stability_score),
        ("instability", result.instability_score),
        ("oscillation", result.oscillation_score),
        ("intensity", result.transition_intensity),
        ("synchronized_fraction", result.synchronized_headline_fraction),
    ):
        if value is not None:
            values[f"regime_transition.{name}"] = value
    for dimension in result.dimension_stability:
        values[f"regime_transition.dimension.{dimension.dimension}.count"] = (
            dimension.transition_count
        )
        if dimension.transition_rate_per_session is not None:
            values[f"regime_transition.dimension.{dimension.dimension}.rate"] = (
                dimension.transition_rate_per_session
            )
    if result.recent_transition_sequence:
        values["regime_transition.previous_headline"] = result.recent_transition_sequence[
            -1
        ].from_regime.value
    if result.dominant_unstable_dimension is not None:
        values["regime_transition.dominant_unstable_dimension"] = result.dominant_unstable_dimension
    payload = base.model_dump(mode="python")
    payload.update(engine_version=__version__, feature_values=values)
    return MarketContextSnapshot.model_validate(payload)
