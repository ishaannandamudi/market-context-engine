"""Schema-1.0 projection; detailed labels remain in typed state and feature names."""

from mce.features.regime.models import MarketRegimeState
from mce.features.regime.taxonomy import BreadthState, HeadlineRegime
from mce.models.context import MarketContextSnapshot
from mce.models.enums import BreadthRegime, DataQuality, MarketRegime
from mce.version import __version__

_HEADLINE_PUBLIC = {
    HeadlineRegime.RISK_ON: MarketRegime.RISK_ON,
    HeadlineRegime.RISK_OFF: MarketRegime.RISK_OFF,
    HeadlineRegime.RANGE_BOUND: MarketRegime.NEUTRAL,
    HeadlineRegime.ROTATION: MarketRegime.MIXED,
    HeadlineRegime.SYSTEMIC_STRESS: MarketRegime.STRESSED,
    HeadlineRegime.LIQUIDITY_STRESS: MarketRegime.STRESSED,
    HeadlineRegime.TRANSITIONAL: MarketRegime.MIXED,
    HeadlineRegime.MIXED: MarketRegime.MIXED,
    HeadlineRegime.UNKNOWN: MarketRegime.UNKNOWN,
}

_BREADTH_PUBLIC = {
    BreadthState.BROAD_POSITIVE: BreadthRegime.STRONG_POSITIVE,
    BreadthState.NARROW_POSITIVE: BreadthRegime.POSITIVE,
    BreadthState.BROAD_NEGATIVE: BreadthRegime.STRONG_NEGATIVE,
    BreadthState.NARROW_NEGATIVE: BreadthRegime.NEGATIVE,
    BreadthState.MIXED: BreadthRegime.NEUTRAL,
    BreadthState.UNKNOWN: BreadthRegime.UNKNOWN,
}


def apply_regime_to_snapshot(
    state: MarketRegimeState,
    snapshot: MarketContextSnapshot | None = None,
) -> MarketContextSnapshot:
    base = snapshot or MarketContextSnapshot(as_of_ns=state.as_of_ns)
    if base.as_of_ns != state.as_of_ns:
        raise ValueError("regime state and snapshot clocks must match")
    values = dict(base.feature_values or {})
    values.update(
        {
            "regime.headline": state.headline_regime.value,
            "regime.direction": state.direction_regime.value,
            "regime.trend": state.trend_regime.value,
            "regime.volatility": state.volatility_regime.value,
            "regime.liquidity": state.liquidity_regime.value,
            "regime.correlation": state.correlation_regime.value,
            "regime.breadth": state.breadth_regime.value,
            "regime.dispersion": state.dispersion_regime.value,
            "regime.stress": state.stress_regime.value,
            "regime.sector_rotation": state.sector_rotation_state.value,
            "regime.evidence_strength": state.evidence_strength,
            "regime.hysteresis_held": state.hysteresis_held,
            "regime.conflict_count": len(state.conflicts),
        }
    )
    if state.selected_rule_id is not None:
        values["regime.selected_rule"] = state.selected_rule_id
    if state.regime_started_at_ns is not None:
        values["regime.started_at_ns"] = state.regime_started_at_ns
    if state.regime_age_observations is not None:
        values["regime.age_observations"] = state.regime_age_observations
    order = {
        DataQuality.UNKNOWN: 0,
        DataQuality.GOOD: 1,
        DataQuality.DEGRADED: 2,
        DataQuality.POOR: 3,
        DataQuality.UNUSABLE: 4,
    }
    quality = max((base.data_quality, state.data_quality), key=order.__getitem__)
    payload = base.model_dump(mode="python")
    payload.update(
        engine_version=__version__,
        market_regime=_HEADLINE_PUBLIC[state.headline_regime],
        trend_regime=state.trend_regime,
        volatility_regime=state.volatility_regime,
        correlation_regime=state.correlation_regime,
        liquidity_regime=state.liquidity_regime,
        dispersion_regime=state.dispersion_regime,
        breadth_regime=_BREADTH_PUBLIC[state.breadth_regime],
        feature_values=values,
        quality_flags=base.quality_flags | state.quality_flags,
        data_quality=quality,
    )
    return MarketContextSnapshot.model_validate(payload)
