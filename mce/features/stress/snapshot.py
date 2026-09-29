"""Project continuous liquidity and stress evidence into schema-1.0 snapshots."""

from mce.features.stress.models import LiquidityStressFeatureSet
from mce.models.context import MarketContextSnapshot
from mce.models.enums import DataQuality
from mce.version import __version__


def apply_liquidity_stress_to_snapshot(
    result: LiquidityStressFeatureSet, snapshot: MarketContextSnapshot | None = None
) -> MarketContextSnapshot:
    base = snapshot or MarketContextSnapshot(as_of_ns=result.as_of_ns)
    if base.as_of_ns != result.as_of_ns:
        raise ValueError("liquidity/stress result and snapshot as_of_ns must match")
    liquid, stress = result.liquidity, result.stress
    values = dict(base.feature_values or {})
    metrics = {
        "liquidity.session_volume": liquid.session_volume,
        "liquidity.dollar_volume": liquid.session_dollar_volume,
        "liquidity.relative_volume": liquid.relative_volume,
        "liquidity.relative_volume_5d": liquid.relative_volume_5d,
        "liquidity.relative_volume_20d": liquid.relative_volume_20d,
        "liquidity.relative_volume_dispersion": liquid.relative_volume_dispersion,
        "liquidity.volume_percentile": liquid.volume_percentile,
        "liquidity.volume_robust_z": liquid.volume_robust_z,
        "liquidity.volume_breadth": liquid.volume_breadth,
        "liquidity.large_move_weak_volume_fraction": liquid.large_move_weak_volume_fraction,
        "liquidity.large_move_high_volume_fraction": liquid.large_move_high_volume_fraction,
        "liquidity.amihud_proxy": liquid.illiquidity_proxy,
        "liquidity.weighted_amihud_proxy": liquid.weighted_illiquidity_proxy,
        "liquidity.amihud_dispersion": liquid.illiquidity_dispersion,
        "liquidity.amihud_percentile": liquid.illiquidity_percentile,
        "liquidity.amihud_robust_z": liquid.illiquidity_robust_z,
        "liquidity.price_impact_proxy": liquid.price_impact_proxy,
        "liquidity.quoted_spread": liquid.quoted_spread,
        "liquidity.relative_quoted_spread": liquid.relative_quoted_spread,
        "liquidity.spread_dispersion": liquid.spread_dispersion,
        "liquidity.dollar_volume_hhi": liquid.dollar_volume_hhi,
        "liquidity.effective_active_names": liquid.effective_active_names,
        "liquidity.change": liquid.liquidity_change,
        "liquidity.score": liquid.liquidity_score,
        "liquidity.member_coverage": liquid.member_coverage,
        "liquidity.quote_coverage": liquid.quote_coverage,
        "liquidity.weight_coverage": liquid.weight_coverage,
        "liquidity.expected_members": liquid.expected_member_count,
        "liquidity.valid_members": liquid.valid_member_count,
        "liquidity.stale_bars": liquid.stale_bar_count,
        "liquidity.stale_quotes": liquid.stale_quote_count,
        "liquidity.evidence_quality": liquid.evidence_quality,
        "stress.volatility": stress.volatility_stress,
        "stress.correlation": stress.correlation_stress,
        "stress.breadth": stress.breadth_stress,
        "stress.liquidity": stress.liquidity_stress,
        "stress.gap": stress.gap_stress,
        "stress.dispersion": stress.dispersion_context_stress,
        "stress.sector": stress.sector_stress,
        "stress.score": stress.stress_score,
        "stress.systemic": stress.systemic_stress,
        "stress.localized": stress.localized_stress,
        "stress.stressed_member_fraction": stress.stressed_member_fraction,
        "stress.gap_up_fraction": stress.gap_up_fraction,
        "stress.gap_down_fraction": stress.gap_down_fraction,
        "stress.gap_magnitude": stress.gap_magnitude,
        "stress.gap_dispersion": stress.gap_dispersion,
        "stress.gap_percentile": stress.gap_percentile,
        "stress.gap_coverage": stress.gap_coverage,
        "stress.percentile": stress.stress_percentile,
        "stress.change": stress.stress_change,
        "stress.acceleration": stress.stress_acceleration,
        "stress.persistence": stress.stress_persistence,
        "stress.shock": stress.stress_shock_score,
        "stress.history_samples": stress.historical_sample_count,
        "stress.evidence_quality": stress.evidence_quality,
    }
    values.update({name: value for name, value in metrics.items() if value is not None})
    values.update(
        {f"liquidity.active_weight.{name}": weight for name, weight in liquid.active_weights}
    )
    values.update(
        {f"stress.active_weight.{name}": weight for name, weight in stress.active_weights}
    )
    order = {
        DataQuality.UNKNOWN: 0,
        DataQuality.GOOD: 1,
        DataQuality.DEGRADED: 2,
        DataQuality.POOR: 3,
        DataQuality.UNUSABLE: 4,
    }
    quality = max((base.data_quality, result.data_quality), key=order.__getitem__)
    payload = base.model_dump(mode="python")
    payload.update(
        engine_version=__version__,
        liquidity_score=liquid.liquidity_score,
        stress_score=stress.stress_score,
        feature_values=values or None,
        contributors=tuple(base.contributors or ()) + liquid.contributors + stress.contributors
        or None,
        quality_flags=base.quality_flags | result.quality_flags,
        data_quality=quality,
    )
    return MarketContextSnapshot.model_validate(payload)
