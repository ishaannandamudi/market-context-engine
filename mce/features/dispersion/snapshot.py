"""Project continuous dispersion evidence into schema-1.0 snapshots."""

from mce.features.dispersion.models import DispersionFeatureSet
from mce.models.context import MarketContextSnapshot
from mce.models.enums import DataQuality
from mce.version import __version__


def apply_dispersion_to_snapshot(
    result: DispersionFeatureSet, snapshot: MarketContextSnapshot | None = None
) -> MarketContextSnapshot:
    base = snapshot or MarketContextSnapshot(as_of_ns=result.as_of_ns)
    if base.as_of_ns != result.as_of_ns:
        raise ValueError("dispersion result and snapshot as_of_ns must match")
    values = dict(base.feature_values or {})
    metrics = {
        "dispersion.return.std": result.return_dispersion,
        "dispersion.return.weighted_std": result.weighted_return_dispersion,
        "dispersion.return.robust_sigma": result.robust_return_dispersion,
        "dispersion.return.mad": result.return_mad,
        "dispersion.return.iqr": result.return_iqr,
        "dispersion.return.tail_spread": result.return_tail_spread,
        "dispersion.return.range": result.return_range,
        "dispersion.return.skew": result.cross_sectional_skewness,
        "dispersion.return.excess_kurtosis": result.cross_sectional_excess_kurtosis,
        "dispersion.upside": result.upside_dispersion,
        "dispersion.downside": result.downside_dispersion,
        "dispersion.downside_upside_spread": result.downside_upside_spread,
        "dispersion.short": result.short_dispersion,
        "dispersion.medium": result.medium_dispersion,
        "dispersion.long": result.long_dispersion,
        "dispersion.short_long_ratio": result.short_long_ratio,
        "dispersion.acceleration": result.dispersion_acceleration,
        "dispersion.change": result.dispersion_change,
        "dispersion.session_change": result.session_change,
        "dispersion.five_session_change": result.five_session_change,
        "dispersion.percentile": result.dispersion_percentile,
        "dispersion.robust_z": result.dispersion_robust_z,
        "dispersion.shock": result.dispersion_shock_score,
        "dispersion.persistence": result.dispersion_persistence,
        "dispersion.sector": result.sector_dispersion,
        "dispersion.sector.relative": result.sector_relative_dispersion,
        "dispersion.sector.relative_mad": result.sector_relative_mad,
        "dispersion.sector.spread": result.sector_spread,
        "dispersion.volatility": result.volatility_dispersion,
        "dispersion.volatility.mad": result.robust_volatility_dispersion,
        "dispersion.volatility.relative": result.relative_volatility_dispersion,
        "dispersion.within_sector": result.within_sector_variance,
        "dispersion.between_sector": result.between_sector_variance,
        "dispersion.total_sector_variance": result.decomposition_total_variance,
        "dispersion.within_sector_share": result.within_sector_share,
        "dispersion.between_sector_share": result.between_sector_share,
        "dispersion.movement_hhi": result.movement_hhi,
        "dispersion.effective_movers": result.effective_movers,
        "dispersion.weighted_movement_hhi": result.weighted_movement_hhi,
        "dispersion.weighted_effective_movers": result.weighted_effective_movers,
        "dispersion.member_coverage": result.member_coverage,
        "dispersion.weight_coverage": result.weight_coverage,
        "dispersion.volatility_coverage": result.volatility_coverage,
        "dispersion.decomposition_coverage": result.decomposition_coverage,
        "dispersion.expected_members": result.expected_member_count,
        "dispersion.valid_members": result.valid_member_count,
        "dispersion.stale_members": len(result.stale_symbols),
        "dispersion.missing_members": len(result.missing_symbols),
        "dispersion.baseline_samples": result.baseline_sample_count,
        "dispersion.evidence_quality": result.evidence_quality,
    }
    values.update({name: value for name, value in metrics.items() if value is not None})
    values.update(
        {
            f"dispersion.horizon.{item.name}.std": item.return_std
            for item in result.horizons
            if item.return_std is not None
        }
    )
    values.update(
        {
            f"dispersion.horizon.{item.name}.coverage": item.member_coverage
            for item in result.horizons
        }
    )
    values.update(
        {
            f"dispersion.sector.{item.sector}.std": item.return_std
            for item in result.sector_return_dispersion
            if item.return_std is not None
        }
    )
    context = result.correlation_context
    if context is not None:
        values.update(
            {
                name: value
                for name, value in {
                    "dispersion.correlation.average": context.average_correlation,
                    "dispersion.correlation.systemic_score": context.systemic_correlation_score,
                    "dispersion.correlation.percentile": context.correlation_percentile,
                    "dispersion.correlation.differentiation": context.differentiation_score,
                    "dispersion.correlation.percentile_spread": (
                        context.dispersion_minus_correlation
                    ),
                }.items()
                if value is not None
            }
        )
    order = {
        DataQuality.UNKNOWN: 0,
        DataQuality.GOOD: 1,
        DataQuality.DEGRADED: 2,
        DataQuality.POOR: 3,
        DataQuality.UNUSABLE: 4,
    }
    quality = result.data_quality
    if base.data_quality != DataQuality.UNKNOWN and order[base.data_quality] > order[quality]:
        quality = base.data_quality
    payload = base.model_dump(mode="python")
    payload.update(
        engine_version=__version__,
        dispersion_score=result.dispersion_score,
        feature_values=values or None,
        contributors=tuple(base.contributors or ()) + result.contributors or None,
        quality_flags=base.quality_flags | result.quality_flags,
        data_quality=quality,
    )
    return MarketContextSnapshot.model_validate(payload)
