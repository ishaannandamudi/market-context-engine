"""Project continuous MCE-7 dependence evidence into schema-1.0 snapshots."""

from mce.features.correlation.models import CorrelationFeatureSet
from mce.models.context import MarketContextSnapshot
from mce.models.enums import DataQuality
from mce.version import __version__


def apply_correlation_to_snapshot(
    result: CorrelationFeatureSet, snapshot: MarketContextSnapshot | None = None
) -> MarketContextSnapshot:
    base = snapshot or MarketContextSnapshot(as_of_ns=result.as_of_ns)
    if base.as_of_ns != result.as_of_ns:
        raise ValueError("correlation result and snapshot as_of_ns must match")
    values = dict(base.feature_values or {})
    metrics = {
        "correlation.average": result.average_pairwise_correlation,
        "correlation.median": result.median_pairwise_correlation,
        "correlation.weighted_average": result.weighted_average_correlation,
        "correlation.dispersion": result.correlation_dispersion,
        "correlation.spearman_average": result.spearman_average,
        "correlation.short": result.short_correlation,
        "correlation.medium": result.medium_correlation,
        "correlation.long": result.long_correlation,
        "correlation.acceleration": result.correlation_acceleration,
        "correlation.change": result.correlation_change,
        "correlation.session_change": result.session_change,
        "correlation.five_session_change": result.five_session_change,
        "correlation.percentile": result.correlation_percentile,
        "correlation.robust_z": result.correlation_robust_z,
        "correlation.first_eigen_share": result.first_eigenvalue_share,
        "correlation.largest_eigenvalue": result.largest_eigenvalue,
        "correlation.effective_dimension": result.effective_dimension,
        "correlation.market_mode": result.market_mode_strength,
        "correlation.systemic_score": result.systemic_co_movement_score,
        "correlation.within_sector": result.within_sector_correlation,
        "correlation.cross_sector": result.cross_sector_correlation,
        "correlation.within_cross_spread": result.within_cross_spread,
        "correlation.symbol_coverage": result.symbol_coverage,
        "correlation.time_coverage": result.time_coverage,
        "correlation.weight_coverage": result.weight_coverage,
        "correlation.expected_symbols": result.expected_symbol_count,
        "correlation.valid_symbols": result.valid_symbol_count,
        "correlation.aligned_samples": result.aligned_sample_count,
        "correlation.matrix_dimension": result.matrix_dimension,
        "correlation.cluster_count": len(result.cluster_groups),
        "correlation.evidence_quality": result.evidence_quality,
    }
    values.update({name: value for name, value in metrics.items() if value is not None})
    values.update(
        {
            f"correlation.horizon.{item.name}": item.average
            for item in result.daily_horizons
            if item.average is not None
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
        correlation_score=result.systemic_co_movement_score,
        feature_values=values or None,
        contributors=tuple(base.contributors or ()) + result.contributors or None,
        quality_flags=base.quality_flags | result.quality_flags,
        data_quality=quality,
    )
    return MarketContextSnapshot.model_validate(payload)
