"""Schema-1.0 projection of causal, continuous factor diagnostics."""

from mce.features.factors.models import FactorContext
from mce.models.context import MarketContextSnapshot
from mce.models.enums import DataQuality
from mce.version import __version__


def apply_factors_to_snapshot(
    result: FactorContext, snapshot: MarketContextSnapshot | None = None
) -> MarketContextSnapshot:
    base = snapshot or MarketContextSnapshot(as_of_ns=result.as_of_ns)
    if base.as_of_ns != result.as_of_ns:
        raise ValueError("factor result and snapshot clocks differ")
    values = {
        name: value
        for name, value in (base.feature_values or {}).items()
        if not name.startswith("factor.")
    }
    values.update(
        {
            "factor.symbol_coverage": result.symbol_coverage,
            "factor.time_coverage": result.time_coverage,
            "factor.aligned_samples": result.aligned_samples,
        }
    )
    if result.causal and result.data_quality not in (DataQuality.POOR, DataQuality.UNUSABLE):
        metrics = {
            "factor.first.explained_variance": result.first_share,
            "factor.top3.explained_variance": result.top3_share,
            "factor.effective_dimension": result.effective_dimension,
            "factor.concentration": result.concentration,
            "factor.dominance": result.dominance,
            "factor.residual_variance": result.decomposition.residual_variance
            if result.decomposition
            else None,
            "factor.residual_variance_share": result.decomposition.median_residual_share
            if result.decomposition
            else None,
            "factor.common_variance_share": result.decomposition.mean_common_share
            if result.decomposition
            else None,
            "factor.residual_dispersion": result.decomposition.residual_dispersion
            if result.decomposition
            else None,
            "factor.loading_stability": result.alignment.mean_similarity
            if result.alignment
            else None,
            "factor.subspace_stability": result.alignment.subspace_similarity
            if result.alignment
            else None,
            "factor.shock": result.shock_z,
            "factor.volatility": result.factor_volatility,
        }
        values.update({name: value for name, value in metrics.items() if value is not None})
        for exposure in result.sector_exposures:
            values.update(
                {
                    f"factor.sector.{exposure.sector}.loading.{index + 1}": loading
                    for index, loading in enumerate(exposure.mean_loadings)
                }
            )
    quality_order = {
        DataQuality.UNKNOWN: 0,
        DataQuality.GOOD: 1,
        DataQuality.DEGRADED: 2,
        DataQuality.POOR: 3,
        DataQuality.UNUSABLE: 4,
    }
    quality = result.data_quality
    if quality_order[base.data_quality] > quality_order[quality]:
        quality = base.data_quality
    payload = base.model_dump(mode="python")
    payload.update(
        engine_version=__version__,
        feature_values=values,
        quality_flags=base.quality_flags | result.quality_flags,
        data_quality=quality,
    )
    return MarketContextSnapshot.model_validate(payload)
