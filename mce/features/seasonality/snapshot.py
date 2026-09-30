"""Additive schema-1.0 projection of session-relative continuous diagnostics."""

from mce.features.seasonality.models import IntradaySeasonalityFeatureSet
from mce.models.context import MarketContextSnapshot
from mce.models.enums import DataQuality
from mce.version import __version__


def apply_seasonality_to_snapshot(
    result: IntradaySeasonalityFeatureSet, snapshot: MarketContextSnapshot | None = None
) -> MarketContextSnapshot:
    base = snapshot or MarketContextSnapshot(as_of_ns=result.as_of_ns)
    if base.as_of_ns != result.as_of_ns:
        raise ValueError("seasonality result and snapshot clocks differ")
    values = {
        name: value
        for name, value in (base.feature_values or {}).items()
        if not name.startswith("seasonality.")
    }
    values["seasonality.member_coverage"] = result.member_coverage
    values["seasonality.expected_symbols"] = len(result.expected_symbols)
    values["seasonality.available_symbols"] = len(result.available_symbols)
    if result.session_time is not None:
        values["seasonality.session_fraction"] = result.session_time.elapsed_fraction
    if result.data_quality not in (DataQuality.POOR, DataQuality.UNUSABLE):
        for metric in result.market_metrics:
            prefix = f"seasonality.{metric.metric}"
            values[f"{prefix}.coverage"] = metric.coverage
            if metric.ratio is not None:
                values[f"{prefix}.ratio"] = metric.ratio
            if metric.percentile is not None:
                values[f"{prefix}.percentile"] = metric.percentile
            if metric.robust_z is not None:
                values[f"{prefix}.robust_z"] = metric.robust_z
            if metric.metric == "cumulative_volume" and metric.ratio is not None:
                values["seasonality.adjusted_rvol"] = metric.ratio
        if result.correlation is not None:
            if result.correlation.ratio is not None:
                values["seasonality.correlation.ratio"] = result.correlation.ratio
            if result.correlation.percentile is not None:
                values["seasonality.correlation.percentile"] = result.correlation.percentile
            if result.correlation.robust_z is not None:
                values["seasonality.correlation.robust_z"] = result.correlation.robust_z
    quality_order = {
        DataQuality.UNKNOWN: 0,
        DataQuality.GOOD: 1,
        DataQuality.DEGRADED: 2,
        DataQuality.POOR: 3,
        DataQuality.UNUSABLE: 4,
    }
    quality = (
        base.data_quality
        if quality_order[base.data_quality] > quality_order[result.data_quality]
        else result.data_quality
    )
    payload = base.model_dump(mode="python")
    payload.update(
        engine_version=__version__,
        feature_values=values,
        quality_flags=base.quality_flags | result.quality_flags,
        data_quality=quality,
    )
    return MarketContextSnapshot.model_validate(payload)
