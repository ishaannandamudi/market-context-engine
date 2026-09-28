"""Additive trend projection into the existing snapshot schema."""

from mce.features.models import TrendFeatureSet
from mce.models.context import MarketContextSnapshot
from mce.models.enums import DataQuality
from mce.version import __version__


def apply_trend_to_snapshot(
    result: TrendFeatureSet, snapshot: MarketContextSnapshot | None = None
) -> MarketContextSnapshot:
    base = snapshot or MarketContextSnapshot(as_of_ns=result.as_of_ns)
    if base.as_of_ns != result.as_of_ns:
        raise ValueError("trend result and snapshot as_of_ns must match")
    values = dict(base.feature_values or {})
    for symbol, item in result.symbols.items():
        for horizon, feature in item.returns.items():
            if feature.value is not None:
                values[f"trend.return.{symbol}.{horizon}"] = feature.value
        for name, value in (
            ("slope.short", item.slope_short.slope_per_hour),
            ("slope.long", item.slope_long.slope_per_hour),
            ("persistence", item.persistence),
            ("path_efficiency", item.path_efficiency),
            ("agreement", item.directional_agreement),
            ("acceleration", item.acceleration_per_hour),
            ("distance", item.distance_from_sma),
        ):
            if value is not None:
                values[f"trend.{name}.{symbol}"] = value
    if result.index_agreement is not None:
        values["trend.index_agreement"] = result.index_agreement
    quality_order = {
        DataQuality.UNKNOWN: 0,
        DataQuality.GOOD: 1,
        DataQuality.DEGRADED: 2,
        DataQuality.POOR: 3,
        DataQuality.UNUSABLE: 4,
    }
    quality = result.data_quality
    if (
        base.data_quality != DataQuality.UNKNOWN
        and quality_order[base.data_quality] > quality_order[quality]
    ):
        quality = base.data_quality
    all_contributors = (
        tuple(base.contributors or ())
        + result.contributors
        + tuple(
            contributor
            for symbol in sorted(result.symbols)
            for contributor in result.symbols[symbol].contributors
        )
    )
    payload = base.model_dump(mode="python")
    payload.update(
        engine_version=__version__,
        market_direction_score=result.market_direction_score,
        trend_score=result.trend_score,
        index_scores=dict(result.index_scores) or None,
        contributors=all_contributors or None,
        feature_values=values or None,
        quality_flags=base.quality_flags | result.quality_flags,
        data_quality=quality,
    )
    return MarketContextSnapshot.model_validate(payload)
