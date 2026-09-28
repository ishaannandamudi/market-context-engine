"""Add realized-volatility evidence to schema-1.0 snapshots without regime labels."""

from mce.features.volatility.models import RealizedVolatilityFeatureSet
from mce.models.context import MarketContextSnapshot
from mce.models.enums import DataQuality
from mce.version import __version__


def apply_volatility_to_snapshot(
    result: RealizedVolatilityFeatureSet, snapshot: MarketContextSnapshot | None = None
) -> MarketContextSnapshot:
    base = snapshot or MarketContextSnapshot(as_of_ns=result.as_of_ns)
    if base.as_of_ns != result.as_of_ns:
        raise ValueError("volatility result and snapshot as_of_ns must match")
    values = dict(base.feature_values or {})
    for symbol in sorted(result.per_symbol):
        item = result.per_symbol[symbol]
        for name, value in (
            ("rv_short", item.rv_short),
            ("rv_medium", item.rv_medium),
            ("rv_long", item.rv_long),
            ("short_long_ratio", item.short_long_ratio),
            ("medium_long_ratio", item.medium_long_ratio),
            ("acceleration", item.volatility_acceleration),
            ("change_1d", item.volatility_change_1d),
            ("percentile", item.volatility_percentile),
            ("robust_z", item.robust_z),
            ("shock", item.volatility_shock_score),
        ):
            if value is not None:
                values[f"volatility.{name}.{symbol}"] = value
        for horizon, window in item.intraday.items():
            for name, value in (
                ("realized", window.annualized_volatility),
                ("realized_variance", window.realized_variance),
                ("bipower", window.bipower_variation),
                ("jump_share", window.jump_share),
                ("downside_share", window.downside_share),
            ):
                if value is not None:
                    values[f"volatility.{name}.{symbol}.{horizon}"] = value
        for horizon, daily_window in item.daily.items():
            for name in (
                "close_to_close",
                "parkinson",
                "garman_klass",
                "rogers_satchell",
                "yang_zhang",
            ):
                estimate = getattr(daily_window, name)
                if estimate is not None:
                    values[f"volatility.{name}.{symbol}.{horizon}"] = estimate.annualized_volatility
            for name, value in (
                ("estimator_agreement", daily_window.estimator_agreement),
                ("estimator_range", daily_window.estimator_range),
                ("overnight_share", daily_window.overnight_variance_share),
                ("intraday_share", daily_window.intraday_variance_share),
                ("gap", daily_window.latest_gap_magnitude),
            ):
                if value is not None:
                    values[f"volatility.{name}.{symbol}.{horizon}"] = value
    if result.cross_index_volatility_range is not None:
        values["volatility.cross_index_range"] = result.cross_index_volatility_range
    if result.cross_index_volatility_ratio is not None:
        values["volatility.cross_index_ratio"] = result.cross_index_volatility_ratio
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
    contributors = (
        tuple(base.contributors or ())
        + result.contributors
        + tuple(
            contributor
            for symbol in sorted(result.per_symbol)
            for contributor in result.per_symbol[symbol].contributors
        )
    )
    payload = base.model_dump(mode="python")
    payload.update(
        engine_version=__version__,
        realized_volatility=result.market_realized_volatility,
        volatility_score=result.volatility_score,
        feature_values=values or None,
        contributors=contributors or None,
        quality_flags=base.quality_flags | result.quality_flags,
        data_quality=quality,
    )
    return MarketContextSnapshot.model_validate(payload)
