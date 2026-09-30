"""Additive schema-1.0 macro projection without changing upstream scores."""

from mce.features.macro.models import MacroContextFeatureSet
from mce.models.context import EventRisk, MarketContextSnapshot, SourceFreshness
from mce.models.enums import DataQuality, EventRiskStatus
from mce.version import __version__


def apply_macro_context_to_snapshot(
    result: MacroContextFeatureSet,
    snapshot: MarketContextSnapshot | None = None,
) -> MarketContextSnapshot:
    base = snapshot or MarketContextSnapshot(as_of_ns=result.as_of_ns)
    if base.as_of_ns != result.as_of_ns:
        raise ValueError("macro context and snapshot clocks differ")
    values = {
        name: value
        for name, value in (base.feature_values or {}).items()
        if not name.startswith("macro.")
    }
    for source in result.sources:
        prefix = f"macro.source.{source.source}"
        values[f"{prefix}.version"] = source.source_version
        values[f"{prefix}.as_of_ns"] = source.as_of_ns
        values[f"{prefix}.available_at_ns"] = source.available_at_ns
        values[f"{prefix}.quality"] = source.quality.value
        if source.source_fingerprint is not None:
            values[f"{prefix}.fingerprint"] = source.source_fingerprint
    for series in result.per_series:
        prefix = f"macro.series.{series.source}.{series.series_id}"
        values[f"{prefix}.category"] = series.category.value
        values[f"{prefix}.unit"] = series.unit.value
        values[f"{prefix}.cadence"] = series.cadence.value
        values[f"{prefix}.stale"] = series.stale
        values[f"{prefix}.release_stale"] = series.release_stale
        values[f"{prefix}.quality"] = series.data_quality.value
        if series.observation is not None:
            values[f"{prefix}.value"] = series.observation.value
            values[f"{prefix}.period_end_ns"] = series.observation.period_end_ns
            values[f"{prefix}.release_at_ns"] = series.observation.release_at_ns
            values[f"{prefix}.vintage_at_ns"] = series.observation.vintage_at_ns
            values[f"{prefix}.revision_status"] = series.observation.revision_status.value
        if series.change is not None:
            values[f"{prefix}.change"] = series.change
        if series.history is not None:
            if series.history.percentile is not None:
                values[f"{prefix}.percentile"] = series.history.percentile
            if series.history.robust_z is not None:
                values[f"{prefix}.robust_z"] = series.history.robust_z
        if series.release is not None:
            values[f"{prefix}.release_event_id"] = series.release.event_id
            values[f"{prefix}.release_at_ns"] = series.release.release_at_ns
            values[f"{prefix}.release_available_at_ns"] = series.release.available_at_ns
            values[f"{prefix}.release_vintage_at_ns"] = series.release.vintage_at_ns
            values[f"{prefix}.release_revision_status"] = series.release.revision_status.value
            if series.release.consensus is not None:
                values[f"{prefix}.consensus"] = series.release.consensus
            if series.release.prior is not None:
                values[f"{prefix}.prior"] = series.release.prior
            if series.release.revised_prior is not None:
                values[f"{prefix}.revised_prior"] = series.release.revised_prior
        if series.surprise is not None:
            if series.surprise.raw is not None:
                values[f"{prefix}.surprise"] = series.surprise.raw
            if series.surprise.normalized_z is not None:
                values[f"{prefix}.surprise_z"] = series.surprise.normalized_z
            if series.surprise.oriented_z is not None:
                values[f"{prefix}.oriented_surprise_z"] = series.surprise.oriented_z
    for dimension in result.dimensions:
        prefix = f"macro.{dimension.name}"
        if dimension.primary_series_id is not None:
            values[f"{prefix}.primary_series_id"] = dimension.primary_series_id
        if dimension.latest_value is not None:
            values[f"{prefix}.value"] = dimension.latest_value
        if dimension.unit is not None:
            values[f"{prefix}.unit"] = dimension.unit.value
        if dimension.latest_surprise is not None:
            values[f"{prefix}.surprise"] = dimension.latest_surprise
        if dimension.latest_surprise_z is not None:
            values[f"{prefix}.surprise_z"] = dimension.latest_surprise_z
        if dimension.normalized_signal is not None:
            values[f"{prefix}.normalized_signal"] = dimension.normalized_signal
    event_risk = base.event_risk
    if result.event is not None:
        event = result.event
        if event.next_event is not None:
            values["macro.event.next.type"] = event.next_event.event_type.value
            values["macro.event.next.id"] = event.next_event.event_id
            values["macro.event.next.scheduled_at_ns"] = event.next_event.scheduled_at_ns
        for name, value in (
            ("next.hours", event.hours_to_next),
            ("next_high.hours", event.hours_to_next_high),
            ("upcoming_count_24h", event.upcoming_count_24h),
            ("high_importance.count.24h", event.high_importance_count_24h),
            ("weighted_count_24h", event.weighted_count_24h),
            ("event_count_window", event.event_count_window),
            ("weighted_count_window", event.weighted_count_window),
            ("proximity_score", event.proximity_score),
            ("recent.seconds_since_release", event.seconds_since_recent_release),
        ):
            if value is not None:
                values[f"macro.event.{name}"] = value
        if event.recent_release is not None:
            values["macro.event.recent.id"] = event.recent_release.event_id
        if event_risk is None:
            status = (
                EventRiskStatus.UNAVAILABLE
                if event.proximity_score is None
                else EventRiskStatus.SCHEDULED
                if event.next_event is not None
                else EventRiskStatus.NONE_SCHEDULED
            )
            event_risk = EventRisk(
                status=status,
                event_type=event.next_event.event_type.value if event.next_event else None,
                event_name=event.next_event.event_id if event.next_event else None,
                scheduled_timestamp_ns=event.next_event.scheduled_at_ns
                if event.next_event
                else None,
                seconds_to_event=event.seconds_to_next,
                risk_score=event.proximity_score,
            )
    if result.policy is not None:
        policy = result.policy
        for name, value in (
            ("rate", policy.current_rate_percent),
            ("delta", policy.latest_change_pp),
            ("trailing_change_count", policy.trailing_change_count),
            ("cumulative_change", policy.cumulative_change_pp),
            ("seconds_since_last_change", policy.seconds_since_last_change),
        ):
            if value is not None:
                values[f"macro.policy.{name}"] = value
    if result.curve is not None:
        curve = result.curve
        for name, value in (
            ("2s10s", curve.spread_2s10s_pp),
            ("3m10y", curve.spread_3m10y_pp),
            ("30y5y", curve.spread_30y5y_pp),
            ("level", curve.level_10y_percent),
            ("mean_level", curve.mean_level_percent),
            ("curvature", curve.curvature_2y5y10y_pp),
            ("inversion", curve.inverted),
            ("inversion_magnitude", curve.inversion_magnitude_pp),
            ("change_10y_1d", curve.change_10y_1d_pp),
            ("change_10y_5d", curve.change_10y_5d_pp),
            ("change_10y_20d", curve.change_10y_20d_pp),
            ("parallel_shift_5d", curve.parallel_shift_5d_pp),
            ("steepening_2s10s_5d", curve.steepening_2s10s_5d_pp),
        ):
            if value is not None:
                values[f"macro.curve.{name}"] = value
        for tenor, rate in curve.yields_percent.items():
            values[f"macro.curve.yield_{tenor}m"] = rate
    if result.stress is not None:
        for name in ("rates", "curve", "surprise", "score"):
            value = getattr(result.stress, name)
            if value is not None:
                values[f"macro.stress.{name}"] = value
    freshness = [
        item for item in (base.source_freshness or ()) if not item.source.startswith("macro.")
    ]
    for source in result.sources:
        child_times = tuple(
            clock
            for item in result.per_series
            if item.source == source.source
            for clock in (
                (item.observation.available_at_ns if item.observation is not None else None),
                (item.release.available_at_ns if item.release is not None else None),
            )
            if clock is not None
        )
        effective = max(child_times, default=source.as_of_ns)
        freshness.append(
            SourceFreshness(
                source=f"macro.{source.source}",
                latest_effective_timestamp_ns=effective,
                age_ns=result.as_of_ns - effective,
                is_stale=any(
                    item.stale or item.release_stale
                    for item in result.per_series
                    if item.source == source.source
                ),
            )
        )
    rank = {
        DataQuality.UNKNOWN: 0,
        DataQuality.GOOD: 1,
        DataQuality.DEGRADED: 2,
        DataQuality.POOR: 3,
        DataQuality.UNUSABLE: 4,
    }
    quality = max((base.data_quality, result.data_quality), key=lambda item: rank[item])
    payload = base.model_dump(mode="python")
    payload.update(
        engine_version=__version__,
        feature_values=values,
        event_risk=event_risk,
        source_freshness=tuple(freshness),
        data_quality=quality,
        quality_flags=base.quality_flags | result.quality_flags,
    )
    return MarketContextSnapshot.model_validate(payload)
