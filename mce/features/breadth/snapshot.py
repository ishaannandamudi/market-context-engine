"""Project continuous breadth evidence into existing schema-1.0 snapshots."""

from mce.features.breadth.models import BreadthFeatureSet
from mce.models.context import MarketContextSnapshot
from mce.models.enums import DataQuality
from mce.version import __version__


def apply_breadth_to_snapshot(
    result: BreadthFeatureSet, snapshot: MarketContextSnapshot | None = None
) -> MarketContextSnapshot:
    base = snapshot or MarketContextSnapshot(as_of_ns=result.as_of_ns)
    if base.as_of_ns != result.as_of_ns:
        raise ValueError("breadth result and snapshot as_of_ns must match")
    values = dict(base.feature_values or {})
    scalars = (
        ("breadth.advance_fraction", result.advance_fraction),
        ("breadth.decline_fraction", result.decline_fraction),
        ("breadth.net_advance_fraction", result.net_advance_fraction),
        ("breadth.advance_decline_ratio", result.advance_decline_ratio),
        ("breadth.equal_weight_return", result.equal_weight_return),
        ("breadth.median_member_return", result.median_member_return),
        ("breadth.weighted_return", result.weighted_return),
        ("breadth.equal_weight_divergence", result.equal_weight_divergence),
        ("breadth.index_proxy_return", result.index_proxy_return),
        ("breadth.index_proxy_divergence", result.index_proxy_divergence),
        ("breadth.directional_index_divergence", result.directional_index_divergence),
        ("breadth.contribution_hhi", result.contribution_hhi),
        ("breadth.effective_contributors", result.effective_contributors),
        ("breadth.participation", result.participation_score),
        ("breadth.momentum", result.breadth_momentum),
        ("breadth.acceleration", result.breadth_acceleration),
        ("breadth.thrust", result.breadth_thrust),
        ("breadth.persistence", result.participation_persistence),
        ("breadth.evidence_quality", result.evidence_quality),
        ("breadth.coverage", result.coverage_ratio),
        ("breadth.weight_coverage", result.weight_coverage),
        ("breadth.expected_members", result.expected_member_count),
        ("breadth.available_members", result.available_member_count),
        ("breadth.stale_members", result.stale_member_count),
        ("breadth.advancers", result.advancers),
        ("breadth.decliners", result.decliners),
        ("breadth.unchanged", result.unchanged),
    )
    values.update({name: value for name, value in scalars if value is not None})
    for name, value in (
        ("new_high_fraction", result.new_high_fraction),
        ("new_low_fraction", result.new_low_fraction),
        ("net_new_high_fraction", result.net_new_high_fraction),
    ):
        if value is not None:
            values[f"breadth.{name}.{result.new_high_low_window_sessions}d"] = value
    for reference in (result.short_reference, result.medium_reference, result.long_reference):
        if reference.fraction is not None:
            values[f"breadth.above_reference.{reference.sessions}d"] = reference.fraction
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
        breadth_score=result.breadth_score,
        feature_values=values or None,
        contributors=tuple(base.contributors or ()) + result.contributors or None,
        quality_flags=base.quality_flags | result.quality_flags,
        data_quality=quality,
    )
    return MarketContextSnapshot.model_validate(payload)
