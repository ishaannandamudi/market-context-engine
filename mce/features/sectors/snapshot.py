"""Project continuous MCE-6 sector evidence into schema-1.0 snapshots."""

from mce.features.sectors.models import SectorFeatureSet
from mce.models.context import MarketContextSnapshot
from mce.models.enums import DataQuality
from mce.version import __version__


def apply_sectors_to_snapshot(
    result: SectorFeatureSet, snapshot: MarketContextSnapshot | None = None
) -> MarketContextSnapshot:
    base = snapshot or MarketContextSnapshot(as_of_ns=result.as_of_ns)
    if base.as_of_ns != result.as_of_ns:
        raise ValueError("sector result and snapshot as_of_ns must match")
    values = dict(base.feature_values or {})
    for sector, item in result.sectors.items():
        prefix = f"sector.{sector}"
        scalars = {
            "relative_strength": item.relative_strength_score,
            "absolute_strength": item.absolute_strength_score,
            "breadth": item.breadth_score,
            "coverage": item.member_coverage,
            "weight_coverage": item.weight_coverage,
            "expected_members": item.expected_member_count,
            "valid_members": item.valid_member_count,
            "stale_members": item.stale_member_count,
            "advancers": item.advancers,
            "decliners": item.decliners,
            "unchanged": item.unchanged,
            "realized_volatility": item.realized_volatility,
            "volatility_short": item.volatility_short,
            "volatility_long": item.volatility_long,
            "volatility_percentile": item.volatility_percentile,
            "relative_volatility": item.relative_volatility,
            "rank": item.rank,
            "percentile": item.percentile,
            "persistence": item.leadership_persistence,
            "rank_change": item.rank_change,
            "relative_strength_change": item.relative_strength_change,
            "momentum": item.sector_momentum,
            "acceleration": item.sector_acceleration,
            "breadth_proxy_divergence": item.breadth_proxy_divergence,
            "evidence_quality": item.evidence_quality,
        }
        values.update(
            {f"{prefix}.{name}": value for name, value in scalars.items() if value is not None}
        )
        for label, mapping in (
            ("return", item.absolute_returns),
            ("weighted_return", item.weighted_returns),
            ("proxy_return", item.proxy_returns),
            ("relative_return", item.relative_returns),
            ("proxy_divergence", item.proxy_divergence),
            ("horizon_coverage", item.horizon_coverage),
            ("weight_coverage", item.weight_coverage_by_horizon),
        ):
            values.update(
                {
                    f"{prefix}.{label}.{name}": value
                    for name, value in mapping.items()
                    if value is not None
                }
            )
        for name, reference in item.breadth_references.items():
            if reference.fraction is not None:
                values[f"{prefix}.above_reference.{name}"] = reference.fraction
    for name, value in (
        ("sector.leadership.spread", result.leadership_spread),
        ("sector.leadership.concentration", result.leadership_concentration),
        ("sector.score_dispersion", result.sector_score_dispersion),
        ("sector.rotation.velocity", result.rotation_velocity),
        ("sector.market_coverage", result.market_coverage),
        ("sector.evidence_quality", result.evidence_quality),
    ):
        if value is not None:
            values[name] = value
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
        sector_scores={
            name: item.relative_strength_score
            for name, item in result.sectors.items()
            if item.relative_strength_score is not None
        }
        or None,
        feature_values=values or None,
        contributors=tuple(base.contributors or ()) + result.contributors or None,
        quality_flags=base.quality_flags | result.quality_flags,
        data_quality=quality,
    )
    return MarketContextSnapshot.model_validate(payload)
