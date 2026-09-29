"""Adapt already-computed causal feature results into one typed scorecard input."""

from statistics import median

from mce.features.breadth.models import BreadthFeatureSet
from mce.features.correlation.models import CorrelationFeatureSet
from mce.features.dispersion.models import DispersionFeatureSet
from mce.features.models import TrendFeatureSet
from mce.features.regime.models import RegimeEvidence
from mce.features.sectors.models import SectorFeatureSet
from mce.features.stress.models import LiquidityStressFeatureSet
from mce.features.volatility.models import RealizedVolatilityFeatureSet
from mce.models.context import MarketContextSnapshot
from mce.models.enums import DataQuality, QualityFlag

_QUALITY_ORDER = {
    DataQuality.GOOD: 0,
    DataQuality.UNKNOWN: 1,
    DataQuality.DEGRADED: 2,
    DataQuality.POOR: 3,
    DataQuality.UNUSABLE: 4,
}


def evidence_from_features(
    as_of_ns: int,
    *,
    trend: TrendFeatureSet | None = None,
    breadth: BreadthFeatureSet | None = None,
    volatility: RealizedVolatilityFeatureSet | None = None,
    correlation: CorrelationFeatureSet | None = None,
    dispersion: DispersionFeatureSet | None = None,
    sectors: SectorFeatureSet | None = None,
    liquidity_stress: LiquidityStressFeatureSet | None = None,
) -> RegimeEvidence:
    contexts = tuple(
        item
        for item in (trend, breadth, volatility, correlation, dispersion, sectors, liquidity_stress)
        if item is not None
    )
    if any(item.as_of_ns != as_of_ns for item in contexts):
        raise ValueError("upstream feature and regime clocks must match")
    sector_breadths = (
        tuple(
            item.breadth_score
            for item in sectors.sectors.values()
            if item.breadth_score is not None
        )
        if sectors is not None
        else ()
    )
    quality = (
        max((item.data_quality for item in contexts), key=_QUALITY_ORDER.__getitem__)
        if contexts
        else DataQuality.UNKNOWN
    )
    coverages = tuple(
        value
        for value in (
            trend.coverage_ratio if trend else None,
            breadth.coverage_ratio if breadth else None,
            liquidity_stress.stress.evidence_quality if liquidity_stress else None,
        )
        if value is not None
    )
    flags: set[QualityFlag] = set()
    for item in contexts:
        flags.update(item.quality_flags)
    return RegimeEvidence(
        as_of_ns=as_of_ns,
        market_direction_score=trend.market_direction_score if trend else None,
        trend_score=trend.trend_score if trend else None,
        breadth_score=breadth.breadth_score if breadth else None,
        breadth_participation_score=breadth.participation_score if breadth else None,
        breadth_coverage=breadth.coverage_ratio if breadth else None,
        index_proxy_divergence=breadth.index_proxy_divergence if breadth else None,
        volatility_score=volatility.volatility_score if volatility else None,
        liquidity_score=liquidity_stress.liquidity.liquidity_score if liquidity_stress else None,
        correlation_score=correlation.systemic_co_movement_score if correlation else None,
        average_correlation=correlation.average_pairwise_correlation if correlation else None,
        dispersion_score=dispersion.dispersion_score if dispersion else None,
        differentiation_score=dispersion.correlation_context.differentiation_score
        if dispersion and dispersion.correlation_context
        else None,
        stress_score=liquidity_stress.stress.stress_score if liquidity_stress else None,
        stressed_member_fraction=liquidity_stress.stress.stressed_member_fraction
        if liquidity_stress
        else None,
        sector_rotation_velocity=sectors.rotation_velocity if sectors else None,
        sector_leadership_spread=sectors.leadership_spread if sectors else None,
        sector_breadth_score=median(sector_breadths) if sector_breadths else None,
        sector_coverage=sectors.market_coverage if sectors else None,
        source_coverage=min(coverages) if coverages else 1,
        data_quality=quality,
        quality_flags=frozenset(flags),
    )


def evidence_from_snapshot(snapshot: MarketContextSnapshot) -> RegimeEvidence:
    values = snapshot.feature_values or {}

    def scalar(name: str) -> float | None:
        value = values.get(name)
        return (
            float(value)
            if isinstance(value, (float, int)) and not isinstance(value, bool)
            else None
        )

    return RegimeEvidence(
        as_of_ns=snapshot.as_of_ns,
        market_direction_score=snapshot.market_direction_score,
        trend_score=snapshot.trend_score,
        breadth_score=snapshot.breadth_score,
        breadth_participation_score=scalar("breadth.participation"),
        volatility_score=snapshot.volatility_score,
        liquidity_score=snapshot.liquidity_score,
        correlation_score=snapshot.correlation_score,
        average_correlation=scalar("correlation.average"),
        dispersion_score=snapshot.dispersion_score,
        stress_score=snapshot.stress_score,
        stressed_member_fraction=scalar("stress.stressed_member_fraction"),
        sector_rotation_velocity=scalar("sector.rotation.velocity"),
        sector_leadership_spread=scalar("sector.leadership.spread"),
        data_quality=snapshot.data_quality,
        quality_flags=snapshot.quality_flags,
    )
