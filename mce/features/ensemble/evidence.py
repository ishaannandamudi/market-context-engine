"""Select one precomputed scalar per family and assign it to a redundancy group."""

from dataclasses import dataclass
from statistics import mean

from mce.config.settings import EnsembleSettings
from mce.features.ensemble.contracts import EnsembleContextInput
from mce.features.ensemble.models import GroupName
from mce.features.ensemble.reliability import reliability
from mce.models.enums import DataQuality

FAMILY_WEIGHTS: dict[str, float] = {
    "trend": 1.0,
    "breadth": 0.70,
    "sectors": 0.30,
    "realized_volatility": 0.40,
    "market_stress": 0.60,
    "liquidity": 1.0,
    "correlation": 0.35,
    "factors": 0.25,
    "network": 0.25,
    "dispersion": 0.15,
    "options": 0.35,
    "macro_stress": 0.35,
    "macro_event": 0.30,
}


@dataclass(frozen=True, slots=True)
class Candidate:
    family: str
    group: GroupName
    value: float
    feature: str
    clock: int
    quality: DataQuality
    coverage: float
    flags: int = 0
    multiplier: float = 1.0

    def reliability(self, as_of_ns: int, settings: EnsembleSettings) -> float:
        return reliability(
            self.quality,
            self.coverage,
            as_of_ns - self.clock,
            settings,
            flags=self.flags,
            multiplier=self.multiplier,
        )


def _mean(values: tuple[float | None, ...]) -> float | None:
    present = tuple(value for value in values if value is not None)
    return mean(present) if present else None


def collect_candidates(
    source: EnsembleContextInput, settings: EnsembleSettings
) -> tuple[Candidate, ...]:
    """Raw source result wins over the schema-1.0 snapshot fallback for each family."""

    result: list[Candidate] = []
    snapshot = source.snapshot
    clock = source.as_of_ns
    snapshot_freshness = (
        0.5
        if snapshot.source_freshness is not None
        and any(
            item.is_stale
            for item in snapshot.source_freshness
            if not item.source.startswith(("macro.", "options."))
        )
        else 1.0
    )

    def add(
        family: str,
        group: GroupName,
        value: float | None,
        feature: str,
        source_clock: int,
        quality: DataQuality,
        coverage: float = 1.0,
        flags: int = 0,
        multiplier: float = 1.0,
    ) -> None:
        if feature.startswith("snapshot."):
            multiplier *= snapshot_freshness
        if value is not None and -1 <= value <= 1:
            result.append(
                Candidate(
                    family,
                    group,
                    value,
                    feature,
                    source_clock,
                    quality,
                    coverage,
                    flags,
                    multiplier,
                )
            )

    trend = source.trend
    add(
        "trend",
        "direction",
        trend.market_direction_score
        if trend
        else snapshot.market_direction_score
        if snapshot.market_direction_score is not None
        else snapshot.trend_score,
        "trend.market_direction_score" if trend else "snapshot.market_direction_score",
        trend.as_of_ns if trend else clock,
        trend.data_quality if trend else snapshot.data_quality,
        trend.coverage_ratio if trend else 1.0,
        len(trend.quality_flags) if trend else 0,
    )
    breadth = source.breadth
    add(
        "breadth",
        "participation",
        breadth.participation_score if breadth else snapshot.breadth_score,
        "breadth.participation_score" if breadth else "snapshot.breadth_score",
        breadth.as_of_ns if breadth else clock,
        breadth.data_quality if breadth else snapshot.data_quality,
        breadth.coverage_ratio if breadth else 1.0,
        len(breadth.quality_flags) if breadth else 0,
    )
    sectors = source.sectors
    snapshot_sector_breadth = tuple(
        float(value)
        for name, value in sorted((snapshot.feature_values or {}).items())
        if name.startswith("sector.")
        and name.endswith(".breadth")
        and isinstance(value, (int, float))
        and not isinstance(value, bool)
    )
    sector_value = (
        _mean(tuple(item.breadth_score for item in sectors.sectors.values()))
        if sectors
        else _mean(snapshot_sector_breadth)
    )
    add(
        "sectors",
        "participation",
        sector_value,
        "sectors.breadth_score" if sectors else "snapshot.sector.*.breadth",
        sectors.as_of_ns if sectors else clock,
        sectors.data_quality if sectors else snapshot.data_quality,
        sectors.market_coverage if sectors else 1.0,
        len(sectors.quality_flags) if sectors else 0,
    )
    seasonal_modifier = 1.0
    seasonal = source.seasonality
    if (
        seasonal is not None
        and seasonal.history_fingerprint is not None
        and reliability(
            seasonal.data_quality,
            seasonal.member_coverage,
            clock - seasonal.as_of_ns,
            settings,
        )
        >= settings.minimum_reliability
    ):
        normal_volatility = any(
            item.metric == "volatility" and item.ratio is not None and 0.8 <= item.ratio <= 1.2
            for item in seasonal.market_metrics
        )
        if normal_volatility:
            seasonal_modifier = 0.75
    volatility = source.volatility
    volatility_value = volatility.volatility_score if volatility else snapshot.volatility_score
    add(
        "realized_volatility",
        "volatility_stress",
        volatility_value,
        "volatility.volatility_score" if volatility else "snapshot.volatility_score",
        volatility.as_of_ns if volatility else clock,
        volatility.data_quality if volatility else snapshot.data_quality,
        volatility.coverage_ratio if volatility else 1.0,
        len(volatility.quality_flags) if volatility else 0,
        seasonal_modifier if volatility_value is not None and volatility_value > 0.6 else 1.0,
    )
    market = source.liquidity_stress
    add(
        "market_stress",
        "volatility_stress",
        market.stress.stress_score if market else snapshot.stress_score,
        "stress.stress_score" if market else "snapshot.stress_score",
        market.as_of_ns if market else clock,
        market.stress.data_quality if market else snapshot.data_quality,
        market.stress.evidence_quality if market else 1.0,
        len(market.stress.quality_flags) if market else 0,
    )
    add(
        "liquidity",
        "liquidity_health",
        market.liquidity.liquidity_score if market else snapshot.liquidity_score,
        "liquidity.liquidity_score" if market else "snapshot.liquidity_score",
        market.as_of_ns if market else clock,
        market.liquidity.data_quality if market else snapshot.data_quality,
        market.liquidity.member_coverage if market else 1.0,
        len(market.liquidity.quality_flags) if market else 0,
    )
    correlation = source.correlation
    add(
        "correlation",
        "systemic_structure",
        correlation.systemic_co_movement_score if correlation else snapshot.correlation_score,
        "correlation.systemic_co_movement_score" if correlation else "snapshot.correlation_score",
        correlation.as_of_ns if correlation else clock,
        correlation.data_quality if correlation else snapshot.data_quality,
        correlation.symbol_coverage * correlation.time_coverage if correlation else 1.0,
        len(correlation.quality_flags) if correlation else 0,
    )
    factors = source.factors
    if factors is not None:
        add(
            "factors",
            "systemic_structure",
            factors.dominance,
            "factors.dominance",
            factors.as_of_ns,
            factors.data_quality,
            factors.symbol_coverage * factors.time_coverage,
            len(factors.quality_flags),
            1.0 if factors.causal else 0.0,
        )
    network = source.network
    if network is not None:
        add(
            "network",
            "systemic_structure",
            network.connectivity_score,
            "network.connectivity_score",
            network.as_of_ns,
            network.data_quality,
            network.coverage,
            len(network.quality_flags),
        )
    dispersion = source.dispersion
    differentiation = (
        dispersion.correlation_context.differentiation_score
        if dispersion is not None and dispersion.correlation_context is not None
        else None
    )
    dispersion_value = (
        1 - differentiation
        if differentiation is not None
        else 1 - snapshot.dispersion_score
        if dispersion is None and snapshot.dispersion_score is not None
        else None
    )
    add(
        "dispersion",
        "systemic_structure",
        dispersion_value,
        "dispersion.differentiation_score" if dispersion else "snapshot.dispersion_score",
        dispersion.as_of_ns if dispersion else clock,
        dispersion.data_quality if dispersion else snapshot.data_quality,
        dispersion.member_coverage if dispersion else 1.0,
        len(dispersion.quality_flags) if dispersion else 0,
    )
    options = source.options
    if options is not None:
        primary = next(
            (
                item
                for item in options.per_underlying
                if item.source.underlying_symbol == options.primary_symbol
            ),
            None,
        )
        mode_factor = (
            {"live": 1.0, "delayed": 0.8, "eod": 0.6, "synthetic": 1.0}[
                primary.source.market_data_mode
            ]
            if primary is not None and primary.headline_eligible
            else 0.0
        )
        add(
            "options",
            "external_risk",
            options.options_stress_score,
            "options.options_stress_score",
            options.as_of_ns,
            options.data_quality,
            options.coverage,
            len(options.quality_flags),
            mode_factor,
        )
    macro = source.macro
    if macro is not None:
        usable = (
            sum(not item.stale and not item.release_stale for item in macro.per_series)
            / len(macro.per_series)
            if macro.per_series
            else 1.0
        )
        for family, value, feature in (
            ("macro_stress", macro.stress.score if macro.stress else None, "macro.stress.score"),
            (
                "macro_event",
                macro.event.proximity_score if macro.event else None,
                "macro.event.proximity_score",
            ),
        ):
            add(
                family,
                "external_risk",
                value,
                feature,
                macro.as_of_ns,
                macro.data_quality,
                usable,
                len(macro.quality_flags),
            )
    return tuple(result)
