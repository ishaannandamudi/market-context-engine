"""Deterministic dimension classification, conflicts, and ordered headline rules."""

from dataclasses import dataclass

from mce.config.settings import RegimeSettings
from mce.features.regime.models import RegimeConflict, RegimeEvidence, RuleEvaluation
from mce.features.regime.taxonomy import (
    BreadthState,
    DirectionState,
    HeadlineRegime,
    SectorRotationState,
    StressState,
)
from mce.models.enums import (
    CorrelationRegime,
    DispersionRegime,
    LiquidityRegime,
    TrendRegime,
    VolatilityRegime,
)


@dataclass(frozen=True)
class DimensionLabels:
    direction: DirectionState
    trend: TrendRegime
    volatility: VolatilityRegime
    liquidity: LiquidityRegime
    correlation: CorrelationRegime
    breadth: BreadthState
    dispersion: DispersionRegime
    stress: StressState
    sector: SectorRotationState


def unknown_dimensions() -> DimensionLabels:
    return DimensionLabels(
        DirectionState.UNKNOWN,
        TrendRegime.UNKNOWN,
        VolatilityRegime.UNKNOWN,
        LiquidityRegime.UNKNOWN,
        CorrelationRegime.UNKNOWN,
        BreadthState.UNKNOWN,
        DispersionRegime.UNKNOWN,
        StressState.UNKNOWN,
        SectorRotationState.UNKNOWN,
    )


def classify_dimensions(e: RegimeEvidence, c: RegimeSettings) -> DimensionLabels:
    trend = TrendRegime.UNKNOWN
    if e.trend_score is not None:
        t = e.trend_score
        trend = (
            TrendRegime.STRONG_UP
            if t >= c.trend_strong
            else TrendRegime.UP
            if t >= c.trend_confirmation
            else TrendRegime.STRONG_DOWN
            if t <= -c.trend_strong
            else TrendRegime.DOWN
            if t <= -c.trend_confirmation
            else TrendRegime.SIDEWAYS
        )
    direction = DirectionState.UNKNOWN
    if e.market_direction_score is not None and e.trend_score is not None:
        d, t, b = e.market_direction_score, e.trend_score, e.breadth_score
        if (
            d >= c.direction_entry
            and t >= c.trend_confirmation
            and (b is None or b > -c.breadth_conflict)
        ):
            direction = DirectionState.POSITIVE
        elif (
            d <= -c.direction_entry
            and t <= -c.trend_confirmation
            and (b is None or b < c.breadth_conflict)
        ):
            direction = DirectionState.NEGATIVE
        elif abs(d) < c.direction_exit and abs(t) < c.trend_confirmation:
            direction = DirectionState.NEUTRAL
        else:
            direction = DirectionState.MIXED
    breadth = BreadthState.UNKNOWN
    if e.breadth_score is not None and (
        e.breadth_coverage is None or e.breadth_coverage >= c.minimum_evidence_quality
    ):
        b, p = e.breadth_score, e.breadth_participation_score
        if b >= c.breadth_confirmation:
            breadth = (
                BreadthState.BROAD_POSITIVE
                if b >= c.breadth_conflict and p is not None and p >= c.breadth_confirmation
                else BreadthState.NARROW_POSITIVE
            )
        elif b <= -c.breadth_confirmation:
            breadth = (
                BreadthState.BROAD_NEGATIVE
                if b <= -c.breadth_conflict and p is not None and p <= -c.breadth_confirmation
                else BreadthState.NARROW_NEGATIVE
            )
        else:
            breadth = BreadthState.MIXED
    volatility = VolatilityRegime.UNKNOWN
    if e.volatility_score is not None:
        v = e.volatility_score
        volatility = (
            VolatilityRegime.EXTREME
            if v >= c.volatility_extreme
            else VolatilityRegime.HIGH
            if v >= c.volatility_high
            else VolatilityRegime.LOW
            if v < c.volatility_low
            else VolatilityRegime.NORMAL
        )
    liquidity = LiquidityRegime.UNKNOWN
    if e.liquidity_score is not None:
        liq = e.liquidity_score
        liquidity = (
            LiquidityRegime.STRESSED
            if liq < c.liquidity_stressed
            else LiquidityRegime.THIN
            if liq < c.liquidity_degraded
            else LiquidityRegime.HEALTHY
            if liq >= c.liquidity_healthy
            else LiquidityRegime.NORMAL
        )
    correlation = CorrelationRegime.UNKNOWN
    if e.correlation_score is not None:
        r = e.correlation_score
        correlation = (
            CorrelationRegime.EXTREME
            if r >= c.correlation_systemic
            else CorrelationRegime.HIGH
            if r >= c.correlation_high
            else CorrelationRegime.LOW
            if r < c.correlation_low
            else CorrelationRegime.NORMAL
        )
    dispersion = DispersionRegime.UNKNOWN
    if e.dispersion_score is not None:
        x = e.dispersion_score
        dispersion = (
            DispersionRegime.EXTREME
            if x >= c.dispersion_extreme
            else DispersionRegime.HIGH
            if x >= c.dispersion_high
            else DispersionRegime.LOW
            if x < c.dispersion_low
            else DispersionRegime.NORMAL
        )
    stress = StressState.UNKNOWN
    if e.stress_score is not None:
        s = e.stress_score
        stress = (
            StressState.SYSTEMIC
            if s >= c.stress_systemic
            and e.stressed_member_fraction is not None
            and e.stressed_member_fraction >= c.systemic_member_fraction
            else StressState.HIGH
            if s >= c.stress_high
            else StressState.ELEVATED
            if s >= c.stress_elevated
            else StressState.CALM
        )
    sector = SectorRotationState.UNKNOWN
    if e.sector_rotation_velocity is not None:
        rotation = e.sector_rotation_velocity
        sector = (
            SectorRotationState.FRAGMENTED
            if rotation >= c.rotation_active
            and e.sector_coverage is not None
            and e.sector_coverage < c.minimum_evidence_quality
            else SectorRotationState.ACTIVE_ROTATION
            if rotation >= c.rotation_active
            else SectorRotationState.STABLE_LEADERSHIP
            if rotation <= c.rotation_stable
            else SectorRotationState.MIXED
        )
    return DimensionLabels(
        direction, trend, volatility, liquidity, correlation, breadth, dispersion, stress, sector
    )


def detect_conflicts(e: RegimeEvidence, c: RegimeSettings) -> tuple[RegimeConflict, ...]:
    conflicts: list[RegimeConflict] = []

    def add(identifier: str, severity: float, features: tuple[str, ...], description: str) -> None:
        conflicts.append(
            RegimeConflict(
                conflict_id=identifier,
                severity=severity,
                source_features=features,
                description=description,
            )
        )

    t, b = e.trend_score, e.breadth_score
    if t is not None and b is not None:
        if t >= c.breadth_conflict and b <= -c.breadth_conflict:
            add(
                "regime.conflict.trend_up_breadth_down",
                1,
                ("trend.score", "breadth.score"),
                "Positive trend contradicts negative breadth",
            )
        if t <= -c.breadth_conflict and b >= c.breadth_conflict:
            add(
                "regime.conflict.trend_down_breadth_up",
                1,
                ("trend.score", "breadth.score"),
                "Negative trend contradicts positive breadth",
            )
    if (
        e.correlation_score is not None
        and e.dispersion_score is not None
        and e.correlation_score >= c.correlation_high
        and e.dispersion_score >= c.dispersion_high
    ):
        add(
            "regime.conflict.correlation_dispersion",
            0.6,
            ("correlation.score", "dispersion.score"),
            "High co-movement coexists with high differentiation",
        )
    if (
        e.stress_score is not None
        and e.liquidity_score is not None
        and e.stress_score >= c.stress_high
        and e.liquidity_score >= c.liquidity_healthy
    ):
        add(
            "regime.conflict.stress_healthy_liquidity",
            0.45,
            ("stress.score", "liquidity.score"),
            "Broad stress coexists with healthy measured liquidity",
        )
    if (
        t is not None
        and e.sector_breadth_score is not None
        and t >= c.breadth_conflict
        and e.sector_breadth_score <= -c.breadth_conflict
    ):
        add(
            "regime.conflict.trend_sector_weakness",
            0.75,
            ("trend.score", "sector.breadth"),
            "Positive trend coexists with negative sector participation",
        )
    if (
        t is not None
        and e.sector_rotation_velocity is not None
        and t >= c.breadth_conflict
        and e.sector_rotation_velocity >= c.rotation_active
    ):
        add(
            "regime.conflict.trend_rotation",
            0.35,
            ("trend.score", "sector.rotation.velocity"),
            "Strong positive trend coexists with rapid sector rotation",
        )
    return tuple(conflicts)


def evaluate_rules(
    e: RegimeEvidence,
    labels: DimensionLabels,
    conflicts: tuple[RegimeConflict, ...],
    c: RegimeSettings,
) -> tuple[HeadlineRegime, str, tuple[RuleEvaluation, ...]]:
    d, b, s, liq, r, x, v, rotation = (
        e.market_direction_score,
        e.breadth_score,
        e.stress_score,
        e.liquidity_score,
        e.correlation_score,
        e.dispersion_score,
        e.volatility_score,
        e.sector_rotation_velocity,
    )
    severe = max((item.severity for item in conflicts), default=0.0)
    rules: dict[
        str,
        tuple[
            HeadlineRegime, bool, float | None, tuple[str, ...], tuple[tuple[str, float], ...], str
        ],
    ] = {
        "regime.systemic_stress": (
            HeadlineRegime.SYSTEMIC_STRESS,
            s is not None
            and s >= c.stress_systemic
            and b is not None
            and b <= -c.breadth_confirmation
            and (
                (r is not None and r >= c.correlation_high)
                or (liq is not None and liq < c.liquidity_stressed)
            ),
            s,
            ("stress.score", "breadth.score", "correlation.score", "liquidity.score"),
            (
                ("stress_systemic", c.stress_systemic),
                ("breadth_confirmation", c.breadth_confirmation),
                ("correlation_high", c.correlation_high),
                ("liquidity_stressed", c.liquidity_stressed),
            ),
            "Systemic stress, weak breadth, and co-movement or liquidity strain",
        ),
        "regime.liquidity_stress": (
            HeadlineRegime.LIQUIDITY_STRESS,
            liq is not None and liq < c.liquidity_stressed and s is not None and s >= c.stress_high,
            1 - liq if liq is not None else None,
            ("liquidity.score", "stress.score"),
            (("liquidity_stressed", c.liquidity_stressed), ("stress_high", c.stress_high)),
            "Severely degraded liquidity and high stress",
        ),
        "regime.transitional": (
            HeadlineRegime.TRANSITIONAL,
            severe >= c.conflict_override_severity,
            severe,
            tuple(feature for item in conflicts for feature in item.source_features),
            (("conflict_override_severity", c.conflict_override_severity),),
            "Severe contradictory evidence",
        ),
        "regime.risk_off": (
            HeadlineRegime.RISK_OFF,
            labels.direction == DirectionState.NEGATIVE
            and b is not None
            and b <= -c.breadth_confirmation
            and s is not None
            and s < c.stress_systemic,
            -d if d is not None and d < 0 else None,
            ("trend.direction", "trend.score", "breadth.score", "stress.score"),
            (
                ("direction_entry", c.direction_entry),
                ("trend_confirmation", c.trend_confirmation),
                ("breadth_confirmation", c.breadth_confirmation),
                ("stress_systemic", c.stress_systemic),
            ),
            "Negative direction confirmed by breadth without systemic stress",
        ),
        "regime.risk_on": (
            HeadlineRegime.RISK_ON,
            labels.direction == DirectionState.POSITIVE
            and b is not None
            and b >= c.breadth_confirmation
            and s is not None
            and s < c.stress_high
            and (liq is None or liq >= c.liquidity_stressed),
            d if d is not None and d > 0 else None,
            ("trend.direction", "trend.score", "breadth.score", "stress.score", "liquidity.score"),
            (
                ("direction_entry", c.direction_entry),
                ("trend_confirmation", c.trend_confirmation),
                ("breadth_confirmation", c.breadth_confirmation),
                ("stress_high", c.stress_high),
                ("liquidity_stressed", c.liquidity_stressed),
            ),
            "Positive direction confirmed by breadth and manageable stress",
        ),
        "regime.rotation": (
            HeadlineRegime.ROTATION,
            labels.direction in (DirectionState.NEUTRAL, DirectionState.MIXED)
            and labels.sector == SectorRotationState.ACTIVE_ROTATION
            and x is not None
            and x >= c.dispersion_high
            and (r is None or r < c.correlation_high)
            and s is not None
            and s < c.stress_high,
            rotation,
            (
                "trend.direction",
                "sector.rotation.velocity",
                "dispersion.score",
                "correlation.score",
                "stress.score",
            ),
            (
                ("rotation_active", c.rotation_active),
                ("dispersion_high", c.dispersion_high),
                ("correlation_high", c.correlation_high),
                ("stress_high", c.stress_high),
            ),
            "Active sector rotation and differentiation without systemic co-movement",
        ),
        "regime.range_bound": (
            HeadlineRegime.RANGE_BOUND,
            labels.direction == DirectionState.NEUTRAL
            and e.trend_score is not None
            and abs(e.trend_score) < c.trend_confirmation
            and s is not None
            and s < c.stress_elevated
            and v is not None
            and v < c.volatility_high,
            1 - abs(d) if d is not None else None,
            ("trend.direction", "trend.score", "stress.score", "volatility.score"),
            (
                ("direction_exit", c.direction_exit),
                ("trend_confirmation", c.trend_confirmation),
                ("stress_elevated", c.stress_elevated),
                ("volatility_high", c.volatility_high),
            ),
            "Weak direction and trend with calm stress and non-high volatility",
        ),
        "regime.mixed": (
            HeadlineRegime.MIXED,
            True,
            0.5,
            (),
            (),
            "No more specific headline rule applies",
        ),
    }
    trace = tuple(
        RuleEvaluation(
            rule_id=identifier,
            precedence=index,
            condition_met=rules[identifier][1],
            score=rules[identifier][2],
            source_features=rules[identifier][3],
            thresholds=rules[identifier][4],
            reason=rules[identifier][5],
        )
        for index, identifier in enumerate(c.precedence)
    )
    selected = next(item for item in trace if item.condition_met)
    return rules[selected.rule_id][0], selected.rule_id, trace


def active_families(e: RegimeEvidence) -> tuple[str, ...]:
    families = (
        ("direction", e.market_direction_score is not None and e.trend_score is not None),
        ("breadth", e.breadth_score is not None),
        ("stress", e.stress_score is not None),
        ("volatility", e.volatility_score is not None),
        ("liquidity", e.liquidity_score is not None),
        ("correlation", e.correlation_score is not None),
        ("dispersion", e.dispersion_score is not None),
        ("sector", e.sector_rotation_velocity is not None),
    )
    return tuple(name for name, present in families if present)


def threshold_margin(
    regime: HeadlineRegime,
    e: RegimeEvidence,
    conflicts: tuple[RegimeConflict, ...],
    c: RegimeSettings,
) -> float:
    if regime == HeadlineRegime.RISK_ON and e.market_direction_score is not None:
        return max(
            0.0, min(1.0, (e.market_direction_score - c.direction_entry) / (1 - c.direction_entry))
        )
    if regime == HeadlineRegime.RISK_OFF and e.market_direction_score is not None:
        return max(
            0.0, min(1.0, (-e.market_direction_score - c.direction_entry) / (1 - c.direction_entry))
        )
    if regime == HeadlineRegime.SYSTEMIC_STRESS and e.stress_score is not None:
        return max(0.0, min(1.0, (e.stress_score - c.stress_systemic) / (1 - c.stress_systemic)))
    if regime == HeadlineRegime.LIQUIDITY_STRESS and e.liquidity_score is not None:
        return max(0.0, min(1.0, (c.liquidity_stressed - e.liquidity_score) / c.liquidity_stressed))
    if regime == HeadlineRegime.ROTATION and e.sector_rotation_velocity is not None:
        return max(
            0.0,
            min(1.0, (e.sector_rotation_velocity - c.rotation_active) / (1 - c.rotation_active)),
        )
    if regime == HeadlineRegime.RANGE_BOUND and e.market_direction_score is not None:
        return max(
            0.0, min(1.0, (c.direction_exit - abs(e.market_direction_score)) / c.direction_exit)
        )
    if regime == HeadlineRegime.TRANSITIONAL:
        return max((item.severity for item in conflicts), default=0.0)
    return 0.0
