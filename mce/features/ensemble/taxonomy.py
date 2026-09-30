"""Ordered, inspectable descriptive headline rules."""

from mce.config.settings import EnsembleSettings
from mce.features.ensemble.models import EnsembleHeadline, EnsembleRuleTrace


def classify(
    scores: dict[str, float | None],
    settings: EnsembleSettings,
    *,
    eligible: bool,
    deterministic_systemic: bool = False,
    rotation_evidence: float | None = None,
    event_proximity: float | None = None,
) -> tuple[EnsembleHeadline, tuple[EnsembleRuleTrace, ...]]:
    t = settings.thresholds
    direction = scores.get("direction")
    participation = scores.get("participation")
    stress = scores.get("volatility_stress")
    liquidity = scores.get("liquidity_health")
    structure = scores.get("systemic_structure")
    external = scores.get("external_risk")
    disagreement = scores.get("disagreement")
    stability = scores.get("temporal_stability")
    core = scores.get("core_market_score")
    rules = {
        "systemic_stress": (
            stress is not None
            and stress >= t.severe_stress
            and structure is not None
            and structure >= t.systemic_structure
            and (liquidity is not None and liquidity <= t.poor_liquidity or deterministic_systemic),
            (("severe_stress", t.severe_stress), ("systemic_structure", t.systemic_structure)),
        ),
        "liquidity_stress": (
            liquidity is not None and liquidity <= t.poor_liquidity,
            (("poor_liquidity", t.poor_liquidity),),
        ),
        "volatility_stress": (
            stress is not None and stress >= t.severe_stress,
            (("severe_stress", t.severe_stress),),
        ),
        "broad_risk_off": (
            direction is not None
            and direction <= -t.positive_direction
            and participation is not None
            and participation <= -t.positive_participation,
            (
                ("negative_direction", -t.positive_direction),
                ("negative_participation", -t.positive_participation),
            ),
        ),
        "broad_risk_on": (
            direction is not None
            and direction >= t.positive_direction
            and participation is not None
            and participation >= t.positive_participation
            and stress is not None
            and stress <= t.elevated_stress
            and liquidity is not None
            and liquidity >= t.healthy_liquidity
            and (disagreement is None or disagreement < t.high_disagreement),
            (
                ("positive_direction", t.positive_direction),
                ("positive_participation", t.positive_participation),
                ("elevated_stress", t.elevated_stress),
                ("healthy_liquidity", t.healthy_liquidity),
            ),
        ),
        "rotation": (
            direction is not None
            and abs(direction) < t.positive_direction
            and structure is not None
            and structure < t.systemic_structure
            and rotation_evidence is not None
            and rotation_evidence >= 0.6,
            (("systemic_structure", t.systemic_structure), ("rotation_evidence", 0.6)),
        ),
        "calm_range": (
            direction is not None
            and abs(direction) <= 0.2
            and stress is not None
            and stress <= t.calm_stress
            and liquidity is not None
            and liquidity >= t.healthy_liquidity,
            (
                ("neutral_direction", 0.2),
                ("calm_stress", t.calm_stress),
                ("healthy_liquidity", t.healthy_liquidity),
            ),
        ),
        "external_event_risk": (
            settings.external_event_headline_enabled
            and external is not None
            and external >= t.external_event
            and event_proximity is not None
            and event_proximity >= t.external_event
            and core is not None
            and abs(core) <= 0.2,
            (("external_event", t.external_event), ("neutral_core", 0.2)),
        ),
        "transitional": (
            stability is not None
            and stability <= t.low_stability
            or disagreement is not None
            and disagreement >= t.high_disagreement,
            (("low_stability", t.low_stability), ("high_disagreement", t.high_disagreement)),
        ),
        "mixed": (True, ()),
    }
    inputs = tuple(sorted((name, value) for name, value in scores.items()))
    trace = tuple(
        EnsembleRuleTrace(
            rule_id=name,
            precedence=position,
            condition={
                "systemic_stress": (
                    "severe stress and systemic structure with poor liquidity "
                    "or deterministic systemic classification"
                ),
                "liquidity_stress": "liquidity health at or below poor threshold",
                "volatility_stress": "volatility/stress at or above severe threshold",
                "broad_risk_off": "negative direction and participation",
                "broad_risk_on": (
                    "positive direction and participation, contained stress, "
                    "healthy liquidity, no severe disagreement"
                ),
                "rotation": "weak net direction, low synchronization, strong rotation evidence",
                "calm_range": "near-neutral direction, calm stress, healthy liquidity",
                "external_event_risk": (
                    "enabled, elevated external and scheduled-event risk with neutral core"
                ),
                "transitional": "low temporal stability or high disagreement",
                "mixed": "sufficient evidence without a higher-precedence match",
            }[name],
            matched=bool(eligible and rules[name][0]),
            inputs=inputs,
            thresholds=rules[name][1],
        )
        for position, name in enumerate(settings.precedence)
    )
    headline = next(
        (EnsembleHeadline(item.rule_id) for item in trace if item.matched),
        EnsembleHeadline.UNKNOWN,
    )
    return headline, trace
