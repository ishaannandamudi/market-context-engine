"""Pure, replayable regime decisions from explicit current evidence and prior states."""

from mce.config.settings import MCEConfig
from mce.features.regime.classifier import (
    active_families,
    classify_dimensions,
    detect_conflicts,
    evaluate_rules,
    threshold_margin,
    unknown_dimensions,
)
from mce.features.regime.hysteresis import stabilize_dimensions, stabilize_headline
from mce.features.regime.models import (
    DimensionTransition,
    MarketRegimeState,
    RegimeEvidence,
    RegimeScorecard,
    RegimeTransition,
)
from mce.features.regime.taxonomy import BreadthState, HeadlineRegime
from mce.models.enums import DataQuality, QualityFlag
from mce.reproducibility import config_fingerprint

_QUALITY_FACTOR = {
    DataQuality.GOOD: 1.0,
    DataQuality.DEGRADED: 0.75,
    DataQuality.UNKNOWN: 0.5,
    DataQuality.POOR: 0.35,
    DataQuality.UNUSABLE: 0.0,
}


class MarketRegimeEngine:
    def __init__(self, config: MCEConfig) -> None:
        self.config = config
        self.fingerprint = config_fingerprint(config)

    def compute(
        self,
        evidence: RegimeEvidence,
        *,
        history: tuple[MarketRegimeState, ...] = (),
    ) -> MarketRegimeState:
        c = self.config.features.regime
        prior_candidates = tuple(
            item
            for item in history
            if item.as_of_ns < evidence.as_of_ns and item.available_at_ns <= evidence.as_of_ns
        )
        if len({item.as_of_ns for item in prior_candidates}) != len(prior_candidates):
            raise ValueError("regime history timestamps must be unique")
        prior = max(prior_candidates, key=lambda item: item.as_of_ns, default=None)
        if prior is not None and prior.config_fingerprint != self.fingerprint:
            raise ValueError("regime history configuration fingerprint differs")
        quality_factor = _QUALITY_FACTOR[evidence.data_quality] * evidence.source_coverage
        raw = (
            classify_dimensions(evidence, c)
            if c.enabled and quality_factor >= c.minimum_evidence_quality
            else unknown_dimensions()
        )
        labels, held_dimensions = stabilize_dimensions(raw, evidence, prior, c)
        if not c.enabled or quality_factor < c.minimum_evidence_quality:
            labels, held_dimensions = unknown_dimensions(), ()
        conflicts = detect_conflicts(evidence, c)
        candidate, candidate_rule, trace = evaluate_rules(evidence, labels, conflicts, c)
        maximum_conflict = max((item.severity for item in conflicts), default=0.0)
        families = active_families(evidence)
        family_factor = 0.75 + 0.25 * len(families) / 8
        critical = {"direction", "breadth", "stress"}.issubset(
            families
        ) and labels.breadth != BreadthState.UNKNOWN
        eligible = (
            c.enabled
            and critical
            and len(families) >= c.minimum_required_families
            and quality_factor >= c.minimum_evidence_quality
        )
        headline, headline_held = (
            stabilize_headline(candidate, evidence, prior, c, maximum_conflict)
            if eligible
            else (HeadlineRegime.UNKNOWN, False)
        )
        margin = threshold_margin(headline, evidence, conflicts, c)
        agreement = 1 - 0.5 * maximum_conflict
        strength = quality_factor * family_factor * (0.5 + 0.5 * margin) * agreement
        if headline_held:
            strength *= 0.85
        strength = max(0.0, min(1.0, strength))
        if not eligible:
            strength = 0.0
        if not eligible or strength < c.minimum_evidence_strength:
            headline = HeadlineRegime.UNKNOWN
            headline_held = False
        selected_rule_id = (
            prior.selected_rule_id
            if headline_held and prior is not None
            else candidate_rule
            if headline != HeadlineRegime.UNKNOWN
            else None
        )
        scorecard = RegimeScorecard(
            evidence=evidence,
            active_families=families,
            quality_factor=quality_factor,
            family_factor=family_factor,
            threshold_margin=margin,
            agreement_factor=agreement,
        )
        transition = None
        dimension_transitions: tuple[DimensionTransition, ...] = ()
        if prior is not None:
            dimensions = (
                ("direction", prior.direction_regime.value, labels.direction.value),
                ("trend", prior.trend_regime.value, labels.trend.value),
                ("volatility", prior.volatility_regime.value, labels.volatility.value),
                ("liquidity", prior.liquidity_regime.value, labels.liquidity.value),
                ("correlation", prior.correlation_regime.value, labels.correlation.value),
                ("breadth", prior.breadth_regime.value, labels.breadth.value),
                ("dispersion", prior.dispersion_regime.value, labels.dispersion.value),
                ("stress", prior.stress_regime.value, labels.stress.value),
                ("sector_rotation", prior.sector_rotation_state.value, labels.sector.value),
            )
            dimension_transitions = tuple(
                DimensionTransition(
                    dimension=name,
                    from_state=before,
                    to_state=after,
                    effective_at_ns=evidence.as_of_ns,
                )
                for name, before, after in dimensions
                if before != after
            )
            if prior.headline_regime != headline:
                selected_trace = next(
                    (item for item in trace if item.rule_id == selected_rule_id), None
                )
                transition = RegimeTransition(
                    from_regime=prior.headline_regime,
                    to_regime=headline,
                    effective_at_ns=evidence.as_of_ns,
                    triggering_rule_id=selected_rule_id or "regime.insufficient_evidence",
                    triggering_evidence=selected_trace.source_features if selected_trace else (),
                    evidence_strength_before=prior.evidence_strength,
                    evidence_strength_after=strength,
                    conflict_ids=tuple(item.conflict_id for item in conflicts),
                )
        if headline == HeadlineRegime.UNKNOWN:
            started_at = None
            age = None
        elif prior is None:
            started_at = evidence.as_of_ns
            age = None
        elif prior.headline_regime == headline:
            started_at = prior.regime_started_at_ns or prior.as_of_ns
            age = (prior.regime_age_observations or 1) + 1
        else:
            started_at = evidence.as_of_ns
            age = 1
        flags = set(evidence.quality_flags)
        if not critical or len(families) < c.minimum_required_families:
            flags.add(QualityFlag.MISSING_DATA)
        if quality_factor < c.minimum_evidence_quality:
            flags.add(QualityFlag.LOW_COVERAGE)
        if headline == HeadlineRegime.UNKNOWN and eligible:
            flags.add(QualityFlag.MISSING_DATA)
        quality = (
            evidence.data_quality
            if headline != HeadlineRegime.UNKNOWN
            else DataQuality.POOR
            if evidence.data_quality != DataQuality.UNUSABLE
            else DataQuality.UNUSABLE
        )
        return MarketRegimeState(
            as_of_ns=evidence.as_of_ns,
            available_at_ns=evidence.as_of_ns,
            config_fingerprint=self.fingerprint,
            headline_regime=headline,
            direction_regime=labels.direction,
            trend_regime=labels.trend,
            volatility_regime=labels.volatility,
            liquidity_regime=labels.liquidity,
            correlation_regime=labels.correlation,
            breadth_regime=labels.breadth,
            dispersion_regime=labels.dispersion,
            stress_regime=labels.stress,
            sector_rotation_state=labels.sector,
            evidence_strength=strength,
            scorecard=scorecard,
            conflicts=conflicts,
            rule_trace=trace,
            candidate_regime=candidate,
            candidate_rule_id=candidate_rule,
            selected_rule_id=selected_rule_id,
            hysteresis_held=headline_held or bool(held_dimensions),
            held_dimensions=held_dimensions,
            transition=transition,
            dimension_transitions=dimension_transitions,
            regime_started_at_ns=started_at,
            regime_age_observations=age,
            data_quality=quality,
            quality_flags=frozenset(flags),
        )
