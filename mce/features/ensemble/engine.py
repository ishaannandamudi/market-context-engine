"""Deterministic synthesis of precomputed, causal market context."""

from mce.config.settings import MCEConfig
from mce.features.ensemble.contracts import EnsembleContextInput
from mce.features.ensemble.disagreement import detect_conflicts, disagreement_score
from mce.features.ensemble.evidence import collect_candidates
from mce.features.ensemble.grouping import CORE_GROUPS, aggregate
from mce.features.ensemble.models import (
    EnsembleContextFeatureSet,
    EnsembleContextReliability,
    EnsembleHeadline,
    EnsembleScorecard,
)
from mce.features.ensemble.reliability import clamp, hmm_reliability, reliability
from mce.features.ensemble.taxonomy import classify
from mce.features.regime.taxonomy import HeadlineRegime
from mce.models.enums import DataQuality, QualityFlag
from mce.reproducibility import config_fingerprint


class EnsembleContextFeatureEngine:
    def __init__(self, config: MCEConfig) -> None:
        self.config = config
        self.settings = config.features.ensemble

    def compute(self, source: EnsembleContextInput) -> EnsembleContextFeatureSet:
        fingerprint = config_fingerprint(self.config)
        if not self.settings.enabled:
            return EnsembleContextFeatureSet(
                as_of_ns=source.as_of_ns,
                config_fingerprint=fingerprint,
                headline=EnsembleHeadline.UNKNOWN,
                scorecard=EnsembleScorecard(),
                data_quality=DataQuality.UNKNOWN,
            )
        candidates = collect_candidates(source, self.settings)
        evidence, groups, contributors, core = aggregate(candidates, source.as_of_ns, self.settings)
        group_values = {item.name: item.score for item in groups}
        active_count = sum(item.adjusted_weight > 0 for item in evidence)
        context_reliability: list[EnsembleContextReliability] = []
        if source.deterministic_regime is not None:
            state = source.deterministic_regime
            value = reliability(
                state.data_quality,
                state.evidence_strength,
                source.as_of_ns - state.available_at_ns,
                self.settings,
                flags=len(state.quality_flags),
            )
            context_reliability.append(
                EnsembleContextReliability(
                    family="deterministic_regime",
                    reliability=value,
                    reason="quality × evidence strength × freshness",
                )
            )
        if source.hmm is not None:
            value = hmm_reliability(source.hmm) * reliability(
                DataQuality.GOOD,
                1.0,
                source.as_of_ns - source.hmm.inference.as_of_ns,
                self.settings,
            )
            context_reliability.append(
                EnsembleContextReliability(
                    family="hmm",
                    reliability=value,
                    reason="filtered entropy × convergence × occupancy × freshness",
                )
            )
        if source.seasonality is not None:
            seasonal = source.seasonality
            value = reliability(
                seasonal.data_quality,
                seasonal.member_coverage,
                source.as_of_ns - seasonal.as_of_ns,
                self.settings,
                flags=len(seasonal.quality_flags),
                multiplier=1.0 if seasonal.history_fingerprint is not None else 0.5,
            )
            context_reliability.append(
                EnsembleContextReliability(
                    family="seasonality",
                    reliability=value,
                    reason="quality × coverage × causal history × freshness",
                )
            )
        if source.regime_transition is not None:
            transition = source.regime_transition
            value = reliability(
                transition.data_quality,
                1.0 if transition.stability_score is not None else 0.0,
                source.as_of_ns - transition.as_of_ns,
                self.settings,
                flags=len(transition.quality_flags),
            )
            context_reliability.append(
                EnsembleContextReliability(
                    family="regime_transition",
                    reliability=value,
                    reason="quality × available stability × freshness",
                )
            )
        context_weights = {item.family: item.reliability for item in context_reliability}
        conflicts = detect_conflicts(
            source,
            evidence,
            groups,
            self.settings.minimum_reliability,
            hmm_weight=context_weights.get("hmm", 0.0),
            regime_weight=context_weights.get("deterministic_regime", 0.0),
        )
        disagreement = disagreement_score(conflicts, active_count)
        temporal = (
            source.regime_transition.stability_score
            if source.regime_transition is not None
            and context_weights.get("regime_transition", 0) >= self.settings.minimum_reliability
            else None
        )
        active_groups = tuple(
            item for item in groups if item.name in CORE_GROUPS and item.adjusted_weight > 0
        )
        critical = {"direction", "participation", "volatility_stress", "liquidity_health"}
        critical_count = sum(item.name in critical for item in active_groups)
        eligible = (
            core is not None
            and len(active_groups) >= self.settings.minimum_active_groups
            and critical_count >= self.settings.minimum_critical_groups
            and active_count >= self.settings.minimum_active_families
        )
        reliability_mean = (
            sum(item.normalized_active_weight * item.reliability for item in active_groups)
            if active_groups
            else 0.0
        )
        coverage = len(active_groups) / len(CORE_GROUPS)
        strength = (
            clamp(reliability_mean * coverage * (1 - (disagreement or 0)) * (0.5 + 0.5 * abs(core)))
            if eligible and core is not None
            else None
        )
        missing_fraction = 1 - coverage
        hmm_uncertainty = (
            1
            - next(
                (item.reliability for item in context_reliability if item.family == "hmm"),
                0.0,
            )
            if source.hmm is not None
            else 0.0
        )
        uncertainty = (
            clamp(
                0.45 * (disagreement or 0)
                + 0.30 * (1 - reliability_mean)
                + 0.15 * missing_fraction
                + 0.05 * hmm_uncertainty
                + 0.05 * (1 - temporal if temporal is not None else 0)
            )
            if core is not None
            else None
        )
        scorecard = EnsembleScorecard(
            direction=group_values["direction"],
            participation=group_values["participation"],
            volatility_stress=group_values["volatility_stress"],
            liquidity_health=group_values["liquidity_health"],
            systemic_structure=group_values["systemic_structure"],
            external_risk=group_values["external_risk"],
            core_market_score=core,
            disagreement=disagreement,
            consensus=1 - disagreement if disagreement is not None else None,
            uncertainty=uncertainty,
            evidence_strength=strength,
            temporal_stability=temporal,
        )
        rotation_evidence = (
            source.sectors.rotation_velocity
            if source.sectors is not None
            else source.snapshot.dispersion_score
        )
        event_proximity = next(
            (
                item.value
                for item in evidence
                if item.family == "macro_event" and item.adjusted_weight > 0
            ),
            None,
        )
        headline, trace = classify(
            {
                "direction": scorecard.direction,
                "participation": scorecard.participation,
                "volatility_stress": scorecard.volatility_stress,
                "liquidity_health": scorecard.liquidity_health,
                "systemic_structure": scorecard.systemic_structure,
                "external_risk": scorecard.external_risk,
                "core_market_score": scorecard.core_market_score,
                "disagreement": scorecard.disagreement,
                "temporal_stability": scorecard.temporal_stability,
            },
            self.settings,
            eligible=eligible,
            deterministic_systemic=source.deterministic_regime is not None
            and context_weights.get("deterministic_regime", 0) >= self.settings.minimum_reliability
            and source.deterministic_regime.headline_regime == HeadlineRegime.SYSTEMIC_STRESS,
            rotation_evidence=rotation_evidence,
            event_proximity=event_proximity,
        )
        flags: set[QualityFlag] = set()
        if not eligible:
            flags.add(QualityFlag.ENSEMBLE_INSUFFICIENT_EVIDENCE)
        if any(item.reliability < self.settings.minimum_reliability for item in evidence) or any(
            item.reliability < self.settings.minimum_reliability for item in context_reliability
        ):
            flags.add(QualityFlag.ENSEMBLE_SOURCE_DEGRADED)
        if disagreement is not None and disagreement >= self.settings.thresholds.high_disagreement:
            flags.add(QualityFlag.ENSEMBLE_HIGH_DISAGREEMENT)
        quality = (
            DataQuality.POOR
            if not eligible
            else DataQuality.DEGRADED
            if flags
            else DataQuality.GOOD
        )
        return EnsembleContextFeatureSet(
            as_of_ns=source.as_of_ns,
            config_fingerprint=fingerprint,
            headline=headline,
            scorecard=scorecard,
            evidence=evidence,
            groups=groups,
            conflicts=conflicts,
            context_reliability=tuple(context_reliability),
            contributors=contributors,
            rule_trace=trace,
            deterministic_regime=source.deterministic_regime.headline_regime.value
            if source.deterministic_regime is not None
            else None,
            hmm_state_id=source.hmm.inference.state_id if source.hmm is not None else None,
            data_quality=quality,
            quality_flags=frozenset(flags),
        )
