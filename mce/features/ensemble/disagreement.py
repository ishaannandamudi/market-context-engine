"""Stable cross-engine contradiction records; missing comparisons are not agreement."""

from mce.features.ensemble.contracts import EnsembleContextInput
from mce.features.ensemble.models import EnsembleConflict, EnsembleEvidence, EnsembleGroup
from mce.features.ensemble.reliability import clamp
from mce.features.regime.taxonomy import HeadlineRegime


def detect_conflicts(
    source: EnsembleContextInput,
    evidence: tuple[EnsembleEvidence, ...],
    groups: tuple[EnsembleGroup, ...],
    minimum_reliability: float,
    *,
    hmm_weight: float = 0.0,
    regime_weight: float = 0.0,
) -> tuple[EnsembleConflict, ...]:
    active = {item.family: item.value for item in evidence if item.adjusted_weight > 0}
    group_values = {item.name: item.score for item in groups}
    conflicts: list[EnsembleConflict] = []

    def record(
        name: str,
        first: str,
        second: str,
        value_a: float | None,
        value_b: float | None,
        severity: float,
        description: str,
    ) -> None:
        if value_a is not None and value_b is not None and severity > 0:
            conflicts.append(
                EnsembleConflict(
                    conflict_id=name,
                    family_a=first,
                    family_b=second,
                    evidence_a=value_a,
                    evidence_b=value_b,
                    severity=clamp(severity),
                    description=description,
                )
            )

    trend, breadth = active.get("trend"), active.get("breadth")
    if trend is not None and breadth is not None and trend * breadth < 0:
        record(
            "direction_vs_breadth",
            "trend",
            "breadth",
            trend,
            breadth,
            min(abs(trend), abs(breadth)),
            "Trend and constituent participation point in opposite directions.",
        )
    realized, options = active.get("realized_volatility"), active.get("options")
    if realized is not None and options is not None and options - realized >= 0.35:
        record(
            "realized_vs_options",
            "realized_volatility",
            "options",
            realized,
            options,
            options - realized,
            "Options stress exceeds realized-volatility stress.",
        )
    correlation, network = active.get("correlation"), active.get("network")
    if correlation is not None and network is not None and abs(correlation - network) >= 0.4:
        record(
            "correlation_vs_network",
            "correlation",
            "network",
            correlation,
            network,
            abs(correlation - network),
            "Dependence and network connectivity differ.",
        )
    factor = active.get("factors")
    if factor is not None and network is not None and abs(factor - network) >= 0.4:
        record(
            "factor_vs_network",
            "factors",
            "network",
            factor,
            network,
            abs(factor - network),
            "Factor dominance and graph connectivity differ.",
        )
    dispersion = active.get("dispersion")
    if correlation is not None and dispersion is not None and abs(correlation - dispersion) >= 0.4:
        record(
            "structure_vs_dispersion",
            "correlation",
            "dispersion",
            correlation,
            dispersion,
            abs(correlation - dispersion),
            "Co-movement and cross-sectional differentiation differ.",
        )
    liquidity = group_values.get("liquidity_health")
    market_stress = active.get("market_stress")
    if (
        liquidity is not None
        and market_stress is not None
        and liquidity >= 0.65
        and market_stress >= 0.65
    ):
        record(
            "liquidity_vs_stress",
            "liquidity",
            "market_stress",
            liquidity,
            market_stress,
            min(liquidity, market_stress),
            "Healthy liquidity coexists with high market stress.",
        )
    macro = active.get("macro_stress")
    if macro is not None and market_stress is not None and macro >= 0.7 and market_stress <= 0.35:
        record(
            "macro_vs_market",
            "macro_stress",
            "market_stress",
            macro,
            market_stress,
            macro - market_stress,
            "Observed macro stress exceeds current market stress.",
        )
    regime = source.deterministic_regime
    hmm = source.hmm
    if (
        regime is not None
        and hmm is not None
        and hmm_weight >= minimum_reliability
        and regime_weight >= minimum_reliability
    ):
        adverse = (
            regime.headline_regime == HeadlineRegime.RISK_ON
            and (
                (hmm.direction is not None and hmm.direction <= -0.4)
                or (hmm.stress is not None and hmm.stress >= 0.7)
            )
        ) or (
            regime.headline_regime == HeadlineRegime.RISK_OFF
            and hmm.direction is not None
            and hmm.direction >= 0.4
        )
        if adverse:
            record(
                "deterministic_vs_hmm",
                "deterministic_regime",
                "hmm",
                1.0 if regime.headline_regime == HeadlineRegime.RISK_ON else -1.0,
                hmm.direction
                if hmm.direction is not None
                else 1 - 2 * hmm.stress
                if hmm.stress is not None
                else None,
                hmm_weight,
                "Deterministic state and filtered HMM profile disagree.",
            )
    core = next((item for item in groups if item.name == "direction"), None)
    if (
        regime is not None
        and core is not None
        and core.score is not None
        and regime_weight >= minimum_reliability
    ):
        risk_on_against_core = (
            regime.headline_regime == HeadlineRegime.RISK_ON and core.score <= -0.4
        )
        risk_off_against_core = (
            regime.headline_regime == HeadlineRegime.RISK_OFF and core.score >= 0.4
        )
        if risk_on_against_core or risk_off_against_core:
            record(
                "deterministic_vs_core",
                "deterministic_regime",
                "direction",
                1.0 if risk_on_against_core else -1.0,
                core.score,
                abs(core.score) * regime_weight,
                "Deterministic classification conflicts with current direction evidence.",
            )
    return tuple(conflicts)


def disagreement_score(
    conflicts: tuple[EnsembleConflict, ...], active_family_count: int
) -> float | None:
    if active_family_count < 2:
        return None
    return clamp(sum(item.severity for item in conflicts) / 2)
