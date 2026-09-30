"""One family vote per redundancy group, then one group vote in the core score."""

from mce.config.settings import EnsembleSettings
from mce.features.ensemble.evidence import FAMILY_WEIGHTS, Candidate
from mce.features.ensemble.models import (
    EnsembleContribution,
    EnsembleEvidence,
    EnsembleGroup,
    GroupName,
)

CORE_GROUPS: tuple[GroupName, ...] = (
    "direction",
    "participation",
    "volatility_stress",
    "liquidity_health",
    "systemic_structure",
)
ALL_GROUPS: tuple[GroupName, ...] = (*CORE_GROUPS, "external_risk")


def aggregate(
    candidates: tuple[Candidate, ...],
    as_of_ns: int,
    settings: EnsembleSettings,
) -> tuple[
    tuple[EnsembleEvidence, ...],
    tuple[EnsembleGroup, ...],
    tuple[EnsembleContribution, ...],
    float | None,
]:
    evidence: list[EnsembleEvidence] = []
    groups: list[EnsembleGroup] = []
    for name in ALL_GROUPS:
        members = tuple(item for item in candidates if item.group == name)
        weighted = tuple(
            (item, item.reliability(as_of_ns, settings), FAMILY_WEIGHTS[item.family])
            for item in members
        )
        active = tuple(
            (item, reliability, configured)
            for item, reliability, configured in weighted
            if reliability >= settings.minimum_reliability
        )
        denominator = sum(reliability * configured for _, reliability, configured in active)
        score = (
            sum(item.value * reliability * configured for item, reliability, configured in active)
            / denominator
            if denominator > 0
            else None
        )
        group_reliability = (
            sum(reliability * configured for _, reliability, configured in active)
            / sum(configured for _, _, configured in active)
            if active
            else 0.0
        )
        configured_group = (
            float(getattr(settings.group_weights, name)) if name != "external_risk" else 0.0
        )
        groups.append(
            EnsembleGroup(
                name=name,
                score=score,
                reliability=group_reliability,
                active_families=tuple(item.family for item, _, _ in active),
                configured_weight=configured_group,
                adjusted_weight=configured_group * group_reliability if score is not None else 0.0,
            )
        )
        for item, reliability, configured in weighted:
            adjusted = (
                configured * reliability if reliability >= settings.minimum_reliability else 0.0
            )
            evidence.append(
                EnsembleEvidence(
                    family=item.family,
                    group=name,
                    value=item.value,
                    source_feature=item.feature,
                    source_as_of_ns=item.clock,
                    source_quality=item.quality,
                    coverage=item.coverage,
                    reliability=reliability,
                    configured_weight=configured,
                    adjusted_weight=adjusted,
                    normalized_active_weight=adjusted / denominator if denominator else 0.0,
                )
            )
    total = sum(item.adjusted_weight for item in groups if item.name in CORE_GROUPS)
    groups = [
        item.model_copy(
            update={
                "normalized_active_weight": item.adjusted_weight / total
                if item.name in CORE_GROUPS and total
                else 0.0
            }
        )
        for item in groups
    ]
    contributors: list[EnsembleContribution] = []
    if (
        total > 0
        and sum(item.score is not None for item in groups if item.name in CORE_GROUPS)
        >= settings.minimum_active_groups
    ):
        for group_item in groups:
            if (
                group_item.name not in CORE_GROUPS
                or group_item.score is None
                or group_item.adjusted_weight == 0
            ):
                continue
            oriented = (
                group_item.score
                if group_item.name in ("direction", "participation")
                else 1 - 2 * group_item.score
                if group_item.name in ("volatility_stress", "systemic_structure")
                else 2 * group_item.score - 1
            )
            contributors.append(
                EnsembleContribution(
                    group=group_item.name,
                    oriented_value=oriented,
                    normalized_weight=group_item.normalized_active_weight,
                    contribution=oriented * group_item.normalized_active_weight,
                )
            )
    core = sum(item.contribution for item in contributors) if contributors else None
    return tuple(evidence), tuple(groups), tuple(contributors), core
