"""Point-in-time sector loadings and separate descriptive label comparisons."""

from collections import defaultdict

import numpy as np

from mce.features.factors.models import (
    FactorArtifact,
    FactorContext,
    FactorGroupSummary,
    SectorFactorExposure,
)
from mce.features.sectors.membership import SectorUniverseProvider
from mce.statistical.models import LatentStateInference


def sector_exposures(
    artifact: FactorArtifact, provider: SectorUniverseProvider | None, as_of_ns: int
) -> tuple[SectorFactorExposure, ...]:
    if provider is None:
        return ()
    snapshot = provider.members(as_of_ns)
    if snapshot.as_of_ns != as_of_ns:
        raise ValueError("sector provider returned a different factor clock")
    if not snapshot.exclusive:
        return ()
    groups: dict[str, list[int]] = defaultdict(list)
    for member in snapshot.members:
        if member.symbol in artifact.symbols:
            groups[member.sector].append(artifact.symbols.index(member.symbol))
    loadings = np.asarray(artifact.loadings)
    return tuple(
        SectorFactorExposure(
            sector,
            tuple(artifact.symbols[index] for index in sorted(indices)),
            tuple(float(value) for value in np.mean(loadings[indices], axis=0)),
            tuple(float(value) for value in np.std(loadings[indices], axis=0)),
        )
        for sector, indices in sorted(groups.items())
    )


def compare_factor_contexts(
    contexts: tuple[FactorContext, ...], labels: dict[int, str | int]
) -> tuple[FactorGroupSummary, ...]:
    """Group finished context by supplied same-clock deterministic or latent labels."""
    groups: dict[str, list[FactorContext]] = defaultdict(list)
    for item in contexts:
        if (
            item.as_of_ns in labels
            and item.causal
            and item.first_share is not None
            and item.dominance is not None
            and item.decomposition is not None
        ):
            groups[str(labels[item.as_of_ns])].append(item)
    summaries: list[FactorGroupSummary] = []
    for label, items in sorted(groups.items()):
        volatility = [
            item.factor_volatility for item in items if item.factor_volatility is not None
        ]
        summaries.append(
            FactorGroupSummary(
                label,
                len(items),
                float(np.mean([item.first_share for item in items])),
                float(np.mean([item.dominance for item in items])),
                float(
                    np.mean(
                        [
                            item.decomposition.residual_dispersion
                            for item in items
                            if item.decomposition is not None
                        ]
                    )
                ),
                float(np.mean(volatility)) if volatility else None,
            )
        )
    return tuple(summaries)


def compare_with_latent_states(
    contexts: tuple[FactorContext, ...], inferences: tuple[LatentStateInference, ...]
) -> tuple[FactorGroupSummary, ...]:
    """Only causal, same-clock filtered HMM states may enter this comparison."""
    if any(not item.causal for item in inferences):
        raise ValueError("factor comparison requires causal filtered latent states")
    labels: dict[int, str | int] = {}
    context_clocks = {item.as_of_ns for item in contexts}
    for item in inferences:
        if item.as_of_ns not in context_clocks:
            raise ValueError("latent-state comparison clock has no factor context")
        if item.as_of_ns in labels:
            raise ValueError("duplicate latent-state comparison clock")
        labels[item.as_of_ns] = item.state_id
    return compare_factor_contexts(contexts, labels)
