"""Explicit bounded quality, coverage, freshness, and model diagnostics transforms."""

from math import log

from mce.config.settings import EnsembleSettings
from mce.features.ensemble.contracts import FilteredHMMContext
from mce.models.enums import DataQuality

QUALITY_FACTOR = {
    DataQuality.GOOD: 1.0,
    DataQuality.DEGRADED: 0.7,
    DataQuality.POOR: 0.2,
    DataQuality.UNUSABLE: 0.0,
    DataQuality.UNKNOWN: 0.0,
}


def clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def reliability(
    quality: DataQuality,
    coverage: float,
    age_ns: int,
    settings: EnsembleSettings,
    *,
    flags: int = 0,
    multiplier: float = 1.0,
) -> float:
    freshness = clamp(1 - max(0, age_ns) / settings.max_source_age_ns)
    return clamp(
        QUALITY_FACTOR[quality]
        * clamp(coverage)
        * freshness
        * (0.85 ** min(flags, 4))
        * clamp(multiplier)
    )


def hmm_reliability(context: FilteredHMMContext) -> float:
    inference = context.inference
    count = len(inference.state_probabilities or ())
    entropy = inference.posterior_entropy
    entropy_factor = clamp(1 - entropy / log(count)) if entropy is not None and count > 1 else 0.75
    quality = context.quality
    diagnostics = (
        (1.0 if quality.converged else 0.35) * (0.5 if quality.degenerate_states else 1.0)
        if quality is not None
        else 0.75
    )
    occupancy = (
        clamp(context.profile.occupancy_fraction / 0.10) if context.profile is not None else 0.75
    )
    return clamp(entropy_factor * diagnostics * occupancy * (0.85 ** len(inference.quality_flags)))
