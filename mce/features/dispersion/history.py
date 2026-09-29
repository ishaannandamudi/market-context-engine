"""Prior-only dispersion ranks, robust shocks and persistence."""

from dataclasses import dataclass
from math import isfinite
from statistics import median


@dataclass(frozen=True)
class HistoricalDispersion:
    sample_count: int
    percentile: float | None
    robust_z: float | None
    shock: float | None
    persistence: float | None


def summarize_history(
    current: float | None,
    slots: tuple[float | None, ...],
    *,
    baseline_sessions: int,
    minimum_samples: int,
    persistence_observations: int,
) -> HistoricalDispersion:
    prior = tuple(value for value in slots[:baseline_sessions] if value is not None)
    if current is None or len(prior) < minimum_samples or any(not isfinite(v) for v in prior):
        return HistoricalDispersion(len(prior), None, None, None, None)
    percentile = (
        sum(value < current for value in prior) + 0.5 * sum(value == current for value in prior)
    ) / len(prior)
    center = median(prior)
    mad = median(abs(value - center) for value in prior)
    z = (current - center) / (1.4826 * mad) if mad > 0 else (0.0 if current == center else None)
    if z is not None and not isfinite(z):
        z = None
    rise = max(0.0, current - center)
    denominator = abs(current) + abs(center)
    shock = max(0.0, 2 * percentile - 1) * (rise / denominator if denominator > 0 else 0.0)
    recent = (current, *slots[: persistence_observations - 1])
    complete = tuple(value for value in recent if value is not None)
    persistence = (
        sum(value > center for value in complete) / persistence_observations
        if len(complete) == persistence_observations
        else None
    )
    return HistoricalDispersion(len(prior), percentile, z, shock, persistence)
