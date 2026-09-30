"""Prior-only midrank and robust historical comparisons."""

from math import isfinite

from mce.features.options_context.models import HistoricalComparison


def _median(ordered: list[float]) -> float:
    middle = len(ordered) // 2
    return ordered[middle] if len(ordered) % 2 else ordered[middle - 1] / 2 + ordered[middle] / 2


def compare_history(
    current: float | None, prior: tuple[float | None, ...], minimum: int
) -> HistoricalComparison:
    values = sorted(value for value in prior if value is not None and isfinite(value))
    count = len(values)
    if current is None or count < minimum:
        return HistoricalComparison(sample_count=count)
    center = _median(values)
    mad = _median(sorted(abs(value - center) for value in values))
    percentile = (
        sum(value < current for value in values) + 0.5 * sum(value == current for value in values)
    ) / count
    scale = 1.4826 * mad
    z = (
        (current - center) / scale
        if scale > 1e-24 and isfinite(scale)
        else 0.0
        if current == center
        else None
    )
    return HistoricalComparison(
        sample_count=count,
        percentile=percentile,
        robust_z=z if z is None or isfinite(z) else None,
        median=center,
    )
