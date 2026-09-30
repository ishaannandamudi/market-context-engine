"""Prior-release-only macro comparisons; no synthetic consensus or forecasts."""

from math import isfinite

from mce.features.macro.contracts import MacroCategory, MacroRelease
from mce.features.macro.models import MacroComparison, MacroSurpriseContext


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    return ordered[middle] if len(ordered) % 2 else ordered[middle - 1] / 2 + ordered[middle] / 2


def compare_values(
    current: float | None, prior: tuple[float, ...], minimum: int
) -> MacroComparison:
    values = sorted(value for value in prior if isfinite(value))
    count = len(values)
    if current is None or count < minimum:
        return MacroComparison(sample_count=count)
    center = _median(values)
    mad = _median([abs(value - center) for value in values])
    percentile = (
        sum(value < current for value in values) + 0.5 * sum(value == current for value in values)
    ) / count
    scale = 1.4826 * mad
    z = (
        (current - center) / scale
        if scale > 1e-12 and isfinite(scale)
        else 0.0
        if current == center
        else None
    )
    return MacroComparison(
        sample_count=count,
        percentile=percentile,
        robust_z=z if z is None or isfinite(z) else None,
        median=center,
    )


def surprise_context(
    release: MacroRelease,
    prior: tuple[MacroRelease, ...],
    minimum: int,
) -> MacroSurpriseContext:
    if release.consensus is None:
        return MacroSurpriseContext()
    raw = release.actual - release.consensus
    if not isfinite(raw):
        return MacroSurpriseContext()
    values = [
        item.actual - item.consensus
        for item in prior
        if item.consensus is not None
        and item.unit == release.unit
        and isfinite(item.actual - item.consensus)
    ]
    count = len(values)
    scale = None
    z = None
    if count >= minimum:
        center = _median(values)
        mad = _median([abs(value - center) for value in values])
        scale = 1.4826 * mad
        if scale > 1e-12 and isfinite(scale):
            candidate = raw / scale
            z = candidate if isfinite(candidate) else None
        elif raw == 0:
            z = 0.0
    axis = "none"
    sign = 1
    if release.category == MacroCategory.INFLATION:
        axis = "inflation_pressure"
    elif release.category in (
        MacroCategory.GROWTH,
        MacroCategory.ACTIVITY,
        MacroCategory.CONSUMER,
        MacroCategory.HOUSING,
    ):
        axis = "growth_strength"
    elif release.category in (MacroCategory.EMPLOYMENT, MacroCategory.WAGES):
        axis = "labor_strength"
    elif release.category == MacroCategory.UNEMPLOYMENT:
        axis = "labor_strength"
        sign = -1
    elif release.category == MacroCategory.POLICY_RATE:
        axis = "policy_tightening"
    return MacroSurpriseContext(
        raw=raw,
        normalized_z=z,
        historical_scale=scale if scale is not None and isfinite(scale) else None,
        historical_sample_count=count,
        orientation_axis=axis,  # type: ignore[arg-type]
        oriented_z=sign * z if z is not None and axis != "none" else None,
    )
