"""Correlation similarity and bounded Mantegna distance."""

from math import isfinite, sqrt


def correlation_distance(correlation: float) -> float:
    if not isfinite(correlation) or correlation < -1 - 1e-9 or correlation > 1 + 1e-9:
        raise ValueError("correlation must be finite and within [-1, 1]")
    return sqrt(2 * (1 - min(1.0, max(-1.0, correlation))))
