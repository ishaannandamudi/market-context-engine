"""Absolute-movement concentration, separate from dispersion of returns."""

from math import isfinite


def movement_concentration(
    returns: tuple[float, ...], weights: tuple[float, ...] | None = None
) -> tuple[float, float] | None:
    if not returns or any(not isfinite(value) for value in returns):
        return None
    if weights is not None and (
        len(weights) != len(returns) or any(not isfinite(w) or w < 0 for w in weights)
    ):
        return None
    magnitudes = tuple(
        abs(value) * (weights[index] if weights is not None else 1.0)
        for index, value in enumerate(returns)
    )
    total = sum(magnitudes)
    if total <= 0 or not isfinite(total):
        return None
    hhi = sum((value / total) ** 2 for value in magnitudes)
    return hhi, 1 / hhi
