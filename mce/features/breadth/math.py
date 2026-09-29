"""Small, transparent cross-sectional breadth calculations."""

from math import isfinite
from statistics import median

from mce.config.settings import BreadthScoreWeights


def direction_counts(
    returns: tuple[float, ...], tolerance: float
) -> tuple[int, int, int, float | None, float | None, float | None, float | None]:
    if tolerance < 0 or not isfinite(tolerance) or any(not isfinite(r) for r in returns):
        raise ValueError("direction inputs must be finite and tolerance nonnegative")
    advance = sum(value > tolerance for value in returns)
    decline = sum(value < -tolerance for value in returns)
    unchanged = len(returns) - advance - decline
    n = len(returns)
    return (
        advance,
        decline,
        unchanged,
        advance / n if n else None,
        decline / n if n else None,
        (advance - decline) / n if n else None,
        advance / decline if decline else None,
    )


def weighted_return(
    pairs: tuple[tuple[float, float], ...], total_universe_weight: float
) -> tuple[float | None, float | None]:
    """Return valid-data weighted mean and original-universe weight coverage."""
    if total_universe_weight <= 0 or not isfinite(total_universe_weight):
        return None, None
    valid_weight = sum(weight / total_universe_weight for _, weight in pairs)
    if valid_weight <= 0 or not isfinite(valid_weight):
        return None, 0.0
    result = sum(value * (weight / total_universe_weight) for value, weight in pairs)
    result /= valid_weight
    return (result if isfinite(result) else None), min(1.0, valid_weight)


def contribution_concentration(
    pairs: tuple[tuple[float, float], ...],
) -> tuple[float, float] | None:
    """HHI of absolute weighted returns, plus reciprocal effective count."""
    if not pairs or any(
        weight < 0 or not isfinite(weight) or not isfinite(r) for r, weight in pairs
    ):
        return None
    maximum = max(abs(r * weight) for r, weight in pairs)
    if maximum == 0 or not isfinite(maximum):
        return None
    scaled = tuple(abs(r * weight) / maximum for r, weight in pairs)
    total = sum(scaled)
    if total <= 0 or not isfinite(total):
        return None
    hhi = sum((value / total) ** 2 for value in scaled)
    return hhi, 1 / hhi


def breadth_score_components(
    net_advance: float | None,
    references: tuple[float | None, float | None, float | None],
    weights: BreadthScoreWeights,
) -> tuple[float | None, tuple[tuple[str, float, float], ...]]:
    """Renormalize only available predefined components; retain exact terms."""
    raw = (
        ("net_advance", net_advance, weights.net_advance),
        ("short_reference", references[0], weights.short_reference),
        ("medium_reference", references[1], weights.medium_reference),
        ("long_reference", references[2], weights.long_reference),
    )
    available = tuple((name, value, weight) for name, value, weight in raw if value is not None)
    weight_sum = sum(weight for _, _, weight in available)
    if not available or weight_sum <= 0:
        return None, ()
    components = tuple(
        (name, value if name == "net_advance" else 2 * value - 1, weight / weight_sum)
        for name, value, weight in available
    )
    score = sum(normalized * weight for _, normalized, weight in components)
    return max(-1.0, min(1.0, score)), components


def median_return(returns: tuple[float, ...]) -> float | None:
    return median(returns) if returns else None
