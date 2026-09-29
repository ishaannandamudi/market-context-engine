"""Exact equal-weight law of total variance over point-in-time sectors."""

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from math import isclose

from mce.features.dispersion.cross_section import cross_section


@dataclass(frozen=True)
class SectorDecomposition:
    member_count: int
    coverage: float
    total_variance: float
    between_variance: float
    within_variance: float
    between_share: float | None
    within_share: float | None
    sector_means: tuple[tuple[str, float], ...]
    sector_std: tuple[tuple[str, float | None], ...]
    sector_counts: tuple[tuple[str, int], ...]


def decompose(
    returns: Mapping[str, float],
    assignments: Mapping[str, str],
    *,
    minimum_sectors: int,
    minimum_coverage: float,
) -> SectorDecomposition | None:
    groups: dict[str, list[float]] = defaultdict(list)
    for symbol, value in returns.items():
        if symbol in assignments:
            groups[assignments[symbol]].append(value)
    count = sum(len(group) for group in groups.values())
    coverage = count / len(returns) if returns else 0.0
    if count < 2 or len(groups) < minimum_sectors or coverage < minimum_coverage:
        return None
    total_values = tuple(value for group in groups.values() for value in group)
    total = cross_section(total_values)
    assert total is not None
    sector_means: list[tuple[str, float]] = []
    sector_std: list[tuple[str, float | None]] = []
    sector_counts: list[tuple[str, int]] = []
    within = between = 0.0
    for sector, values in sorted(groups.items()):
        weight = len(values) / count
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        between += weight * (mean - total.mean) ** 2
        within += weight * variance
        sector_means.append((sector, mean))
        sector_std.append((sector, variance**0.5 if len(values) >= 2 else None))
        sector_counts.append((sector, len(values)))
    total_variance = total.std**2
    if not isclose(total_variance, between + within, rel_tol=1e-9, abs_tol=1e-18):
        raise ArithmeticError("sector variance decomposition failed")
    return SectorDecomposition(
        member_count=count,
        coverage=coverage,
        total_variance=total_variance,
        between_variance=between,
        within_variance=within,
        between_share=between / total_variance if total_variance > 1e-24 else None,
        within_share=within / total_variance if total_variance > 1e-24 else None,
        sector_means=tuple(sector_means),
        sector_std=tuple(sector_std),
        sector_counts=tuple(sector_counts),
    )
