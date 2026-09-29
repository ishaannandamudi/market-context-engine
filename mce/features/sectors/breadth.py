"""Sector participation using the MCE-5 directional and reference primitives."""

from math import isclose

from mce.features.breadth.math import direction_counts
from mce.features.breadth.models import ReferenceParticipation


def sector_direction(
    returns: tuple[float, ...], tolerance: float
) -> tuple[int, int, int, float | None]:
    advances, declines, unchanged, _, _, net, _ = direction_counts(returns, tolerance)
    return advances, declines, unchanged, net


def reference_participation(
    prices_and_prior: tuple[tuple[float, tuple[float | None, ...]], ...],
    sessions: int,
) -> ReferenceParticipation:
    comparable = 0
    above = 0
    for price, prior in prices_and_prior:
        sample = prior[:sessions]
        if len(sample) != sessions or any(item is None for item in sample):
            continue
        reference = sum(item / sessions for item in sample if item is not None)
        comparable += 1
        above += int(price > reference and not isclose(price, reference, rel_tol=1e-12))
    return ReferenceParticipation(
        sessions=sessions,
        above_count=above,
        comparable_count=comparable,
        fraction=above / comparable if comparable else None,
    )
