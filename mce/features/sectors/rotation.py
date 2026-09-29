"""Deterministic ranking and explicit causal sector-rank history."""

from math import ceil


def rank_scores(scores: dict[str, float | None]) -> dict[str, tuple[int, float]]:
    ordered = sorted(
        ((name, score) for name, score in scores.items() if score is not None),
        key=lambda item: (-item[1], item[0]),
    )
    count = len(ordered)
    return {
        name: (index + 1, (count - index - 1) / (count - 1) if count > 1 else 0.5)
        for index, (name, _) in enumerate(ordered)
    }


def leadership_persistence(
    sector: str,
    current: dict[str, tuple[int, float]],
    prior_newest_first: tuple[dict[str, tuple[int, float]], ...],
    window: int,
) -> float | None:
    observations = (current, *prior_newest_first[: window - 1])
    if len(observations) != window or any(
        sector not in ranks or len(ranks) < 2 for ranks in observations
    ):
        return None
    return sum(ranks[sector][0] <= max(1, ceil(len(ranks) / 4)) for ranks in observations) / window


def rotation_velocity(
    current: dict[str, tuple[int, float]],
    prior_newest_first: tuple[dict[str, tuple[int, float]], ...],
    transitions: int,
) -> float | None:
    observations = (current, *prior_newest_first[:transitions])
    if len(observations) != transitions + 1:
        return None
    values: list[float] = []
    for newer, older in zip(observations, observations[1:], strict=False):
        common = sorted(newer.keys() & older.keys())
        if len(common) < 2:
            return None
        scale = max(len(newer), len(older)) - 1
        values.append(
            sum(abs(newer[name][0] - older[name][0]) / scale for name in common) / len(common)
        )
    return sum(values) / len(values)


def leadership_hhi(scores: dict[str, float | None]) -> float | None:
    positive = tuple(value for value in scores.values() if value is not None and value > 0)
    total = sum(positive)
    return sum((value / total) ** 2 for value in positive) if total > 0 else None
