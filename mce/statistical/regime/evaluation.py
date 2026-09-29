"""Descriptive unsupervised diagnostics; no predictive or trading scores."""

from collections import Counter, defaultdict
from dataclasses import dataclass
from math import log

import numpy as np

from mce.statistical.dataset import StatisticalDataset
from mce.statistical.models import LatentStateProfile


@dataclass(frozen=True, slots=True)
class RegimeEvaluation:
    profiles: tuple[LatentStateProfile, ...]
    transition_counts: tuple[tuple[int, int, int], ...]
    dwell_lengths: tuple[tuple[int, tuple[int, ...]], ...]
    session_weighted_occupancy: tuple[tuple[int, float], ...]
    exit_entropy_nats: tuple[tuple[int, float | None], ...]
    weighted_exit_entropy_nats: float | None
    persistence_fraction: float | None
    deterministic_contingency: tuple[tuple[int, str, int], ...]
    deterministic_purity: float | None


def evaluate_states(dataset: StatisticalDataset, states: tuple[int, ...]) -> RegimeEvaluation:
    if not states or len(states) != len(dataset.rows) or any(state < 0 for state in states):
        raise ValueError("states must align one-to-one with nonempty dataset rows")
    occupancy = Counter(states)
    transitions = Counter(zip(states, states[1:], strict=False))
    dwell: dict[int, list[int]] = defaultdict(list)
    current, length = states[0], 1
    for state in states[1:]:
        if state == current:
            length += 1
        else:
            dwell[current].append(length)
            current, length = state, 1
    dwell[current].append(length)
    by_day = Counter(row.trading_date for row in dataset.rows)
    weighted: dict[int, float] = defaultdict(float)
    for state, row in zip(states, dataset.rows, strict=True):
        weighted[state] += 1.0 / by_day[row.trading_date]
    total_days = len(by_day)
    exits: dict[int, Counter[int]] = defaultdict(Counter)
    for (source, destination), count in transitions.items():
        if source != destination:
            exits[source][destination] = count
    entropies: dict[int, float | None] = {}
    exit_totals: dict[int, int] = {}
    for state in sorted(occupancy):
        total = sum(exits[state].values())
        exit_totals[state] = total
        entropies[state] = (
            -sum((count / total) * log(count / total) for count in exits[state].values())
            if total
            else None
        )
    total_exits = sum(exit_totals.values())
    aggregate_entropy = (
        sum(exit_totals[state] * (entropies[state] or 0.0) for state in occupancy) / total_exits
        if total_exits
        else None
    )
    contingency = Counter(
        (state, row.deterministic_regime)
        for state, row in zip(states, dataset.rows, strict=True)
        if row.deterministic_regime is not None
    )
    labeled = sum(contingency.values())
    purity = (
        sum(
            max((count for (s, _), count in contingency.items() if s == state), default=0)
            for state in occupancy
        )
        / labeled
        if labeled
        else None
    )
    matrix = np.array([row.feature_vector for row in dataset.rows], dtype=np.float64)
    state_array = np.asarray(states, dtype=np.int64)
    profiles: list[LatentStateProfile] = []
    for state in sorted(occupancy):
        subset = matrix[state_array == state]
        centroid: list[float | None] = []
        median: list[float | None] = []
        dispersion: list[float | None] = []
        for col in range(subset.shape[1]):
            observed = subset[np.isfinite(subset[:, col]), col]
            centroid.append(float(np.mean(observed)) if observed.size else None)
            median.append(float(np.median(observed)) if observed.size else None)
            dispersion.append(float(np.std(observed)) if observed.size else None)
        profiles.append(
            LatentStateProfile(
                state_id=state,
                observations=occupancy[state],
                occupancy_fraction=occupancy[state] / len(states),
                centroid=tuple(centroid),
                median=tuple(median),
                dispersion=tuple(dispersion),
                average_dwell_observations=sum(dwell[state]) / len(dwell[state]),
                deterministic_overlap=tuple(
                    sorted(
                        (regime, count) for (s, regime), count in contingency.items() if s == state
                    )
                ),
            )
        )
    return RegimeEvaluation(
        profiles=tuple(profiles),
        transition_counts=tuple(sorted((a, b, count) for (a, b), count in transitions.items())),
        dwell_lengths=tuple((state, tuple(dwell[state])) for state in sorted(occupancy)),
        session_weighted_occupancy=tuple(
            (state, weighted[state] / total_days) for state in sorted(occupancy)
        ),
        exit_entropy_nats=tuple((state, entropies[state]) for state in sorted(occupancy)),
        weighted_exit_entropy_nats=aggregate_entropy,
        persistence_fraction=sum(count for (a, b), count in transitions.items() if a == b)
        / (len(states) - 1)
        if len(states) > 1
        else None,
        deterministic_contingency=tuple(
            sorted((state, regime, count) for (state, regime), count in contingency.items())
        ),
        deterministic_purity=purity,
    )
