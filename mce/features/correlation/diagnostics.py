"""Transparent matrix summaries, shrinkage, eigen diagnostics and grouping."""

from math import isfinite, sqrt
from statistics import median, pstdev

import numpy as np
from numpy.typing import NDArray


def pair_values(matrix: NDArray[np.float64]) -> NDArray[np.float64]:
    return matrix[np.triu_indices(matrix.shape[0], k=1)]


def weighted_average(matrix: NDArray[np.float64], weights: tuple[float, ...]) -> float | None:
    if len(weights) != matrix.shape[0] or any(
        value < 0 or not isfinite(value) for value in weights
    ):
        return None
    pair_weights = np.outer(np.asarray(weights), np.asarray(weights))
    upper = np.triu_indices(len(weights), k=1)
    denominator = float(pair_weights[upper].sum())
    return (
        float(np.dot(pair_weights[upper], matrix[upper]) / denominator) if denominator > 0 else None
    )


def shrink_identity(matrix: NDArray[np.float64], strength: float) -> NDArray[np.float64]:
    if not 0 <= strength <= 1:
        raise ValueError("shrinkage strength must be in [0,1]")
    return np.asarray((1 - strength) * matrix + strength * np.eye(matrix.shape[0]))


def eigen_diagnostics(matrix: NDArray[np.float64]) -> tuple[float, float, float] | None:
    if matrix.ndim != 2 or matrix.shape[0] < 2 or matrix.shape[0] != matrix.shape[1]:
        return None
    values = np.linalg.eigvalsh(matrix)
    if not np.all(np.isfinite(values)) or float(values.min()) < -1e-10:
        return None
    values = np.clip(values, 0, None)
    total = float(values.sum())
    return (float(values[-1]), float(values[-1] / total), float(total**2 / np.dot(values, values)))


def correlation_distance(value: float) -> float:
    if not isfinite(value):
        raise ValueError("correlation distance requires a finite correlation")
    return sqrt(2 * (1 - max(-1.0, min(1.0, value))))


def complete_link_groups(
    matrix: NDArray[np.float64], symbols: tuple[str, ...], threshold: float
) -> tuple[tuple[str, ...], ...]:
    """Lexical greedy complete-link threshold groups; stable under input ordering."""
    if len(symbols) != matrix.shape[0] or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix and symbols differ")
    indices = {symbol: index for index, symbol in enumerate(symbols)}
    groups: list[list[str]] = []
    for symbol in sorted(symbols):
        for group in groups:
            if all(matrix[indices[symbol], indices[member]] >= threshold for member in group):
                group.append(symbol)
                break
        else:
            groups.append([symbol])
    return tuple(tuple(group) for group in groups)


def summary(matrix: NDArray[np.float64]) -> tuple[float, float, float]:
    pairs = tuple(float(value) for value in pair_values(matrix))
    return sum(pairs) / len(pairs), median(pairs), pstdev(pairs)
