"""Covariance, Pearson and rank dependence on a single common sample."""

from dataclasses import dataclass
from math import isfinite

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class MatrixEstimate:
    symbols: tuple[str, ...]
    sample_count: int
    expected_samples: int
    matrix: tuple[tuple[float, ...], ...]
    spearman_matrix: tuple[tuple[float, ...], ...] | None


def sample_covariance(samples: NDArray[np.float64]) -> NDArray[np.float64] | None:
    if samples.ndim != 2 or samples.shape[0] < 2 or samples.shape[1] < 2:
        return None
    centered = samples - samples.mean(axis=0)
    return np.asarray(centered.T @ centered / (samples.shape[0] - 1), dtype=np.float64)


def covariance_to_correlation(covariance: NDArray[np.float64]) -> NDArray[np.float64] | None:
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        return None
    diagonal = np.diag(covariance)
    if not np.all(np.isfinite(covariance)) or np.any(diagonal <= 1e-24):
        return None
    standard = np.sqrt(diagonal)
    matrix = covariance / np.outer(standard, standard)
    matrix = (matrix + matrix.T) / 2
    if np.any(np.abs(matrix) > 1 + 1e-10):
        return None
    np.fill_diagonal(matrix, 1.0)
    matrix = np.clip(matrix, -1.0, 1.0)
    return matrix if np.linalg.eigvalsh(matrix).min() >= -1e-10 else None


def average_ranks(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Midranks for ties, independently in each column."""
    ranks = np.empty_like(values)
    for column in range(values.shape[1]):
        order = np.argsort(values[:, column], kind="stable")
        sorted_values = values[order, column]
        position = 0
        while position < len(order):
            finish = position + 1
            while finish < len(order) and sorted_values[finish] == sorted_values[position]:
                finish += 1
            ranks[order[position:finish], column] = (position + finish - 1) / 2 + 1
            position = finish
    return ranks


def estimate(
    returns: dict[str, dict[int, float]],
    expected_samples: int,
    minimum_samples: int,
    robust: bool = True,
) -> MatrixEstimate | None:
    if len(returns) < 2:
        return None
    symbols = tuple(sorted(returns))
    aligned = set.intersection(*(set(returns[symbol]) for symbol in symbols))
    if len(aligned) < minimum_samples:
        return None
    times = sorted(aligned)
    samples = np.asarray([[returns[symbol][time] for symbol in symbols] for time in times])
    covariance = sample_covariance(samples)
    matrix = covariance_to_correlation(covariance) if covariance is not None else None
    if matrix is None:
        return None
    rank_matrix: NDArray[np.float64] | None = None
    if robust:
        rank_covariance = sample_covariance(average_ranks(samples))
        rank_matrix = (
            covariance_to_correlation(rank_covariance) if rank_covariance is not None else None
        )
    native = tuple(tuple(float(value) for value in row) for row in matrix)
    ranked = (
        tuple(tuple(float(value) for value in row) for row in rank_matrix)
        if rank_matrix is not None
        else None
    )
    if not all(isfinite(value) for row in native for value in row):
        return None
    return MatrixEstimate(symbols, len(times), expected_samples, native, ranked)
