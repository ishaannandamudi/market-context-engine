"""Deterministic sparse threshold and union top-k neighbor selection."""

import numpy as np
from numpy.typing import NDArray


def threshold_pairs(matrix: NDArray[np.float64], threshold: float) -> tuple[tuple[int, int], ...]:
    return tuple(
        (i, j)
        for i in range(len(matrix))
        for j in range(i + 1, len(matrix))
        if matrix[i, j] >= threshold
    )


def top_k_pairs(matrix: NDArray[np.float64], count: int) -> tuple[tuple[int, int], ...]:
    if count < 1:
        raise ValueError("top-k neighbor count must be positive")
    pairs: set[tuple[int, int]] = set()
    for i in range(len(matrix)):
        neighbors = sorted(
            (j for j in range(len(matrix)) if j != i),
            key=lambda j: (-float(matrix[i, j]), j),
        )[:count]
        pairs.update((min(i, j), max(i, j)) for j in neighbors)
    return tuple(sorted(pairs))
