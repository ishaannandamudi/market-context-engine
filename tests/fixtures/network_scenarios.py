"""Deterministic PSD dependence matrices for graph reference tests."""

import numpy as np

from mce.features.correlation.models import CorrelationMatrix
from mce.features.network.models import DependenceInput


def dependence(
    values: np.ndarray,
    *,
    as_of_ns: int = 100,
    symbols: tuple[str, ...] | None = None,
    expected: tuple[str, ...] | None = None,
) -> DependenceInput:
    names = symbols or tuple(chr(65 + index) * 3 for index in range(len(values)))
    matrix = CorrelationMatrix(
        symbols=names,
        values=tuple(tuple(float(value) for value in row) for row in values),
    )
    return DependenceInput(as_of_ns, expected or names, matrix, "raw", aligned_samples=40)


def equicorrelation(size: int = 4, rho: float = 0.8) -> np.ndarray:
    matrix = np.full((size, size), rho)
    np.fill_diagonal(matrix, 1)
    return matrix


def blocks(*, within: float = 0.85, cross: float = 0.1) -> np.ndarray:
    matrix = equicorrelation(4, cross)
    matrix[0, 1] = matrix[1, 0] = within
    matrix[2, 3] = matrix[3, 2] = within
    return matrix


def star(size: int = 4, center: int = 0, rho: float = 0.7) -> np.ndarray:
    matrix = equicorrelation(size, rho * rho)
    matrix[center, :] = rho
    matrix[:, center] = rho
    matrix[center, center] = 1
    return matrix
