"""Common-versus-idiosyncratic return decomposition on a fitted PCA window."""

import numpy as np
from numpy.typing import NDArray

from mce.features.factors.models import FactorArtifact, FactorDecomposition, ReturnPanel

Matrix = NDArray[np.float64]


def reconstruct(artifact: FactorArtifact, scores: Matrix) -> Matrix:
    values = np.asarray(scores, dtype=np.float64)
    if (
        values.ndim != 2
        or values.shape[1] != artifact.selected_count
        or not np.isfinite(values).all()
    ):
        raise ValueError("factor scores and artifact dimensions differ")
    standardized = values @ np.asarray(artifact.loadings).T
    return np.asarray(
        standardized * np.asarray(artifact.scales) + np.asarray(artifact.means),
        dtype=np.float64,
    )


def decompose(panel: ReturnPanel, artifact: FactorArtifact, scores: Matrix) -> FactorDecomposition:
    if panel.symbols != artifact.symbols or len(panel.returns) < 2:
        raise ValueError("factor decomposition requires aligned fitted returns")
    original = np.asarray(panel.returns, dtype=np.float64)
    fitted = reconstruct(artifact, scores)
    if fitted.shape != original.shape:
        raise ValueError("factor reconstruction dimensions differ")
    residual = original - fitted
    total_by_symbol = np.var(original, axis=0, ddof=1)
    residual_by_symbol = np.var(residual, axis=0, ddof=1)
    common_by_symbol = np.var(fitted, axis=0, ddof=1)
    total = float(np.sum(total_by_symbol))
    explained = float(np.sum(common_by_symbol))
    unexplained = float(np.sum(residual_by_symbol))
    if abs(total - explained - unexplained) > 1e-7 * max(1.0, total):
        raise ValueError("factor variance decomposition identity failed")
    shares = tuple(
        float(min(1.0, max(0.0, residual_by_symbol[index] / variance)))
        if variance > 1e-24
        else None
        for index, variance in enumerate(total_by_symbol)
    )
    common = tuple(1 - value if value is not None else None for value in shares)
    valid_residual = [value for value in shares if value is not None]
    valid_common = [value for value in common if value is not None]
    return FactorDecomposition(
        shares,
        common,
        float(np.mean(valid_common)) if valid_common else None,
        float(np.median(valid_residual)) if valid_residual else None,
        float(np.mean(np.std(original, axis=1))),
        float(np.mean(np.std(residual, axis=1))),
        explained,
        unexplained,
        total,
    )
