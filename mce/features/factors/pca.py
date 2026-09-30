"""Transparent covariance/correlation PCA with deterministic factor orientation."""

from hashlib import sha256
from math import isfinite, sqrt

import numpy as np
from numpy.typing import NDArray

from mce.config.settings import FactorSettings
from mce.features.factors.models import MODEL_VERSION, FactorArtifact, ReturnPanel
from mce.serialization.canonical import canonical_json_bytes

Matrix = NDArray[np.float64]


def canonical_signs(vectors: Matrix) -> Matrix:
    """Make the first largest-absolute loading positive in every factor."""
    result = np.asarray(vectors, dtype=np.float64).copy()
    if result.ndim != 2 or not np.isfinite(result).all():
        raise ValueError("factor vectors must be a finite matrix")
    for column in range(result.shape[1]):
        pivot = int(np.argmax(np.abs(result[:, column])))
        if result[pivot, column] < 0:
            result[:, column] *= -1
    return result


def pca_spectrum(standardized: Matrix) -> tuple[Matrix, Matrix, str]:
    """Return descending eigenvalues and eigenvectors; SVD avoids large N×N matrices."""
    x = np.asarray(standardized, dtype=np.float64)
    if x.ndim != 2 or min(x.shape) < 2 or not np.isfinite(x).all():
        raise ValueError("PCA needs a finite matrix with at least two rows and columns")
    rows, columns = x.shape
    if columns > rows:
        _, singular, right = np.linalg.svd(x / sqrt(rows - 1), full_matrices=False)
        values = singular * singular
        vectors = right.T
        method = "svd"
    else:
        covariance = (x.T @ x) / (rows - 1)
        covariance = (covariance + covariance.T) / 2
        values, vectors = np.linalg.eigh(covariance)
        order = np.argsort(-values, kind="stable")
        values, vectors = values[order], vectors[:, order]
        method = "eigh"
    scale = max(1.0, float(np.max(np.abs(values))))
    if float(np.min(values)) < -1e-9 * scale:
        raise ValueError("PCA matrix is not positive semidefinite")
    values = np.clip(values, 0, None)
    if len(values) < columns:
        values = np.pad(values, (0, columns - len(values)))
    return np.asarray(values), canonical_signs(np.asarray(vectors)), method


def fit_pca(
    panel: ReturnPanel, settings: FactorSettings, *, config_fingerprint: str
) -> tuple[FactorArtifact, Matrix]:
    if (
        len(panel.symbols) < settings.minimum_symbols
        or len(panel.returns) < settings.minimum_samples
        or panel.symbol_coverage < settings.minimum_symbol_coverage
        or panel.time_coverage < settings.minimum_time_coverage
    ):
        raise ValueError("factor panel below fit coverage or sample threshold")
    if len(config_fingerprint) != 64:
        raise ValueError("invalid factor configuration fingerprint")
    x = np.asarray(panel.returns, dtype=np.float64)
    mean = x.mean(axis=0)
    centered = x - mean
    variance = np.sum(centered * centered, axis=0) / (len(x) - 1)
    if np.any(variance <= 1e-24):
        raise ValueError("constant factor input symbol")
    scales = np.sqrt(variance) if settings.matrix_type == "correlation" else np.ones(len(mean))
    standardized = centered / scales
    eigenvalues, vectors, method = pca_spectrum(standardized)
    total = float(np.sum(eigenvalues))
    if not isfinite(total) or total <= 1e-24:
        raise ValueError("factor matrix has no positive variance")
    ratios = eigenvalues / total
    positive = int(np.sum(eigenvalues > max(eigenvalues[0] * 1e-10, 1e-14)))
    if positive == 0:
        raise ValueError("factor matrix has zero effective rank")
    cap = min(settings.factor_count, positive, vectors.shape[1])
    count = cap
    if settings.variance_threshold is not None:
        count = min(cap, int(np.searchsorted(np.cumsum(ratios), settings.variance_threshold)) + 1)
    chosen = vectors[:, :count]
    # A repeated eigenvalue makes individual directions unidentifiable even if
    # their span is stable. Include the selected/unselected boundary in the check.
    limit = min(len(eigenvalues) - 1, max(count, 1))
    weak = any(
        (eigenvalues[index] - eigenvalues[index + 1]) / max(eigenvalues[0], 1e-24)
        <= settings.eigen_gap_tolerance
        for index in range(limit)
        if eigenvalues[index] > 1e-14
    )
    artifact = FactorArtifact(
        MODEL_VERSION,
        panel.symbols,
        settings.matrix_type,
        tuple(float(v) for v in mean),
        tuple(float(v) for v in scales),
        tuple(float(v) for v in eigenvalues),
        tuple(float(v) for v in ratios),
        tuple(tuple(float(v) for v in row) for row in chosen),
        count,
        panel.timestamps_ns[0],
        panel.timestamps_ns[-1],
        panel.fingerprint,
        config_fingerprint,
        sha256(canonical_json_bytes(settings)).hexdigest(),
        weak,
        method,
    )
    scores = standardized @ chosen
    if not np.isfinite(scores).all():
        raise ValueError("nonfinite factor scores")
    return artifact, np.asarray(scores)


def transform(artifact: FactorArtifact, panel: ReturnPanel) -> Matrix:
    """Project on frozen training parameters; caller decides if rows are out-of-sample."""
    if panel.symbols != artifact.symbols:
        raise ValueError("frozen factor transform requires the same ordered symbols")
    x = np.asarray(panel.returns, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != len(artifact.symbols):
        raise ValueError("factor transform dimensions differ")
    return np.asarray(
        ((x - np.asarray(artifact.means)) / np.asarray(artifact.scales))
        @ np.asarray(artifact.loadings)
    )
