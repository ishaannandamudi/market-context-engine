"""Deterministic cross-window factor matching and causal score diagnostics."""

from functools import lru_cache

import numpy as np
from numpy.typing import NDArray

from mce.features.factors.models import FactorAlignment, FactorArtifact

Matrix = NDArray[np.float64]


def _assignment(similarity: Matrix) -> tuple[int, ...]:
    """Exact maximum-weight matching for small K, lexicographic on equal scores."""
    if similarity.ndim != 2 or similarity.shape[0] != similarity.shape[1]:
        raise ValueError("factor matching needs a square similarity matrix")
    size = similarity.shape[0]

    @lru_cache(None)
    def best(index: int, used: int) -> tuple[float, tuple[int, ...]]:
        if index == size:
            return 0.0, ()
        choices = []
        for prior in range(size):
            if used & (1 << prior):
                continue
            value, suffix = best(index + 1, used | (1 << prior))
            choices.append((float(similarity[index, prior]) + value, (prior,) + suffix))
        return min(choices, key=lambda item: (-round(item[0], 12), item[1]))

    return best(0, 0)[1]


def align_factors(
    current: FactorArtifact, previous: FactorArtifact
) -> tuple[FactorAlignment | None, tuple[str, ...], tuple[tuple[float, ...], ...]]:
    shared = tuple(sorted(set(current.symbols) & set(previous.symbols)))
    count = min(current.selected_count, previous.selected_count)
    if len(shared) < max(2, count):
        return None, (), ()
    current_indices = [current.symbols.index(symbol) for symbol in shared]
    previous_indices = [previous.symbols.index(symbol) for symbol in shared]
    first = np.asarray(current.loadings)[current_indices, :count]
    second = np.asarray(previous.loadings)[previous_indices, :count]
    first_norms = np.linalg.norm(first, axis=0)
    second_norms = np.linalg.norm(second, axis=0)
    if np.any(first_norms < 1e-12) or np.any(second_norms < 1e-12):
        return None, (), ()
    cosine = (first / first_norms).T @ (second / second_norms)
    mapping = _assignment(np.abs(cosine))
    signs = tuple(1 if cosine[index, prior] >= 0 else -1 for index, prior in enumerate(mapping))
    similarities = tuple(float(abs(cosine[index, prior])) for index, prior in enumerate(mapping))
    q_first, _ = np.linalg.qr(first)
    q_second, _ = np.linalg.qr(second)
    singular = np.linalg.svd(q_first.T @ q_second, compute_uv=False)
    subspace = float(np.mean(np.clip(singular, 0, 1) ** 2))
    drift = tuple(
        tuple(
            float(signs[index] * first[row, index] - second[row, prior])
            for row in range(len(shared))
        )
        for index, prior in enumerate(mapping)
    )
    alignment = FactorAlignment(
        mapping, signs, similarities, float(np.mean(similarities)), subspace
    )
    return alignment, shared, drift


def factor_shock_and_volatility(
    scores: Matrix, minimum_prior: int
) -> tuple[float | None, float | None]:
    series = np.asarray(scores, dtype=np.float64)
    if series.ndim != 2 or series.shape[1] < 1 or not np.isfinite(series).all():
        raise ValueError("factor score history must be a finite matrix")
    volatility = float(np.std(series[:, 0], ddof=1)) if len(series) >= 2 else None
    if len(series) - 1 < minimum_prior:
        return None, volatility
    prior = series[:-1, 0]
    spread = float(np.std(prior, ddof=1))
    shock = float((series[-1, 0] - np.mean(prior)) / spread) if spread > 1e-12 else None
    return shock, volatility


def spectrum_diagnostics(eigenvalues: tuple[float, ...]) -> tuple[float, float, float, float]:
    values = np.asarray(eigenvalues, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2 or np.any(values < 0) or not np.isfinite(values).all():
        raise ValueError("invalid factor eigenvalue spectrum")
    total = float(values.sum())
    if total <= 0:
        raise ValueError("zero factor eigenvalue spectrum")
    shares = values / total
    hhi = float(np.dot(shares, shares))
    effective = 1 / hhi
    first = float(shares[0])
    top3 = float(np.sum(shares[:3]))
    dominance = 0.6 * first + 0.4 * hhi
    return first, top3, effective, dominance
