"""Numerically stable diagonal-Gaussian HMM recursions."""

from math import pi

import numpy as np
from numpy.typing import NDArray

Matrix = NDArray[np.float64]


def logsumexp(values: Matrix, axis: int | None = None) -> NDArray[np.float64]:
    """Stable log(sum(exp(values))); all-impossible slices remain -inf."""
    array = np.asarray(values, dtype=np.float64)
    maximum = np.max(array, axis=axis, keepdims=True)
    safe = np.where(np.isfinite(maximum), maximum, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        result = safe + np.log(np.sum(np.exp(array - safe), axis=axis, keepdims=True))
    return np.asarray(np.squeeze(result, axis=axis) if axis is not None else result.item())


def diagonal_gaussian_log_density(
    observations: Matrix, means: Matrix, variances: Matrix, covariance_floor: float
) -> Matrix:
    x = np.asarray(observations, dtype=np.float64)
    mu = np.asarray(means, dtype=np.float64)
    var = np.asarray(variances, dtype=np.float64)
    if (
        x.ndim != 2
        or mu.ndim != 2
        or var.shape != mu.shape
        or x.shape[1] != mu.shape[1]
        or x.shape[0] == 0
        or mu.shape[0] == 0
        or covariance_floor <= 0
        or not np.isfinite(x).all()
        or not np.isfinite(mu).all()
        or not np.isfinite(var).all()
        or np.any(var <= 0)
    ):
        raise ValueError("invalid diagonal Gaussian dimensions or parameters")
    safe_var = np.maximum(var, covariance_floor)
    delta = x[:, None, :] - mu[None, :, :]
    quadratic = np.sum(delta * delta / safe_var[None, :, :], axis=2)
    log_det = np.sum(np.log(safe_var), axis=1)
    result = -0.5 * (x.shape[1] * np.log(2 * pi) + log_det[None, :] + quadratic)
    if not np.isfinite(result).all():
        raise ValueError("nonfinite Gaussian log density")
    return np.asarray(result, dtype=np.float64)


def _log_probabilities(initial: Matrix, transition: Matrix) -> tuple[Matrix, Matrix]:
    p = np.asarray(initial, dtype=np.float64)
    a = np.asarray(transition, dtype=np.float64)
    if (
        p.ndim != 1
        or a.shape != (len(p), len(p))
        or len(p) < 2
        or not np.isfinite(p).all()
        or not np.isfinite(a).all()
        or np.any(p < 0)
        or np.any(a < 0)
        or not np.isclose(p.sum(), 1.0, atol=1e-8)
        or not np.allclose(a.sum(axis=1), 1.0, atol=1e-8)
    ):
        raise ValueError("invalid HMM probabilities")
    with np.errstate(divide="ignore"):
        return np.asarray(np.log(p)), np.asarray(np.log(a))


def forward_log(
    log_emissions: Matrix,
    initial: Matrix,
    transition: Matrix,
    *,
    previous_filtered: Matrix | None = None,
) -> tuple[Matrix, float]:
    emissions = np.asarray(log_emissions, dtype=np.float64)
    log_initial, log_transition = _log_probabilities(initial, transition)
    if emissions.ndim != 2 or emissions.shape[1] != len(log_initial) or len(emissions) == 0:
        raise ValueError("invalid emission matrix")
    if np.isnan(emissions).any() or np.isposinf(emissions).any():
        raise ValueError("invalid emission log density")
    if previous_filtered is not None:
        prior = np.asarray(previous_filtered, dtype=np.float64)
        _log_probabilities(prior, transition)
        with np.errstate(divide="ignore"):
            log_initial = np.log(prior @ transition)
    alpha = np.empty_like(emissions)
    alpha[0] = log_initial + emissions[0]
    for t in range(1, len(emissions)):
        alpha[t] = emissions[t] + logsumexp(alpha[t - 1, :, None] + log_transition, axis=0)
    likelihood = float(logsumexp(alpha[-1]))
    if not np.isfinite(likelihood):
        raise ValueError("sequence has no finite likelihood")
    return alpha, likelihood


def backward_log(log_emissions: Matrix, transition: Matrix) -> Matrix:
    emissions = np.asarray(log_emissions, dtype=np.float64)
    k = emissions.shape[1] if emissions.ndim == 2 else 0
    _, log_transition = _log_probabilities(np.full(k, 1.0 / k), transition)
    if len(emissions) == 0 or np.isnan(emissions).any() or np.isposinf(emissions).any():
        raise ValueError("invalid emission matrix")
    beta = np.zeros_like(emissions)
    for t in range(len(emissions) - 2, -1, -1):
        beta[t] = logsumexp(
            log_transition + emissions[t + 1][None, :] + beta[t + 1][None, :], axis=1
        )
    return beta


def posteriors(
    log_emissions: Matrix,
    log_alpha: Matrix,
    log_beta: Matrix,
    transition: Matrix,
    log_likelihood: float,
) -> tuple[Matrix, NDArray[np.float64]]:
    if log_alpha.shape != log_beta.shape or log_alpha.shape != log_emissions.shape:
        raise ValueError("forward/backward shapes differ")
    k = log_alpha.shape[1]
    _, log_transition = _log_probabilities(np.full(k, 1.0 / k), transition)
    log_gamma = log_alpha + log_beta - log_likelihood
    gamma = np.exp(log_gamma - logsumexp(log_gamma, axis=1)[:, None])
    log_xi = (
        log_alpha[:-1, :, None]
        + log_transition[None, :, :]
        + log_emissions[1:, None, :]
        + log_beta[1:, None, :]
        - log_likelihood
    )
    xi = np.exp(log_xi)
    if not np.allclose(gamma.sum(axis=1), 1.0, atol=1e-7) or (
        len(xi) and not np.allclose(xi.sum(axis=(1, 2)), 1.0, atol=1e-7)
    ):
        raise ValueError("posterior normalization failed")
    return np.asarray(gamma), np.asarray(xi)


def viterbi_path(log_emissions: Matrix, initial: Matrix, transition: Matrix) -> tuple[int, ...]:
    emissions = np.asarray(log_emissions, dtype=np.float64)
    log_initial, log_transition = _log_probabilities(initial, transition)
    if emissions.ndim != 2 or emissions.shape[1] != len(log_initial) or len(emissions) == 0:
        raise ValueError("invalid emission matrix")
    if np.isnan(emissions).any() or np.isposinf(emissions).any():
        raise ValueError("invalid emission log density")
    score = np.empty_like(emissions)
    back = np.zeros(emissions.shape, dtype=np.int64)
    score[0] = log_initial + emissions[0]
    for t in range(1, len(emissions)):
        candidates = score[t - 1, :, None] + log_transition
        back[t] = np.argmax(candidates, axis=0)
        score[t] = emissions[t] + np.max(candidates, axis=0)
    if not np.isfinite(score[-1]).any():
        raise ValueError("sequence has no finite Viterbi path")
    path = np.zeros(len(emissions), dtype=np.int64)
    path[-1] = int(np.argmax(score[-1]))
    for t in range(len(emissions) - 2, -1, -1):
        path[t] = back[t + 1, path[t + 1]]
    return tuple(int(value) for value in path)
