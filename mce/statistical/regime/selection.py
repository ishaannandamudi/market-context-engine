"""Descriptive model-size and deterministic-regime comparison reports."""

from collections import Counter
from dataclasses import dataclass
from math import comb, isfinite, log, sqrt

import numpy as np

from mce.config.settings import HMMSettings, KMeansSettings
from mce.statistical.dataset import StatisticalDataset
from mce.statistical.normalization import FittedNormalizer
from mce.statistical.regime.clustering import KMeansRegimeModel
from mce.statistical.regime.evaluation import RegimeEvaluation, evaluate_states
from mce.statistical.regime.hmm import GaussianHMMRegimeModel


def adjusted_rand_index(first: tuple[object, ...], second: tuple[object, ...]) -> float:
    if not first or len(first) != len(second):
        raise ValueError("label sequences must be nonempty and aligned")
    n = len(first)
    if n == 1:
        return 1.0
    pairs = Counter(zip(first, second, strict=True))
    a, b = Counter(first), Counter(second)
    index = sum(comb(count, 2) for count in pairs.values())
    first_pairs = sum(comb(count, 2) for count in a.values())
    second_pairs = sum(comb(count, 2) for count in b.values())
    expected = first_pairs * second_pairs / comb(n, 2)
    maximum = (first_pairs + second_pairs) / 2
    denominator = maximum - expected
    if abs(denominator) < 1e-12:
        return 1.0 if index == maximum else 0.0
    return float((index - expected) / denominator)


def normalized_mutual_information(first: tuple[object, ...], second: tuple[object, ...]) -> float:
    """NMI = mutual information / geometric mean of label entropies."""
    if not first or len(first) != len(second):
        raise ValueError("label sequences must be nonempty and aligned")
    n = len(first)
    a, b = Counter(first), Counter(second)
    joint = Counter(zip(first, second, strict=True))
    entropy_a = -sum((count / n) * log(count / n) for count in a.values())
    entropy_b = -sum((count / n) * log(count / n) for count in b.values())
    if entropy_a == 0 or entropy_b == 0:
        return 1.0 if entropy_a == entropy_b == 0 else 0.0
    information = sum(
        (count / n) * log(count * n / (a[left] * b[right]))
        for (left, right), count in joint.items()
    )
    return float(max(0.0, min(1.0, information / sqrt(entropy_a * entropy_b))))


@dataclass(frozen=True, slots=True)
class RegimeComparison:
    model_type: str
    evaluation: RegimeEvaluation
    adjusted_rand: float | None
    normalized_mutual_information: float | None


def compare_assignments(
    dataset: StatisticalDataset, states: tuple[int, ...], *, model_type: str
) -> RegimeComparison:
    evaluation = evaluate_states(dataset, states)
    pairs = tuple(
        (state, row.deterministic_regime)
        for state, row in zip(states, dataset.rows, strict=True)
        if row.deterministic_regime is not None
    )
    ari: float | None
    nmi: float | None
    if pairs:
        latent = tuple(state for state, _ in pairs)
        deterministic = tuple(label for _, label in pairs)
        ari = adjusted_rand_index(latent, deterministic)
        nmi = normalized_mutual_information(latent, deterministic)
    else:
        ari = nmi = None
    return RegimeComparison(model_type, evaluation, ari, nmi)


@dataclass(frozen=True, slots=True)
class HMMSelectionEntry:
    state_count: int
    log_likelihood: float
    aic: float
    bic: float
    converged: bool
    iterations: int
    occupancy_balance: float
    mean_dwell_observations: float
    transition_entropy_nats: float | None
    minimum_centroid_separation: float
    degenerate_states: tuple[int, ...]
    model_fingerprint: str


@dataclass(frozen=True, slots=True)
class KMeansSelectionEntry:
    cluster_count: int
    inertia: float
    converged: bool
    iterations: int
    occupancy_balance: float
    empty_clusters: int
    model_fingerprint: str


def compare_kmeans_cluster_counts(
    training: StatisticalDataset,
    normalizer: FittedNormalizer,
    counts: tuple[int, ...],
    *,
    settings: KMeansSettings | None = None,
    random_seed: int = 0,
) -> tuple[KMeansSelectionEntry, ...]:
    if (
        not counts
        or tuple(sorted(set(counts))) != counts
        or any(not 2 <= count <= 6 for count in counts)
    ):
        raise ValueError("k-means cluster counts must be unique, sorted, and between 2 and 6")
    base = settings or KMeansSettings()
    entries: list[KMeansSelectionEntry] = []
    for count in counts:
        model = KMeansRegimeModel(base.model_copy(update={"cluster_count": count}))
        artifact = model.fit(training, normalizer, random_seed=random_seed)
        labels = model.infer(training)
        occupancy = [labels.count(state) for state in range(count)]
        diagnostic = artifact.restarts[artifact.selected_restart]
        entries.append(
            KMeansSelectionEntry(
                count,
                diagnostic.inertia,
                diagnostic.converged,
                diagnostic.iterations,
                min(occupancy) / max(occupancy) if max(occupancy) else 0.0,
                diagnostic.empty_clusters,
                artifact.fingerprint,
            )
        )
    return tuple(entries)


def compare_hmm_state_counts(
    training: StatisticalDataset,
    normalizer: FittedNormalizer,
    counts: tuple[int, ...],
    *,
    settings: HMMSettings | None = None,
    random_seed: int = 0,
) -> tuple[HMMSelectionEntry, ...]:
    if (
        not counts
        or tuple(sorted(set(counts))) != counts
        or any(not 2 <= count <= 6 for count in counts)
    ):
        raise ValueError("HMM state counts must be unique, sorted, and between 2 and 6")
    base = settings or HMMSettings()
    entries: list[HMMSelectionEntry] = []
    for count in counts:
        model = GaussianHMMRegimeModel(base.model_copy(update={"state_count": count}))
        artifact = model.fit(training, normalizer, random_seed=random_seed)
        quality = model.quality(training)
        states = tuple(item.state_id for item in model.infer(training, mode="filtered"))
        evaluation = evaluate_states(training, states)
        occupancy = [
            next(
                (
                    profile.observations
                    for profile in evaluation.profiles
                    if profile.state_id == state
                ),
                0,
            )
            for state in range(count)
        ]
        balance = min(occupancy) / max(occupancy) if max(occupancy) else 0.0
        dwell = [length for _, lengths in evaluation.dwell_lengths for length in lengths]
        means = np.asarray(artifact.means)
        variances = np.asarray(artifact.variances)
        separation = min(
            float(
                np.linalg.norm(means[i] - means[j])
                / sqrt(float(np.mean((variances[i] + variances[j]) / 2)))
            )
            for i in range(count)
            for j in range(i)
        )
        if not isfinite(separation):
            raise ValueError("nonfinite HMM state separation")
        entries.append(
            HMMSelectionEntry(
                count,
                quality.log_likelihood,
                quality.aic,
                quality.bic,
                quality.converged,
                artifact.restarts[artifact.selected_restart].iterations,
                balance,
                sum(dwell) / len(dwell),
                evaluation.weighted_exit_entropy_nats,
                separation,
                quality.degenerate_states,
                artifact.fingerprint,
            )
        )
    return tuple(entries)
