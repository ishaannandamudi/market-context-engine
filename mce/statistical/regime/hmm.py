"""Diagonal Gaussian HMM with seeded EM and explicit causal inference modes."""

import json
from dataclasses import asdict, dataclass
from hashlib import sha256
from math import isfinite, log
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from mce.config.settings import HMMSettings
from mce.serialization.canonical import canonical_json_bytes, canonical_json_str
from mce.statistical.artifacts import ModelArtifactMetadata, parameter_fingerprint
from mce.statistical.dataset import StatisticalDataset
from mce.statistical.features import STATISTICAL_FEATURE_SCHEMA
from mce.statistical.models import LatentStateInference, LatentStateProfile
from mce.statistical.normalization import FittedNormalizer
from mce.statistical.regime.alignment import canonical_state_mapping
from mce.statistical.regime.clustering import fit_kmeans_array
from mce.statistical.regime.evaluation import evaluate_states
from mce.statistical.regime.hmm_math import (
    backward_log,
    diagonal_gaussian_log_density,
    forward_log,
    logsumexp,
    posteriors,
    viterbi_path,
)

Matrix = NDArray[np.float64]
MODEL_VERSION = "diagonal-gaussian-hmm/1"
TRANSITION_PRIOR = 1e-3


@dataclass(frozen=True, slots=True)
class HMMRestart:
    seed: int
    iterations: int
    converged: bool
    log_likelihood: float | None
    likelihood_history: tuple[float, ...]
    error: str | None = None


@dataclass(frozen=True, slots=True)
class HMMArtifact:
    metadata: ModelArtifactMetadata
    settings_fingerprint: str
    initial_probabilities: tuple[float, ...]
    transition_matrix: tuple[tuple[float, ...], ...]
    means: tuple[tuple[float, ...], ...]
    variances: tuple[tuple[float, ...], ...]
    covariance_floor: float
    state_mapping: tuple[int, ...]
    training_end_filtered: tuple[float, ...]
    selected_restart: int
    restarts: tuple[HMMRestart, ...]

    def __post_init__(self) -> None:
        k = len(self.initial_probabilities)
        d = len(self.means[0]) if self.means else 0
        if (
            k < 2
            or d < 1
            or self.metadata.model_type != "diagonal_gaussian_hmm"
            or self.metadata.model_version != MODEL_VERSION
            or self.metadata.feature_schema_version != STATISTICAL_FEATURE_SCHEMA.version
        ):
            raise ValueError("invalid HMM artifact model version or dimensions")
        if not isfinite(self.covariance_floor) or self.covariance_floor <= 0:
            raise ValueError("invalid covariance floor")
        if (
            len(self.transition_matrix) != k
            or len(self.means) != k
            or len(self.variances) != k
            or len(self.training_end_filtered) != k
            or len(self.state_mapping) != k
            or set(self.state_mapping) != set(range(k))
            or any(len(row) != k for row in self.transition_matrix)
            or any(len(row) != d for row in self.means + self.variances)
        ):
            raise ValueError("invalid HMM artifact dimensions or mapping")
        probabilities = (
            self.initial_probabilities,
            self.training_end_filtered,
            *self.transition_matrix,
        )
        if any(
            any(not isfinite(p) or p < 0 for p in row) or abs(sum(row) - 1) > 1e-8
            for row in probabilities
        ):
            raise ValueError("invalid HMM artifact probabilities")
        if any(not isfinite(value) for row in self.means for value in row) or any(
            not isfinite(value) or value < self.covariance_floor - 1e-12
            for row in self.variances
            for value in row
        ):
            raise ValueError("invalid HMM artifact emissions")
        if not self.restarts or not 0 <= self.selected_restart < len(self.restarts):
            raise ValueError("invalid HMM restart selection")
        if len(self.settings_fingerprint) != 64:
            raise ValueError("invalid HMM settings fingerprint")
        parameters = {
            "initial": self.initial_probabilities,
            "transition": self.transition_matrix,
            "means": self.means,
            "variances": self.variances,
            "mapping": self.state_mapping,
        }
        if parameter_fingerprint(parameters) != self.metadata.parameter_hash:
            raise ValueError("HMM artifact parameter hash mismatch")

    @property
    def fingerprint(self) -> str:
        return sha256(canonical_json_bytes(asdict(self))).hexdigest()

    def serialize(self) -> str:
        return canonical_json_str(asdict(self))

    @classmethod
    def from_json(
        cls, payload: str, *, schema_fingerprint: str, normalizer_fingerprint: str
    ) -> "HMMArtifact":
        data = json.loads(payload)
        data["metadata"] = ModelArtifactMetadata(**data["metadata"])
        data["restarts"] = tuple(
            HMMRestart(**(item | {"likelihood_history": tuple(item["likelihood_history"])}))
            for item in data["restarts"]
        )
        for key in ("transition_matrix", "means", "variances"):
            data[key] = tuple(tuple(row) for row in data[key])
        for key in ("initial_probabilities", "state_mapping", "training_end_filtered"):
            data[key] = tuple(data[key])
        artifact = cls(**data)
        if (
            artifact.metadata.feature_schema_fingerprint != schema_fingerprint
            or artifact.metadata.normalizer_fingerprint != normalizer_fingerprint
        ):
            raise ValueError("HMM artifact feature schema or normalizer mismatch")
        return artifact


@dataclass(frozen=True, slots=True)
class HMMQuality:
    converged: bool
    log_likelihood: float
    aic: float
    bic: float
    minimum_occupancy: float
    mean_posterior_entropy: float
    excessive_switching: bool
    degenerate_states: tuple[int, ...]


def parameter_count(state_count: int, dimension: int) -> int:
    if state_count < 2 or dimension < 1:
        raise ValueError("invalid HMM parameter dimensions")
    return (state_count - 1) + state_count * (state_count - 1) + 2 * state_count * dimension


def information_criteria(
    log_likelihood: float, rows: int, state_count: int, dimension: int
) -> tuple[float, float]:
    if rows < 1 or not isfinite(log_likelihood):
        raise ValueError("invalid likelihood or row count")
    count = parameter_count(state_count, dimension)
    return 2 * count - 2 * log_likelihood, count * log(rows) - 2 * log_likelihood


def _initial_parameters(
    x: Matrix, k: int, seed: int, floor: float
) -> tuple[Matrix, Matrix, Matrix, Matrix]:
    fit = fit_kmeans_array(x, k, seed=seed, max_iterations=30)
    labels = fit.labels
    global_var = np.maximum(np.var(x, axis=0), floor)
    means = fit.centers.copy()
    variances = np.empty_like(means)
    for state in range(k):
        members = x[labels == state]
        variances[state] = (
            np.maximum(np.var(members, axis=0), floor) if len(members) else global_var
        )
    initial = np.full(k, TRANSITION_PRIOR)
    initial[int(labels[0])] += 1.0
    initial /= initial.sum()
    transition = np.full((k, k), TRANSITION_PRIOR)
    np.add.at(transition, (labels[:-1], labels[1:]), 1.0)
    transition /= transition.sum(axis=1, keepdims=True)
    return initial, transition, means, variances


def _fit_restart(
    x: Matrix, k: int, seed: int, settings: HMMSettings
) -> tuple[tuple[Matrix, Matrix, Matrix, Matrix], HMMRestart]:
    initial, transition, means, variances = _initial_parameters(
        x, k, seed, settings.covariance_floor
    )
    history: list[float] = []
    converged = False
    for _ in range(settings.max_iterations):
        emissions = diagonal_gaussian_log_density(x, means, variances, settings.covariance_floor)
        alpha, likelihood = forward_log(emissions, initial, transition)
        history.append(likelihood)
        beta = backward_log(emissions, transition)
        gamma, xi = posteriors(emissions, alpha, beta, transition, likelihood)
        weight = gamma.sum(axis=0)
        new_initial = (gamma[0] + TRANSITION_PRIOR) / (1 + k * TRANSITION_PRIOR)
        trans_counts = xi.sum(axis=0)
        new_transition = trans_counts + TRANSITION_PRIOR
        new_transition /= new_transition.sum(axis=1, keepdims=True)
        new_means = means.copy()
        new_variances = variances.copy()
        active = weight > 1e-8
        new_means[active] = gamma[:, active].T @ x / weight[active, None]
        second = gamma[:, active].T @ (x * x) / weight[active, None]
        new_variances[active] = np.maximum(
            second - new_means[active] ** 2, settings.covariance_floor
        )
        candidate_emissions = diagonal_gaussian_log_density(
            x, new_means, new_variances, settings.covariance_floor
        )
        _, candidate_likelihood = forward_log(candidate_emissions, new_initial, new_transition)
        delta = candidate_likelihood - likelihood
        if delta < -1e-6 * (1 + abs(likelihood)):
            raise ValueError("EM likelihood decreased materially")
        initial, transition, means, variances = (
            new_initial,
            new_transition,
            new_means,
            new_variances,
        )
        if abs(delta) <= settings.tolerance * (1 + abs(likelihood)):
            history.append(candidate_likelihood)
            converged = True
            break
    if not converged:
        emissions = diagonal_gaussian_log_density(x, means, variances, settings.covariance_floor)
        _, final_likelihood = forward_log(emissions, initial, transition)
        history.append(final_likelihood)
    result = HMMRestart(seed, len(history) - 1, converged, history[-1], tuple(history))
    return (initial, transition, means, variances), result


class GaussianHMMRegimeModel:
    def __init__(self, settings: HMMSettings | None = None) -> None:
        self.settings = settings or HMMSettings()
        self.artifact: HMMArtifact | None = None
        self.normalizer: FittedNormalizer | None = None

    def fit(
        self,
        training: StatisticalDataset,
        normalizer: FittedNormalizer,
        *,
        random_seed: int,
        split_fingerprint: str | None = None,
    ) -> HMMArtifact:
        if normalizer.training_fingerprint != training.fingerprint or random_seed < 0:
            raise ValueError("HMM fit requires exact training dataset and nonnegative seed")
        x, _ = normalizer.transform(training)
        k = self.settings.state_count
        if len(x) < k:
            raise ValueError("training rows fewer than HMM states")
        candidates: list[tuple[Matrix, Matrix, Matrix, Matrix] | None] = []
        diagnostics: list[HMMRestart] = []
        for restart in range(self.settings.restarts):
            seed = random_seed + restart
            try:
                params, diag = _fit_restart(x, k, seed, self.settings)
                candidates.append(params)
                diagnostics.append(diag)
            except ValueError as exc:
                candidates.append(None)
                diagnostics.append(HMMRestart(seed, 0, False, None, (), str(exc)))
        eligible = [
            i for i, item in enumerate(candidates) if item is not None and diagnostics[i].converged
        ]
        if not eligible:
            eligible = [i for i, item in enumerate(candidates) if item is not None]
        if not eligible:
            raise ValueError("all HMM restarts failed")
        selected = max(
            eligible,
            key=lambda index: (
                diagnostics[index].log_likelihood
                if diagnostics[index].log_likelihood is not None
                else -1e300,
                -index,
            ),
        )
        selected_params = candidates[selected]
        assert selected_params is not None
        initial, transition, means, variances = selected_params
        raw_means = means * np.asarray(normalizer.scales) + np.asarray(normalizer.centers)
        profiles = tuple(
            LatentStateProfile(
                state, 0, 0.0, tuple(float(v) for v in raw_means[state]), (), (), 0.0, ()
            )
            for state in range(k)
        )
        mapping_dict = canonical_state_mapping(profiles, training.schema.names)
        raw_to_canonical = tuple(mapping_dict[state] for state in range(k))
        order = np.argsort(raw_to_canonical)
        initial = initial[order]
        transition = transition[np.ix_(order, order)]
        means = means[order]
        variances = variances[order]
        emissions = diagonal_gaussian_log_density(
            x, means, variances, self.settings.covariance_floor
        )
        alpha, _ = forward_log(emissions, initial, transition)
        filtered = np.exp(alpha[-1] - logsumexp(alpha[-1]))
        parameters = {
            "initial": initial.tolist(),
            "transition": transition.tolist(),
            "means": means.tolist(),
            "variances": variances.tolist(),
            "mapping": raw_to_canonical,
        }
        metadata = ModelArtifactMetadata.from_fit(
            model_type="diagonal_gaussian_hmm",
            model_version=MODEL_VERSION,
            config_fingerprint=training.config_fingerprint,
            dataset_fingerprint=training.fingerprint,
            split_fingerprint=split_fingerprint or training.fingerprint,
            normalizer=normalizer,
            random_seed=random_seed,
            parameters=parameters,
        )
        artifact = HMMArtifact(
            metadata=metadata,
            settings_fingerprint=sha256(canonical_json_bytes(self.settings)).hexdigest(),
            initial_probabilities=tuple(float(v) for v in initial),
            transition_matrix=tuple(tuple(float(v) for v in row) for row in transition),
            means=tuple(tuple(float(v) for v in row) for row in means),
            variances=tuple(tuple(float(v) for v in row) for row in variances),
            covariance_floor=self.settings.covariance_floor,
            state_mapping=raw_to_canonical,
            training_end_filtered=tuple(float(v) for v in filtered),
            selected_restart=selected,
            restarts=tuple(diagnostics),
        )
        self.artifact, self.normalizer = artifact, normalizer
        return artifact

    def load_artifact(self, payload: str, normalizer: FittedNormalizer) -> HMMArtifact:
        artifact = HMMArtifact.from_json(
            payload,
            schema_fingerprint=normalizer.feature_schema_fingerprint,
            normalizer_fingerprint=normalizer.fingerprint,
        )
        if artifact.metadata.config_fingerprint != normalizer.config_fingerprint:
            raise ValueError("HMM artifact configuration mismatch")
        if artifact.settings_fingerprint != sha256(canonical_json_bytes(self.settings)).hexdigest():
            raise ValueError("HMM artifact settings mismatch")
        self.artifact, self.normalizer = artifact, normalizer
        return artifact

    def _matrices(self, dataset: StatisticalDataset) -> tuple[Matrix, Matrix, Matrix, Matrix]:
        if self.artifact is None or self.normalizer is None:
            raise ValueError("HMM model is not fitted")
        if dataset.schema.fingerprint != self.artifact.metadata.feature_schema_fingerprint:
            raise ValueError("HMM feature schema mismatch")
        x, _ = self.normalizer.transform(dataset)
        return (
            x,
            np.asarray(self.artifact.initial_probabilities),
            np.asarray(self.artifact.transition_matrix),
            diagonal_gaussian_log_density(
                x,
                np.asarray(self.artifact.means),
                np.asarray(self.artifact.variances),
                self.artifact.covariance_floor,
            )
            if len(x)
            else np.empty((0, len(self.artifact.initial_probabilities))),
        )

    def infer(
        self,
        dataset: StatisticalDataset,
        *,
        mode: Literal["filtered", "smoothed", "viterbi"] = "filtered",
        previous_filtered: tuple[float, ...] | None = None,
    ) -> tuple[LatentStateInference, ...]:
        if mode not in ("filtered", "smoothed", "viterbi"):
            raise ValueError("unknown HMM inference mode")
        if previous_filtered is not None and mode != "filtered":
            raise ValueError("previous belief is only valid for filtering")
        _, initial, transition, emissions = self._matrices(dataset)
        if not len(emissions):
            return ()
        alpha, likelihood = forward_log(
            emissions,
            initial,
            transition,
            previous_filtered=np.asarray(previous_filtered)
            if previous_filtered is not None
            else None,
        )
        if mode == "filtered":
            probabilities = np.exp(alpha - logsumexp(alpha, axis=1)[:, None])
            path = np.argmax(probabilities, axis=1)
        elif mode == "smoothed":
            beta = backward_log(emissions, transition)
            probabilities, _ = posteriors(emissions, alpha, beta, transition, likelihood)
            path = np.argmax(probabilities, axis=1)
        else:
            probabilities = None
            path = viterbi_path(emissions, initial, transition)
        assert self.artifact is not None
        model_fingerprint = self.artifact.fingerprint
        result: list[LatentStateInference] = []
        for index, row in enumerate(dataset.rows):
            probs = (
                tuple(float(p) for p in probabilities[index]) if probabilities is not None else None
            )
            entropy = -sum(p * log(p) for p in probs if p > 0) if probs is not None else None
            result.append(
                LatentStateInference(
                    as_of_ns=row.as_of_ns,
                    state_id=int(path[index]),
                    state_probabilities=probs,
                    mode=mode,
                    retrospective=(
                        mode != "filtered" or row.as_of_ns <= self.artifact.metadata.training_end_ns
                    ),
                    training_end_ns=self.artifact.metadata.training_end_ns,
                    model_version=MODEL_VERSION,
                    feature_schema_version=dataset.schema.version,
                    quality_flags=row.quality_flags,
                    posterior_entropy=entropy,
                    model_fingerprint=model_fingerprint,
                )
            )
        return tuple(result)

    def state_profiles(self, training: StatisticalDataset) -> tuple[LatentStateProfile, ...]:
        if self.artifact is None or self.normalizer is None:
            raise ValueError("HMM model is not fitted")
        if training.fingerprint != self.normalizer.training_fingerprint:
            raise ValueError("state profiles require original training dataset")
        states = tuple(item.state_id for item in self.infer(training, mode="filtered"))
        observed = {
            profile.state_id: profile for profile in evaluate_states(training, states).profiles
        }
        raw_means = np.asarray(self.artifact.means) * np.asarray(
            self.normalizer.scales
        ) + np.asarray(self.normalizer.centers)
        return tuple(
            observed.get(state)
            or LatentStateProfile(
                state, 0, 0.0, tuple(float(v) for v in raw_means[state]), (), (), 0.0, ()
            )
            for state in range(len(self.artifact.means))
        )

    def quality(self, training: StatisticalDataset) -> HMMQuality:
        if self.artifact is None:
            raise ValueError("HMM model is not fitted")
        fitted = self.artifact
        outputs = self.infer(training, mode="filtered")
        probs = np.asarray([item.state_probabilities for item in outputs], dtype=np.float64)
        occupancy = probs.mean(axis=0)
        entropy = float(np.mean([item.posterior_entropy or 0.0 for item in outputs]))
        chosen = fitted.restarts[fitted.selected_restart]
        assert chosen.log_likelihood is not None
        aic, bic = information_criteria(
            chosen.log_likelihood, len(training.rows), len(occupancy), len(fitted.means[0])
        )
        states = tuple(item.state_id for item in outputs)
        switching = sum(a != b for a, b in zip(states, states[1:], strict=False)) / max(
            1, len(states) - 1
        )
        collapsed = {
            i for i, value in enumerate(occupancy) if value < self.settings.minimum_occupancy
        }
        for i in range(len(occupancy)):
            if all(value <= fitted.covariance_floor * 1.0001 for value in fitted.variances[i]):
                collapsed.add(i)
        for i in range(len(occupancy)):
            for j in range(i):
                if np.linalg.norm(np.asarray(fitted.means[i]) - np.asarray(fitted.means[j])) < 1e-3:
                    collapsed.add(i)
        return HMMQuality(
            chosen.converged,
            chosen.log_likelihood,
            aic,
            bic,
            float(np.min(occupancy)),
            entropy,
            switching > 0.75,
            tuple(sorted(collapsed)),
        )

    def serialize_artifact(self) -> bytes:
        if self.artifact is None:
            raise ValueError("HMM model is not fitted")
        return self.artifact.serialize().encode("utf-8")

    def diagnostics(self) -> dict[str, float | int | bool]:
        if self.artifact is None:
            raise ValueError("HMM model is not fitted")
        selected = self.artifact.restarts[self.artifact.selected_restart]
        assert selected.log_likelihood is not None
        return {
            "log_likelihood": selected.log_likelihood,
            "iterations": selected.iterations,
            "converged": selected.converged,
            "selected_restart": self.artifact.selected_restart,
        }
