"""Seeded, deterministic k-means baseline without temporal-state assumptions."""

import json
from dataclasses import asdict, dataclass
from hashlib import sha256
from math import isfinite

import numpy as np
from numpy.typing import NDArray

from mce.config.settings import KMeansSettings
from mce.serialization.canonical import canonical_json_bytes, canonical_json_str
from mce.statistical.dataset import StatisticalDataset
from mce.statistical.models import LatentStateProfile
from mce.statistical.normalization import FittedNormalizer
from mce.statistical.regime.alignment import canonical_state_mapping

Matrix = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class KMeansRestart:
    seed: int
    inertia: float
    iterations: int
    converged: bool
    empty_clusters: int


@dataclass(frozen=True, slots=True)
class KMeansFit:
    centers: Matrix
    labels: NDArray[np.int64]
    diagnostic: KMeansRestart


def squared_distances(x: Matrix, centers: Matrix) -> Matrix:
    result = np.sum((x[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    if not np.isfinite(result).all():
        raise ValueError("nonfinite cluster distances")
    return np.asarray(result)


def fit_kmeans_array(
    matrix: Matrix, cluster_count: int, *, seed: int, max_iterations: int = 100
) -> KMeansFit:
    x = np.asarray(matrix, dtype=np.float64)
    if (
        x.ndim != 2
        or x.shape[0] < cluster_count
        or x.shape[1] == 0
        or cluster_count < 2
        or max_iterations < 1
        or seed < 0
        or not np.isfinite(x).all()
    ):
        raise ValueError("invalid k-means data or settings")
    rng = np.random.default_rng(seed)
    centers = np.empty((cluster_count, x.shape[1]), dtype=np.float64)
    centers[0] = x[int(rng.integers(0, len(x)))]
    nearest = squared_distances(x, centers[:1])[:, 0]
    for index in range(1, cluster_count):
        chosen = int(np.argmax(nearest))
        centers[index] = x[chosen]
        nearest = np.minimum(nearest, squared_distances(x, centers[index : index + 1])[:, 0])
    converged = False
    empty = 0
    iterations = 0
    for iteration in range(1, max_iterations + 1):
        iterations = iteration
        distances = squared_distances(x, centers)
        labels = np.argmin(distances, axis=1)
        updated = centers.copy()
        empty = 0
        for state in range(cluster_count):
            members = x[labels == state]
            if len(members):
                updated[state] = np.mean(members, axis=0)
            else:
                empty += 1
                # Farthest point from its assigned center; stable argmax tie-break.
                updated[state] = x[int(np.argmax(np.min(distances, axis=1)))]
        if np.allclose(updated, centers, rtol=0, atol=1e-10):
            centers = updated
            converged = True
            break
        centers = updated
    distances = squared_distances(x, centers)
    labels = np.asarray(np.argmin(distances, axis=1), dtype=np.int64)
    empty = sum(not np.any(labels == state) for state in range(cluster_count))
    inertia = float(np.sum(distances[np.arange(len(x)), labels]))
    return KMeansFit(centers, labels, KMeansRestart(seed, inertia, iterations, converged, empty))


@dataclass(frozen=True, slots=True)
class KMeansArtifact:
    centers: tuple[tuple[float, ...], ...]
    mapping: tuple[int, ...]
    feature_schema_fingerprint: str
    normalizer_fingerprint: str
    training_fingerprint: str
    training_end_ns: int
    config_fingerprint: str
    seed: int
    selected_restart: int
    restarts: tuple[KMeansRestart, ...]
    settings_fingerprint: str
    parameter_hash: str

    def __post_init__(self) -> None:
        k = len(self.centers)
        if k < 2 or len(self.mapping) != k or set(self.mapping) != set(range(k)):
            raise ValueError("invalid k-means state mapping")
        if not self.centers[0] or any(len(row) != len(self.centers[0]) for row in self.centers):
            raise ValueError("invalid k-means center dimensions")
        if any(not isfinite(value) for row in self.centers for value in row):
            raise ValueError("nonfinite k-means centers")
        if not self.restarts or not 0 <= self.selected_restart < len(self.restarts):
            raise ValueError("invalid k-means restart selection")
        if len(self.settings_fingerprint) != 64:
            raise ValueError("invalid k-means settings fingerprint")
        if (
            sha256(
                canonical_json_bytes({"centers": self.centers, "mapping": self.mapping})
            ).hexdigest()
            != self.parameter_hash
        ):
            raise ValueError("k-means parameter hash mismatch")

    @property
    def fingerprint(self) -> str:
        return sha256(canonical_json_bytes(asdict(self))).hexdigest()

    def serialize(self) -> str:
        return canonical_json_str(asdict(self))

    @classmethod
    def from_json(
        cls, payload: str, *, schema_fingerprint: str, normalizer_fingerprint: str
    ) -> "KMeansArtifact":
        data = json.loads(payload)
        data["centers"] = tuple(tuple(row) for row in data["centers"])
        data["mapping"] = tuple(data["mapping"])
        data["restarts"] = tuple(KMeansRestart(**item) for item in data["restarts"])
        artifact = cls(**data)
        if (
            artifact.feature_schema_fingerprint != schema_fingerprint
            or artifact.normalizer_fingerprint != normalizer_fingerprint
        ):
            raise ValueError("k-means artifact schema or normalizer mismatch")
        return artifact


class KMeansRegimeModel:
    def __init__(self, settings: KMeansSettings | None = None) -> None:
        self.settings = settings or KMeansSettings()
        self.artifact: KMeansArtifact | None = None
        self.normalizer: FittedNormalizer | None = None

    def fit(
        self, training: StatisticalDataset, normalizer: FittedNormalizer, *, random_seed: int
    ) -> KMeansArtifact:
        if normalizer.training_fingerprint != training.fingerprint or random_seed < 0:
            raise ValueError("k-means fit requires exact training data and nonnegative seed")
        x, _ = normalizer.transform(training)
        fits = tuple(
            fit_kmeans_array(
                x,
                self.settings.cluster_count,
                seed=random_seed + restart,
                max_iterations=self.settings.max_iterations,
            )
            for restart in range(self.settings.restarts)
        )
        chosen = min(range(len(fits)), key=lambda index: (fits[index].diagnostic.inertia, index))
        fit = fits[chosen]
        raw_centers = fit.centers * np.asarray(normalizer.scales) + np.asarray(normalizer.centers)
        profiles = tuple(
            LatentStateProfile(
                state,
                int(np.sum(fit.labels == state)),
                0.0,
                tuple(float(v) for v in raw_centers[state]),
                (),
                (),
                0.0,
                (),
            )
            for state in range(self.settings.cluster_count)
        )
        mapping_dict = canonical_state_mapping(profiles, training.schema.names)
        mapping = tuple(mapping_dict[state] for state in range(self.settings.cluster_count))
        artifact = KMeansArtifact(
            centers=tuple(tuple(float(v) for v in row) for row in fit.centers),
            mapping=mapping,
            feature_schema_fingerprint=training.schema.fingerprint,
            normalizer_fingerprint=normalizer.fingerprint,
            training_fingerprint=training.fingerprint,
            training_end_ns=training.rows[-1].as_of_ns,
            config_fingerprint=training.config_fingerprint,
            seed=random_seed,
            selected_restart=chosen,
            restarts=tuple(item.diagnostic for item in fits),
            settings_fingerprint=sha256(canonical_json_bytes(self.settings)).hexdigest(),
            parameter_hash=sha256(
                canonical_json_bytes(
                    {
                        "centers": tuple(tuple(float(v) for v in row) for row in fit.centers),
                        "mapping": mapping,
                    }
                )
            ).hexdigest(),
        )
        self.artifact, self.normalizer = artifact, normalizer
        return artifact

    def load_artifact(self, payload: str, normalizer: FittedNormalizer) -> KMeansArtifact:
        artifact = KMeansArtifact.from_json(
            payload,
            schema_fingerprint=normalizer.feature_schema_fingerprint,
            normalizer_fingerprint=normalizer.fingerprint,
        )
        if artifact.config_fingerprint != normalizer.config_fingerprint:
            raise ValueError("k-means artifact configuration mismatch")
        if artifact.settings_fingerprint != sha256(canonical_json_bytes(self.settings)).hexdigest():
            raise ValueError("k-means artifact settings mismatch")
        self.artifact, self.normalizer = artifact, normalizer
        return artifact

    def infer(self, dataset: StatisticalDataset) -> tuple[int, ...]:
        if self.artifact is None or self.normalizer is None:
            raise ValueError("k-means model is not fitted")
        if dataset.schema.fingerprint != self.artifact.feature_schema_fingerprint:
            raise ValueError("k-means feature schema mismatch")
        x, _ = self.normalizer.transform(dataset)
        if not len(x):
            return ()
        centers = np.asarray(self.artifact.centers)
        raw = np.argmin(squared_distances(x, centers), axis=1)
        return tuple(self.artifact.mapping[int(state)] for state in raw)

    def serialize_artifact(self) -> bytes:
        if self.artifact is None:
            raise ValueError("k-means model is not fitted")
        return self.artifact.serialize().encode("utf-8")

    def diagnostics(self) -> dict[str, float | int | bool]:
        if self.artifact is None:
            raise ValueError("k-means model is not fitted")
        chosen = self.artifact.restarts[self.artifact.selected_restart]
        return {
            "inertia": chosen.inertia,
            "iterations": chosen.iterations,
            "converged": chosen.converged,
            "empty_clusters": chosen.empty_clusters,
        }
