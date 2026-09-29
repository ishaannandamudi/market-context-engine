"""Latent-state contracts; no fitted statistical model is shipped in MCE-12."""

from dataclasses import dataclass
from math import isfinite
from typing import Literal, Protocol

from mce.models.enums import QualityFlag
from mce.statistical.dataset import StatisticalDataset
from mce.statistical.normalization import FittedNormalizer


@dataclass(frozen=True, slots=True)
class LatentStateInference:
    as_of_ns: int
    state_id: int
    state_probabilities: tuple[float, ...] | None
    mode: Literal["filtered", "smoothed", "viterbi"]
    retrospective: bool
    training_end_ns: int
    model_version: str
    feature_schema_version: str
    quality_flags: frozenset[QualityFlag] = frozenset()

    def __post_init__(self) -> None:
        if self.state_id < 0 or self.training_end_ns < 0 or self.as_of_ns < 0:
            raise ValueError("state id and clocks must be nonnegative")
        if self.mode != "filtered" and not self.retrospective:
            raise ValueError("smoothed and Viterbi inferences are retrospective")
        if self.training_end_ns >= self.as_of_ns and not self.retrospective:
            raise ValueError("in-sample or future-trained inference is retrospective")
        if self.state_probabilities is not None:
            probs = self.state_probabilities
            if self.state_id >= len(probs) or any(not isfinite(p) or p < 0 or p > 1 for p in probs):
                raise ValueError("invalid state probabilities")
            if abs(sum(probs) - 1.0) > 1e-8:
                raise ValueError("state probabilities must sum to one")


@dataclass(frozen=True, slots=True)
class LatentStateProfile:
    state_id: int
    observations: int
    occupancy_fraction: float
    centroid: tuple[float | None, ...]
    median: tuple[float | None, ...]
    dispersion: tuple[float | None, ...]
    average_dwell_observations: float
    deterministic_overlap: tuple[tuple[str, int], ...]


class StatisticalRegimeModel(Protocol):
    """Implementations must expose fitted metadata and explicit inference mode."""

    def fit(
        self, training: StatisticalDataset, normalizer: FittedNormalizer, *, random_seed: int
    ) -> None: ...

    def infer(
        self, dataset: StatisticalDataset, *, mode: Literal["filtered", "smoothed", "viterbi"]
    ) -> tuple[LatentStateInference, ...]: ...

    def serialize_artifact(self) -> bytes: ...

    def diagnostics(self) -> dict[str, float | int | bool]: ...
