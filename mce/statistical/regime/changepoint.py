"""Transparent adjacent-window mean-shift scores for offline research."""

from dataclasses import dataclass
from hashlib import sha256
from math import isfinite, sqrt

import numpy as np

from mce.config.settings import ChangePointSettings
from mce.serialization.canonical import canonical_json_bytes
from mce.statistical.dataset import StatisticalDataset
from mce.statistical.normalization import FittedNormalizer


@dataclass(frozen=True, slots=True)
class ChangePoint:
    index: int
    as_of_ns: int
    score: float
    right_window_end_ns: int
    causal: bool
    model_fingerprint: str


@dataclass(frozen=True, slots=True)
class CausalChangeScore:
    index: int
    as_of_ns: int
    score: float
    causal: bool


class MeanShiftDetector:
    def __init__(self, settings: ChangePointSettings | None = None) -> None:
        self.settings = settings or ChangePointSettings()

    def _matrix(self, dataset: StatisticalDataset, normalizer: FittedNormalizer) -> np.ndarray:
        matrix, _ = normalizer.transform(dataset)
        return np.asarray(matrix)

    def offline(
        self, dataset: StatisticalDataset, normalizer: FittedNormalizer
    ) -> tuple[ChangePoint, ...]:
        """Each score at t uses the right window t:t+right_window, so is retrospective."""
        x = self._matrix(dataset, normalizer)
        left, right = self.settings.left_window, self.settings.right_window
        if len(x) < left + right:
            return ()
        sums = np.vstack((np.zeros((1, x.shape[1])), np.cumsum(x, axis=0)))
        candidates: list[tuple[int, float]] = []
        for index in range(left, len(x) - right + 1):
            before = (sums[index] - sums[index - left]) / left
            after = (sums[index + right] - sums[index]) / right
            score = float(np.linalg.norm(after - before) / sqrt(x.shape[1]))
            if not isfinite(score):
                raise ValueError("nonfinite change-point score")
            if score >= self.settings.threshold:
                candidates.append((index, score))
        selected: list[tuple[int, float]] = []
        for index, score in sorted(candidates, key=lambda item: (-item[1], item[0])):
            if all(abs(index - other) >= self.settings.minimum_separation for other, _ in selected):
                selected.append((index, score))
        fingerprint = sha256(
            canonical_json_bytes(
                {
                    "settings": self.settings,
                    "normalizer": normalizer.fingerprint,
                    "dataset": dataset.fingerprint,
                }
            )
        ).hexdigest()
        return tuple(
            ChangePoint(
                index,
                dataset.rows[index].as_of_ns,
                score,
                dataset.rows[index + right - 1].as_of_ns,
                False,
                fingerprint,
            )
            for index, score in sorted(selected)
        )

    def causal_scores(
        self, dataset: StatisticalDataset, normalizer: FittedNormalizer
    ) -> tuple[CausalChangeScore, ...]:
        """Trailing-window comparison; fitted normalizer must predate every scored row."""
        if dataset.rows and normalizer.training_end_ns >= dataset.rows[0].as_of_ns:
            raise ValueError("causal change scores require a prior-fitted normalizer")
        x = self._matrix(dataset, normalizer)
        left, right = self.settings.left_window, self.settings.right_window
        if len(x) < left + right:
            return ()
        sums = np.vstack((np.zeros((1, x.shape[1])), np.cumsum(x, axis=0)))
        output: list[CausalChangeScore] = []
        for index in range(left + right - 1, len(x)):
            boundary = index - right + 1
            before = (sums[boundary] - sums[boundary - left]) / left
            recent = (sums[index + 1] - sums[boundary]) / right
            score = float(np.linalg.norm(recent - before) / sqrt(x.shape[1]))
            if not isfinite(score):
                raise ValueError("nonfinite causal change score")
            output.append(CausalChangeScore(index, dataset.rows[index].as_of_ns, score, True))
        return tuple(output)
