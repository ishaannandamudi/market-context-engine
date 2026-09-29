"""Training-only median imputation and fitted feature scaling."""

from dataclasses import asdict, dataclass
from hashlib import sha256
from math import isfinite
from typing import Literal, cast

import numpy as np
from numpy.typing import NDArray

from mce.serialization.canonical import canonical_json_bytes
from mce.statistical.dataset import StatisticalDataset
from mce.statistical.features import NormalizeMethod

FloatMatrix = NDArray[np.float64]
BoolMatrix = NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class FittedNormalizer:
    feature_schema_version: str
    feature_schema_fingerprint: str
    config_fingerprint: str
    feature_names: tuple[str, ...]
    methods: tuple[NormalizeMethod, ...]
    impute_medians: tuple[float, ...]
    centers: tuple[float, ...]
    scales: tuple[float, ...]
    constant_features: tuple[bool, ...]
    training_start_ns: int
    training_end_ns: int
    training_rows: int
    training_fingerprint: str

    def __post_init__(self) -> None:
        width = len(self.feature_names)
        if not width or any(
            len(values) != width
            for values in (
                self.methods,
                self.impute_medians,
                self.centers,
                self.scales,
                self.constant_features,
            )
        ):
            raise ValueError("normalizer vectors must match feature names")
        if any(scale <= 0 or not isfinite(scale) for scale in self.scales):
            raise ValueError("normalizer scales must be finite and positive")
        if any(not isfinite(value) for value in self.impute_medians + self.centers):
            raise ValueError("normalizer parameters must be finite")
        if self.training_start_ns > self.training_end_ns or self.training_rows <= 0:
            raise ValueError("invalid normalizer training interval")

    @property
    def fingerprint(self) -> str:
        return sha256(canonical_json_bytes(asdict(self))).hexdigest()

    def transform(self, dataset: StatisticalDataset) -> tuple[FloatMatrix, BoolMatrix]:
        if (
            dataset.schema.version != self.feature_schema_version
            or dataset.schema.fingerprint != self.feature_schema_fingerprint
            or dataset.schema.names != self.feature_names
        ):
            raise ValueError("normalizer feature schema mismatch")
        if dataset.config_fingerprint != self.config_fingerprint:
            raise ValueError("normalizer configuration mismatch")
        if not dataset.rows:
            width = len(self.feature_names)
            return np.empty((0, width), dtype=np.float64), np.empty((0, width), dtype=np.bool_)
        raw = np.array([row.feature_vector for row in dataset.rows], dtype=np.float64)
        mask = np.array([row.feature_mask for row in dataset.rows], dtype=np.bool_)
        filled = np.where(mask, raw, np.asarray(self.impute_medians, dtype=np.float64))
        result = (filled - np.asarray(self.centers, dtype=np.float64)) / np.asarray(
            self.scales, dtype=np.float64
        )
        if not np.isfinite(result).all():
            raise ValueError("normalization produced nonfinite values")
        return result, mask


def fit_normalizer(
    dataset: StatisticalDataset,
    train_indices: tuple[int, ...],
    method: Literal["schema", "none", "standard", "robust"] | None = None,
) -> FittedNormalizer:
    """Fit on explicit training indices; the caller owns the temporal split."""
    if not train_indices or train_indices != tuple(sorted(set(train_indices))):
        raise ValueError("training indices must be unique, ordered, and nonempty")
    training = dataset.take(train_indices)
    chosen = method or dataset.normalization_method
    if chosen not in ("schema", "none", "standard", "robust"):
        raise ValueError("unsupported normalization method")
    raw = np.array([row.feature_vector for row in training.rows], dtype=np.float64)
    mask = np.array([row.feature_mask for row in training.rows], dtype=np.bool_)
    medians: list[float] = []
    centers: list[float] = []
    scales: list[float] = []
    constants: list[bool] = []
    methods: list[NormalizeMethod] = []
    for col, spec in enumerate(dataset.schema.specs):
        observed = raw[mask[:, col], col]
        if observed.size == 0:
            raise ValueError(f"training feature has no observed values: {spec.name}")
        median = float(np.median(observed))
        current: NormalizeMethod = (
            spec.normalization if chosen == "schema" else cast(NormalizeMethod, chosen)
        )
        if current == "standard":
            center, scale = float(np.mean(observed)), float(np.std(observed))
        elif current == "robust":
            center, scale = median, float(1.4826 * np.median(np.abs(observed - median)))
        else:
            center, scale = 0.0, 1.0
        constant = bool(np.all(observed == observed[0])) or (current != "none" and scale <= 0.0)
        if scale <= 0.0:
            scale = 1.0
        medians.append(median)
        centers.append(center)
        scales.append(scale)
        constants.append(constant)
        methods.append(current)
    return FittedNormalizer(
        feature_schema_version=dataset.schema.version,
        feature_schema_fingerprint=dataset.schema.fingerprint,
        config_fingerprint=dataset.config_fingerprint,
        feature_names=dataset.schema.names,
        methods=tuple(methods),
        impute_medians=tuple(medians),
        centers=tuple(centers),
        scales=tuple(scales),
        constant_features=tuple(constants),
        training_start_ns=training.rows[0].as_of_ns,
        training_end_ns=training.rows[-1].as_of_ns,
        training_rows=len(training.rows),
        training_fingerprint=training.fingerprint,
    )
