"""Deterministic experiment metadata, independent of wall-clock creation time."""

from dataclasses import dataclass
from hashlib import sha256
from typing import Any

from mce.serialization.canonical import canonical_json_bytes
from mce.statistical.normalization import FittedNormalizer
from mce.version import __version__


def parameter_fingerprint(parameters: Any) -> str:
    return sha256(canonical_json_bytes(parameters)).hexdigest()


@dataclass(frozen=True, slots=True)
class ModelArtifactMetadata:
    model_type: str
    model_version: str
    feature_schema_version: str
    feature_schema_fingerprint: str
    config_fingerprint: str
    dataset_fingerprint: str
    split_fingerprint: str
    normalizer_fingerprint: str
    training_start_ns: int
    training_end_ns: int
    training_rows: int
    random_seed: int
    parameter_hash: str
    created_by_version: str = __version__

    def __post_init__(self) -> None:
        if not self.model_type or not self.model_version or self.training_rows <= 0:
            raise ValueError("incomplete model artifact metadata")
        if self.training_start_ns > self.training_end_ns or self.random_seed < 0:
            raise ValueError("invalid training window or seed")
        for digest in (
            self.feature_schema_fingerprint,
            self.config_fingerprint,
            self.dataset_fingerprint,
            self.split_fingerprint,
            self.normalizer_fingerprint,
            self.parameter_hash,
        ):
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError("artifact fingerprints must be lowercase SHA-256")

    @property
    def fingerprint(self) -> str:
        from dataclasses import asdict

        return sha256(canonical_json_bytes(asdict(self))).hexdigest()

    @classmethod
    def from_fit(
        cls,
        *,
        model_type: str,
        model_version: str,
        config_fingerprint: str,
        dataset_fingerprint: str,
        split_fingerprint: str,
        normalizer: FittedNormalizer,
        random_seed: int,
        parameters: Any,
    ) -> "ModelArtifactMetadata":
        if config_fingerprint != normalizer.config_fingerprint:
            raise ValueError("model and normalizer configuration mismatch")
        return cls(
            model_type=model_type,
            model_version=model_version,
            feature_schema_version=normalizer.feature_schema_version,
            feature_schema_fingerprint=normalizer.feature_schema_fingerprint,
            config_fingerprint=config_fingerprint,
            dataset_fingerprint=dataset_fingerprint,
            split_fingerprint=split_fingerprint,
            normalizer_fingerprint=normalizer.fingerprint,
            training_start_ns=normalizer.training_start_ns,
            training_end_ns=normalizer.training_end_ns,
            training_rows=normalizer.training_rows,
            random_seed=random_seed,
            parameter_hash=parameter_fingerprint(parameters),
        )
