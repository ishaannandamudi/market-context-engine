"""Immutable point-in-time factor panel, artifact and descriptive context contracts."""

import json
from dataclasses import asdict, dataclass
from hashlib import sha256
from math import isfinite

import numpy as np

from mce.models.enums import DataQuality, QualityFlag
from mce.serialization.canonical import canonical_json_bytes, canonical_json_str

MODEL_VERSION = "rolling-pca/1"


@dataclass(frozen=True, slots=True)
class ReturnPanel:
    as_of_ns: int
    timestamps_ns: tuple[int, ...]
    expected_symbols: tuple[str, ...]
    symbols: tuple[str, ...]
    returns: tuple[tuple[float, ...], ...]
    expected_samples: int
    missing_symbols: tuple[str, ...] = ()
    stale_symbols: tuple[str, ...] = ()
    source_quality: DataQuality = DataQuality.GOOD
    quality_flags: frozenset[QualityFlag] = frozenset()
    membership_fingerprint: str | None = None
    row_available_at_ns: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.as_of_ns < 0 or self.expected_samples < 1:
            raise ValueError("invalid factor panel clock or sample count")
        if self.expected_symbols != tuple(sorted(set(self.expected_symbols))):
            raise ValueError("expected factor symbols must be sorted and unique")
        if self.symbols != tuple(sorted(set(self.symbols))) or not set(self.symbols) <= set(
            self.expected_symbols
        ):
            raise ValueError("factor symbols must be sorted members of expected universe")
        if (
            len(self.timestamps_ns) != len(self.returns)
            or len(self.returns) > self.expected_samples
        ):
            raise ValueError("factor panel rows and timestamps differ")
        if any(a >= b for a, b in zip(self.timestamps_ns, self.timestamps_ns[1:], strict=False)):
            raise ValueError("factor timestamps must increase")
        if self.timestamps_ns and self.timestamps_ns[-1] > self.as_of_ns:
            raise ValueError("factor panel contains future data")
        if any(
            len(row) != len(self.symbols) or any(not isfinite(v) for v in row)
            for row in self.returns
        ):
            raise ValueError("factor returns must be complete finite aligned rows")
        if self.membership_fingerprint is not None and len(self.membership_fingerprint) != 64:
            raise ValueError("invalid panel membership fingerprint")
        if self.row_available_at_ns and (
            len(self.row_available_at_ns) != len(self.timestamps_ns)
            or any(
                available < timestamp or available > self.as_of_ns
                for timestamp, available in zip(
                    self.timestamps_ns, self.row_available_at_ns, strict=True
                )
            )
        ):
            raise ValueError("factor row availability is inconsistent")

    @property
    def symbol_coverage(self) -> float:
        return len(self.symbols) / len(self.expected_symbols) if self.expected_symbols else 0.0

    @property
    def time_coverage(self) -> float:
        return len(self.returns) / self.expected_samples

    @property
    def fingerprint(self) -> str:
        return sha256(canonical_json_bytes(asdict(self))).hexdigest()

    def take(self, indices: tuple[int, ...]) -> "ReturnPanel":
        if not indices or any(i < 0 or i >= len(self.returns) for i in indices):
            raise ValueError("factor panel selection is empty or out of range")
        return ReturnPanel(
            self.timestamps_ns[indices[-1]],
            tuple(self.timestamps_ns[i] for i in indices),
            self.expected_symbols,
            self.symbols,
            tuple(self.returns[i] for i in indices),
            len(indices),
            self.missing_symbols,
            self.stale_symbols,
            self.source_quality,
            self.quality_flags,
            self.membership_fingerprint,
            tuple(self.row_available_at_ns[i] for i in indices) if self.row_available_at_ns else (),
        )


@dataclass(frozen=True, slots=True)
class FactorArtifact:
    model_version: str
    symbols: tuple[str, ...]
    matrix_type: str
    means: tuple[float, ...]
    scales: tuple[float, ...]
    eigenvalues: tuple[float, ...]
    explained_variance: tuple[float, ...]
    loadings: tuple[tuple[float, ...], ...]
    selected_count: int
    fit_start_ns: int
    fit_end_ns: int
    training_fingerprint: str
    config_fingerprint: str
    settings_fingerprint: str
    near_degenerate: bool
    method: str
    input_schema_version: str = "return-panel/1"

    def __post_init__(self) -> None:
        n, k = len(self.symbols), self.selected_count
        if (
            self.model_version != MODEL_VERSION
            or self.input_schema_version != "return-panel/1"
            or self.matrix_type not in ("covariance", "correlation")
            or self.method not in ("eigh", "svd")
        ):
            raise ValueError("unsupported factor artifact model")
        if n < 2 or k < 1 or k > n or self.symbols != tuple(sorted(set(self.symbols))):
            raise ValueError("invalid factor artifact symbols or factor count")
        if self.fit_start_ns < 0 or self.fit_start_ns > self.fit_end_ns:
            raise ValueError("invalid factor fit interval")
        if len(self.means) != n or len(self.scales) != n or len(self.loadings) != n:
            raise ValueError("factor artifact dimensions differ")
        if any(len(row) != k for row in self.loadings):
            raise ValueError("factor loading dimensions differ")
        if len(self.eigenvalues) != n or len(self.explained_variance) != n:
            raise ValueError("factor spectrum dimensions differ")
        if any(
            not isfinite(v)
            for v in self.means + self.scales + self.eigenvalues + self.explained_variance
        ):
            raise ValueError("nonfinite factor artifact value")
        if any(not isfinite(v) for row in self.loadings for v in row):
            raise ValueError("nonfinite factor loading")
        if any(v <= 0 for v in self.scales) or any(v < -1e-10 for v in self.eigenvalues):
            raise ValueError("invalid factor scales or eigenvalues")
        if any(a + 1e-10 < b for a, b in zip(self.eigenvalues, self.eigenvalues[1:], strict=False)):
            raise ValueError("factor eigenvalues must descend")
        if any(v < -1e-10 or v > 1 + 1e-10 for v in self.explained_variance):
            raise ValueError("invalid explained variance ratio")
        if abs(sum(self.explained_variance) - 1) > 1e-7:
            raise ValueError("explained variance ratios must sum to one")
        total_variance = sum(self.eigenvalues)
        if total_variance <= 0 or any(
            abs(ratio - value / total_variance) > 1e-7
            for ratio, value in zip(self.explained_variance, self.eigenvalues, strict=True)
        ):
            raise ValueError("factor spectrum and explained variance disagree")
        basis = np.asarray(self.loadings)
        if not np.allclose(basis.T @ basis, np.eye(k), atol=1e-7):
            raise ValueError("factor loadings must be orthonormal")
        if any(
            basis[int(np.argmax(np.abs(basis[:, index]))), index] < -1e-12 for index in range(k)
        ):
            raise ValueError("factor loading signs are not canonical")
        if any(
            len(item) != 64 or any(c not in "0123456789abcdef" for c in item)
            for item in (
                self.training_fingerprint,
                self.config_fingerprint,
                self.settings_fingerprint,
            )
        ):
            raise ValueError("invalid factor artifact fingerprint")

    @property
    def fingerprint(self) -> str:
        return sha256(canonical_json_bytes(asdict(self))).hexdigest()

    def serialize(self) -> str:
        return canonical_json_str(asdict(self))

    @classmethod
    def from_json(
        cls,
        payload: str,
        *,
        settings_fingerprint: str | None = None,
        expected_fingerprint: str | None = None,
        expected_symbols: tuple[str, ...] | None = None,
        config_fingerprint: str | None = None,
    ) -> "FactorArtifact":
        data = json.loads(payload)
        for key in ("symbols", "means", "scales", "eigenvalues", "explained_variance"):
            data[key] = tuple(data[key])
        data["loadings"] = tuple(tuple(row) for row in data["loadings"])
        artifact = cls(**data)
        if (
            settings_fingerprint is not None
            and artifact.settings_fingerprint != settings_fingerprint
        ):
            raise ValueError("factor artifact settings mismatch")
        if expected_fingerprint is not None and artifact.fingerprint != expected_fingerprint:
            raise ValueError("factor artifact fingerprint mismatch")
        if expected_symbols is not None and artifact.symbols != expected_symbols:
            raise ValueError("factor artifact symbol schema mismatch")
        if config_fingerprint is not None and artifact.config_fingerprint != config_fingerprint:
            raise ValueError("factor artifact configuration mismatch")
        return artifact


@dataclass(frozen=True, slots=True)
class FactorDecomposition:
    residual_variance_shares: tuple[float | None, ...]
    common_variance_shares: tuple[float | None, ...]
    mean_common_share: float | None
    median_residual_share: float | None
    raw_dispersion: float
    residual_dispersion: float
    explained_variance: float
    residual_variance: float
    total_variance: float


@dataclass(frozen=True, slots=True)
class FactorAlignment:
    current_to_previous: tuple[int, ...]
    signs: tuple[int, ...]
    similarities: tuple[float, ...]
    mean_similarity: float
    subspace_similarity: float


@dataclass(frozen=True, slots=True)
class SectorFactorExposure:
    sector: str
    symbols: tuple[str, ...]
    mean_loadings: tuple[float, ...]
    loading_dispersion: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class FactorContext:
    as_of_ns: int
    panel_fingerprint: str
    artifact: FactorArtifact | None
    expected_symbols: tuple[str, ...]
    valid_symbols: tuple[str, ...]
    missing_symbols: tuple[str, ...]
    stale_symbols: tuple[str, ...]
    symbol_coverage: float
    aligned_samples: int
    expected_samples: int
    time_coverage: float
    first_share: float | None
    top3_share: float | None
    effective_dimension: float | None
    concentration: float | None
    dominance: float | None
    current_scores: tuple[float, ...]
    score_history: tuple[tuple[float, ...], ...]
    decomposition: FactorDecomposition | None
    alignment: FactorAlignment | None
    drift_symbols: tuple[str, ...]
    loading_drift: tuple[tuple[float, ...], ...]
    sector_exposures: tuple[SectorFactorExposure, ...]
    shock_z: float | None
    factor_volatility: float | None
    quality_flags: frozenset[QualityFlag]
    data_quality: DataQuality

    @property
    def causal(self) -> bool:
        return self.artifact is not None and self.artifact.fit_end_ns <= self.as_of_ns


@dataclass(frozen=True, slots=True)
class FactorGroupSummary:
    label: str
    observations: int
    mean_first_share: float
    mean_dominance: float
    mean_residual_dispersion: float
    mean_factor_volatility: float | None
