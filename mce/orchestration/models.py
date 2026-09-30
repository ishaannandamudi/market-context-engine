"""Quantitative lineage and separate operational diagnostics."""

from enum import StrEnum
from hashlib import sha256

from pydantic import Field

from mce.features.factors.models import FactorContext
from mce.features.network.models import NetworkContext
from mce.features.regime.models import MarketRegimeState
from mce.models._base import FrozenModel
from mce.models.context import MarketContextSnapshot
from mce.models.enums import DataQuality
from mce.models.time import TimestampNs
from mce.serialization.canonical import canonical_json_bytes


class PipelineErrorCode(StrEnum):
    INVALID_REQUEST = "INVALID_REQUEST"
    INVALID_AS_OF = "INVALID_AS_OF"
    INSUFFICIENT_MARKET_DATA = "INSUFFICIENT_MARKET_DATA"
    PROVIDER_UNAVAILABLE = "PROVIDER_UNAVAILABLE"
    SCHEMA_MISMATCH = "SCHEMA_MISMATCH"
    MODEL_ARTIFACT_UNAVAILABLE = "MODEL_ARTIFACT_UNAVAILABLE"
    REPLAY_RANGE_INVALID = "REPLAY_RANGE_INVALID"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class PipelineError(Exception):
    def __init__(self, code: PipelineErrorCode, message: str) -> None:
        super().__init__(message)
        self.code = code


class StageStatus(StrEnum):
    OK = "ok"
    DEGRADED = "degraded"
    WITHHELD = "withheld"


class FailurePolicy(StrEnum):
    FAIL_SNAPSHOT = "fail_snapshot"
    WITHHOLD_FEATURE = "withhold_feature"
    DEGRADE_QUALITY = "degrade_quality"


class PipelineStageDiagnostic(FrozenModel):
    stage_id: str
    status: StageStatus
    policy: FailurePolicy
    duration_ns: int = Field(ge=0)
    output_quality: DataQuality | None = None
    reason_code: str | None = None


class PipelineDiagnostics(FrozenModel):
    as_of_ns: TimestampNs
    duration_ns: int = Field(ge=0)
    stage_results: tuple[PipelineStageDiagnostic, ...]
    quality: DataQuality
    config_fingerprint: str

    @property
    def withheld_stages(self) -> tuple[str, ...]:
        return tuple(
            item.stage_id for item in self.stage_results if item.status == StageStatus.WITHHELD
        )

    @property
    def degraded_stages(self) -> tuple[str, ...]:
        return tuple(
            item.stage_id for item in self.stage_results if item.status == StageStatus.DEGRADED
        )


class SnapshotLineage(FrozenModel):
    config_fingerprint: str
    market_data_fingerprint: str
    membership_fingerprints: tuple[tuple[str, str], ...] = ()
    source_fingerprints: tuple[tuple[str, str], ...] = ()
    engine_version: str
    schema_version: str


class PipelineResult(FrozenModel):
    snapshot: MarketContextSnapshot
    lineage: SnapshotLineage
    snapshot_fingerprint: str
    diagnostics: PipelineDiagnostics
    regime_state: MarketRegimeState | None = Field(default=None, exclude=True)
    factor_context: FactorContext | None = Field(default=None, exclude=True)
    network_context: NetworkContext | None = Field(default=None, exclude=True)


def snapshot_fingerprint(snapshot: MarketContextSnapshot, lineage: SnapshotLineage) -> str:
    """Operational timing and request IDs are deliberately absent."""

    return sha256(canonical_json_bytes({"snapshot": snapshot, "lineage": lineage})).hexdigest()
