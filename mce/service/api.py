"""Validated, bounded service requests with stable, safe errors."""

import json
import logging
from collections.abc import Callable
from threading import Lock
from time import perf_counter_ns
from typing import Any, Literal

from pydantic import ValidationError

from mce.features.macro.contracts import MACRO_CONTEXT_SCHEMA_VERSION
from mce.features.options_context.contracts import OPTIONS_CONTEXT_SCHEMA_VERSION
from mce.models._base import FrozenModel
from mce.models.context import MarketContextSnapshot
from mce.models.enums import MarketSession
from mce.models.market_data import BarObservation
from mce.models.time import TimestampNs
from mce.orchestration.context import SuppliedContexts
from mce.orchestration.models import PipelineError, PipelineErrorCode
from mce.orchestration.pipeline import MarketContextPipeline
from mce.orchestration.replay import MarketContextReplay, ReplayRequest
from mce.providers.base import ProviderStatus
from mce.replay.core import HistoricalReplay
from mce.replay.store import Observation
from mce.serialization.canonical import canonical_json_bytes
from mce.statistical.features import STATISTICAL_FEATURE_SCHEMA
from mce.version import SCHEMA_VERSION, __version__

_LOG = logging.getLogger(__name__)
MAX_REQUEST_BYTES = 65_536
MAX_RESPONSE_BYTES = 8 * 1024 * 1024
FEATURE_FAMILIES = (
    "trend",
    "volatility",
    "breadth",
    "sectors",
    "correlation",
    "dispersion",
    "liquidity_stress",
    "regime",
    "regime_transition",
    "statistical",
    "factors",
    "network",
    "seasonality",
    "options",
    "macro",
    "ensemble",
)
_PUBLIC_ERRORS = {
    PipelineErrorCode.INVALID_REQUEST: "invalid request",
    PipelineErrorCode.INVALID_AS_OF: "invalid snapshot clock",
    PipelineErrorCode.INSUFFICIENT_MARKET_DATA: "insufficient market data",
    PipelineErrorCode.PROVIDER_UNAVAILABLE: "provider unavailable",
    PipelineErrorCode.SCHEMA_MISMATCH: "unsupported schema version",
    PipelineErrorCode.MODEL_ARTIFACT_UNAVAILABLE: "model artifact unavailable",
    PipelineErrorCode.REPLAY_RANGE_INVALID: "invalid replay range or size",
    PipelineErrorCode.INTERNAL_ERROR: "internal service error",
}


class SnapshotRequest(FrozenModel):
    as_of_ns: TimestampNs
    schema_version: Literal["1.0"] = "1.0"
    include_diagnostics: bool = False


class _Counters:
    def __init__(self) -> None:
        self._lock = Lock()
        self.snapshots_total = 0
        self.snapshots_failed = 0
        self.degraded_snapshots = 0
        self.replay_snapshots = 0

    def add(self, name: str, count: int = 1) -> None:
        with self._lock:
            setattr(self, name, getattr(self, name) + count)

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return {
                "snapshots_total": self.snapshots_total,
                "snapshots_failed": self.snapshots_failed,
                "degraded_snapshots": self.degraded_snapshots,
                "replay_snapshots": self.replay_snapshots,
            }


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


class MarketContextService:
    """The configured observation batch is the only request-accessible source."""

    def __init__(
        self,
        pipeline: MarketContextPipeline,
        observations: tuple[Observation, ...],
        supplied_at: Callable[[int], SuppliedContexts] | None = None,
    ) -> None:
        self.pipeline = pipeline
        self.observations = observations
        self.supplied_at = supplied_at
        self.metrics = _Counters()
        broad = set(self.pipeline.config.market.universe.broad_market)
        source_id = self.pipeline.config.provider.source
        self.has_market_bar = any(
            isinstance(item, BarObservation)
            and item.symbol in broad
            and item.source == source_id
            and item.bar_start_ns is not None
            and item.bar_end_ns is not None
            and self.pipeline.calendar.session_for_timestamp(item.bar_start_ns).session
            == MarketSession.REGULAR
            and self.pipeline.calendar.session_for_timestamp(item.bar_end_ns - 1).session
            == MarketSession.REGULAR
            for item in self.observations
        )

    def readiness(self) -> dict[str, object]:
        provider = self.pipeline.providers.market_health
        market = (
            provider.health().status
            if provider is not None
            else (ProviderStatus.HEALTHY if self.has_market_bar else ProviderStatus.UNAVAILABLE)
        )
        optional = {}
        for name, source in (
            ("options", self.pipeline.providers.options_health),
            ("macro", self.pipeline.providers.macro_health),
        ):
            optional[name] = (
                source.health().status.value if source is not None else "not_configured"
            )
        ready = market in (ProviderStatus.HEALTHY, ProviderStatus.DEGRADED) and self.has_market_bar
        return {"ready": ready, "market": market.value, "optional": optional}

    @staticmethod
    def _response(status: int, value: object) -> tuple[int, bytes]:
        payload = canonical_json_bytes(value)
        if len(payload) > MAX_RESPONSE_BYTES:
            raise PipelineError(PipelineErrorCode.REPLAY_RANGE_INVALID, "response too large")
        return status, payload

    @staticmethod
    def _error(status: int, code: PipelineErrorCode, message: str) -> tuple[int, bytes]:
        return MarketContextService._response(status, {"error": {"code": code, "message": message}})

    def _payload(self, body: bytes) -> dict[str, Any]:
        if len(body) > MAX_REQUEST_BYTES:
            raise PipelineError(PipelineErrorCode.INVALID_REQUEST, "request payload too large")
        value = json.loads(body.decode("utf-8"), object_pairs_hook=_unique_object)
        if not isinstance(value, dict):
            raise ValueError("JSON object required")
        if value.get("schema_version", SCHEMA_VERSION) != SCHEMA_VERSION:
            raise PipelineError(PipelineErrorCode.SCHEMA_MISMATCH, "unsupported snapshot schema")
        return value

    def handle(self, method: str, path: str, body: bytes = b"") -> tuple[int, bytes]:
        started = perf_counter_ns()
        try:
            if method == "GET" and path == "/health":
                return self._response(200, {"status": "ok"})
            if method == "GET" and path == "/ready":
                readiness = self.readiness()
                return self._response(200 if readiness["ready"] else 503, readiness)
            if method == "GET" and path == "/version":
                return self._response(
                    200,
                    {
                        "engine_version": __version__,
                        "snapshot_schema_version": SCHEMA_VERSION,
                        "statistical_feature_schema_version": STATISTICAL_FEATURE_SCHEMA.version,
                        "options_context_schema_version": OPTIONS_CONTEXT_SCHEMA_VERSION,
                        "macro_context_schema_version": MACRO_CONTEXT_SCHEMA_VERSION,
                        "feature_families": FEATURE_FAMILIES,
                    },
                )
            if method == "GET" and path == "/api/context/schema":
                return self._response(
                    200,
                    {
                        "schema_version": SCHEMA_VERSION,
                        "snapshot_fields": tuple(sorted(MarketContextSnapshot.model_fields)),
                    },
                )
            if method == "POST" and path == "/api/context/snapshot":
                input_payload = self._payload(body)
                raw_clock = input_payload.get("as_of_ns")
                if type(raw_clock) is int and raw_clock < 0:
                    raise PipelineError(
                        PipelineErrorCode.INVALID_AS_OF, "as_of_ns must be nonnegative"
                    )
                request = SnapshotRequest.model_validate(input_payload)
                replay = HistoricalReplay(
                    self.observations,
                    max_observations=self.pipeline.config.replay.max_observations,
                    start_ns=request.as_of_ns,
                )
                supplied = self.supplied_at(request.as_of_ns) if self.supplied_at else None
                result = self.pipeline.compute_snapshot(request.as_of_ns, replay.store, supplied)
                self.metrics.add("snapshots_total")
                if result.snapshot.data_quality.value != "good":
                    self.metrics.add("degraded_snapshots")
                _LOG.info(
                    "snapshot_complete",
                    extra={
                        "as_of_ns": request.as_of_ns,
                        "duration_ms": (perf_counter_ns() - started) / 1e6,
                        "quality": result.snapshot.data_quality.value,
                    },
                )
                payload: dict[str, object] = {
                    "snapshot": result.snapshot,
                    "lineage": result.lineage,
                    "snapshot_fingerprint": result.snapshot_fingerprint,
                }
                if request.include_diagnostics:
                    payload["diagnostics"] = result.diagnostics
                return self._response(200, payload)
            if method == "POST" and path == "/api/context/replay":
                input_payload = self._payload(body)
                start_ns = input_payload.get("start_ns")
                end_ns = input_payload.get("end_ns")
                if (
                    type(start_ns) is int
                    and type(end_ns) is int
                    and (start_ns < 0 or end_ns < start_ns)
                ):
                    raise PipelineError(
                        PipelineErrorCode.REPLAY_RANGE_INVALID, "invalid replay range"
                    )
                replay_request = ReplayRequest.model_validate_json(
                    canonical_json_bytes(input_payload)
                )
                if replay_request.max_snapshots > 100:
                    raise PipelineError(
                        PipelineErrorCode.REPLAY_RANGE_INVALID, "service replay limit is 100"
                    )
                replay_result = MarketContextReplay(self.pipeline).run(
                    self.observations, replay_request, self.supplied_at
                )
                self.metrics.add("replay_snapshots", len(replay_result.results))
                _LOG.info(
                    "replay_complete",
                    extra={
                        "as_of_ns": replay_request.end_ns,
                        "duration_ms": (perf_counter_ns() - started) / 1e6,
                        "count": len(replay_result.results),
                    },
                )
                return self._response(
                    200,
                    {
                        "schema_version": SCHEMA_VERSION,
                        "snapshots": tuple(item.snapshot for item in replay_result.results),
                        "fingerprints": tuple(
                            item.snapshot_fingerprint for item in replay_result.results
                        ),
                    },
                )
            return self._error(404, PipelineErrorCode.INVALID_REQUEST, "unknown endpoint")
        except PipelineError as error:
            self.metrics.add("snapshots_failed")
            status = 413 if str(error) == "request payload too large" else 422
            if error.code == PipelineErrorCode.PROVIDER_UNAVAILABLE:
                status = 503
            _LOG.warning("request_rejected", extra={"status": status, "code": error.code.value})
            message = "request payload too large" if status == 413 else _PUBLIC_ERRORS[error.code]
            return self._error(status, error.code, message)
        except (ValidationError, ValueError, UnicodeDecodeError, json.JSONDecodeError):
            self.metrics.add("snapshots_failed")
            _LOG.warning("invalid_request")
            return self._error(422, PipelineErrorCode.INVALID_REQUEST, "invalid request or context")
        except Exception:
            self.metrics.add("snapshots_failed")
            _LOG.error("snapshot_service_failure")
            return self._error(500, PipelineErrorCode.INTERNAL_ERROR, "internal service error")
