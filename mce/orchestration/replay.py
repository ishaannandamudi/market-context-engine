"""Bounded deterministic replay over explicit historical clocks."""

from collections.abc import Callable
from dataclasses import replace
from datetime import timedelta
from typing import Literal
from zoneinfo import ZoneInfo

from pydantic import Field, model_validator

from mce.features.regime.models import MarketRegimeState
from mce.models._base import FrozenModel
from mce.models.market_data import BarObservation
from mce.models.time import TimestampNs, ns_to_datetime
from mce.orchestration.context import SuppliedContexts
from mce.orchestration.models import PipelineError, PipelineErrorCode, PipelineResult
from mce.orchestration.pipeline import MarketContextPipeline
from mce.replay.core import HistoricalReplay
from mce.replay.store import Observation


class ReplayRequest(FrozenModel):
    start_ns: TimestampNs
    end_ns: TimestampNs
    schema_version: Literal["1.0"] = "1.0"
    cadence: Literal["completed_bar", "n_minute", "session_close", "timestamps"]
    cadence_minutes: int | None = Field(default=None, ge=1, le=390)
    timestamps: tuple[TimestampNs, ...] = ()
    max_snapshots: int = Field(default=100, ge=1, le=1000)

    @model_validator(mode="after")
    def validate_request(self) -> "ReplayRequest":
        if self.end_ns < self.start_ns or self.end_ns - self.start_ns > 3650 * 86_400_000_000_000:
            raise ValueError("invalid replay range")
        if (self.cadence == "n_minute") != (self.cadence_minutes is not None):
            raise ValueError("n_minute cadence requires cadence_minutes only")
        if self.cadence == "timestamps":
            if not self.timestamps or tuple(sorted(set(self.timestamps))) != self.timestamps:
                raise ValueError("timestamps must be nonempty, sorted, and unique")
            if self.timestamps[0] < self.start_ns or self.timestamps[-1] > self.end_ns:
                raise ValueError("replay timestamps must lie in range")
        elif self.timestamps:
            raise ValueError("timestamps are only allowed with timestamps cadence")
        return self


class ReplayResult(FrozenModel):
    request: ReplayRequest
    results: tuple[PipelineResult, ...]


class MarketContextReplay:
    def __init__(self, pipeline: MarketContextPipeline) -> None:
        self.pipeline = pipeline

    def schedule(
        self, observations: tuple[Observation, ...], request: ReplayRequest
    ) -> tuple[int, ...]:
        if request.cadence == "timestamps":
            return request.timestamps
        if request.cadence == "completed_bar":
            broad = set(self.pipeline.config.market.universe.broad_market)
            source = self.pipeline.config.provider.source
            return tuple(
                sorted(
                    {
                        item.effective_timestamp_ns
                        for item in observations
                        if isinstance(item, BarObservation)
                        and item.symbol in broad
                        and item.source == source
                        and item.bar_start_ns is not None
                        and item.bar_end_ns is not None
                        and request.start_ns <= item.effective_timestamp_ns <= request.end_ns
                        and self.pipeline.calendar.session_for_timestamp(
                            item.bar_start_ns
                        ).session.value
                        == "regular"
                        and self.pipeline.calendar.session_for_timestamp(
                            item.bar_end_ns - 1
                        ).session.value
                        == "regular"
                    }
                )
            )
        local_zone = ZoneInfo("America/New_York")
        day = ns_to_datetime(request.start_ns).astimezone(local_zone).date()
        last = ns_to_datetime(request.end_ns).astimezone(local_zone).date()
        times: list[int] = []
        while day <= last:
            if self.pipeline.calendar.is_trading_day(day):
                candidate: tuple[int, ...]
                if request.cadence == "session_close":
                    candidate = (self.pipeline.calendar.regular_session_bounds(day)[1],)
                else:
                    assert request.cadence_minutes is not None
                    candidate = self.pipeline.calendar.snapshot_times(
                        day, request.cadence_minutes * 60_000_000_000
                    )
                times.extend(
                    timestamp
                    for timestamp in candidate
                    if request.start_ns <= timestamp <= request.end_ns
                )
                if len(times) > request.max_snapshots:
                    raise PipelineError(
                        PipelineErrorCode.REPLAY_RANGE_INVALID, "too many snapshots"
                    )
            day += timedelta(days=1)
        return tuple(times)

    def run(
        self,
        observations: tuple[Observation, ...],
        request: ReplayRequest,
        supplied_at: Callable[[int], SuppliedContexts] | None = None,
    ) -> ReplayResult:
        times = self.schedule(observations, request)
        if len(times) > request.max_snapshots:
            raise PipelineError(PipelineErrorCode.REPLAY_RANGE_INVALID, "too many snapshots")
        replay = HistoricalReplay(
            observations, max_observations=self.pipeline.config.replay.max_observations
        )
        outputs: list[PipelineResult] = []
        regimes: tuple[MarketRegimeState, ...] = ()
        previous_factor = None
        previous_network = None
        for timestamp in times:
            replay.advance_to(timestamp)
            supplied = supplied_at(timestamp) if supplied_at is not None else SuppliedContexts()
            current = replace(
                supplied,
                regime_history=(*supplied.regime_history, *regimes),
                previous_factor=supplied.previous_factor or previous_factor,
                previous_network=supplied.previous_network or previous_network,
            )
            result = self.pipeline.compute_snapshot(timestamp, replay.store, current)
            outputs.append(result)
            if result.regime_state is not None:
                regimes = (*regimes, result.regime_state)
            previous_factor = result.factor_context or previous_factor
            previous_network = result.network_context or previous_network
        return ReplayResult(request=request, results=tuple(outputs))
