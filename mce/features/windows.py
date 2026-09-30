"""Typed windows and causal completed-bar selection."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from itertools import islice
from typing import TYPE_CHECKING

from pydantic import Field, field_validator, model_validator

from mce.errors import CalendarRangeError
from mce.models._base import FrozenModel
from mce.models.enums import MarketSession
from mce.models.market_data import BarObservation
from mce.replay.store import CausalDataStore

if TYPE_CHECKING:
    from mce.calendar.service import MarketSessionService


class WindowUnit(StrEnum):
    ELAPSED = "elapsed"
    OBSERVATIONS = "observations"
    SESSION = "session"
    TRADING_DAYS = "trading_days"


class WindowSpec(FrozenModel):
    name: str = Field(min_length=1, pattern=r"^[a-z0-9][a-z0-9_]*$")
    unit: WindowUnit
    length: int = Field(gt=0)

    @field_validator("unit", mode="before")
    @classmethod
    def parse_unit(cls, value: object) -> WindowUnit:
        if isinstance(value, WindowUnit):
            return value
        if isinstance(value, str):
            return WindowUnit(value)
        raise ValueError("window unit must be a known string")

    @model_validator(mode="after")
    def validate_unit(self) -> WindowSpec:
        if self.unit == WindowUnit.SESSION and self.length != 1:
            raise ValueError("session window length must be 1")
        return self


@dataclass(frozen=True)
class WindowSample:
    """Reference is observed, never synthesized; bars end no later than as_of."""

    bars: tuple[BarObservation, ...]
    reference_price: float | None
    expected_count: int | None
    coverage_ratio: float | None

    @property
    def sample_count(self) -> int:
        return len(self.bars)


class FeatureWindowReader:
    def __init__(
        self,
        store: CausalDataStore,
        calendar: MarketSessionService,
        cadence_ns: int,
        source: str | None = None,
    ) -> None:
        if cadence_ns <= 0:
            raise ValueError("cadence_ns must be positive")
        self.store = store
        self.calendar = calendar
        self.cadence_ns = cadence_ns
        self.source = source
        self._regular_cache: dict[tuple[int, int], bool] = {}

    def _regular_interval(self, start_ns: int, end_ns: int) -> bool:
        key = (start_ns, end_ns)
        cached = self._regular_cache.get(key)
        if cached is not None:
            return cached
        result = (
            self.calendar.session_for_timestamp(start_ns).session == MarketSession.REGULAR
            and self.calendar.session_for_timestamp(end_ns - 1).session == MarketSession.REGULAR
        )
        if len(self._regular_cache) >= 4096:
            self._regular_cache.clear()
        self._regular_cache[key] = result
        return result

    def _is_regular(self, bar: BarObservation, as_of_ns: int) -> bool:
        return (
            (self.source is None or bar.source == self.source)
            and bar.bar_start_ns is not None
            and bar.bar_end_ns is not None
            and bar.bar_end_ns <= as_of_ns
            and self._regular_interval(bar.bar_start_ns, bar.bar_end_ns)
        )

    def _regular(
        self, bars: tuple[BarObservation, ...], as_of_ns: int
    ) -> tuple[BarObservation, ...]:
        return tuple(bar for bar in bars if self._is_regular(bar, as_of_ns))

    def current_bar(self, symbol: str, as_of_ns: int) -> BarObservation | None:
        return next(
            (
                bar
                for bar in self.store.bars_reverse(symbol, as_of_ns)
                if self._is_regular(bar, as_of_ns)
            ),
            None,
        )

    def _session_bars(self, symbol: str, day: date, as_of_ns: int) -> tuple[BarObservation, ...]:
        opened, closed = self.calendar.regular_session_bounds(day)
        return self._regular(self.store.bars(symbol, opened, closed + 1, as_of_ns), as_of_ns)

    def select(self, symbol: str, as_of_ns: int, spec: WindowSpec) -> WindowSample:
        if as_of_ns < 0:
            raise ValueError("as_of_ns must be nonnegative")
        current = self.current_bar(symbol, as_of_ns)
        if current is None or current.bar_end_ns is None:
            return WindowSample((), None, None, None)
        assert current.bar_start_ns is not None
        session = self.calendar.session_for_timestamp(current.bar_start_ns)
        if session.trading_date is None:
            return WindowSample((), None, None, None)
        if spec.unit == WindowUnit.ELAPSED:
            boundary = max(0, as_of_ns - spec.length)
            maximum_reference_lag = min(self.cadence_ns, spec.length)
            candidate = self._regular(
                self.store.bars(
                    symbol,
                    max(0, boundary - maximum_reference_lag - self.cadence_ns),
                    as_of_ns + 1,
                    as_of_ns,
                ),
                as_of_ns,
            )
            anchors = [
                bar
                for bar in candidate
                if bar.bar_end_ns is not None
                and boundary - maximum_reference_lag <= bar.bar_end_ns <= boundary
            ]
            anchor = anchors[-1] if anchors else None
            bars = tuple(
                bar
                for bar in candidate
                if bar.bar_end_ns is not None and bar.bar_end_ns >= boundary
            )
            expected = spec.length // self.cadence_ns + 1
            observed = len(bars) + int(anchor is not None and anchor not in bars)
            return WindowSample(
                bars,
                anchor.close if anchor else None,
                expected,
                min(1.0, observed / expected) if expected else None,
            )
        if spec.unit == WindowUnit.OBSERVATIONS:
            selected = tuple(
                reversed(
                    tuple(
                        islice(
                            (
                                bar
                                for bar in self.store.bars_reverse(symbol, as_of_ns)
                                if self._is_regular(bar, as_of_ns)
                            ),
                            spec.length,
                        )
                    )
                )
            )
            return WindowSample(
                selected,
                selected[0].close if selected else None,
                spec.length,
                len(selected) / spec.length,
            )
        if spec.unit == WindowUnit.SESSION:
            opened, _ = self.calendar.regular_session_bounds(session.trading_date)
            bars = self._session_bars(symbol, session.trading_date, as_of_ns)
            expected = max(1, (min(as_of_ns, current.bar_end_ns) - opened) // self.cadence_ns)
            return WindowSample(
                bars,
                bars[0].open if bars else None,
                expected,
                min(1.0, len(bars) / expected),
            )
        reference_day = session.trading_date
        try:
            for _ in range(spec.length):
                reference_day = self.calendar.previous_trading_day(reference_day)
        except CalendarRangeError:
            return WindowSample((current,), None, 2, 0.5)
        prior = self._session_bars(symbol, reference_day, as_of_ns)
        _, prior_close = self.calendar.regular_session_bounds(reference_day)
        completed = [bar for bar in prior if bar.bar_end_ns == prior_close]
        return WindowSample(
            (current,), completed[-1].close if completed else None, 2, 1.0 if completed else 0.5
        )
