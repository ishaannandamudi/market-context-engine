"""Explicit-clock replay of canonical observations in causal availability order."""

from collections.abc import Iterable

from mce.errors import CausalityError
from mce.models.market_data import BarObservation
from mce.replay.store import CausalDataStore, Observation, observation_order_key


class ReplayClock:
    def __init__(self, start_ns: int = 0) -> None:
        if start_ns < 0:
            raise ValueError("start_ns must be nonnegative")
        self._as_of_ns = start_ns

    @property
    def as_of_ns(self) -> int:
        return self._as_of_ns

    def advance_to(self, as_of_ns: int) -> None:
        if as_of_ns < self._as_of_ns:
            raise CausalityError("replay clock cannot move backward; reset explicitly")
        self._as_of_ns = as_of_ns

    def reset(self, as_of_ns: int = 0) -> None:
        if as_of_ns < 0:
            raise ValueError("as_of_ns must be nonnegative")
        self._as_of_ns = as_of_ns


class HistoricalReplay:
    """Sort an entire historical batch once, then advance a monotone clock."""

    def __init__(
        self,
        observations: Iterable[Observation],
        max_observations: int = 100_000,
        start_ns: int = 0,
    ) -> None:
        self._observations = tuple(sorted(observations, key=observation_order_key))
        for observation in self._observations:
            if isinstance(observation, BarObservation) and observation.bar_end_ns is None:
                raise ValueError("historical replay requires completed bar bounds")
        self.clock = ReplayClock()
        self.store = CausalDataStore(max_observations)
        self._cursor = 0
        self.advance_to(start_ns)

    def advance_to(self, as_of_ns: int) -> int:
        self.clock.advance_to(as_of_ns)
        accepted = 0
        while (
            self._cursor < len(self._observations)
            and self._observations[self._cursor].effective_timestamp_ns <= as_of_ns
        ):
            if self.store.add(self._observations[self._cursor]):
                accepted += 1
            self._cursor += 1
        return accepted

    def reset(self, to_ns: int = 0) -> int:
        self.clock.reset()
        self.store = CausalDataStore(self.store.max_observations)
        self._cursor = 0
        return self.advance_to(to_ns)
