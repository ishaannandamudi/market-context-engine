"""Bounded, event-indexed observation store with as-of availability filtering."""

from bisect import bisect_left
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass, field
from hashlib import sha256
from typing import TypeAlias, TypeVar, cast

from mce.models.market_data import (
    BarObservation,
    MarketObservation,
    QuoteObservation,
    TradeObservation,
)
from mce.models.symbol import normalize_symbol
from mce.serialization.canonical import canonical_json_bytes

Observation: TypeAlias = BarObservation | QuoteObservation | TradeObservation
ObservationT = TypeVar("ObservationT", BarObservation, QuoteObservation, TradeObservation)
OrderKey: TypeAlias = tuple[int, int, str, str, str, bytes]
EventKey: TypeAlias = tuple[int, int, str, str, str, bytes]


def observation_order_key(observation: Observation) -> OrderKey:
    """Stable replay order; canonical bytes distinguish same-time values and sequences."""
    return (
        observation.effective_timestamp_ns,
        observation.event_timestamp_ns,
        observation.source,
        observation.symbol,
        type(observation).__name__,
        canonical_json_bytes(observation),
    )


def _event_key(observation: Observation) -> EventKey:
    effective, event, source, symbol, kind, payload = observation_order_key(observation)
    return (event, effective, source, symbol, kind, payload)


@dataclass
class _Bucket:
    keys: list[EventKey] = field(default_factory=list)
    observations: list[Observation] = field(default_factory=list)


class CausalDataStore:
    """Keep at most max_observations, evicting oldest accepted observations."""

    def __init__(self, max_observations: int = 100_000) -> None:
        if max_observations <= 0:
            raise ValueError("max_observations must be positive")
        self.max_observations = max_observations
        self._buckets: dict[tuple[type[MarketObservation], str], _Bucket] = {}
        self._arrival: deque[tuple[tuple[type[MarketObservation], str], EventKey]] = deque()
        self._dedupe: set[bytes] = set()

    def __len__(self) -> int:
        return len(self._arrival)

    def causal_fingerprint(self, as_of_ns: int) -> str:
        """Hash only observations available by the requested replay clock."""

        if as_of_ns < 0:
            raise ValueError("as_of_ns must be nonnegative")
        visible = sorted(
            (
                item
                for bucket in self._buckets.values()
                for item in bucket.observations
                if item.effective_timestamp_ns <= as_of_ns
            ),
            key=observation_order_key,
        )
        digest = sha256()
        for item in visible:
            digest.update(canonical_json_bytes(item))
            digest.update(b"\n")
        return digest.hexdigest()

    def add(self, observation: Observation) -> bool:
        """Return False only for an exact repeated bar/quote; trades are never deduped."""
        if isinstance(observation, BarObservation) and observation.bar_end_ns is None:
            raise ValueError("causal bar ingestion requires bar_start_ns and bar_end_ns")
        if not isinstance(observation, (BarObservation, QuoteObservation, TradeObservation)):
            raise TypeError("unsupported observation type")
        payload = canonical_json_bytes(observation)
        dedupable = not isinstance(observation, TradeObservation)
        if dedupable and payload in self._dedupe:
            return False
        bucket_key = (type(observation), observation.symbol)
        bucket = self._buckets.setdefault(bucket_key, _Bucket())
        key = _event_key(observation)
        position = bisect_left(bucket.keys, key)
        bucket.keys.insert(position, key)
        bucket.observations.insert(position, observation)
        self._arrival.append((bucket_key, key))
        if dedupable:
            self._dedupe.add(payload)
        if len(self._arrival) > self.max_observations:
            self._evict_oldest()
        return True

    def _evict_oldest(self) -> None:
        bucket_key, key = self._arrival.popleft()
        bucket = self._buckets[bucket_key]
        position = bisect_left(bucket.keys, key)
        removed = bucket.observations.pop(position)
        bucket.keys.pop(position)
        if not bucket.keys:
            del self._buckets[bucket_key]
        if not isinstance(removed, TradeObservation):
            self._dedupe.remove(canonical_json_bytes(removed))

    def _range(
        self,
        kind: type[ObservationT],
        symbol: str,
        start_ns: int,
        end_ns: int,
        as_of_ns: int,
    ) -> tuple[ObservationT, ...]:
        if min(start_ns, end_ns, as_of_ns) < 0 or end_ns < start_ns:
            raise ValueError("invalid query time range")
        bucket = self._buckets.get((kind, normalize_symbol(symbol)))
        if bucket is None:
            return ()
        start = bisect_left(bucket.keys, (start_ns,))
        end = bisect_left(bucket.keys, (end_ns,))
        return tuple(
            cast(ObservationT, item)
            for item in bucket.observations[start:end]
            if item.effective_timestamp_ns <= as_of_ns
        )

    def bars(
        self, symbol: str, start_ns: int, end_ns: int, as_of_ns: int
    ) -> tuple[BarObservation, ...]:
        return self._range(BarObservation, symbol, start_ns, end_ns, as_of_ns)

    def trades(
        self, symbol: str, start_ns: int, end_ns: int, as_of_ns: int
    ) -> tuple[TradeObservation, ...]:
        return self._range(TradeObservation, symbol, start_ns, end_ns, as_of_ns)

    def quotes(
        self, symbol: str, start_ns: int, end_ns: int, as_of_ns: int
    ) -> tuple[QuoteObservation, ...]:
        return self._range(QuoteObservation, symbol, start_ns, end_ns, as_of_ns)

    def _latest(
        self, kind: type[MarketObservation], symbol: str, as_of_ns: int
    ) -> Observation | None:
        if as_of_ns < 0:
            raise ValueError("as_of_ns must be nonnegative")
        bucket = self._buckets.get((kind, normalize_symbol(symbol)))
        if bucket is None:
            return None
        return next(
            (
                item
                for item in reversed(bucket.observations)
                if item.effective_timestamp_ns <= as_of_ns
            ),
            None,
        )

    def latest_bar(self, symbol: str, as_of_ns: int) -> BarObservation | None:
        result = self._latest(BarObservation, symbol, as_of_ns)
        return result if isinstance(result, BarObservation) else None

    def bars_reverse(self, symbol: str, as_of_ns: int) -> Iterator[BarObservation]:
        """Iterate available bars newest event first without copying the bucket."""
        if as_of_ns < 0:
            raise ValueError("as_of_ns must be nonnegative")
        bucket = self._buckets.get((BarObservation, normalize_symbol(symbol)))
        if bucket is not None:
            for item in reversed(bucket.observations):
                if item.effective_timestamp_ns <= as_of_ns:
                    yield cast(BarObservation, item)

    def latest_quote(self, symbol: str, as_of_ns: int) -> QuoteObservation | None:
        result = self._latest(QuoteObservation, symbol, as_of_ns)
        return result if isinstance(result, QuoteObservation) else None
