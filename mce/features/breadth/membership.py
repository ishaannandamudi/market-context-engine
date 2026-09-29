"""Immutable, deterministic point-in-time breadth membership."""

from collections import defaultdict
from dataclasses import dataclass
from typing import Protocol

from pydantic import Field, model_validator

from mce.models._base import FrozenModel
from mce.models.enums import DataQuality
from mce.models.quality import QualityTaggedModel, SourceId
from mce.models.symbol import Symbol
from mce.models.time import TimestampNs


class UniverseMembership(FrozenModel):
    symbol: Symbol
    effective_from_ns: TimestampNs = 0
    effective_to_ns: TimestampNs | None = None
    available_at_ns: TimestampNs = 0
    weight: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    source: SourceId = "config"

    @model_validator(mode="after")
    def validate_interval(self) -> "UniverseMembership":
        if self.effective_to_ns is not None and self.effective_to_ns <= self.effective_from_ns:
            raise ValueError("membership end must follow its start")
        return self


class UniverseSnapshot(QualityTaggedModel):
    as_of_ns: TimestampNs
    members: tuple[UniverseMembership, ...]
    source: SourceId
    source_quality: DataQuality

    @model_validator(mode="after")
    def validate_members(self) -> "UniverseSnapshot":
        if len({member.symbol for member in self.members}) != len(self.members):
            raise ValueError("active universe symbols must be unique")
        if any(
            member.effective_from_ns > self.as_of_ns
            or member.available_at_ns > self.as_of_ns
            or (member.effective_to_ns is not None and self.as_of_ns >= member.effective_to_ns)
            for member in self.members
        ):
            raise ValueError("universe snapshot contains inactive or unavailable membership")
        object.__setattr__(
            self, "members", tuple(sorted(self.members, key=lambda item: item.symbol))
        )
        return self


class UniverseProvider(Protocol):
    def members(self, as_of_ns: int) -> UniverseSnapshot: ...


@dataclass(frozen=True, init=False)
class MembershipTimeline:
    """Reject overlapping intervals; updates become visible only at their own as-of time."""

    records: tuple[UniverseMembership, ...]
    source: str
    source_quality: DataQuality

    def __init__(
        self,
        records: tuple[UniverseMembership, ...],
        *,
        source: SourceId = "config",
        source_quality: DataQuality = DataQuality.GOOD,
    ) -> None:
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "source_quality", source_quality)
        by_symbol: dict[str, list[UniverseMembership]] = defaultdict(list)
        for record in records:
            by_symbol[record.symbol].append(record)
        for group in by_symbol.values():
            group.sort(key=lambda member: member.effective_from_ns)
            for earlier, later in zip(group, group[1:], strict=False):
                if (
                    earlier.effective_to_ns is None
                    or earlier.effective_to_ns > later.effective_from_ns
                ):
                    raise ValueError("membership intervals for one symbol must not overlap")
        object.__setattr__(
            self,
            "records",
            tuple(sorted(records, key=lambda item: (item.symbol, item.effective_from_ns))),
        )

    @classmethod
    def from_symbols(cls, symbols: tuple[str, ...]) -> "MembershipTimeline":
        return cls(tuple(UniverseMembership(symbol=symbol) for symbol in symbols))

    def members(self, as_of_ns: int) -> UniverseSnapshot:
        if as_of_ns < 0:
            raise ValueError("as_of_ns must be nonnegative")
        active = tuple(
            item
            for item in self.records
            if item.available_at_ns <= as_of_ns
            and item.effective_from_ns <= as_of_ns
            and (item.effective_to_ns is None or as_of_ns < item.effective_to_ns)
        )
        return UniverseSnapshot(
            as_of_ns=as_of_ns,
            members=active,
            source=self.source,
            source_quality=self.source_quality,
        )
