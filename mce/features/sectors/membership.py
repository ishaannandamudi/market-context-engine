"""Point-in-time sector assignment, reclassification, and weight history."""

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

from pydantic import model_validator

from mce.features.breadth.membership import UniverseMembership
from mce.models.enums import DataQuality
from mce.models.quality import QualityTaggedModel, SourceId
from mce.models.sector import SectorName
from mce.models.time import TimestampNs


class SectorMembership(UniverseMembership):
    sector: SectorName


class SectorUniverseSnapshot(QualityTaggedModel):
    as_of_ns: TimestampNs
    members: tuple[SectorMembership, ...]
    source: SourceId
    source_quality: DataQuality
    exclusive: bool = True

    @model_validator(mode="after")
    def validate_snapshot(self) -> "SectorUniverseSnapshot":
        assignments = [(item.sector, item.symbol) for item in self.members]
        if len(assignments) != len(set(assignments)):
            raise ValueError("active sector assignments must be unique")
        if self.exclusive and len({item.symbol for item in self.members}) != len(self.members):
            raise ValueError("exclusive sector snapshot assigns one symbol twice")
        if any(
            item.effective_from_ns > self.as_of_ns
            or item.available_at_ns > self.as_of_ns
            or (item.effective_to_ns is not None and self.as_of_ns >= item.effective_to_ns)
            for item in self.members
        ):
            raise ValueError("sector snapshot contains inactive or unavailable membership")
        object.__setattr__(
            self,
            "members",
            tuple(sorted(self.members, key=lambda item: (item.sector, item.symbol))),
        )
        return self


class SectorUniverseProvider(Protocol):
    def members(self, as_of_ns: int) -> SectorUniverseSnapshot: ...


@dataclass(frozen=True, init=False)
class SectorMembershipTimeline:
    records: tuple[SectorMembership, ...]
    source: str
    source_quality: DataQuality
    exclusive: bool

    def __init__(
        self,
        records: tuple[SectorMembership, ...],
        *,
        source: SourceId = "config",
        source_quality: DataQuality = DataQuality.GOOD,
        exclusive: bool = True,
    ) -> None:
        by_symbol: dict[str, list[SectorMembership]] = defaultdict(list)
        by_assignment: dict[tuple[str, str], list[SectorMembership]] = defaultdict(list)
        for item in records:
            by_symbol[item.symbol].append(item)
            by_assignment[(item.symbol, item.sector)].append(item)
        groups = by_symbol.values() if exclusive else by_assignment.values()
        for group in groups:
            group.sort(key=lambda item: (item.effective_from_ns, item.sector))
            for earlier, later in zip(group, group[1:], strict=False):
                overlapping = (
                    earlier.effective_to_ns is None
                    or earlier.effective_to_ns > later.effective_from_ns
                )
                if overlapping:
                    raise ValueError("sector membership intervals overlap")
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "source_quality", source_quality)
        object.__setattr__(self, "exclusive", exclusive)
        object.__setattr__(
            self,
            "records",
            tuple(
                sorted(records, key=lambda item: (item.sector, item.symbol, item.effective_from_ns))
            ),
        )

    @classmethod
    def from_groups(
        cls, groups: Mapping[str, tuple[str, ...]], *, exclusive: bool = True
    ) -> "SectorMembershipTimeline":
        return cls(
            tuple(
                SectorMembership(sector=sector, symbol=symbol)
                for sector, symbols in sorted(groups.items())
                for symbol in symbols
            ),
            exclusive=exclusive,
        )

    def members(self, as_of_ns: int) -> SectorUniverseSnapshot:
        if as_of_ns < 0:
            raise ValueError("as_of_ns must be nonnegative")
        return SectorUniverseSnapshot(
            as_of_ns=as_of_ns,
            members=tuple(
                item
                for item in self.records
                if item.available_at_ns <= as_of_ns
                and item.effective_from_ns <= as_of_ns
                and (item.effective_to_ns is None or as_of_ns < item.effective_to_ns)
            ),
            source=self.source,
            source_quality=self.source_quality,
            exclusive=self.exclusive,
        )
