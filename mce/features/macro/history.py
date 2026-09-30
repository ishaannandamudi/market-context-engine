"""Immutable availability-indexed vintages and point-in-time event calendars."""

from bisect import bisect_right
from dataclasses import dataclass
from itertools import groupby
from types import MappingProxyType
from typing import Generic, TypeVar

from mce.features.macro.contracts import (
    MacroContextInput,
    MacroObservation,
    MacroRelease,
    ScheduledMacroEvent,
)

RecordT = TypeVar("RecordT", MacroObservation, MacroRelease)
SeriesKey = tuple[str, str]
EventKey = tuple[str, str]


@dataclass(frozen=True)
class VintageTimeline(Generic[RecordT]):
    clocks: tuple[int, ...]
    states: tuple[tuple[RecordT, ...], ...]

    @classmethod
    def build(cls, records: tuple[tuple[int, RecordT], ...]) -> "VintageTimeline[RecordT]":
        clocks: list[int] = []
        states: list[tuple[RecordT, ...]] = []
        active: dict[int, RecordT] = {}
        ordered = sorted(
            records, key=lambda pair: (pair[0], pair[1].period_end_ns, pair[1].vintage_at_ns)
        )
        for clock, group in groupby(ordered, key=lambda pair: pair[0]):
            for _, record in group:
                previous = active.get(record.period_end_ns)
                if (
                    previous is not None
                    and record.vintage_at_ns == previous.vintage_at_ns
                    and record != previous
                ):
                    raise ValueError("ambiguous same-vintage macro revision")
                if previous is None or record.vintage_at_ns > previous.vintage_at_ns:
                    active[record.period_end_ns] = record
            clocks.append(clock)
            states.append(tuple(active[period] for period in sorted(active)))
        return cls(tuple(clocks), tuple(states))

    def at(self, as_of_ns: int) -> tuple[RecordT, ...]:
        position = bisect_right(self.clocks, as_of_ns) - 1
        return self.states[position] if position >= 0 else ()

    def latest(self, as_of_ns: int) -> RecordT | None:
        records = self.at(as_of_ns)
        return records[-1] if records else None

    def prior(self, period_end_ns: int, as_of_ns: int, limit: int) -> tuple[RecordT, ...]:
        if limit <= 0:
            raise ValueError("history limit must be positive")
        records = self.at(as_of_ns)
        ends = tuple(record.period_end_ns for record in records)
        return records[
            max(0, bisect_right(ends, period_end_ns - 1) - limit) : bisect_right(
                ends, period_end_ns - 1
            )
        ]


@dataclass(frozen=True)
class CalendarState:
    events: tuple[ScheduledMacroEvent, ...]
    times: tuple[int, ...]

    def upcoming(
        self, as_of_ns: int, horizon_ns: int | None = None
    ) -> tuple[ScheduledMacroEvent, ...]:
        start = bisect_right(self.times, as_of_ns - 1)
        end = (
            len(self.events)
            if horizon_ns is None
            else bisect_right(self.times, as_of_ns + horizon_ns)
        )
        return self.events[start:end]


class MacroHistoryIndex:
    """Build once; per-snapshot series/calendar lookup uses indexed immutable states."""

    def __init__(self, inputs: tuple[MacroContextInput, ...]) -> None:
        unique_batches: dict[tuple[str, int, int], MacroContextInput] = {}
        for batch in inputs:
            batch_key = (batch.source, batch.as_of_ns, batch.available_at_ns)
            previous = unique_batches.get(batch_key)
            if previous is not None and previous != batch:
                raise ValueError("ambiguous same-clock macro source revision")
            unique_batches[batch_key] = batch
        batches = tuple(
            sorted(
                unique_batches.values(),
                key=lambda item: (item.available_at_ns, item.source, item.as_of_ns),
            )
        )
        source_groups: dict[str, list[MacroContextInput]] = {}
        observations: dict[SeriesKey, list[tuple[int, MacroObservation]]] = {}
        releases: dict[SeriesKey, list[tuple[int, MacroRelease]]] = {}
        series_contracts: dict[SeriesKey, tuple[object, object, object, int | None]] = {}
        event_versions: dict[tuple[EventKey, int], ScheduledMacroEvent] = {}
        calendar_known_clocks: list[int] = []
        for batch in batches:
            source_groups.setdefault(batch.source, []).append(batch)
            if batch.calendar_coverage_declared or batch.scheduled_events:
                calendar_known_clocks.append(batch.available_at_ns)
            for record in batch.observations:
                series_key = (record.source, record.series_id)
                descriptor = (
                    record.category,
                    record.unit,
                    record.cadence,
                    record.tenor_months,
                )
                prior = series_contracts.get(series_key)
                if prior is not None and prior[:3] != descriptor[:3]:
                    raise ValueError("macro series category, unit, or cadence changed")
                if prior is not None and prior[3] is not None and prior[3] != record.tenor_months:
                    raise ValueError("Treasury series tenor changed")
                series_contracts[series_key] = descriptor
                observations.setdefault(series_key, []).append((batch.available_at_ns, record))
            for release in batch.releases:
                series_key = (release.source, release.series_id)
                descriptor = (release.category, release.unit, release.cadence, None)
                prior = series_contracts.get(series_key)
                if prior is not None and prior[:3] != descriptor[:3]:
                    raise ValueError("macro series category, unit, or cadence changed")
                if prior is None:
                    series_contracts[series_key] = descriptor
                releases.setdefault(series_key, []).append((batch.available_at_ns, release))
            for event in batch.scheduled_events:
                event_version_key = ((event.source, event.event_id), event.calendar_available_at_ns)
                prior_event = event_versions.get(event_version_key)
                if prior_event is not None and prior_event != event:
                    raise ValueError("ambiguous same-clock event schedule")
                event_versions[event_version_key] = event
        self.series_keys = tuple(sorted(series_contracts))
        self.observations = MappingProxyType(
            {key: VintageTimeline.build(tuple(values)) for key, values in observations.items()}
        )
        self.releases = MappingProxyType(
            {key: VintageTimeline.build(tuple(values)) for key, values in releases.items()}
        )
        self.sources = MappingProxyType(
            {name: tuple(values) for name, values in source_groups.items()}
        )
        self.source_clocks = MappingProxyType(
            {
                name: tuple(item.available_at_ns for item in values)
                for name, values in source_groups.items()
            }
        )
        self.calendar_known_at_ns = min(calendar_known_clocks) if calendar_known_clocks else None
        changes = sorted(
            (
                (max(event.calendar_available_at_ns, batch.available_at_ns), event)
                for batch in batches
                for event in batch.scheduled_events
            ),
            key=lambda pair: (
                pair[0],
                pair[1].source,
                pair[1].event_id,
                pair[1].calendar_available_at_ns,
            ),
        )
        active: dict[EventKey, ScheduledMacroEvent] = {}
        clocks: list[int] = []
        states: list[CalendarState] = []
        for clock, group in groupby(changes, key=lambda pair: pair[0]):
            for _, event in group:
                active_key = (event.source, event.event_id)
                prior_event = active.get(active_key)
                if (
                    prior_event is not None
                    and event.calendar_available_at_ns < prior_event.calendar_available_at_ns
                ):
                    continue
                if (
                    prior_event is not None
                    and event.calendar_available_at_ns == prior_event.calendar_available_at_ns
                    and event != prior_event
                ):
                    raise ValueError("ambiguous event schedule revision")
                active[active_key] = event
            ordered = tuple(
                sorted(
                    (event for event in active.values() if event.status == "scheduled"),
                    key=lambda item: (item.scheduled_at_ns, item.source, item.event_id),
                )
            )
            clocks.append(clock)
            states.append(CalendarState(ordered, tuple(item.scheduled_at_ns for item in ordered)))
        self.calendar_clocks = tuple(clocks)
        self.calendar_states = tuple(states)

    def visible_sources(self, as_of_ns: int) -> tuple[MacroContextInput, ...]:
        visible = []
        for name in sorted(self.sources):
            position = bisect_right(self.source_clocks[name], as_of_ns) - 1
            if position >= 0:
                visible.append(self.sources[name][position])
        return tuple(visible)

    def observation(self, key: SeriesKey, as_of_ns: int) -> MacroObservation | None:
        timeline = self.observations.get(key)
        return timeline.latest(as_of_ns) if timeline is not None else None

    def observation_state(self, key: SeriesKey, as_of_ns: int) -> tuple[MacroObservation, ...]:
        timeline = self.observations.get(key)
        return timeline.at(as_of_ns) if timeline is not None else ()

    def prior_observations(
        self, key: SeriesKey, period_end_ns: int, as_of_ns: int, limit: int
    ) -> tuple[MacroObservation, ...]:
        timeline = self.observations.get(key)
        return timeline.prior(period_end_ns, as_of_ns, limit) if timeline is not None else ()

    def release(self, key: SeriesKey, as_of_ns: int) -> MacroRelease | None:
        timeline = self.releases.get(key)
        return timeline.latest(as_of_ns) if timeline is not None else None

    def release_state(self, key: SeriesKey, as_of_ns: int) -> tuple[MacroRelease, ...]:
        timeline = self.releases.get(key)
        return timeline.at(as_of_ns) if timeline is not None else ()

    def prior_releases(
        self, key: SeriesKey, period_end_ns: int, as_of_ns: int, limit: int
    ) -> tuple[MacroRelease, ...]:
        timeline = self.releases.get(key)
        return timeline.prior(period_end_ns, as_of_ns, limit) if timeline is not None else ()

    def calendar(self, as_of_ns: int) -> CalendarState | None:
        if self.calendar_known_at_ns is None or as_of_ns < self.calendar_known_at_ns:
            return None
        position = bisect_right(self.calendar_clocks, as_of_ns) - 1
        return self.calendar_states[position] if position >= 0 else CalendarState((), ())
