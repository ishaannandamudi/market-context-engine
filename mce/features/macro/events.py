"""Scheduled information proximity and separately observed release aftermath."""

from mce.config.settings import MacroSettings
from mce.features.macro.contracts import DAY_NS, EventImportance, MacroRelease, ScheduledMacroEvent
from mce.features.macro.history import CalendarState
from mce.features.macro.models import MacroEventContext
from mce.models.enums import DataQuality

HOUR_NS = 3_600_000_000_000


def _importance(event: ScheduledMacroEvent, settings: MacroSettings) -> float:
    return float(getattr(settings.importance_weights, event.importance.value))


def event_context(
    calendar: CalendarState | None,
    releases: tuple[MacroRelease, ...],
    as_of_ns: int,
    settings: MacroSettings,
    eligible_sources: frozenset[str],
) -> MacroEventContext:
    def eligible(item: ScheduledMacroEvent) -> bool:
        return item.source in eligible_sources and (
            item.quality == DataQuality.GOOD
            or (settings.minimum_quality == "degraded" and item.quality == DataQuality.DEGRADED)
        )

    recent_candidates = tuple(
        item
        for item in releases
        if item.release_at_ns <= as_of_ns
        and 0 <= as_of_ns - item.release_at_ns <= settings.recent_release_horizon_ns
    )
    recent = max(
        recent_candidates, key=lambda item: (item.release_at_ns, item.event_id), default=None
    )
    since = as_of_ns - recent.release_at_ns if recent is not None else None
    if calendar is None:
        return MacroEventContext(
            recent_release=recent,
            seconds_since_recent_release=since // 1_000_000_000 if since is not None else None,
        )
    upcoming = calendar.upcoming(as_of_ns)
    next_event = upcoming[0] if upcoming else None
    high = next((item for item in upcoming if item.importance == EventImportance.HIGH), None)
    day_events = calendar.upcoming(as_of_ns, DAY_NS)
    valid_day = tuple(item for item in day_events if eligible(item))
    complete_day = len(valid_day) == len(day_events)
    weighted_24h = sum(_importance(item, settings) for item in valid_day)
    density_events = calendar.upcoming(as_of_ns, settings.event_density_horizon_ns)
    density = tuple(item for item in density_events if eligible(item))
    complete_density = len(density) == len(density_events)
    weighted_window = sum(_importance(item, settings) for item in density)
    near = calendar.upcoming(as_of_ns, settings.event_proximity_horizon_ns)
    complete_near = all(eligible(item) for item in near)
    remaining = 1.0
    for event in near:
        if eligible(event):
            decay = 1 - (event.scheduled_at_ns - as_of_ns) / settings.event_proximity_horizon_ns
            remaining *= 1 - _importance(event, settings) * max(0.0, decay)
    score = min(1.0, max(0.0, 1 - remaining))
    return MacroEventContext(
        next_event=next_event,
        next_high_importance=high,
        recent_release=recent,
        seconds_to_next=(next_event.scheduled_at_ns - as_of_ns) // 1_000_000_000
        if next_event
        else None,
        hours_to_next=(next_event.scheduled_at_ns - as_of_ns) / HOUR_NS if next_event else None,
        hours_to_next_high=(high.scheduled_at_ns - as_of_ns) / HOUR_NS if high else None,
        seconds_since_recent_release=since // 1_000_000_000 if since is not None else None,
        upcoming_count_24h=len(valid_day) if complete_day else None,
        high_importance_count_24h=sum(item.importance == EventImportance.HIGH for item in valid_day)
        if complete_day
        else None,
        weighted_count_24h=weighted_24h if complete_day else None,
        event_count_window=len(density) if complete_density else None,
        weighted_count_window=weighted_window if complete_density else None,
        proximity_score=score if complete_near else None,
    )
