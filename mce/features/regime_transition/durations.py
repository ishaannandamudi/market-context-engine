"""Completed dwell distributions and observed regular-session occupancy."""

from bisect import bisect_left, bisect_right
from collections import defaultdict
from statistics import mean, median

from mce.calendar.service import MarketSessionService
from mce.config.settings import RegimeTransitionSettings
from mce.features.regime.taxonomy import HeadlineRegime
from mce.features.regime_transition.history import RegimeHistory
from mce.features.regime_transition.models import RegimeDurationSummary, RegimeOccupancy
from mce.features.regime_transition.time_axis import SessionAxis


def observed_occupancy(
    history: RegimeHistory,
    end: int,
    start_ns: int,
    as_of_ns: int,
    axis: SessionAxis,
    config: RegimeTransitionSettings,
) -> tuple[tuple[RegimeOccupancy, ...], int, float, bool]:
    """Assign each sufficiently sampled interval to its last observed headline."""
    amounts: dict[HeadlineRegime, int] = defaultdict(int)
    covered_ns = 0
    covered_sessions = 0.0
    gap_seen = False
    first_index = max(0, bisect_right(history.timestamps, start_ns) - 1)
    for index in range(first_index, end):
        point = history.points[index]
        next_ns = history.points[index + 1].as_of_ns if index + 1 < end else as_of_ns
        if next_ns <= start_ns or point.as_of_ns >= as_of_ns:
            continue
        segment_start = max(point.as_of_ns, start_ns)
        segment_end = min(next_ns, as_of_ns)
        span = axis.trading_ns_between(segment_start, segment_end)
        if span > config.maximum_gap_trading_ns:
            gap_seen = True
            continue
        if span:
            amounts[point.headline_regime] += span
            covered_ns += span
            covered_sessions += axis.session_equivalents_between(segment_start, segment_end)
    occupancy = (
        tuple(
            RegimeOccupancy(regime=regime, observed_trading_ns=amount, fraction=amount / covered_ns)
            for regime, amount in sorted(amounts.items(), key=lambda item: item[0].value)
        )
        if covered_ns
        else ()
    )
    return occupancy, covered_ns, covered_sessions, gap_seen


def completed_durations(
    history: RegimeHistory,
    end: int,
    start_ns: int,
    axis: SessionAxis,
    calendar: MarketSessionService,
    config: RegimeTransitionSettings,
) -> tuple[RegimeDurationSummary, ...]:
    episodes: dict[HeadlineRegime, list[tuple[int, int]]] = defaultdict(list)
    for index in range(bisect_left(history.step_timestamps, start_ns), end - 1):
        step = history.steps[index]
        if not step.headline_changed or step.at_ns < start_ns:
            continue
        first_index = history.episode_start_indices[index]
        first = history.points[first_index]
        if first.as_of_ns < start_ns:
            continue  # left-censored by the selected window
        if any(
            axis.trading_ns_between(
                history.points[part].as_of_ns, history.points[part + 1].as_of_ns
            )
            > config.maximum_gap_trading_ns
            for part in range(first_index, index + 1)
        ):
            continue
        episodes[first.headline_regime].append(
            (
                axis.sessions_spanned(first.as_of_ns, step.at_ns, calendar),
                axis.trading_ns_between(first.as_of_ns, step.at_ns),
            )
        )
    return tuple(
        RegimeDurationSummary(
            regime=regime,
            completed_episodes=len(values),
            mean_sessions=mean(item[0] for item in values)
            if len(values) >= config.minimum_completed_episodes
            else None,
            median_sessions=median(item[0] for item in values)
            if len(values) >= config.minimum_completed_episodes
            else None,
            minimum_sessions=min(item[0] for item in values)
            if len(values) >= config.minimum_completed_episodes
            else None,
            maximum_sessions=max(item[0] for item in values)
            if len(values) >= config.minimum_completed_episodes
            else None,
            mean_trading_ns=mean(item[1] for item in values)
            if len(values) >= config.minimum_completed_episodes
            else None,
        )
        for regime, values in sorted(episodes.items(), key=lambda item: item[0].value)
    )
