"""Causal descriptive analytics over explicit, preclassified regime history."""

from bisect import bisect_left
from collections import Counter
from collections.abc import Iterable

from mce.calendar.service import MarketSessionService
from mce.config.settings import MCEConfig, RegimeTransitionSettings
from mce.features.regime.models import MarketRegimeState
from mce.features.regime.taxonomy import HeadlineRegime
from mce.features.regime_transition.durations import completed_durations, observed_occupancy
from mce.features.regime_transition.frequencies import (
    dominant_pair,
    headline_steps,
    observed_frequencies,
    reversal_metrics,
)
from mce.features.regime_transition.history import (
    DIMENSION_NAMES,
    HistoryStep,
    RegimeHistory,
    RegimeHistoryPoint,
)
from mce.features.regime_transition.intensity import directed_intensity
from mce.features.regime_transition.models import (
    DimensionTransitionSummary,
    RegimeTransitionFeatureSet,
    StabilityContributor,
    TransitionSequenceEntry,
    TransitionWindowSummary,
)
from mce.features.regime_transition.stability import (
    normalized_churn,
    temporal_stability,
    transition_intensity,
)
from mce.features.regime_transition.time_axis import SessionAxis, local_date, window_start
from mce.models.enums import DataQuality, QualityFlag
from mce.reproducibility import config_fingerprint


def _window(
    history: RegimeHistory,
    end: int,
    start_ns: int,
    as_of_ns: int,
    sessions: int,
    axis: SessionAxis,
    calendar: MarketSessionService,
    config: RegimeTransitionSettings,
) -> tuple[TransitionWindowSummary, tuple[HistoryStep, ...], float, bool]:
    first_point = bisect_left(history.timestamps, start_ns)
    points = history.points[first_point:end]
    first_step = bisect_left(history.step_timestamps, start_ns)
    steps = history.steps[first_step : max(0, end - 1)]
    transitions = headline_steps(steps)
    occupancy, covered_ns, covered_sessions, gap_seen = observed_occupancy(
        history, end, start_ns, as_of_ns, axis, config
    )
    total_ns = axis.trading_ns_between(start_ns, as_of_ns)
    rate = len(transitions) / covered_sessions if covered_sessions > 0 else None
    reversals, reversal_fraction = reversal_metrics(steps, axis, config)
    trading_dates = set(axis.dates)
    summary = TransitionWindowSummary(
        sessions=sessions,
        start_ns=start_ns,
        observations=len(points),
        observed_sessions=len(
            {local_date(item.as_of_ns, calendar) for item in points} & trading_dates
        ),
        headline_transitions=len(transitions),
        transition_rate_per_session=rate,
        headline_churn=normalized_churn(rate, config),
        unique_regimes=len({item.headline_regime for item in points}),
        reversal_count=reversals,
        reversal_fraction=reversal_fraction,
        oscillation_score=reversal_fraction,
        time_coverage=covered_ns / total_ns if total_ns else None,
        unknown_fraction=sum(item.headline_regime == HeadlineRegime.UNKNOWN for item in points)
        / len(points)
        if points
        else None,
        usable_quality_fraction=sum(
            item.data_quality in (DataQuality.GOOD, DataQuality.DEGRADED) for item in points
        )
        / len(points)
        if points
        else None,
        occupancy=occupancy,
        dominant_transition_pair=dominant_pair(steps),
    )
    return summary, steps, covered_sessions, gap_seen


class RegimeTransitionAnalyticsEngine:
    def __init__(self, config: MCEConfig, calendar: MarketSessionService | None = None) -> None:
        if calendar is not None and calendar.settings.model_dump(
            exclude={"universe"}
        ) != config.market.model_dump(exclude={"universe"}):
            raise ValueError("calendar settings must match result-relevant market configuration")
        self.config = config
        self.calendar = calendar or MarketSessionService(config.market)
        self.fingerprint = config_fingerprint(config)

    def compute(
        self,
        history: RegimeHistory | Iterable[MarketRegimeState | RegimeHistoryPoint],
        as_of_ns: int,
    ) -> RegimeTransitionFeatureSet:
        if as_of_ns < 0:
            raise ValueError("as_of_ns must be nonnegative")
        config = self.config.features.regime_transition
        if not config.enabled:
            return RegimeTransitionFeatureSet(
                as_of_ns=as_of_ns,
                config_fingerprint=self.fingerprint,
                primary_window_sessions=config.primary_window_sessions,
                windows=(),
                quality_flags=frozenset({QualityFlag.MISSING_DATA}),
                data_quality=DataQuality.UNUSABLE,
            )
        indexed = history if isinstance(history, RegimeHistory) else RegimeHistory.build(history)
        indexed, end = indexed.causal_prefix(as_of_ns)
        starts = {
            count: window_start(self.calendar, as_of_ns, count)
            for count in config.recent_windows_sessions
        }
        current = indexed.points[end - 1] if end else None
        current_start_index = indexed.episode_start_indices[end - 1] if end else None
        current_start = (
            indexed.points[current_start_index].as_of_ns
            if current_start_index is not None
            else None
        )
        dimension_start = (
            min(
                indexed.points[index].as_of_ns for index in indexed.dimension_start_indices[end - 1]
            )
            if end
            else None
        )
        axis_start = min(starts.values())
        if current_start is not None:
            axis_start = min(axis_start, current_start)
        if dimension_start is not None:
            axis_start = min(axis_start, dimension_start)
        axis = SessionAxis.build(self.calendar, axis_start, as_of_ns)
        calculations = tuple(
            _window(indexed, end, starts[count], as_of_ns, count, axis, self.calendar, config)
            for count in config.recent_windows_sessions
        )
        windows = tuple(item[0] for item in calculations)
        primary_index = config.recent_windows_sessions.index(config.primary_window_sessions)
        primary, steps, covered_sessions, gap_seen = calculations[primary_index]
        first_primary_point = bisect_left(indexed.timestamps, primary.start_ns)
        primary_points = indexed.points[first_primary_point:end]
        flags = {flag for point in primary_points for flag in point.quality_flags}
        if gap_seen:
            flags.add(QualityFlag.INSUFFICIENT_WINDOW_COVERAGE)
        if (
            primary.observations < config.minimum_observations
            or primary.observed_sessions < config.minimum_sessions
        ):
            flags.add(QualityFlag.INSUFFICIENT_HISTORY)
        if primary.time_coverage is None or primary.time_coverage < config.minimum_time_coverage:
            flags.add(QualityFlag.LOW_COVERAGE)
        if (
            primary.unknown_fraction is None
            or primary.unknown_fraction > config.maximum_unknown_fraction
        ):
            flags.add(QualityFlag.MISSING_DATA)
        if (
            primary.usable_quality_fraction is None
            or primary.usable_quality_fraction < config.minimum_usable_quality_fraction
        ):
            flags.add(QualityFlag.PROVIDER_DEGRADED)
        eligible = not flags.intersection(
            {
                QualityFlag.INSUFFICIENT_HISTORY,
                QualityFlag.LOW_COVERAGE,
                QualityFlag.MISSING_DATA,
                QualityFlag.PROVIDER_DEGRADED,
            }
        )
        age_sessions = (
            axis.sessions_spanned(current_start, as_of_ns, self.calendar)
            if current_start is not None
            else None
        )
        if age_sessions == 0:
            age_sessions = None
        dimensional = []
        dimension_counts = Counter(name for step in steps for name in step.changed_dimensions)
        headline_count = primary.headline_transitions
        synchronized_count = sum(
            step.headline_changed and bool(step.changed_dimensions) for step in steps
        )
        co_transition_count = sum(len(step.changed_dimensions) >= 2 for step in steps)
        for dimension_index, name in enumerate(DIMENSION_NAMES):
            count = dimension_counts[name]
            rate = count / covered_sessions if covered_sessions > 0 else None
            dimension_start = (
                indexed.points[indexed.dimension_start_indices[end - 1][dimension_index]].as_of_ns
                if end
                else None
            )
            dimension_age = (
                axis.sessions_spanned(dimension_start, as_of_ns, self.calendar)
                if dimension_start is not None
                else None
            )
            synchronized = sum(
                step.headline_changed and name in step.changed_dimensions for step in steps
            )
            dimension_churn = normalized_churn(rate, config)
            dimensional.append(
                DimensionTransitionSummary(
                    dimension=name,
                    transition_count=count,
                    transition_rate_per_session=rate,
                    current_age_sessions=dimension_age or None,
                    stability_score=1 - dimension_churn
                    if eligible and dimension_churn is not None
                    else None,
                    synchronized_headline_count=synchronized,
                    synchronized_headline_fraction=synchronized / headline_count
                    if headline_count
                    else None,
                )
            )
        dominant_dimension = min(
            (name for name in DIMENSION_NAMES if dimension_counts[name] > 0),
            key=lambda name: (-dimension_counts[name], name),
            default=None,
        )
        total_dimension_rate = (
            sum(dimension_counts.values()) / (len(DIMENSION_NAMES) * covered_sessions)
            if covered_sessions > 0
            else None
        )
        dimension_churn = normalized_churn(total_dimension_rate, config)
        stability = None
        intensity = None
        contributors: tuple[StabilityContributor, ...] = ()
        if (
            eligible
            and age_sessions is not None
            and primary.headline_churn is not None
            and dimension_churn is not None
        ):
            stability, contributors = temporal_stability(
                age_sessions,
                config.primary_window_sessions,
                primary.headline_churn,
                dimension_churn,
                primary.oscillation_score,
                config,
            )
            intensity = transition_intensity(primary.headline_churn, dimension_churn)
        first_step = bisect_left(indexed.step_timestamps, primary.start_ns)
        recent = headline_steps(indexed.steps[first_step : max(0, end - 1)])[
            -config.max_recent_transitions :
        ]
        quality = (
            DataQuality.POOR
            if not eligible
            else DataQuality.DEGRADED
            if gap_seen or any(point.data_quality != DataQuality.GOOD for point in primary_points)
            else DataQuality.GOOD
        )
        return RegimeTransitionFeatureSet(
            as_of_ns=as_of_ns,
            config_fingerprint=self.fingerprint,
            classifier_fingerprint=indexed.points[0].config_fingerprint if end else None,
            current_regime=current.headline_regime if current is not None else None,
            current_regime_started_at_ns=current_start,
            current_age_ns=as_of_ns - current_start if current_start is not None else None,
            current_age_trading_ns=axis.trading_ns_between(current_start, as_of_ns)
            if current_start is not None
            else None,
            current_age_sessions=age_sessions,
            current_age_observations=end - current_start_index
            if current_start_index is not None
            else None,
            primary_window_sessions=config.primary_window_sessions,
            windows=windows,
            observed_transition_frequency=observed_frequencies(steps),
            historical_durations=completed_durations(
                indexed, end, primary.start_ns, axis, self.calendar, config
            ),
            recent_transition_sequence=tuple(
                TransitionSequenceEntry(
                    at_ns=step.at_ns,
                    from_regime=step.from_regime,
                    to_regime=step.to_regime,
                    changed_dimensions=step.changed_dimensions,
                    changed_dimension_fraction=len(step.changed_dimensions) / len(DIMENSION_NAMES),
                )
                for step in recent
            ),
            directed_transition_intensity=directed_intensity(steps),
            dimension_stability=tuple(dimensional),
            synchronized_headline_fraction=synchronized_count / headline_count
            if headline_count
            else None,
            co_transition_count=co_transition_count,
            dominant_unstable_dimension=dominant_dimension,
            stability_score=stability,
            instability_score=1 - stability if stability is not None else None,
            oscillation_score=primary.oscillation_score,
            transition_intensity=intensity,
            contributors=contributors,
            data_quality=quality,
            quality_flags=frozenset(flags),
        )
