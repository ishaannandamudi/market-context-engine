"""Deterministic MCE-11 transition, duration, occupancy, and causality scenarios."""

from dataclasses import replace
from datetime import date

import pytest

from mce.calendar.service import MarketSessionService
from mce.config.settings import (
    FeatureSettings,
    MCEConfig,
    RegimeTransitionSettings,
    RegimeTransitionStabilityWeights,
)
from mce.features.regime import MarketRegimeEngine, RegimeEvidence
from mce.features.regime.taxonomy import HeadlineRegime
from mce.features.regime_transition import (
    RegimeHistory,
    RegimeHistoryPoint,
    RegimeTransitionAnalyticsEngine,
    apply_regime_transition_to_snapshot,
)
from mce.features.regime_transition.history import DIMENSION_NAMES
from mce.features.regime_transition.time_axis import SessionAxis
from mce.models.context import MarketContextSnapshot
from mce.models.enums import DataQuality, MarketRegime, QualityFlag
from mce.reproducibility import config_fingerprint
from mce.serialization import canonical_json_str

FINGERPRINT = "a" * 64
BASE_DIMENSIONS = (
    "positive",
    "up",
    "normal",
    "healthy",
    "normal",
    "broad_positive",
    "normal",
    "calm",
    "stable_leadership",
)


def _point(
    timestamp_ns: int,
    regime: HeadlineRegime,
    *,
    dimensions: tuple[str, ...] = BASE_DIMENSIONS,
    quality: DataQuality = DataQuality.GOOD,
    available_at_ns: int | None = None,
    fingerprint: str = FINGERPRINT,
) -> RegimeHistoryPoint:
    return RegimeHistoryPoint(
        as_of_ns=timestamp_ns,
        available_at_ns=available_at_ns if available_at_ns is not None else timestamp_ns,
        headline_regime=regime,
        dimensions=dimensions,
        evidence_strength=0.8,
        data_quality=quality,
        config_fingerprint=fingerprint,
    )


def _times(
    calendar: MarketSessionService, count: int, start: date = date(2024, 6, 24)
) -> tuple[int, ...]:
    values = []
    day = start
    while len(values) < count:
        opened, _ = calendar.regular_session_bounds(day)
        values.append(opened + 60 * 60 * 1_000_000_000)
        day = calendar.next_trading_day(day)
    return tuple(values)


def _engine(
    calendar: MarketSessionService, settings: RegimeTransitionSettings | None = None
) -> RegimeTransitionAnalyticsEngine:
    config = MCEConfig(
        _env_file=None,
        features=FeatureSettings(regime_transition=settings or RegimeTransitionSettings()),
    )
    return RegimeTransitionAnalyticsEngine(config, calendar)


def test_known_transition_frequencies_dwell_and_reversal(calendar: MarketSessionService) -> None:
    times = _times(calendar, 7)
    states = (
        HeadlineRegime.RISK_ON,
        HeadlineRegime.RISK_OFF,
        HeadlineRegime.RISK_ON,
        HeadlineRegime.RISK_OFF,
        HeadlineRegime.RISK_ON,
        HeadlineRegime.ROTATION,
        HeadlineRegime.ROTATION,
    )
    result = _engine(calendar).compute(
        RegimeHistory.build(_point(t, state) for t, state in zip(times, states, strict=True)),
        times[-1],
    )
    matrix = {row.from_regime: row for row in result.observed_transition_frequency}
    risk_on = matrix[HeadlineRegime.RISK_ON]
    assert risk_on.observed_exits == 3
    assert {item.to_regime: item.count for item in risk_on.targets} == {
        HeadlineRegime.RISK_OFF: 2,
        HeadlineRegime.ROTATION: 1,
    }
    assert {item.to_regime: item.observed_frequency for item in risk_on.targets} == pytest.approx(
        {HeadlineRegime.RISK_OFF: 2 / 3, HeadlineRegime.ROTATION: 1 / 3}
    )
    assert all(
        sum(item.observed_frequency for item in row.targets) == pytest.approx(1)
        for row in matrix.values()
    )
    assert HeadlineRegime.ROTATION not in matrix  # no observed exit
    assert result.current_regime == HeadlineRegime.ROTATION
    assert result.current_age_sessions == 2
    assert result.current_age_observations == 2
    assert result.windows[2].headline_transitions == 5
    assert result.windows[2].reversal_count == 3
    assert result.windows[2].oscillation_score == pytest.approx(3 / 4)
    pair = result.windows[2].dominant_transition_pair
    assert pair is not None and pair.count == 4
    duration = {item.regime: item for item in result.historical_durations}
    assert duration[HeadlineRegime.RISK_ON].completed_episodes == 3
    assert duration[HeadlineRegime.RISK_ON].mean_sessions is not None
    assert duration[HeadlineRegime.RISK_OFF].completed_episodes == 2
    assert HeadlineRegime.ROTATION not in duration  # current episode is censored
    assert result.recent_transition_sequence[-1].to_regime == HeadlineRegime.ROTATION


def test_calendar_age_holiday_weekend_and_early_close(calendar: MarketSessionService) -> None:
    days = (date(2024, 7, 3), date(2024, 7, 5), date(2024, 7, 8))
    times = tuple(calendar.regular_session_bounds(day)[0] + 150 * 60_000_000_000 for day in days)
    history = RegimeHistory.build(_point(t, HeadlineRegime.RISK_ON) for t in times)
    result = _engine(calendar).compute(history, times[-1])
    assert result.config_fingerprint == config_fingerprint(_engine(calendar).config)
    assert result.classifier_fingerprint == FINGERPRINT
    assert result.current_age_sessions == 3
    assert result.current_age_trading_ns == 10 * 60 * 60 * 1_000_000_000
    assert result.current_age_ns == times[-1] - times[0]
    assert result.current_age_observations == 3
    assert result.stability_score is None  # fewer than five observations
    early_open, early_close = calendar.regular_session_bounds(date(2024, 7, 3))
    axis = SessionAxis.build(calendar, early_open, early_close)
    assert axis.session_equivalents_between(early_open, early_close) == 1
    assert axis.trading_ns_between(early_open, early_close) == 210 * 60_000_000_000
    weekend_ns = calendar.regular_session_bounds(date(2024, 7, 5))[1] + 2 * 86_400_000_000_000
    weekend = _engine(calendar).compute(history, weekend_ns)
    assert weekend.current_age_sessions == 2  # July 8 observation is still in the future


def test_irregular_time_weighted_occupancy(calendar: MarketSessionService) -> None:
    first_open, first_close = calendar.regular_session_bounds(date(2024, 6, 24))
    _, next_close = calendar.regular_session_bounds(date(2024, 6, 25))
    times = (first_open + 60_000_000_000, first_close, next_close)
    regimes = (HeadlineRegime.RISK_ON, HeadlineRegime.RISK_OFF, HeadlineRegime.MIXED)
    result = _engine(calendar).compute(
        RegimeHistory.build(_point(t, regime) for t, regime in zip(times, regimes, strict=True)),
        times[-1],
    )
    occupancy = {item.regime: item for item in result.windows[2].occupancy}
    assert occupancy[HeadlineRegime.RISK_ON].observed_trading_ns == 389 * 60_000_000_000
    assert occupancy[HeadlineRegime.RISK_OFF].observed_trading_ns == 390 * 60_000_000_000
    assert HeadlineRegime.MIXED not in occupancy  # current instant has zero elapsed time
    assert sum(item.fraction for item in occupancy.values()) == pytest.approx(1)
    assert occupancy[HeadlineRegime.RISK_ON].fraction == pytest.approx(389 / 779)


def test_stability_churn_oscillation_and_dimensions(calendar: MarketSessionService) -> None:
    times = _times(calendar, 6)
    stable = RegimeHistory.build(_point(t, HeadlineRegime.RISK_ON) for t in times)
    alternating = RegimeHistory.build(
        _point(t, HeadlineRegime.RISK_ON if index % 2 == 0 else HeadlineRegime.RISK_OFF)
        for index, t in enumerate(times)
    )
    settings = RegimeTransitionSettings(recent_windows_sessions=(1, 5), primary_window_sessions=5)
    engine = _engine(calendar, settings)
    steady = engine.compute(stable, times[-1])
    volatile = engine.compute(alternating, times[-1])
    assert steady.stability_score is not None and volatile.stability_score is not None
    assert steady.stability_score > volatile.stability_score
    assert steady.windows[1].headline_churn == 0
    assert volatile.windows[1].headline_churn is not None
    assert volatile.windows[1].headline_churn > 0
    assert volatile.windows[1].oscillation_score == 1
    assert steady.oscillation_score is None
    assert volatile.instability_score == pytest.approx(1 - volatile.stability_score)
    assert sum(item.contribution for item in volatile.contributors) == pytest.approx(
        volatile.stability_score
    )
    changed = list(stable.points)
    dimensions = list(BASE_DIMENSIONS)
    dimensions[2] = "high"
    changed[-2] = replace(changed[-2], dimensions=tuple(dimensions))
    changed[-1] = replace(changed[-1], dimensions=tuple(dimensions))
    dimension_result = engine.compute(RegimeHistory.build(changed), times[-1])
    volatility = next(
        item for item in dimension_result.dimension_stability if item.dimension == "volatility"
    )
    assert dimension_result.windows[1].headline_transitions == 0
    assert volatility.transition_count == 1
    assert dimension_result.stability_score is not None
    assert dimension_result.stability_score < steady.stability_score
    assert dimension_result.dominant_unstable_dimension == "volatility"


def test_synchronized_transitions_and_intensity(calendar: MarketSessionService) -> None:
    times = _times(calendar, 6)
    points = [_point(t, HeadlineRegime.RISK_ON) for t in times]
    new_dimensions = list(BASE_DIMENSIONS)
    new_dimensions[0] = "negative"
    new_dimensions[5] = "broad_negative"
    points[2] = replace(
        points[2], headline_regime=HeadlineRegime.RISK_OFF, dimensions=tuple(new_dimensions)
    )
    points[3] = replace(
        points[3], headline_regime=HeadlineRegime.RISK_OFF, dimensions=tuple(new_dimensions)
    )
    points[4] = replace(points[4], headline_regime=HeadlineRegime.RISK_ON)
    settings = RegimeTransitionSettings(recent_windows_sessions=(1, 5), primary_window_sessions=5)
    result = _engine(calendar, settings).compute(RegimeHistory.build(points), times[-1])
    assert result.synchronized_headline_fraction == 1
    assert result.co_transition_count == 2
    direction = next(item for item in result.dimension_stability if item.dimension == "direction")
    assert direction.transition_count == 2
    assert direction.synchronized_headline_count == 2
    assert direction.synchronized_headline_fraction == 1
    assert result.transition_intensity is not None and result.transition_intensity > 0
    directed = {
        (item.from_regime, item.to_regime): item for item in result.directed_transition_intensity
    }
    assert directed[
        (HeadlineRegime.RISK_ON, HeadlineRegime.RISK_OFF)
    ].mean_changed_dimension_fraction == pytest.approx(2 / 9)
    assert directed[
        (HeadlineRegime.RISK_OFF, HeadlineRegime.RISK_ON)
    ].mean_changed_dimension_fraction == pytest.approx(2 / 9)
    assert result.recent_transition_sequence[-1].changed_dimensions == ("direction", "breadth")


def test_future_history_delayed_availability_sorting_and_replay(
    calendar: MarketSessionService,
) -> None:
    times = _times(calendar, 7)
    points = tuple(_point(t, HeadlineRegime.RISK_ON) for t in times[:5])
    settings = RegimeTransitionSettings(recent_windows_sessions=(1, 5), primary_window_sessions=5)
    engine = _engine(calendar, settings)
    reference = engine.compute(RegimeHistory.build(points), times[4])
    future = _point(times[5], HeadlineRegime.UNKNOWN, dimensions=("unknown",) * 9)
    later = _point(times[6], HeadlineRegime.RISK_OFF)
    all_history = RegimeHistory.build((later, *reversed(points), future))
    assert canonical_json_str(engine.compute(all_history, times[4])) == canonical_json_str(
        reference
    )
    delayed = replace(points[2], available_at_ns=times[6])
    with_delayed = RegimeHistory.build((*points[:2], delayed, *points[3:]))
    expected = engine.compute(RegimeHistory.build((*points[:2], *points[3:])), times[4])
    assert canonical_json_str(engine.compute(with_delayed, times[4])) == canonical_json_str(
        expected
    )
    assert canonical_json_str(engine.compute(all_history, times[4])) == canonical_json_str(
        engine.compute(all_history, times[4])
    )
    with pytest.raises(ValueError, match="duplicate"):
        RegimeHistory.build((*points, points[0]))
    changed_fingerprint = replace(future, config_fingerprint="b" * 64)
    changing = RegimeHistory.build((*points, changed_fingerprint))
    assert canonical_json_str(engine.compute(changing, times[4])) == canonical_json_str(reference)
    with pytest.raises(ValueError, match="fingerprints"):
        engine.compute(changing, times[5])


def test_quality_gates_gaps_unknown_and_upstream_quality(calendar: MarketSessionService) -> None:
    times = _times(calendar, 8)
    settings = RegimeTransitionSettings(recent_windows_sessions=(1, 5), primary_window_sessions=5)
    engine = _engine(calendar, settings)
    short = engine.compute(
        RegimeHistory.build(_point(t, HeadlineRegime.RISK_ON) for t in times[:2]), times[1]
    )
    assert short.stability_score is None
    assert QualityFlag.INSUFFICIENT_HISTORY in short.quality_flags
    unknown = RegimeHistory.build(
        _point(t, HeadlineRegime.UNKNOWN if index >= 3 else HeadlineRegime.RISK_ON)
        for index, t in enumerate(times)
    )
    unknown_result = engine.compute(unknown, times[-1])
    assert unknown_result.stability_score is None
    assert QualityFlag.MISSING_DATA in unknown_result.quality_flags
    poor = RegimeHistory.build(
        _point(t, HeadlineRegime.RISK_ON, quality=DataQuality.POOR) for t in times
    )
    poor_result = engine.compute(poor, times[-1])
    assert poor_result.stability_score is None
    assert QualityFlag.PROVIDER_DEGRADED in poor_result.quality_flags
    gap_settings = RegimeTransitionSettings(
        recent_windows_sessions=(1, 5),
        primary_window_sessions=5,
        maximum_gap_trading_ns=60_000_000_000,
    )
    gap = _engine(calendar, gap_settings).compute(
        RegimeHistory.build(_point(t, HeadlineRegime.RISK_ON) for t in times), times[-1]
    )
    assert QualityFlag.INSUFFICIENT_WINDOW_COVERAGE in gap.quality_flags
    assert gap.windows[1].time_coverage == 0
    assert gap.stability_score is None


def test_empty_and_one_session_history(calendar: MarketSessionService) -> None:
    first = _times(calendar, 1)[0]
    settings = RegimeTransitionSettings(recent_windows_sessions=(1, 5), primary_window_sessions=5)
    engine = _engine(calendar, settings)
    empty = engine.compute(RegimeHistory.build(()), first)
    assert empty.current_regime is None
    assert empty.classifier_fingerprint is None
    assert empty.observed_transition_frequency == ()
    assert empty.stability_score is None
    assert QualityFlag.INSUFFICIENT_HISTORY in empty.quality_flags
    one_session = RegimeHistory.build(
        _point(first + index * 60_000_000_000, HeadlineRegime.RISK_ON) for index in range(5)
    )
    result = engine.compute(one_session, one_session.points[-1].as_of_ns)
    assert result.windows[1].observations == 5
    assert result.windows[1].observed_sessions == 1
    assert QualityFlag.INSUFFICIENT_HISTORY in result.quality_flags
    assert result.stability_score is None


def test_unknown_mixed_stress_episodes_and_bounded_sequence(calendar: MarketSessionService) -> None:
    times = _times(calendar, 8)
    regimes = (
        HeadlineRegime.RISK_ON,
        HeadlineRegime.MIXED,
        HeadlineRegime.UNKNOWN,
        HeadlineRegime.TRANSITIONAL,
        HeadlineRegime.RISK_OFF,
        HeadlineRegime.SYSTEMIC_STRESS,
        HeadlineRegime.LIQUIDITY_STRESS,
        HeadlineRegime.RANGE_BOUND,
    )
    settings = RegimeTransitionSettings(max_recent_transitions=2)
    result = _engine(calendar, settings).compute(
        RegimeHistory.build(_point(t, regime) for t, regime in zip(times, regimes, strict=True)),
        times[-1],
    )
    primary = result.windows[2]
    assert primary.headline_transitions == 7
    assert primary.reversal_count == 0 and primary.oscillation_score == 0
    assert len(result.recent_transition_sequence) == 2
    assert result.recent_transition_sequence[-1].to_regime == HeadlineRegime.RANGE_BOUND
    assert {row.from_regime for row in result.observed_transition_frequency} == set(regimes[:-1])
    assert {item.regime for item in primary.occupancy} == set(regimes[:-1])
    assert all(item.mean_sessions is None for item in result.historical_durations)
    assert result.current_regime == HeadlineRegime.RANGE_BOUND


def test_config_snapshot_and_classifier_source_contract(calendar: MarketSessionService) -> None:
    times = _times(calendar, 5)
    config = MCEConfig(_env_file=None)
    classifier = MarketRegimeEngine(config)
    state = classifier.compute(
        RegimeEvidence(
            as_of_ns=times[0],
            market_direction_score=0.8,
            trend_score=0.8,
            breadth_score=0.7,
            volatility_score=0.3,
            stress_score=0.2,
        )
    )
    source = RegimeHistory.build((state,))
    assert source.points[0].headline_regime == state.headline_regime
    assert source.points[0].dimensions[0] == state.direction_regime.value
    history = RegimeHistory.build(_point(t, HeadlineRegime.RISK_ON) for t in times)
    result = _engine(calendar).compute(history, times[-1])
    original = MarketContextSnapshot(
        as_of_ns=times[-1], market_regime=MarketRegime.RISK_OFF, confidence=0.8
    )
    projected = apply_regime_transition_to_snapshot(result, original)
    assert projected.schema_version == "1.0" and projected.engine_version == "0.15.0"
    assert projected.market_regime == MarketRegime.RISK_OFF
    assert projected.confidence == 0.8
    assert projected.feature_values is not None
    assert projected.feature_values["regime_transition.age_sessions"] == result.current_age_sessions
    assert canonical_json_str(
        MarketContextSnapshot.model_validate_json(canonical_json_str(projected))
    ) == canonical_json_str(projected)
    with pytest.raises(ValueError):
        apply_regime_transition_to_snapshot(result, MarketContextSnapshot(as_of_ns=times[-1] + 1))
    disabled = RegimeTransitionSettings(enabled=False)
    assert _engine(calendar, disabled).compute(history, times[-1]).windows == ()
    with pytest.raises(ValueError):
        RegimeTransitionSettings(recent_windows_sessions=(5, 1))
    with pytest.raises(ValueError):
        RegimeTransitionSettings(primary_window_sessions=7)
    with pytest.raises(ValueError):
        RegimeTransitionSettings(primary_window_sessions=1)
    with pytest.raises(ValueError):
        RegimeTransitionStabilityWeights(dwell=0.5)
    changed = MCEConfig(
        _env_file=None,
        features=FeatureSettings(
            regime_transition=RegimeTransitionSettings(
                recent_windows_sessions=(1, 5, 21, 60), primary_window_sessions=21
            )
        ),
    )
    assert config_fingerprint(config) != config_fingerprint(changed)
    with pytest.raises(ValueError):
        RegimeHistoryPoint(
            as_of_ns=times[0],
            available_at_ns=times[0],
            headline_regime=HeadlineRegime.RISK_ON,
            dimensions=("only_one",),
            evidence_strength=0.8,
            data_quality=DataQuality.GOOD,
            config_fingerprint=FINGERPRINT,
        )
    with pytest.raises(ValueError):
        _point(times[0], HeadlineRegime.RISK_ON, dimensions=("invalid",) + BASE_DIMENSIONS[1:])
    with pytest.raises(ValueError):
        _point(times[0], HeadlineRegime.RISK_ON, available_at_ns=times[0] - 1)
    assert tuple(DIMENSION_NAMES) == tuple(item.dimension for item in result.dimension_stability)
    assert (
        SessionAxis.build(calendar, times[0], times[-1]).sessions_spanned(
            times[0], times[-1], calendar
        )
        == 5
    )
