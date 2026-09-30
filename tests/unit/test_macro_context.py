"""MCE-18 contract, event-calendar, vintage, surprise, rate, and replay tests."""

from itertools import permutations
from json import dumps

import pytest
from pydantic import ValidationError

from mce.features.macro import (
    MacroCategory,
    MacroContextAdapter,
    MacroContextFeatureEngine,
    MacroContextInput,
    MacroEventType,
    MacroHistoryIndex,
    MacroObservation,
    MacroRelease,
    MacroUnit,
    RevisionStatus,
    ScheduledMacroEvent,
    SeriesCadence,
    apply_macro_context_to_snapshot,
)
from mce.features.macro.contracts import DAY_NS
from mce.features.macro.surprises import compare_values, surprise_context
from mce.models.context import EventRisk, MarketContextSnapshot
from mce.models.enums import DataQuality, EventRiskStatus, MarketRegime, QualityFlag
from tests.fixtures.macro_scenarios import (
    BASE,
    HOUR,
    MINUTE,
    SOURCE,
    batch,
    event,
    full_context,
    macro_config,
    observation,
    release,
)


def _compute(inputs: tuple[MacroContextInput, ...] | MacroHistoryIndex):
    return MacroContextFeatureEngine(macro_config()).compute(inputs, BASE)


def _series(result, name: str):
    return next(item for item in result.per_series if item.series_id == name)


def test_adapter_schema_source_metadata_and_optional_fields() -> None:
    source = full_context()
    payload = source.model_dump(mode="json")
    payload["future_additive_field"] = {"ignored": True}
    parsed = MacroContextAdapter().translate(payload)
    assert parsed == source
    assert parsed.macro_context_schema_version == "1.0"
    assert parsed.source == SOURCE and parsed.source_version == "1.2.3"
    assert parsed.source_fingerprint is not None
    for patch in (
        {"macro_context_schema_version": "2.0"},
        {"source_version": "bad"},
        {"available_at_ns": BASE - 1},
    ):
        with pytest.raises(ValidationError):
            MacroContextAdapter().translate({**payload, **patch})
    no_optional = {key: value for key, value in payload.items() if key != "source_fingerprint"}
    assert MacroContextAdapter().translate(no_optional).source_fingerprint is None
    with pytest.raises(ValidationError):
        MacroContextAdapter().translate(
            {key: value for key, value in payload.items() if key != "macro_context_schema_version"}
        )


def test_contract_rejects_invalid_clocks_units_tenors_consensus_and_child_source() -> None:
    source = full_context()
    obs = source.observations[0].model_dump(mode="json")
    rel = source.releases[0].model_dump(mode="json")
    cal = source.scheduled_events[0].model_dump(mode="json")
    for patch in (
        {"value": float("nan")},
        {"period_end_ns": obs["release_at_ns"] + 1},
        {"available_at_ns": obs["vintage_at_ns"] - 1},
        {"tenor_months": 120},
    ):
        with pytest.raises(ValidationError):
            MacroObservation.model_validate_json(dumps({**obs, **patch}))
    with pytest.raises(ValidationError):
        MacroObservation.model_validate_json(
            dumps({**obs, "category": "treasury_yield", "unit": "index_level", "tenor_months": 120})
        )
    for patch in (
        {"consensus": None},
        {"consensus_available_at_ns": rel["release_at_ns"] + 1},
        {"revised_prior": 1},
        {"actual": float("inf")},
    ):
        with pytest.raises(ValidationError):
            MacroRelease.model_validate_json(dumps({**rel, **patch}))
    with pytest.raises(ValidationError):
        ScheduledMacroEvent.model_validate_json(
            dumps({**cal, "previous_scheduled_at_ns": cal["scheduled_at_ns"]})
        )
    with pytest.raises(ValidationError):
        MacroContextAdapter().translate(
            {**source.model_dump(mode="json"), "observations": [{**obs, "source": "other"}]}
        )
    with pytest.raises(ValidationError):
        MacroContextAdapter().translate(
            {
                **source.model_dump(mode="json"),
                "scheduled_events": [{**cal, "calendar_available_at_ns": BASE + 1}],
            }
        )


@pytest.mark.parametrize(
    ("minutes", "expected"),
    [(5, 1 - 5 / (24 * 60)), (60, 1 - 60 / (24 * 60)), (24 * 60, 0)],
)
def test_single_high_event_proximity(minutes: int, expected: float) -> None:
    scheduled = BASE + minutes * MINUTE
    source = batch(events=(event("major", MacroEventType.CPI, scheduled),))
    result = _compute((source,))
    assert result.event is not None
    assert result.event.hours_to_next == pytest.approx(minutes / 60)
    assert result.event.proximity_score == pytest.approx(expected)
    assert 0 <= result.event.proximity_score <= 1
    assert result.event.upcoming_count_24h == 1
    assert (
        result.event.next_event is not None
        and result.event.next_event.event_type == MacroEventType.CPI
    )


def test_event_cluster_next_session_empty_calendar_and_unannounced_event() -> None:
    one = _compute((batch(events=(event("cpi", MacroEventType.CPI, BASE + 5 * MINUTE),)),))
    clustered = _compute(
        (
            batch(
                events=(
                    event("cpi", MacroEventType.CPI, BASE + 5 * MINUTE),
                    event("fomc", MacroEventType.FOMC_RATE_DECISION, BASE + HOUR),
                    event("nfp", MacroEventType.NONFARM_PAYROLLS, BASE + DAY_NS + HOUR),
                )
            ),
        )
    )
    assert one.event is not None and clustered.event is not None
    assert clustered.event.proximity_score > one.event.proximity_score
    assert clustered.event.high_importance_count_24h == 2
    assert clustered.event.next_high_importance is not None
    assert clustered.event.next_high_importance.event_id == "cpi"
    empty = _compute((batch(),))
    assert empty.event is not None and empty.event.proximity_score == 0
    assert empty.event.upcoming_count_24h == 0
    assert empty.event.next_event is None
    unknown = _compute(())
    assert unknown.event is not None and unknown.event.proximity_score is None
    assert QualityFlag.MACRO_EVENT_CONTEXT_UNAVAILABLE in unknown.quality_flags
    announced_later = batch(
        clock=BASE + HOUR,
        events=(event("late", MacroEventType.CPI, BASE + 2 * HOUR, announced=BASE + HOUR),),
    )
    before = MacroContextFeatureEngine(macro_config()).compute((batch(), announced_later), BASE)
    after = MacroContextFeatureEngine(macro_config()).compute(
        (batch(), announced_later), BASE + HOUR
    )
    assert before.event is not None and before.event.next_event is None
    assert after.event is not None and after.event.next_event is not None


def test_recent_release_aftermath_and_schedule_revision_are_point_in_time() -> None:
    past_release = release(
        "CPI_YOY",
        MacroCategory.INFLATION,
        3.4,
        3.2,
        period_end=BASE - 30 * DAY_NS,
        released=BASE - 5 * MINUTE,
    )
    current = batch(
        releases=(past_release,), events=(event("cpi-old", MacroEventType.CPI, BASE - 5 * MINUTE),)
    )
    just = _compute((current,))
    assert just.event is not None and just.event.seconds_since_recent_release == 300
    assert just.event.next_event is None
    yesterday = MacroContextFeatureEngine(macro_config()).compute((current,), BASE + DAY_NS)
    assert yesterday.event is not None and yesterday.event.seconds_since_recent_release == 86_700
    original = batch(
        clock=BASE - DAY_NS,
        events=(
            event("fomc", MacroEventType.FOMC_RATE_DECISION, BASE + HOUR, announced=BASE - DAY_NS),
        ),
    )
    revised = batch(
        clock=BASE + 30 * MINUTE,
        events=(
            event(
                "fomc",
                MacroEventType.FOMC_RATE_DECISION,
                BASE + 2 * HOUR,
                announced=BASE + 30 * MINUTE,
                previous=BASE + HOUR,
            ),
        ),
    )
    index = MacroHistoryIndex((revised, original))
    before = MacroContextFeatureEngine(macro_config()).compute(index, BASE)
    after = MacroContextFeatureEngine(macro_config()).compute(index, BASE + 31 * MINUTE)
    assert before.event is not None and before.event.hours_to_next == 1
    assert after.event is not None and after.event.hours_to_next == pytest.approx(
        (2 * HOUR - 31 * MINUTE) / HOUR
    )
    assert (
        MacroContextFeatureEngine(macro_config()).compute((original,), BASE).fingerprint
        == before.fingerprint
    )
    with pytest.raises(ValueError, match="ambiguous"):
        MacroHistoryIndex(
            (original, original.model_copy(update={"source_fingerprint": "different"}))
        )


def test_surprise_positive_zero_negative_missing_and_category_orientation() -> None:
    prior = tuple(
        release(
            "CPI_YOY",
            MacroCategory.INFLATION,
            actual,
            3.0,
            period_end=BASE - (i + 2) * 30 * DAY_NS,
            released=BASE - (i + 1) * 30 * DAY_NS,
        )
        for i, actual in enumerate((2.9, 3.0, 3.1, 3.2))
    )
    for actual, expected in ((3.3, 0.3), (3.0, 0), (2.7, -0.3)):
        item = release(
            "CPI_YOY",
            MacroCategory.INFLATION,
            actual,
            3.0,
            period_end=BASE - 30 * DAY_NS,
            released=BASE,
        )
        result = surprise_context(item, prior, 3)
        assert result.raw == pytest.approx(expected)
        assert result.normalized_z is not None
        assert result.orientation_axis == "inflation_pressure"
        assert result.oriented_z == result.normalized_z
    missing = release(
        "CPI_YOY", MacroCategory.INFLATION, 3.3, None, period_end=BASE - 30 * DAY_NS, released=BASE
    )
    assert surprise_context(missing, prior, 3).raw is None
    unemployment = release(
        "UNEMPLOYMENT",
        MacroCategory.UNEMPLOYMENT,
        4.2,
        4.0,
        period_end=BASE - 30 * DAY_NS,
        released=BASE,
    )
    previous = tuple(
        release(
            "UNEMPLOYMENT",
            MacroCategory.UNEMPLOYMENT,
            value,
            4.0,
            period_end=BASE - (i + 2) * 30 * DAY_NS,
            released=BASE - (i + 1) * 30 * DAY_NS,
        )
        for i, value in enumerate((3.9, 4.0, 4.1, 4.2))
    )
    labor = surprise_context(unemployment, previous, 3)
    assert labor.raw == pytest.approx(0.2)
    assert labor.oriented_z is not None and labor.oriented_z < 0
    assert compare_values(2, (1, 2, 3), 3).percentile == 0.5
    assert compare_values(5, (1, 2, 3), 4).percentile is None


def test_insufficient_surprise_history_future_consensus_and_unit_mismatch() -> None:
    current = release(
        "GDP", MacroCategory.GROWTH, 2.1, 2.0, period_end=BASE - 90 * DAY_NS, released=BASE
    )
    prior = (
        release(
            "GDP",
            MacroCategory.GROWTH,
            1.9,
            2.0,
            period_end=BASE - 180 * DAY_NS,
            released=BASE - 90 * DAY_NS,
        ),
    )
    result = surprise_context(current, prior, 3)
    assert result.raw == pytest.approx(0.1) and result.normalized_z is None
    wrong_unit = prior[0].model_copy(update={"unit": MacroUnit.MILLIONS})
    assert surprise_context(current, (wrong_unit,), 1).historical_sample_count == 0
    initial = current.model_copy(update={"consensus": None, "consensus_available_at_ns": None})
    first_batch = batch(releases=(initial,))
    late_consensus = current.model_copy(
        update={"vintage_at_ns": BASE + DAY_NS, "available_at_ns": BASE + DAY_NS}
    )
    later_batch = batch(clock=BASE + DAY_NS, releases=(late_consensus,))
    engine = MacroContextFeatureEngine(macro_config())
    before = engine.compute((first_batch, later_batch), BASE)
    after = engine.compute((first_batch, later_batch), BASE + DAY_NS)
    assert _series(before, "GDP").surprise is not None
    assert _series(before, "GDP").surprise.raw is None
    assert _series(after, "GDP").surprise is not None
    assert _series(after, "GDP").surprise.raw == pytest.approx(0.1)
    with pytest.raises(ValueError, match="unit"):
        MacroHistoryIndex(
            (
                batch(
                    observations=(
                        observation(
                            "GDP",
                            MacroCategory.GROWTH,
                            2,
                            period_end=BASE - 90 * DAY_NS,
                            released=BASE - DAY_NS,
                        ),
                    )
                ),
                batch(
                    clock=BASE + DAY_NS,
                    releases=(
                        release(
                            "GDP",
                            MacroCategory.GROWTH,
                            2,
                            2,
                            period_end=BASE - 90 * DAY_NS,
                            released=BASE,
                            unit=MacroUnit.INDEX_LEVEL,
                        ),
                    ),
                ),
            )
        )


def test_policy_level_hike_cut_unchanged_and_replay() -> None:
    first = observation(
        "FED_TARGET_MID",
        MacroCategory.POLICY_RATE,
        4.0,
        period_end=BASE - 2 * DAY_NS,
        released=BASE - 2 * DAY_NS,
        cadence=SeriesCadence.EVENT_DRIVEN,
    )
    hike = observation(
        "FED_TARGET_MID",
        MacroCategory.POLICY_RATE,
        4.25,
        period_end=BASE - DAY_NS,
        released=BASE - DAY_NS,
        cadence=SeriesCadence.EVENT_DRIVEN,
    )
    cut = observation(
        "FED_TARGET_MID",
        MacroCategory.POLICY_RATE,
        3.75,
        period_end=BASE + DAY_NS,
        released=BASE + DAY_NS,
        cadence=SeriesCadence.EVENT_DRIVEN,
    )
    before_batch = batch(clock=BASE - DAY_NS, observations=(first, hike))
    later_batch = batch(clock=BASE + DAY_NS, observations=(cut,))
    engine = MacroContextFeatureEngine(macro_config())
    before = engine.compute((before_batch, later_batch), BASE)
    after = engine.compute((before_batch, later_batch), BASE + DAY_NS)
    assert before.policy is not None and before.policy.current_rate_percent == 4.25
    assert before.policy.latest_change_pp == pytest.approx(0.25)
    assert before.policy.trailing_change_count == 1
    assert after.policy is not None and after.policy.latest_change_pp == pytest.approx(-0.5)
    unchanged = hike.model_copy(
        update={
            "period_end_ns": BASE,
            "release_at_ns": BASE,
            "available_at_ns": BASE,
            "vintage_at_ns": BASE,
        }
    )
    flat = _compute((batch(observations=(first, hike, unchanged)),))
    assert flat.policy is not None and flat.policy.latest_change_pp == 0
    assert flat.policy.trailing_change_count == 1
    assert engine.compute((before_batch,), BASE).fingerprint == before.fingerprint


def test_yield_curve_reference_spreads_curvature_changes_and_inversion() -> None:
    result = _compute((full_context(),))
    curve = result.curve
    assert curve is not None
    assert curve.spread_2s10s_pp == pytest.approx(0.4)
    assert curve.spread_3m10y_pp == pytest.approx(-0.1)
    assert curve.spread_30y5y_pp == pytest.approx(0.6)
    assert curve.curvature_2y5y10y_pp == pytest.approx(0.05)
    assert curve.level_10y_percent == 4.4
    assert curve.inverted is True and curve.inversion_magnitude_pp == pytest.approx(0.1)
    assert curve.change_10y_1d_pp == pytest.approx(0.1)
    assert curve.change_10y_5d_pp == pytest.approx(0.2)
    assert curve.change_10y_20d_pp == pytest.approx(0.3)
    assert curve.parallel_shift_5d_pp == pytest.approx(0.2)
    assert curve.steepening_2s10s_5d_pp == pytest.approx(0)
    assert result.policy is not None and result.policy.current_rate_percent == 4.25
    assert result.stress is not None and result.stress.score is not None
    assert 0 <= result.stress.score <= 1
    assert result.event is not None and result.event.proximity_score is not None
    assert result.stress.score != result.event.proximity_score


def test_curve_flat_steepening_flattening_missing_tenor_and_future_yield() -> None:
    old2 = observation(
        "UST2Y",
        MacroCategory.TREASURY_YIELD,
        4,
        period_end=BASE - 6 * DAY_NS,
        released=BASE - 6 * DAY_NS,
        cadence=SeriesCadence.DAILY,
        tenor_months=24,
    )
    old10 = observation(
        "UST10Y",
        MacroCategory.TREASURY_YIELD,
        4,
        period_end=BASE - 6 * DAY_NS,
        released=BASE - 6 * DAY_NS,
        cadence=SeriesCadence.DAILY,
        tenor_months=120,
    )
    new2 = observation(
        "UST2Y",
        MacroCategory.TREASURY_YIELD,
        4,
        period_end=BASE - DAY_NS,
        released=BASE - DAY_NS,
        cadence=SeriesCadence.DAILY,
        tenor_months=24,
    )
    new10 = observation(
        "UST10Y",
        MacroCategory.TREASURY_YIELD,
        4,
        period_end=BASE - DAY_NS,
        released=BASE - DAY_NS,
        cadence=SeriesCadence.DAILY,
        tenor_months=120,
    )
    flat = _compute((batch(observations=(old2, old10, new2, new10)),))
    assert flat.curve is not None and flat.curve.spread_2s10s_pp == 0
    assert flat.curve.inverted is False
    steep = _compute(
        (batch(observations=(old2, old10, new2, new10.model_copy(update={"value": 4.5}))),)
    )
    assert steep.curve is not None and steep.curve.steepening_2s10s_5d_pp == pytest.approx(0.5)
    flat_again = _compute(
        (batch(observations=(old2, old10, new2.model_copy(update={"value": 4.5}), new10)),)
    )
    assert (
        flat_again.curve is not None
        and flat_again.curve.steepening_2s10s_5d_pp == pytest.approx(-0.5)
    )
    missing = _compute((batch(observations=(new10,)),))
    assert missing.curve is not None and missing.curve.spread_2s10s_pp is None
    assert QualityFlag.MACRO_PARTIAL_CURVE in missing.quality_flags
    future = observation(
        "UST10Y",
        MacroCategory.TREASURY_YIELD,
        9,
        period_end=BASE + DAY_NS,
        released=BASE + DAY_NS,
        cadence=SeriesCadence.DAILY,
        tenor_months=120,
    )
    future_batch = batch(clock=BASE + DAY_NS, observations=(future,))
    engine = MacroContextFeatureEngine(macro_config())
    baseline = engine.compute((batch(observations=(old2, old10, new2, new10)),), BASE)
    augmented = engine.compute((batch(observations=(old2, old10, new2, new10)), future_batch), BASE)
    assert augmented.fingerprint == baseline.fingerprint


def test_vintage_revision_and_future_history_do_not_rewrite_past() -> None:
    period = BASE - 90 * DAY_NS
    first = observation(
        "GDP",
        MacroCategory.GROWTH,
        2.0,
        period_end=period,
        released=BASE - DAY_NS,
        cadence=SeriesCadence.QUARTERLY,
        status=RevisionStatus.PRELIMINARY,
    )
    revised = observation(
        "GDP",
        MacroCategory.GROWTH,
        1.5,
        period_end=period,
        released=BASE - DAY_NS,
        vintage=BASE + DAY_NS,
        cadence=SeriesCadence.QUARTERLY,
        status=RevisionStatus.REVISED,
    )
    first_batch = batch(clock=BASE - DAY_NS, observations=(first,))
    revised_batch = batch(clock=BASE + DAY_NS, observations=(revised,))
    future_history = observation(
        "GDP",
        MacroCategory.GROWTH,
        99,
        period_end=BASE + DAY_NS,
        released=BASE + DAY_NS,
        cadence=SeriesCadence.QUARTERLY,
    )
    future_batch = batch(clock=BASE + 2 * DAY_NS, observations=(future_history,))
    engine = MacroContextFeatureEngine(macro_config())
    baseline = engine.compute((first_batch,), BASE)
    for order in permutations((first_batch, revised_batch, future_batch)):
        assert engine.compute(order, BASE).fingerprint == baseline.fingerprint
    after = engine.compute((first_batch, revised_batch), BASE + DAY_NS)
    assert _series(baseline, "GDP").observation.value == 2.0
    assert _series(after, "GDP").observation.value == 1.5
    assert _series(after, "GDP").observation.revision_status == RevisionStatus.REVISED
    with pytest.raises(ValueError, match="same-vintage"):
        MacroHistoryIndex(
            (
                first_batch,
                batch(clock=BASE, observations=(first.model_copy(update={"value": 3.0}),)),
            )
        )


def test_cadence_quality_dimensions_and_snapshot_preservation() -> None:
    source = full_context()
    result = _compute((source,))
    assert QualityFlag.MACRO_SERIES_STALE not in result.quality_flags
    inflation = next(item for item in result.dimensions if item.name == "inflation")
    growth = next(item for item in result.dimensions if item.name == "growth")
    labor = next(item for item in result.dimensions if item.name == "labor")
    assert inflation.latest_value == 3.5 and inflation.latest_surprise == pytest.approx(0.3)
    assert growth.latest_surprise == pytest.approx(-0.4)
    assert labor.contributing_series == 2
    assert _series(result, "CPI_YOY").history is not None
    assert _series(result, "CPI_YOY").history.percentile is not None
    snapshot = apply_macro_context_to_snapshot(
        result,
        MarketContextSnapshot(
            as_of_ns=BASE,
            market_regime=MarketRegime.RISK_OFF,
            stress_score=0.77,
            feature_values={"options.iv.atm": 0.2, "macro.old": 99},
        ),
    )
    assert snapshot.schema_version == "1.0" and snapshot.engine_version == "1.0.0"
    assert snapshot.market_regime == MarketRegime.RISK_OFF and snapshot.stress_score == 0.77
    assert snapshot.feature_values is not None
    assert snapshot.feature_values["options.iv.atm"] == 0.2
    assert "macro.old" not in snapshot.feature_values
    assert snapshot.feature_values["macro.curve.2s10s"] == pytest.approx(0.4)
    assert snapshot.feature_values["macro.inflation.surprise"] == pytest.approx(0.3)
    assert (
        snapshot.event_risk is not None and snapshot.event_risk.status == EventRiskStatus.SCHEDULED
    )
    assert snapshot.source_freshness is not None
    assert MarketContextSnapshot.model_validate_json(snapshot.model_dump_json()) == snapshot
    original_risk = EventRisk(status=EventRiskStatus.UNAVAILABLE)
    retained = apply_macro_context_to_snapshot(
        result, MarketContextSnapshot(as_of_ns=BASE, event_risk=original_risk)
    )
    assert retained.event_risk == original_risk
    with pytest.raises(ValueError):
        apply_macro_context_to_snapshot(result, MarketContextSnapshot(as_of_ns=BASE + 1))


def test_stale_monthly_vs_genuinely_stale_and_low_quality_source() -> None:
    recent_period = BASE - 40 * DAY_NS
    release_time = BASE - DAY_NS
    monthly = observation(
        "CPI_YOY", MacroCategory.INFLATION, 3.0, period_end=recent_period, released=release_time
    )
    valid = _compute((batch(observations=(monthly,)),))
    assert not _series(valid, "CPI_YOY").stale
    stale_obs = observation(
        "CPI_YOY",
        MacroCategory.INFLATION,
        3.0,
        period_end=BASE - 90 * DAY_NS,
        released=BASE - 80 * DAY_NS,
    )
    stale = _compute((batch(observations=(stale_obs,)),))
    assert _series(stale, "CPI_YOY").stale
    assert QualityFlag.MACRO_SERIES_STALE in stale.quality_flags
    poor_source = batch(observations=(monthly,)).model_copy(update={"quality": DataQuality.POOR})
    poor = _compute((poor_source,))
    assert QualityFlag.MACRO_SOURCE_DEGRADED in poor.quality_flags
    assert _series(poor, "CPI_YOY").observation.value == 3.0
    assert _series(poor, "CPI_YOY").data_quality == DataQuality.POOR
    empty = _compute(())
    assert empty.data_quality == DataQuality.POOR
    config = macro_config()
    disabled = config.features.macro.model_copy(update={"enabled": False})
    config = config.model_copy(
        update={"features": config.features.model_copy(update={"macro": disabled})}
    )
    assert (
        MacroContextFeatureEngine(config).compute((full_context(),), BASE).data_quality
        == DataQuality.UNKNOWN
    )


def test_release_only_staleness_and_fresh_observation_with_old_release() -> None:
    old = release(
        "CPI_YOY",
        MacroCategory.INFLATION,
        3.2,
        3.0,
        period_end=BASE - 90 * DAY_NS,
        released=BASE - 80 * DAY_NS,
    )
    release_only = _compute((batch(releases=(old,)),))
    stale_series = _series(release_only, "CPI_YOY")
    assert stale_series.stale and stale_series.release_stale
    assert QualityFlag.MACRO_SERIES_STALE in release_only.quality_flags
    assert release_only.stress is not None and release_only.stress.surprise is None
    assert (
        next(item for item in release_only.dimensions if item.name == "inflation").latest_surprise
        is None
    )

    current = observation(
        "CPI_YOY",
        MacroCategory.INFLATION,
        3.5,
        period_end=BASE - 30 * DAY_NS,
        released=BASE - DAY_NS,
    )
    mixed = _compute((batch(observations=(current,), releases=(old,)),))
    mixed_series = _series(mixed, "CPI_YOY")
    assert not mixed_series.stale and mixed_series.release_stale
    inflation = next(item for item in mixed.dimensions if item.name == "inflation")
    assert inflation.latest_value == 3.5
    assert inflation.latest_surprise is None


def test_finite_macro_inputs_cannot_overflow_raw_change() -> None:
    earlier = observation(
        "GDP",
        MacroCategory.GROWTH,
        -1e308,
        period_end=BASE - 180 * DAY_NS,
        released=BASE - 90 * DAY_NS,
        cadence=SeriesCadence.QUARTERLY,
    )
    current = observation(
        "GDP",
        MacroCategory.GROWTH,
        1e308,
        period_end=BASE - 90 * DAY_NS,
        released=BASE - DAY_NS,
        cadence=SeriesCadence.QUARTERLY,
    )
    result = _compute((batch(observations=(earlier, current)),))
    assert _series(result, "GDP").change is None


def test_finite_yield_levels_cannot_overflow_spread() -> None:
    two = observation(
        "UST2Y",
        MacroCategory.TREASURY_YIELD,
        -1e308,
        period_end=BASE - DAY_NS,
        released=BASE - DAY_NS,
        cadence=SeriesCadence.DAILY,
        tenor_months=24,
    )
    ten = observation(
        "UST10Y",
        MacroCategory.TREASURY_YIELD,
        1e308,
        period_end=BASE - DAY_NS,
        released=BASE - DAY_NS,
        cadence=SeriesCadence.DAILY,
        tenor_months=120,
    )
    result = _compute((batch(observations=(two, ten)),))
    assert result.curve is not None
    assert result.curve.spread_2s10s_pp is None
    assert result.curve.level_10y_percent == 1e308


def test_config_fingerprint_and_duplicate_event_defense() -> None:
    source = full_context()
    baseline = _compute((source,))
    altered = macro_config(minimum_history=4)
    assert (
        MacroContextFeatureEngine(altered).compute((source,), BASE).config_fingerprint
        != baseline.config_fingerprint
    )
    duplicate = source.model_copy(
        update={"scheduled_events": (*source.scheduled_events, source.scheduled_events[0])}
    )
    same = _compute((duplicate,))
    assert same.event is not None and baseline.event is not None
    assert same.event.upcoming_count_24h == baseline.event.upcoming_count_24h
    conflicting_event = source.scheduled_events[0].model_copy(
        update={"scheduled_at_ns": BASE + HOUR}
    )
    conflict = source.model_copy(
        update={"scheduled_events": (*source.scheduled_events, conflicting_event)}
    )
    with pytest.raises(ValueError, match="ambiguous"):
        MacroHistoryIndex((conflict,))
    assert (
        MacroContextFeatureEngine(macro_config()).compute((source,), BASE).fingerprint
        == baseline.fingerprint
    )


def test_rate_units_and_unrepresentable_decimal_rate() -> None:
    from mce.features.macro.rates import rate_percent

    assert rate_percent(0.0425, MacroUnit.DECIMAL_RATE) == pytest.approx(4.25)
    assert rate_percent(425, MacroUnit.BASIS_POINTS) == pytest.approx(4.25)
    assert rate_percent(4.25, MacroUnit.PERCENT) == 4.25
    with pytest.raises(ValueError, match="convertible"):
        rate_percent(1, MacroUnit.INDEX_LEVEL)
    with pytest.raises(ValidationError):
        observation(
            "UST10Y",
            MacroCategory.TREASURY_YIELD,
            1e308,
            period_end=BASE - DAY_NS,
            released=BASE - DAY_NS,
            unit=MacroUnit.DECIMAL_RATE,
            cadence=SeriesCadence.DAILY,
            tenor_months=120,
        )
    with pytest.raises(ValidationError):
        release(
            "POLICY",
            MacroCategory.POLICY_RATE,
            1e308,
            None,
            period_end=BASE - DAY_NS,
            released=BASE,
            unit=MacroUnit.DECIMAL_RATE,
            cadence=SeriesCadence.EVENT_DRIVEN,
        )


def test_revised_prior_metadata_and_cancelled_schedule() -> None:
    item = release(
        "GDP", MacroCategory.GROWTH, 2.1, 2.0, period_end=BASE - 90 * DAY_NS, released=BASE - DAY_NS
    )
    revised = item.model_copy(
        update={"prior": 1.8, "revised_prior": 1.6, "revision_status": RevisionStatus.REVISED}
    )
    result = _compute((batch(releases=(revised,)),))
    context = _series(result, "GDP")
    assert context.release is not None and context.release.revised_prior == 1.6
    assert context.surprise is not None and context.surprise.raw == pytest.approx(0.1)
    snapshot = apply_macro_context_to_snapshot(result)
    assert snapshot.feature_values is not None
    assert snapshot.feature_values[f"macro.series.{SOURCE}.GDP.prior"] == 1.8
    assert snapshot.feature_values[f"macro.series.{SOURCE}.GDP.revised_prior"] == 1.6
    scheduled = event("fomc", MacroEventType.FOMC_RATE_DECISION, BASE + HOUR)
    cancelled = scheduled.model_copy(
        update={"status": "cancelled", "calendar_available_at_ns": BASE + MINUTE}
    )
    first = batch(events=(scheduled,))
    second = batch(clock=BASE + MINUTE, events=(cancelled,))
    engine = MacroContextFeatureEngine(macro_config())
    assert engine.compute((first, second), BASE).event.next_event is not None
    after = engine.compute((first, second), BASE + MINUTE)
    assert after.event is not None and after.event.next_event is None
    assert after.event.proximity_score == 0


def test_future_revision_does_not_change_prior_percentile_or_calendar() -> None:
    first = full_context()
    original = _compute((first,))
    current_cpi = next(
        item
        for item in first.observations
        if item.series_id == "CPI_YOY" and item.period_end_ns == BASE - 30 * DAY_NS
    )
    prior_cpi = next(
        item
        for item in first.observations
        if item.series_id == "CPI_YOY" and item.period_end_ns == BASE - 60 * DAY_NS
    )
    future_revision = prior_cpi.model_copy(
        update={
            "value": 50.0,
            "vintage_at_ns": BASE + DAY_NS,
            "available_at_ns": BASE + DAY_NS,
            "revision_status": RevisionStatus.REVISED,
        }
    )
    future_event = event(
        "announced-later", MacroEventType.CPI, BASE + 2 * DAY_NS, announced=BASE + DAY_NS
    )
    later = batch(clock=BASE + DAY_NS, observations=(future_revision,), events=(future_event,))
    engine = MacroContextFeatureEngine(macro_config())
    replay = engine.compute((later, first), BASE)
    assert replay.fingerprint == original.fingerprint
    assert (
        _series(replay, "CPI_YOY").history.percentile
        == _series(original, "CPI_YOY").history.percentile
    )
    assert replay.event is not None and original.event is not None
    assert replay.event.upcoming_count_24h == original.event.upcoming_count_24h
    assert current_cpi.value == 3.5


def test_future_consensus_cannot_change_prior_surprise_scale() -> None:
    first = full_context()
    current = _compute((first,))
    future_release = release(
        "CPI_YOY",
        MacroCategory.INFLATION,
        9.0,
        3.0,
        period_end=BASE,
        released=BASE + DAY_NS,
    )
    later = batch(clock=BASE + DAY_NS, releases=(future_release,))
    replay = _compute((first, later))
    baseline_surprise = _series(current, "CPI_YOY").surprise
    replay_surprise = _series(replay, "CPI_YOY").surprise
    assert replay_surprise == baseline_surprise
    assert replay.fingerprint == current.fingerprint


def test_strict_source_and_event_quality_withholds_proximity_score() -> None:
    source = batch(events=(event("fomc", MacroEventType.FOMC_RATE_DECISION, BASE + MINUTE),))
    poor = source.model_copy(update={"quality": DataQuality.POOR})
    poor_result = _compute((poor,))
    assert poor_result.event is not None and poor_result.event.next_event is not None
    assert poor_result.event.proximity_score is None
    assert poor_result.data_quality == DataQuality.DEGRADED
    config = macro_config()
    strict_settings = config.features.macro.model_copy(update={"minimum_quality": "good"})
    strict_config = config.model_copy(
        update={"features": config.features.model_copy(update={"macro": strict_settings})}
    )
    degraded_event = source.scheduled_events[0].model_copy(update={"quality": DataQuality.DEGRADED})
    result = MacroContextFeatureEngine(strict_config).compute(
        (source.model_copy(update={"scheduled_events": (degraded_event,)}),), BASE
    )
    assert result.event is not None and result.event.next_event is not None
    assert result.event.proximity_score is None
    assert result.event.upcoming_count_24h is None
    assert QualityFlag.MACRO_SOURCE_DEGRADED in result.quality_flags
