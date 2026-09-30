"""Causal, point-in-time session normalization with real synthetic observations."""

from datetime import date

import pytest

from mce.calendar.service import MarketSessionService
from mce.features.breadth.membership import MembershipTimeline, UniverseMembership
from mce.features.correlation.models import CorrelationFeatureSet
from mce.features.seasonality.engine import IntradaySeasonalityEngine
from mce.features.seasonality.readers import read_session_metrics
from mce.features.seasonality.snapshot import apply_seasonality_to_snapshot
from mce.models.context import MarketContextSnapshot
from mce.models.enums import DataQuality, MarketRegime, QualityFlag
from mce.models.market_data import BarObservation, QuoteObservation
from mce.replay.store import CausalDataStore
from tests.fixtures.seasonal_scenarios import MINUTE, SYMBOLS, add_session, seasonal_config

PRIOR = (date(2024, 1, 3), date(2024, 1, 4), date(2024, 1, 5))
TARGET = date(2024, 1, 8)


def _prepared(
    *,
    current_minutes: int = 30,
    spike: frozenset[int] = frozenset(),
    missing: frozenset[tuple[str, int]] = frozenset(),
    quote_until: int | None = None,
    crossed: frozenset[int] = frozenset(),
    prior_minutes: int = 30,
    volume_scale: float = 1.0,
) -> tuple[MarketSessionService, CausalDataStore, IntradaySeasonalityEngine, int]:
    calendar = MarketSessionService()
    store = CausalDataStore(max_observations=20_000)
    for day in PRIOR:
        add_session(store, calendar, day, minutes=prior_minutes)
    add_session(
        store,
        calendar,
        TARGET,
        minutes=current_minutes,
        spike_minutes=spike,
        missing=missing,
        volume_scale=volume_scale,
        quote_until=quote_until,
        crossed_minutes=crossed,
    )
    engine = IntradaySeasonalityEngine(seasonal_config(), calendar=calendar)
    opened, _ = calendar.regular_session_bounds(TARGET)
    return calendar, store, engine, opened


def _metric(result: object, symbol: str, metric: str) -> object:
    from mce.features.seasonality.models import IntradaySeasonalityFeatureSet

    assert isinstance(result, IntradaySeasonalityFeatureSet)
    return next(
        item
        for member in result.per_symbol
        if member.symbol == symbol
        for item in member.metrics
        if item.metric == metric
    )


def test_opening_u_shape_and_cumulative_adjusted_rvol() -> None:
    _, store, engine, opened = _prepared()
    history = engine.build_history(store, opened)
    opening = engine.compute(store, opened + 5 * MINUTE, history)
    assert opening.data_quality == DataQuality.GOOD
    for metric in (
        "volume",
        "cumulative_volume",
        "dollar_volume",
        "volatility",
        "quoted_spread",
        "spread",
    ):
        value = _metric(opening, "AAA", metric)
        assert value.ratio == pytest.approx(1)  # type: ignore[attr-defined]
        assert value.percentile == 0.5  # type: ignore[attr-defined]
    assert _metric(opening, "AAA", "volume").raw_value == 500  # type: ignore[attr-defined]
    midday = engine.compute(store, opened + 30 * MINUTE, history)
    assert _metric(midday, "AAA", "volume").raw_value == 500  # type: ignore[attr-defined]
    assert _metric(midday, "AAA", "volume").ratio == pytest.approx(1)  # type: ignore[attr-defined]
    assert midday.market_metrics[0].coverage == 1
    snapshot = apply_seasonality_to_snapshot(
        opening,
        MarketContextSnapshot(
            as_of_ns=opening.as_of_ns,
            market_regime=MarketRegime.RISK_OFF,
            feature_values={"liquidity.relative_volume": 0.25, "seasonality.volume.ratio": 999},
        ),
    )
    assert snapshot.schema_version == "1.0" and snapshot.engine_version == "0.17.0"
    assert snapshot.market_regime == MarketRegime.RISK_OFF
    assert snapshot.feature_values is not None
    assert snapshot.feature_values["liquidity.relative_volume"] == 0.25
    assert snapshot.feature_values["seasonality.adjusted_rvol"] == pytest.approx(1)
    assert snapshot.feature_values["seasonality.volume.ratio"] == pytest.approx(1)
    assert opening.fingerprint == engine.compute(store, opening.as_of_ns, history).fingerprint


def test_unusual_volume_volatility_spread_without_opening_false_alarm() -> None:
    _, store, engine, opened = _prepared(spike=frozenset(range(5, 10)))
    history = engine.build_history(store, opened)
    opening = engine.compute(store, opened + 5 * MINUTE, history)
    unusual = engine.compute(store, opened + 10 * MINUTE, history)
    assert _metric(opening, "AAA", "volume").ratio == 1  # type: ignore[attr-defined]
    assert _metric(unusual, "AAA", "volume").ratio == 5  # type: ignore[attr-defined]
    assert _metric(unusual, "AAA", "volatility").ratio == pytest.approx(25)  # type: ignore[attr-defined]
    assert _metric(unusual, "AAA", "quoted_spread").ratio > 4.9  # type: ignore[attr-defined]
    assert _metric(unusual, "AAA", "spread").ratio == pytest.approx(5)  # type: ignore[attr-defined]
    assert _metric(unusual, "AAA", "cumulative_volume").ratio == pytest.approx(3)  # type: ignore[attr-defined]
    assert _metric(unusual, "AAA", "volume").percentile == 1  # type: ignore[attr-defined]


def test_same_absolute_volume_is_abnormal_at_midday() -> None:
    _, store, engine, opened = _prepared(
        current_minutes=185, prior_minutes=185, spike=frozenset(range(180, 185))
    )
    history = engine.build_history(store, opened)
    opening = engine.compute(store, opened + 5 * MINUTE, history)
    midday = engine.compute(store, opened + 185 * MINUTE, history)
    assert _metric(opening, "AAA", "volume").raw_value == 500  # type: ignore[attr-defined]
    assert _metric(midday, "AAA", "volume").raw_value == 500  # type: ignore[attr-defined]
    assert _metric(opening, "AAA", "volume").ratio == 1  # type: ignore[attr-defined]
    assert _metric(midday, "AAA", "volume").ratio == 5  # type: ignore[attr-defined]
    _, low_store, low_engine, low_open = _prepared(volume_scale=0.5)
    low_history = low_engine.build_history(low_store, low_open)
    low = low_engine.compute(low_store, low_open + 5 * MINUTE, low_history)
    assert _metric(low, "AAA", "volume").ratio == pytest.approx(0.5)  # type: ignore[attr-defined]


def test_missing_bar_and_stale_or_crossed_quote_withhold_normalization() -> None:
    _, store, engine, opened = _prepared(missing=frozenset({("AAA", 7)}), quote_until=5)
    history = engine.build_history(store, opened)
    result = engine.compute(store, opened + 10 * MINUTE, history)
    assert _metric(result, "AAA", "volume").raw_value is None  # type: ignore[attr-defined]
    assert _metric(result, "AAA", "cumulative_volume").ratio is None  # type: ignore[attr-defined]
    assert _metric(result, "BBB", "volume").ratio == pytest.approx(1)  # type: ignore[attr-defined]
    assert _metric(result, "BBB", "spread").ratio is None  # type: ignore[attr-defined]
    assert result.member_coverage == pytest.approx(2 / 3)
    assert QualityFlag.LOW_COVERAGE in result.quality_flags
    assert QualityFlag.MISSING_DATA in result.quality_flags
    crossed_calendar, crossed_store, crossed_engine, crossed_open = _prepared(
        crossed=frozenset({4})
    )
    del crossed_calendar
    crossed_history = crossed_engine.build_history(crossed_store, crossed_open)
    crossed_result = crossed_engine.compute(
        crossed_store, crossed_open + 5 * MINUTE, crossed_history
    )
    assert _metric(crossed_result, "AAA", "spread").ratio is None  # type: ignore[attr-defined]
    assert QualityFlag.REJECTED_DATA in crossed_result.quality_flags
    _, stale_store, stale_engine, stale_open = _prepared(quote_until=6)
    stale_history = stale_engine.build_history(stale_store, stale_open)
    stale_result = stale_engine.compute(stale_store, stale_open + 10 * MINUTE, stale_history)
    assert _metric(stale_result, "AAA", "spread").ratio is None  # type: ignore[attr-defined]
    assert QualityFlag.STALE_DATA in stale_result.quality_flags


def test_future_bar_quote_and_membership_do_not_change_past() -> None:
    _, store, engine, opened = _prepared()
    future = opened + 40 * MINUTE
    provider = MembershipTimeline(
        tuple(UniverseMembership(symbol=symbol, effective_to_ns=future) for symbol in SYMBOLS)
        + tuple(UniverseMembership(symbol=symbol, effective_from_ns=future) for symbol in SYMBOLS)
        + (UniverseMembership(symbol="DDD", effective_from_ns=future, available_at_ns=future),)
    )
    engine = IntradaySeasonalityEngine(engine.config, provider, engine.calendar)
    history = engine.build_history(store, opened)
    asof = opened + 10 * MINUTE
    original = engine.compute(store, asof, history)
    before = original.fingerprint
    store.add(
        BarObservation(
            symbol="AAA",
            source="synthetic",
            event_timestamp_ns=asof,
            effective_timestamp_ns=future,
            bar_start_ns=asof - MINUTE,
            bar_end_ns=asof,
            open=100,
            high=100,
            low=100,
            close=100,
            volume=1_000_000,
        )
    )
    store.add(
        QuoteObservation(
            symbol="AAA",
            source="synthetic",
            event_timestamp_ns=asof,
            effective_timestamp_ns=future,
            bid=80,
            ask=120,
        )
    )
    add_session(store, engine.calendar, TARGET, minutes=45)
    assert engine.compute(store, asof, history).fingerprint == before
    assert engine.compute(store, future + 5 * MINUTE, history).expected_symbols == SYMBOLS + (
        "DDD",
    )


def test_historical_membership_and_short_history_are_causal() -> None:
    calendar, store, original, opened = _prepared()
    _, jan4_close = calendar.regular_session_bounds(PRIOR[1])
    jan5_open, _ = calendar.regular_session_bounds(PRIOR[2])
    provider = MembershipTimeline(
        (
            UniverseMembership(symbol="AAA"),
            UniverseMembership(symbol="BBB"),
            UniverseMembership(symbol="CCC", effective_from_ns=jan5_open),
        )
    )
    assert provider.members(jan4_close).members[-1].symbol == "BBB"
    engine = IntradaySeasonalityEngine(original.config, provider, calendar)
    history = engine.build_history(store, opened)
    assert len(history.samples("volume", "CCC", 0)) == 1
    result = engine.compute(store, opened + 5 * MINUTE, history)
    assert _metric(result, "CCC", "volume").raw_value == 500  # type: ignore[attr-defined]
    assert _metric(result, "CCC", "volume").ratio is None  # type: ignore[attr-defined]
    assert result.market_metrics[0].coverage == pytest.approx(2 / 3)
    with pytest.raises(ValueError, match="mismatch"):
        engine.compute(
            store, opened + 5 * MINUTE, engine.build_history(store, opened + 24 * 60 * MINUTE)
        )


def _correlation(asof: int, value: float) -> CorrelationFeatureSet:
    return CorrelationFeatureSet(
        as_of_ns=asof,
        expected_symbols=SYMBOLS,
        valid_symbols=SYMBOLS,
        expected_symbol_count=3,
        valid_symbol_count=3,
        symbol_coverage=1,
        expected_observations=20,
        aligned_sample_count=20,
        time_coverage=1,
        matrix_dimension=0,
        average_pairwise_correlation=value,
        evidence_quality=1,
        data_quality=DataQuality.GOOD,
    )


def test_same_bucket_upstream_correlation_only() -> None:
    calendar, store, engine, opened = _prepared()
    prior_contexts = tuple(
        _correlation(calendar.regular_session_bounds(day)[0] + 5 * MINUTE, value)
        for day, value in zip(PRIOR, (0.2, 0.3, 0.4), strict=True)
    )
    history = engine.build_history(store, opened, correlation_history=prior_contexts)
    current = engine.compute(
        store, opened + 5 * MINUTE, history, correlation=_correlation(opened + 5 * MINUTE, 0.6)
    )
    assert current.correlation is not None
    assert current.correlation.ratio == pytest.approx(2)
    assert current.correlation.percentile == 1
    future_context = _correlation(opened + 10 * MINUTE, 0.9)
    with pytest.raises(ValueError, match="future correlation"):
        engine.compute(store, opened + 5 * MINUTE, history, correlation=future_context)
    ignored = engine.compute(
        store, opened + 10 * MINUTE, history, correlation=_correlation(opened + 9 * MINUTE, 0.7)
    )
    assert ignored.correlation is None
    assert apply_seasonality_to_snapshot(current).feature_values[
        "seasonality.correlation.ratio"
    ] == pytest.approx(2)  # type: ignore[index]


def test_missing_historical_bucket_and_future_session_are_absent() -> None:
    calendar = MarketSessionService()
    store = CausalDataStore(max_observations=20_000)
    add_session(store, calendar, PRIOR[0], minutes=10)
    add_session(store, calendar, PRIOR[1], minutes=10, missing=frozenset({("AAA", 2)}))
    add_session(store, calendar, PRIOR[2], minutes=10)
    add_session(store, calendar, TARGET, minutes=10)
    engine = IntradaySeasonalityEngine(seasonal_config(), calendar=calendar)
    opened, _ = calendar.regular_session_bounds(TARGET)
    history = engine.build_history(store, opened)
    sample = _metric(engine.compute(store, opened + 5 * MINUTE, history), "AAA", "volume")
    assert sample.baseline.sample_count == 2  # type: ignore[attr-defined]
    assert sample.baseline.coverage == pytest.approx(2 / 3)  # type: ignore[attr-defined]
    first_fingerprint = history.fingerprint
    add_session(store, calendar, date(2024, 1, 9), minutes=10, volume_scale=100)
    assert engine.build_history(store, opened).fingerprint == first_fingerprint
    assert history.samples("volume", "AAA", 0)[0].trading_date < TARGET


def test_shortened_session_fraction_cumulative_and_wall_clock_missing_late_bucket() -> None:
    calendar = MarketSessionService()
    store = CausalDataStore(max_observations=30_000)
    prior_days = (date(2024, 11, 25), date(2024, 11, 26), date(2024, 11, 27))
    early = date(2024, 11, 29)
    for day in prior_days:
        add_session(store, calendar, day, minutes=260)
    add_session(store, calendar, early, minutes=210)
    fraction = IntradaySeasonalityEngine(
        seasonal_config(
            bucket_mode="session_fraction", bucket_minutes=None, session_fraction_bins=7
        ),
        calendar=calendar,
    )
    opened, closed = calendar.regular_session_bounds(early)
    history = fraction.build_history(store, opened)
    result = fraction.compute(store, opened + 30 * MINUTE, history)
    value = _metric(result, "AAA", "cumulative_volume")
    assert value.raw_value == 3000  # type: ignore[attr-defined]
    assert value.ratio is not None  # type: ignore[attr-defined]
    assert value.ratio < 1  # type: ignore[attr-defined]
    assert result.session_time is not None and result.session_time.is_early_close
    last = fraction.compute(store, closed, history)
    assert last.session_time is not None and last.session_time.completed_bucket is not None
    assert last.session_time.completed_bucket.index == 6
    # The following normal session has two comparable late wall-clock samples;
    # Thanksgiving Friday had no bucket after its 13:00 ET close.
    add_session(store, calendar, date(2024, 12, 2), minutes=260)
    wall = IntradaySeasonalityEngine(seasonal_config(), calendar=calendar)
    dec2_open, _ = calendar.regular_session_bounds(date(2024, 12, 2))
    wall_history = wall.build_history(store, dec2_open)
    assert date(2024, 11, 29) in wall_history.prior_dates
    assert len(wall_history.samples("volume", "AAA", 50)) == 2


def test_extreme_finite_inputs_do_not_emit_nonfinite_metrics() -> None:
    calendar = MarketSessionService()
    opened, _ = calendar.regular_session_bounds(TARGET)
    store = CausalDataStore()
    store.add(
        BarObservation(
            symbol="AAA",
            source="synthetic",
            event_timestamp_ns=opened + MINUTE,
            effective_timestamp_ns=opened + MINUTE,
            bar_start_ns=opened,
            bar_end_ns=opened + MINUTE,
            open=1e308,
            high=1e308,
            low=1e308,
            close=1e308,
            volume=1e308,
        )
    )
    config = seasonal_config(bucket_minutes=1)
    read = read_session_metrics(
        store, calendar, config, "AAA", TARGET, opened + MINUTE, current=True
    )
    assert read.missing_bars and not read.observations
