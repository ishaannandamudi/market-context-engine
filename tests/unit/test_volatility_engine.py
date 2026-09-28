from datetime import date
from math import log, sqrt

import pytest
from pydantic import ValidationError

from mce.calendar.service import MarketSessionService
from mce.config.settings import (
    EngineSettings,
    FeatureSettings,
    MarketSettings,
    MCEConfig,
    UniverseSettings,
    VolatilitySettings,
)
from mce.features.volatility.engine import RealizedVolatilityEngine
from mce.features.volatility.estimators import yang_zhang
from mce.features.volatility.intraday import completed_intraday_bars, measure_intraday
from mce.features.volatility.models import RealizedVolatilityFeatureSet
from mce.features.volatility.rolling import CompletedSessionReader
from mce.features.volatility.snapshot import apply_volatility_to_snapshot
from mce.models.context import MarketContextSnapshot
from mce.models.enums import DataQuality, QualityFlag
from mce.replay.store import CausalDataStore
from mce.reproducibility import config_fingerprint
from mce.serialization import canonical_json_bytes, canonical_json_str
from tests.fixtures.volatility_scenarios import (
    MINUTE,
    daily_bar,
    daily_store,
    intraday_bars,
    patterned_daily_store,
)


def _engine(
    calendar: MarketSessionService,
    *,
    symbols: tuple[str, ...] = ("SPY", "QQQ", "IWM"),
    volatility: VolatilitySettings | None = None,
) -> RealizedVolatilityEngine:
    return RealizedVolatilityEngine(
        MCEConfig(
            _env_file=None,
            market=MarketSettings(universe=UniverseSettings(broad_market=symbols)),
            features=FeatureSettings(volatility=volatility or VolatilitySettings()),
        ),
        calendar,
    )


def test_intraday_known_variance_and_annualization(calendar: MarketSessionService) -> None:
    day = date(2024, 1, 3)
    bars = intraday_bars(calendar, day, (0.01, -0.01, 0.02, 0.0, 0.01))
    opened, closed = calendar.regular_session_bounds(day)
    result = measure_intraday(bars, opened, closed, bars[-1].bar_end_ns, 5 * MINUTE, MINUTE, 252)
    expected_rv = sum(log(1 + item) ** 2 for item in (0.01, -0.01, 0.02, 0, 0.01))
    assert result.realized_variance == pytest.approx(expected_rv)
    assert result.annualized_volatility == pytest.approx(sqrt(expected_rv * 252 * 390 / 5))
    assert result.observed_return_count == result.expected_return_count == 5
    assert result.coverage_ratio == 1
    assert result.downside_semivariance == pytest.approx(log(0.99) ** 2)
    assert result.downside_share == pytest.approx(log(0.99) ** 2 / expected_rv)


def test_intraday_missing_irregular_and_delayed_bar(calendar: MarketSessionService) -> None:
    day = date(2024, 1, 3)
    complete = intraday_bars(calendar, day, (0.01, 0.01, 0.01, 0.01, 0.01))
    missing = tuple(bar for index, bar in enumerate(complete) if index != 2)
    opened, closed = calendar.regular_session_bounds(day)
    result = measure_intraday(
        missing, opened, closed, complete[-1].bar_end_ns, 5 * MINUTE, MINUTE, 252
    )
    assert result.observed_return_count == 4
    assert result.expected_return_count == 5
    assert result.coverage_ratio == 0.8
    assert result.realized_variance is not None
    trailing_missing = measure_intraday(
        complete[:-1], opened, closed, complete[-1].bar_end_ns, 5 * MINUTE, MINUTE, 252
    )
    assert trailing_missing.expected_return_count == 5
    assert trailing_missing.observed_return_count == 4
    assert trailing_missing.coverage_ratio == 0.8
    assert trailing_missing.realized_variance == pytest.approx(4 * log(1.01) ** 2)
    no_bars = measure_intraday((), opened, closed, complete[-1].bar_end_ns, None, MINUTE, 252)
    assert no_bars.realized_variance is None
    assert no_bars.expected_return_count == 5
    delayed = intraday_bars(calendar, day, (0.01,) * 5, delayed_last_ns=MINUTE)
    store = CausalDataStore()
    for bar in delayed:
        store.add(bar)
    visible, _ = completed_intraday_bars(
        store, calendar, "synthetic", "SPY", day, delayed[-1].bar_end_ns, MINUTE
    )
    assert len(visible) == 4
    visible_after, _ = completed_intraday_bars(
        store, calendar, "synthetic", "SPY", day, delayed[-1].effective_timestamp_ns, MINUTE
    )
    assert len(visible_after) == 5
    shifted = complete[0].model_copy(
        update={
            "bar_start_ns": opened + MINUTE // 2,
            "bar_end_ns": opened + 3 * MINUTE // 2,
            "event_timestamp_ns": opened + MINUTE // 2,
            "effective_timestamp_ns": opened + 3 * MINUTE // 2,
        }
    )
    shifted_measure = measure_intraday(
        (shifted,), opened, closed, shifted.bar_end_ns, None, MINUTE, 252
    )
    assert shifted_measure.observed_return_count == shifted_measure.expected_return_count == 1
    assert shifted_measure.realized_variance == pytest.approx(log(1.01) ** 2)


def test_early_close_session_and_five_minute_cadence(calendar: MarketSessionService) -> None:
    day = date(2024, 11, 29)
    returns = tuple(0.001 if index % 2 else -0.001 for index in range(42))
    bars = intraday_bars(calendar, day, returns, cadence_ns=5 * MINUTE)
    opened, closed = calendar.regular_session_bounds(day)
    assert closed - opened == 210 * MINUTE
    measured = measure_intraday(bars, opened, closed, closed, None, 5 * MINUTE, 252)
    assert measured.observed_return_count == measured.expected_return_count == 42
    assert measured.coverage_ratio == 1
    assert measured.realized_variance is not None
    assert measured.annualized_volatility == pytest.approx(sqrt(measured.realized_variance * 252))


def test_completed_session_reader_uses_exchange_adjacency(calendar: MarketSessionService) -> None:
    friday = date(2024, 5, 24)
    tuesday = date(2024, 5, 28)
    store = CausalDataStore()
    for day, price in ((friday, 100.0), (tuesday, 103.0)):
        store.add(daily_bar(calendar, day, opening=price, closing=price))
    reader = CompletedSessionReader(store, calendar, "synthetic", MINUTE)
    as_of = calendar.regular_session_bounds(tuesday)[1]
    history = reader.history("SPY", tuesday, 2, as_of)
    assert [item.day for item in history if item is not None] == [friday, tuesday]
    assert reader.session("SPY", tuesday, as_of - 1) is None


def test_yang_zhang_holiday_and_early_close_session_chain(
    calendar: MarketSessionService,
) -> None:
    dates = (
        date(2024, 11, 26),
        date(2024, 11, 27),
        date(2024, 11, 29),
        date(2024, 12, 2),
    )
    store = CausalDataStore()
    for index, day in enumerate(dates):
        store.add(
            daily_bar(
                calendar,
                day,
                opening=100 + index,
                closing=101 + index,
                high=102 + index,
                low=99 + index,
            )
        )
    reader = CompletedSessionReader(store, calendar, "synthetic", MINUTE)
    as_of = calendar.regular_session_bounds(dates[-1])[1]
    history = reader.history("SPY", dates[-1], 4, as_of)
    assert tuple(item.day for item in history if item is not None) == dates
    assert yang_zhang(tuple(item for item in history if item is not None), 252) is not None
    missing = CausalDataStore()
    for day in (dates[0], dates[2], dates[3]):
        missing.add(daily_bar(calendar, day))
    assert reader.session("SPY", dates[2], as_of) is not None
    assert (
        CompletedSessionReader(missing, calendar, "synthetic", MINUTE).history(
            "SPY", dates[-1], 4, as_of
        )[1]
        is None
    )


def test_intraday_duplicate_interval_is_flagged(calendar: MarketSessionService) -> None:
    day = date(2024, 1, 3)
    bars = intraday_bars(calendar, day, (0.001,) * 5)
    replacement = bars[-1].model_copy(
        update={
            "close": bars[-1].close * 1.001,
            "high": bars[-1].high * 1.001,
            "effective_timestamp_ns": bars[-1].effective_timestamp_ns + 1,
        }
    )
    store = CausalDataStore()
    for bar in (*bars, replacement):
        store.add(bar)
    visible, duplicate = completed_intraday_bars(
        store, calendar, "synthetic", "SPY", day, replacement.effective_timestamp_ns, MINUTE
    )
    assert duplicate
    assert visible[-1].close == replacement.close
    result = _engine(calendar, symbols=("SPY",)).compute(store, replacement.effective_timestamp_ns)
    assert (
        QualityFlag.TIMESTAMP_MISMATCH in result.per_symbol["SPY"].intraday["session"].quality_flags
    )


def test_full_intraday_aggregation_and_incomplete_rejection(calendar: MarketSessionService) -> None:
    day = date(2024, 1, 3)
    bars = intraday_bars(calendar, day, (0.0001,) * 390)
    store = CausalDataStore()
    for bar in bars:
        store.add(bar)
    reader = CompletedSessionReader(store, calendar, "synthetic", MINUTE)
    closed = calendar.regular_session_bounds(day)[1]
    summary = reader.session("SPY", day, closed)
    assert summary is not None
    assert summary.open == bars[0].open
    assert summary.close == bars[-1].close
    assert reader.session("SPY", day, closed - 1) is None
    incomplete = CausalDataStore()
    for bar in bars[:-1]:
        incomplete.add(bar)
    assert (
        CompletedSessionReader(incomplete, calendar, "synthetic", MINUTE).session(
            "SPY", day, closed
        )
        is None
    )


def test_daily_engine_estimators_and_broad_market(calendar: MarketSessionService) -> None:
    store, as_of = daily_store(calendar)
    result = _engine(calendar).compute(store, as_of)
    assert result.market_realized_volatility is not None
    assert result.volatility_score is not None
    assert result.available_symbol_count == result.expected_symbol_count == 3
    assert result.coverage_ratio == 1
    symbol = result.per_symbol["SPY"]
    assert (
        symbol.rv_short is not None and symbol.rv_medium is not None and symbol.rv_long is not None
    )
    assert symbol.short_long_ratio is not None
    assert symbol.volatility_acceleration is not None
    assert symbol.volatility_change_1d is not None
    assert symbol.volatility_percentile is not None
    assert symbol.baseline_sample_count >= 20
    window = symbol.daily["20d"]
    assert all(
        getattr(window, name) is not None
        for name in ("close_to_close", "parkinson", "garman_klass", "rogers_satchell", "yang_zhang")
    )
    assert window.estimator_range is not None and window.estimator_agreement is not None
    assert window.overnight_variance_share == 0
    assert window.intraday_variance_share == 1
    assert canonical_json_str(
        RealizedVolatilityFeatureSet.model_validate_json(canonical_json_str(result))
    ) == canonical_json_str(result)


def test_high_vs_low_and_index_disagreement(calendar: MarketSessionService) -> None:
    store, as_of = daily_store(calendar, amplitudes=(0.001, 0.006, 0.001))
    result = _engine(calendar).compute(store, as_of)
    qqq = result.per_symbol["QQQ"].rv_medium
    spy = result.per_symbol["SPY"].rv_medium
    assert qqq is not None and spy is not None and qqq > spy * 5
    assert (
        result.cross_index_volatility_range is not None and result.cross_index_volatility_range > 0
    )
    assert (
        result.cross_index_volatility_ratio is not None and result.cross_index_volatility_ratio > 5
    )
    all_low_store, _ = daily_store(calendar, amplitudes=(0.001, 0.001, 0.001))
    all_high_store, _ = daily_store(calendar, amplitudes=(0.01, 0.01, 0.01))
    low_market = _engine(calendar).compute(all_low_store, as_of).market_realized_volatility
    high_market = _engine(calendar).compute(all_high_store, as_of).market_realized_volatility
    assert low_market is not None and high_market is not None
    assert high_market > low_market * 9


def test_time_varying_volatility_and_prior_only_shock(calendar: MarketSessionService) -> None:
    rising_path = tuple(0.001 + 0.009 * index / 89 for index in range(90))
    falling_path = tuple(reversed(rising_path))
    rising_store, as_of = patterned_daily_store(calendar, rising_path)
    falling_store, _ = patterned_daily_store(calendar, falling_path)
    engine = _engine(calendar, symbols=("SPY",))
    rising = engine.compute(rising_store, as_of).per_symbol["SPY"]
    falling = engine.compute(falling_store, as_of).per_symbol["SPY"]
    assert rising.short_long_ratio is not None and rising.short_long_ratio > 1
    assert falling.short_long_ratio is not None and falling.short_long_ratio < 1
    assert rising.volatility_acceleration is not None and rising.volatility_acceleration > 0
    assert falling.volatility_acceleration is not None and falling.volatility_acceleration < 0
    stable_store, _ = patterned_daily_store(calendar, (0.002,) * 90)
    spike_store, _ = patterned_daily_store(calendar, (0.002,) * 89 + (0.08,))
    stable = engine.compute(stable_store, as_of).per_symbol["SPY"]
    spike = engine.compute(spike_store, as_of).per_symbol["SPY"]
    assert stable.volatility_shock_score is not None
    assert spike.volatility_shock_score == 1
    assert spike.volatility_percentile == 1
    assert spike.rv_short is not None and stable.rv_short is not None
    assert spike.rv_short > stable.rv_short


def test_overnight_vs_intraday_paths_and_zero_ratio(calendar: MarketSessionService) -> None:
    engine = _engine(calendar, symbols=("SPY",))
    overnight_store, as_of = patterned_daily_store(calendar, (0.001,) * 90, gaps=(0.02,) * 90)
    intraday_store, _ = patterned_daily_store(calendar, (0.02,) * 90, gaps=(0.001,) * 90)
    overnight = engine.compute(overnight_store, as_of).per_symbol["SPY"].daily["20d"]
    intraday = engine.compute(intraday_store, as_of).per_symbol["SPY"].daily["20d"]
    assert overnight.overnight_variance_share is not None
    assert intraday.overnight_variance_share is not None
    assert overnight.overnight_variance_share > 0.95
    assert intraday.overnight_variance_share < 0.05
    flat_store, _ = patterned_daily_store(calendar, (0.0,) * 90)
    flat = engine.compute(flat_store, as_of).per_symbol["SPY"]
    assert flat.rv_short == 0
    assert flat.short_long_ratio is None
    assert flat.volatility_acceleration is None


def test_partial_market_coverage_custom_weights_and_missing_baseline(
    calendar: MarketSessionService,
) -> None:
    store, as_of = daily_store(calendar, symbols=("SPY", "QQQ"), amplitudes=(0.001, 0.004))
    default = _engine(calendar).compute(store, as_of)
    assert default.market_realized_volatility is None
    assert default.coverage_ratio == pytest.approx(2 / 3)
    assert QualityFlag.PARTIAL_UNIVERSE in default.quality_flags
    relaxed = VolatilitySettings(minimum_market_coverage=0.6)
    equal = _engine(calendar, volatility=relaxed).compute(store, as_of)
    weighted = _engine(
        calendar,
        volatility=VolatilitySettings(
            minimum_market_coverage=0.6,
            index_weights={"spy": 3.0, "QQQ": 1.0, "IWM": 1.0},
        ),
    ).compute(store, as_of)
    assert equal.market_realized_volatility is not None
    assert weighted.market_realized_volatility is not None
    assert weighted.market_realized_volatility < equal.market_realized_volatility
    short_store, short_time = daily_store(calendar, count=8)
    short = _engine(calendar).compute(short_store, short_time)
    assert short.volatility_score is None
    assert short.per_symbol["SPY"].volatility_percentile is None


def test_one_stale_symbol_preserves_other_results(calendar: MarketSessionService) -> None:
    store, as_of = daily_store(calendar)
    missing_last = CausalDataStore()
    for symbol in ("SPY", "QQQ", "IWM"):
        bars = store.bars(symbol, 0, as_of + 1, as_of)
        for bar in bars[:-1] if symbol == "SPY" else bars:
            missing_last.add(bar)
    result = _engine(calendar).compute(missing_last, as_of)
    assert QualityFlag.STALE_DATA in result.per_symbol["SPY"].quality_flags
    assert QualityFlag.STALE_DATA not in result.per_symbol["QQQ"].quality_flags
    assert result.per_symbol["QQQ"].rv_medium is not None
    assert result.per_symbol["SPY"].rv_medium is None
    assert result.available_symbol_count == 2


def test_future_daily_bar_cannot_leak_into_intraday_snapshot(
    calendar: MarketSessionService,
) -> None:
    store, _ = daily_store(calendar, end=date(2024, 6, 27), count=90)
    day = date(2024, 6, 28)
    opened, closed = calendar.regular_session_bounds(day)
    as_of = opened + 60 * MINUTE
    engine = _engine(calendar)
    baseline = canonical_json_bytes(engine.compute(store, as_of))
    for symbol in ("SPY", "QQQ", "IWM"):
        store.add(
            daily_bar(calendar, day, symbol=symbol, opening=100, closing=200, high=200, low=100)
        )
    assert canonical_json_bytes(engine.compute(store, as_of)) == baseline
    after_close = engine.compute(store, closed)
    assert after_close.per_symbol["SPY"].daily["20d"].close_to_close is not None
    assert canonical_json_bytes(after_close) != baseline


def test_future_daily_session_cannot_change_historical_percentile(
    calendar: MarketSessionService,
) -> None:
    store, as_of = daily_store(calendar, end=date(2024, 6, 27))
    engine = _engine(calendar)
    before = canonical_json_bytes(engine.compute(store, as_of))
    for symbol in ("SPY", "QQQ", "IWM"):
        store.add(
            daily_bar(
                calendar,
                date(2024, 6, 28),
                symbol=symbol,
                opening=100,
                closing=200,
                high=200,
                low=100,
            )
        )
    assert canonical_json_bytes(engine.compute(store, as_of)) == before


def test_future_intraday_bar_and_order_invariance(calendar: MarketSessionService) -> None:
    store, daily_as_of = daily_store(calendar)
    day = date(2024, 6, 28)
    bars = intraday_bars(calendar, day, (0.001,) * 30)
    for symbol in ("SPY", "QQQ", "IWM"):
        for bar in bars:
            store.add(bar.model_copy(update={"symbol": symbol}))
    as_of = bars[-1].bar_end_ns
    engine = _engine(calendar)
    baseline = canonical_json_bytes(engine.compute(store, as_of))
    future = intraday_bars(calendar, day, (0.001,) * 31)[-1]
    for symbol in ("SPY", "QQQ", "IWM"):
        store.add(future.model_copy(update={"symbol": symbol}))
    assert canonical_json_bytes(engine.compute(store, as_of)) == baseline
    reversed_store = CausalDataStore()
    for symbol in ("SPY", "QQQ", "IWM"):
        for bar in reversed(store.bars(symbol, 0, daily_as_of + 1, as_of)):
            reversed_store.add(bar)
    assert canonical_json_bytes(engine.compute(reversed_store, as_of)) == baseline


def test_stale_daily_input_and_snapshot_mapping(calendar: MarketSessionService) -> None:
    store, as_of = daily_store(calendar)
    result = _engine(calendar).compute(store, as_of + 10 * MINUTE)
    assert QualityFlag.STALE_DATA in result.quality_flags
    base = MarketContextSnapshot(
        as_of_ns=as_of + 10 * MINUTE, trend_score=0.4, feature_values={"trend.existing": 0.4}
    )
    snapshot = apply_volatility_to_snapshot(result, base)
    assert snapshot.realized_volatility == result.market_realized_volatility
    assert snapshot.volatility_score == result.volatility_score
    assert snapshot.trend_score == 0.4
    assert snapshot.volatility_regime.value == "unknown"
    assert snapshot.index_scores is None
    assert snapshot.feature_values is not None
    assert snapshot.feature_values["trend.existing"] == 0.4
    assert "volatility.close_to_close.SPY.20d" in snapshot.feature_values
    assert canonical_json_str(
        MarketContextSnapshot.model_validate_json(canonical_json_str(snapshot))
    ) == canonical_json_str(snapshot)
    with pytest.raises(ValueError):
        apply_volatility_to_snapshot(result, MarketContextSnapshot(as_of_ns=as_of))


def test_volatility_settings_validation_and_fingerprint(monkeypatch: pytest.MonkeyPatch) -> None:
    baseline = MCEConfig(_env_file=None)
    changed = MCEConfig(
        _env_file=None,
        features=FeatureSettings(
            volatility=VolatilitySettings(
                trading_sessions_per_year=250,
            )
        ),
    )
    assert config_fingerprint(changed) != config_fingerprint(baseline)
    assert config_fingerprint(baseline) == config_fingerprint(
        MCEConfig(_env_file=None, engine=EngineSettings(log_level="DEBUG"))
    )
    monkeypatch.setenv("MCE_FEATURES__VOLATILITY__TRADING_SESSIONS_PER_YEAR", "250")
    assert MCEConfig(_env_file=None).features.volatility.trading_sessions_per_year == 250
    with pytest.raises(ValidationError):
        VolatilitySettings(short_window_sessions=20, medium_window_sessions=5)
    with pytest.raises(ValidationError):
        VolatilitySettings(enabled_estimators=("parkinson", "parkinson"))
    with pytest.raises(ValidationError):
        VolatilitySettings(index_weights={"spy": 1, "SPY": 1})
    with pytest.raises(ValidationError):
        VolatilitySettings(minimum_window_coverage=1.1)
    with pytest.raises(ValidationError):
        VolatilitySettings(minimum_baseline_samples=100)
    with pytest.raises(ValidationError):
        VolatilitySettings(intraday_horizons_ns={"session": MINUTE})


def test_disabled_and_empty_market(calendar: MarketSessionService) -> None:
    disabled = _engine(calendar, volatility=VolatilitySettings(enabled=False)).compute(
        CausalDataStore(), 1
    )
    assert disabled.data_quality == DataQuality.UNKNOWN
    with pytest.raises(ValueError):
        _engine(calendar, symbols=()).compute(CausalDataStore(), 1)
    with pytest.raises(ValueError):
        _engine(calendar).compute(CausalDataStore(), -1)
    with pytest.raises(ValueError):
        _engine(calendar, volatility=VolatilitySettings(index_weights={"SPY": 1})).compute(
            CausalDataStore(), 1
        )
