from datetime import date
from math import isclose

import pytest
from pydantic import ValidationError

from mce.calendar.service import MarketSessionService
from mce.features.normalization import signed_tanh
from mce.features.returns import log_return, simple_return
from mce.features.trend import (
    distance_from_sma,
    log_price_regression,
    path_efficiency,
    persistence,
    sign_agreement,
)
from mce.features.windows import FeatureWindowReader, WindowSpec, WindowUnit
from mce.replay.store import CausalDataStore
from tests.fixtures.feature_scenarios import MINUTE, bars_for_day, trend_store


@pytest.mark.parametrize(
    "start,end,expected",
    [
        (100.0, 110.0, 0.1),
        (100.0, 90.0, -0.1),
        (100.0, 100.0, 0.0),
        (None, 100.0, None),
        (0.0, 1.0, None),
        (100.0, None, None),
    ],
)
def test_simple_return(start: float | None, end: float | None, expected: float | None) -> None:
    actual = simple_return(start, end)
    assert (actual is None and expected is None) or (
        actual is not None and expected is not None and isclose(actual, expected)
    )


def test_log_return_and_normalization_boundaries() -> None:
    assert log_return(100, 100) == 0
    assert log_return(100, 110) is not None and log_return(100, 110) > 0
    assert log_return(100, 90) is not None and log_return(100, 90) < 0
    assert log_return(0, 1) is None
    assert log_return(1, 0) is None
    assert signed_tanh(0, 1) == 0
    assert 0 < signed_tanh(1, 1) < 1
    with pytest.raises(ValueError):
        signed_tanh(float("nan"), 1)
    with pytest.raises(ValueError):
        signed_tanh(1, 0)


@pytest.mark.parametrize("direction", [1, -1, 0])
def test_regression_direction_and_scale(direction: int, calendar: MarketSessionService) -> None:
    prices = tuple(100 + direction * index * 0.1 for index in range(10))
    bars = bars_for_day(calendar, date(2024, 1, 3), prices=prices)
    result = log_price_regression(bars)
    assert result is not None
    assert result.sample_count == 10
    assert result.slope_per_hour * direction >= 0
    if direction == 0:
        assert result.slope_per_hour == 0
        assert result.r_squared == 1
    else:
        assert result.slope_per_hour != 0
        scaled = bars_for_day(calendar, date(2024, 1, 3), prices=tuple(p * 4 for p in prices))
        scaled_result = log_price_regression(scaled)
        assert scaled_result is not None
        assert isclose(result.slope_per_hour, scaled_result.slope_per_hour, rel_tol=1e-10)


def test_regression_irregular_noisy_and_insufficient(calendar: MarketSessionService) -> None:
    bars = bars_for_day(calendar, date(2024, 1, 3), prices=(100, 101, 100.5, 103, 104))
    irregular = (bars[0], bars[2], bars[4])
    result = log_price_regression(irregular)
    assert result is not None and result.slope_per_hour > 0
    assert 0 < result.r_squared < 1
    assert log_price_regression((bars[0],)) is None
    assert persistence(irregular) == 1
    assert distance_from_sma(irregular) is not None
    assert path_efficiency(irregular) == 1


def test_regression_rejects_nonpositive_and_duplicate_times(calendar: MarketSessionService) -> None:
    bars = bars_for_day(calendar, date(2024, 1, 3), prices=(100, 101, 102))
    assert log_price_regression((bars[0], bars[0])) is None
    zero = bars[0].model_copy(update={"close": 0.0})
    assert log_price_regression((zero, bars[1])) is None
    assert persistence((zero, bars[1])) is None
    assert path_efficiency((zero, bars[1])) is None
    assert distance_from_sma(()) is None
    assert sign_agreement((1, 0, -1))[1]
    with pytest.raises(ValueError):
        sign_agreement((float("inf"),))


def test_persistence_agreement_and_flat(calendar: MarketSessionService) -> None:
    day = date(2024, 1, 3)
    up = bars_for_day(calendar, day, prices=(100, 101, 102, 103))
    down = bars_for_day(calendar, day, prices=(103, 102, 101, 100))
    alternating = bars_for_day(calendar, day, prices=(100, 101, 100, 101, 100))
    flat = bars_for_day(calendar, day, prices=(100, 100, 100))
    assert persistence(up) == persistence(down) == 1
    assert persistence(alternating) == 0
    assert persistence(flat) == path_efficiency(flat) == distance_from_sma(flat) == 0
    assert sign_agreement((1, 2, 3)) == (1, False)
    assert sign_agreement((-1, -2)) == (1, False)
    assert sign_agreement((1, -1)) == (0, True)
    assert sign_agreement(()) == (None, False)
    assert sign_agreement((0, 0)) == (0, False)


def test_window_validation() -> None:
    with pytest.raises(ValidationError):
        WindowSpec(name="bad name", unit=WindowUnit.ELAPSED, length=1)
    with pytest.raises(ValidationError):
        WindowSpec(name="x", unit=WindowUnit.SESSION, length=2)
    with pytest.raises(ValidationError):
        WindowSpec(name="x", unit=WindowUnit.OBSERVATIONS, length=0)


def test_elapsed_and_count_windows_are_causal(calendar: MarketSessionService) -> None:
    store, as_of = trend_store(calendar, symbols=("SPY",), points=5)
    reader = FeatureWindowReader(store, calendar, MINUTE)
    minute = WindowSpec(name="1m", unit=WindowUnit.ELAPSED, length=MINUTE)
    sample = reader.select("SPY", as_of, minute)
    assert sample.bars[-1].bar_end_ns == as_of
    assert sample.sample_count == 2
    assert sample.reference_price == 100.15
    delayed, _ = trend_store(calendar, symbols=("SPY",), points=6, delayed_last_ns=MINUTE)
    assert (
        FeatureWindowReader(delayed, calendar, MINUTE)
        .select("SPY", as_of, minute)
        .bars[-1]
        .bar_end_ns
        == as_of
    )
    count = WindowSpec(name="three", unit=WindowUnit.OBSERVATIONS, length=3)
    assert reader.select("SPY", as_of, count).sample_count == 3
    assert reader.select("SPY", as_of, count).coverage_ratio == 1
    assert reader.select("MISSING", as_of, minute).sample_count == 0


def test_elapsed_boundary_with_off_grid_clock_and_five_minute_bars(
    calendar: MarketSessionService,
) -> None:
    store, as_of = trend_store(calendar, symbols=("SPY",), points=5)
    reader = FeatureWindowReader(store, calendar, MINUTE)
    minute = WindowSpec(name="1m", unit=WindowUnit.ELAPSED, length=MINUTE)
    off_grid = reader.select("SPY", as_of + 20_000_000_000, minute)
    assert off_grid.reference_price == 100.15
    assert off_grid.bars[-1].close == 100.2
    five_minute = CausalDataStore()
    day = date(2024, 1, 3)
    bars = bars_for_day(calendar, day, prices=(100, 101))
    opened, _ = calendar.regular_session_bounds(day)
    first = bars[0].model_copy(
        update={
            "bar_end_ns": opened + 5 * MINUTE,
            "effective_timestamp_ns": opened + 5 * MINUTE,
        }
    )
    second = bars[1].model_copy(
        update={
            "bar_start_ns": opened + 5 * MINUTE,
            "bar_end_ns": opened + 10 * MINUTE,
            "event_timestamp_ns": opened + 5 * MINUTE,
            "effective_timestamp_ns": opened + 10 * MINUTE,
        }
    )
    five_minute.add(first)
    five_minute.add(second)
    five_reader = FeatureWindowReader(five_minute, calendar, 5 * MINUTE)
    assert five_reader.select("SPY", opened + 10 * MINUTE, minute).reference_price is None
    five = WindowSpec(name="5m", unit=WindowUnit.ELAPSED, length=5 * MINUTE)
    assert five_reader.select("SPY", opened + 10 * MINUTE, five).reference_price == 100


def test_session_early_close_and_trading_day_reference(calendar: MarketSessionService) -> None:
    store = CausalDataStore()
    early = date(2024, 11, 29)
    bars = bars_for_day(calendar, early, prices=(100, 101, 102))
    for bar in bars:
        store.add(bar)
    reader = FeatureWindowReader(store, calendar, MINUTE)
    session = reader.select(
        "SPY", bars[-1].bar_end_ns, WindowSpec(name="session", unit=WindowUnit.SESSION, length=1)
    )
    assert session.reference_price == 100
    assert session.bars[-1].close == 102
    assert (
        reader.select(
            "SPY",
            bars[-1].bar_end_ns,
            WindowSpec(name="1d", unit=WindowUnit.TRADING_DAYS, length=1),
        ).reference_price
        is None
    )
    opened, closed = calendar.regular_session_bounds(early)
    assert closed - opened == 210 * MINUTE


def test_five_trading_days_cross_holiday(calendar: MarketSessionService) -> None:
    target = date(2024, 12, 30)
    reference = target
    for _ in range(5):
        reference = calendar.previous_trading_day(reference)
    assert reference == date(2024, 12, 20)
    store = CausalDataStore()
    opened, closed = calendar.regular_session_bounds(reference)
    last = bars_for_day(calendar, reference, prices=(99,))[0].model_copy(
        update={
            "bar_start_ns": closed - MINUTE,
            "bar_end_ns": closed,
            "event_timestamp_ns": closed - MINUTE,
            "effective_timestamp_ns": closed,
        }
    )
    store.add(last)
    today = bars_for_day(calendar, target, prices=(100,))[0]
    store.add(today)
    reader = FeatureWindowReader(store, calendar, MINUTE)
    sample = reader.select(
        "SPY", today.bar_end_ns, WindowSpec(name="5d", unit=WindowUnit.TRADING_DAYS, length=5)
    )
    assert sample.reference_price == 99
    assert sample.bars[-1] == today
    assert opened < closed


def test_early_close_is_daily_reference(calendar: MarketSessionService) -> None:
    early = date(2024, 11, 29)
    next_day = calendar.next_trading_day(early)
    assert next_day == date(2024, 12, 2)
    _, early_close = calendar.regular_session_bounds(early)
    prior = bars_for_day(calendar, early, prices=(99,))[0].model_copy(
        update={
            "bar_start_ns": early_close - MINUTE,
            "bar_end_ns": early_close,
            "event_timestamp_ns": early_close - MINUTE,
            "effective_timestamp_ns": early_close,
        }
    )
    current = bars_for_day(calendar, next_day, prices=(101,))[0]
    store = CausalDataStore()
    store.add(prior)
    store.add(current)
    result = FeatureWindowReader(store, calendar, MINUTE).select(
        "SPY", current.bar_end_ns, WindowSpec(name="1d", unit=WindowUnit.TRADING_DAYS, length=1)
    )
    assert result.reference_price == 99
    assert result.bars[-1].close == 101
