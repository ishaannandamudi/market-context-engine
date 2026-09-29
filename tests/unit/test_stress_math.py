from datetime import date
from math import log

import pytest

from mce.calendar.service import MarketSessionService
from mce.config.settings import LiquiditySettings, StressScoreWeights, StressSettings
from mce.features.stress.composite import (
    dollar_volume_concentration,
    gap_stress,
    liquidity_score,
    weighted_score,
)
from mce.features.stress.history import prior_rank, summarize_stress_history
from mce.features.stress.models import StressHistoryPoint
from mce.features.stress.primitives import read_session, read_spread
from mce.models.market_data import BarObservation, QuoteObservation
from mce.replay.store import CausalDataStore
from tests.fixtures.breadth_scenarios import MINUTE
from tests.fixtures.stress_scenarios import stress_store


def test_reference_volume_amihud_gap_and_quoted_spread(calendar: MarketSessionService) -> None:
    store, as_of = stress_store(
        calendar,
        volumes=(100,) * 6,
        returns=(0.01,) * 6,
        gaps=(-0.1,) * 6,
        quote_width=0.04,
    )
    day = calendar.session_for_timestamp(as_of).trading_date
    assert day is not None
    sample = read_session(store, calendar, "T1", day, as_of, "synthetic", 2 * MINUTE, MINUTE)
    assert sample.volume == 300
    assert sample.dollar_volume == pytest.approx(27090)
    assert sample.session_return == pytest.approx(0.01)
    assert sample.amihud == pytest.approx(0.01 / 27090)
    assert sample.gap_return == pytest.approx(log(0.9))
    opened, closed = calendar.regular_session_bounds(day)
    quote = read_spread(store, "T1", as_of, opened, closed, "synthetic", 10_000_000_000)
    assert quote.absolute == pytest.approx(0.04)
    assert quote.relative == pytest.approx(0.04 / 90.9)
    assert not quote.stale and not quote.invalid


def test_score_reference_vectors_and_missing_components() -> None:
    settings = LiquiditySettings(minimum_score_components=2)
    score, weights, contributors = liquidity_score(0.5, 1.0, 0.001, settings)
    assert score == pytest.approx(0.5)
    assert sum(value for _, value in weights) == pytest.approx(1)
    assert sum(item.contribution or 0 for item in contributors) == pytest.approx(score)
    partial, weights, terms = liquidity_score(0.8, 2.0, None, settings)
    assert partial == pytest.approx((0.45 * 0.2 + 0.35 * (2 / 3)) / 0.8)
    assert len(weights) == len(terms) == 2
    assert liquidity_score(None, None, 0.001, settings)[0] is None
    composite, active, components = weighted_score(
        (("a", 1.0, 0.25), ("b", 0.0, 0.75)), 2, "stress"
    )
    assert composite == pytest.approx(0.25)
    assert active == (("a", 0.25), ("b", 0.75))
    assert sum(item.contribution or 0 for item in components) == pytest.approx(0.25)
    assert gap_stress(1, 0.02, 1, StressSettings(gap_scale=0.02)) == 1
    assert gap_stress(0, 0.02, 1, StressSettings(gap_scale=0.02)) == 0
    assert gap_stress(None, None, None, StressSettings()) is None


def test_history_midrank_changes_acceleration_persistence_future_exclusion() -> None:
    rank = prior_rank(3, (1, 2, 3, 4, 5), 5)
    assert rank.percentile == 0.5 and rank.median == 3
    assert rank.robust_z == 0
    assert prior_rank(2, (1,) * 5, 5).robust_z is None
    assert prior_rank(1, (1, 2), 3).percentile is None
    assert prior_rank(1e308, (1e308, 1e308), 2).median is None
    base = tuple(
        StressHistoryPoint(as_of_ns=index, available_at_ns=index, stress_score=value)
        for index, value in enumerate((0.2, 0.4, 0.6, 0.8, 0.1), start=1)
    )
    future = StressHistoryPoint(as_of_ns=20, available_at_ns=20, stress_score=1)
    result = summarize_stress_history(0.9, base + (future,), 10, 5, 3, 3)
    assert result.sample_count == 5 and result.percentile == 1
    assert result.change == pytest.approx(0.8)
    assert result.acceleration == pytest.approx(1.5)
    assert result.persistence == pytest.approx(2 / 3)
    assert result.shock is not None and result.shock > 0
    assert summarize_stress_history(0.9, base, 10, 5, 3, 3) == result


def test_concentration_config_and_quote_safety(calendar: MarketSessionService) -> None:
    assert dollar_volume_concentration((1, 1)) == pytest.approx((0.5, 2))
    assert dollar_volume_concentration((0, 0)) is None
    assert dollar_volume_concentration((1, -1)) is None
    with pytest.raises(ValueError):
        StressScoreWeights(volatility=0.5)
    with pytest.raises(ValueError):
        LiquiditySettings(historical_sessions=2, minimum_history_samples=3)
    with pytest.raises(ValueError):
        StressSettings(historical_baseline_observations=2, minimum_history_samples=3)
    store, as_of = stress_store(calendar, quote_width=None)
    opened, closed = calendar.regular_session_bounds(
        calendar.session_for_timestamp(as_of).trading_date
    )
    assert (
        read_spread(store, "T1", as_of, opened, closed, "synthetic", 10_000_000_000).relative
        is None
    )
    store.add(
        QuoteObservation(
            symbol="T1",
            source="synthetic",
            event_timestamp_ns=as_of,
            effective_timestamp_ns=as_of,
            bid=101,
            ask=100,
        )
    )
    assert read_spread(store, "T1", as_of, opened, closed, "synthetic", 10_000_000_000).invalid


def test_spread_zero_bid_zero_width_staleness_and_future_quote(
    calendar: MarketSessionService,
) -> None:
    store, as_of = stress_store(calendar, quote_width=None)
    day = calendar.session_for_timestamp(as_of).trading_date
    assert day is not None
    opened, closed = calendar.regular_session_bounds(day)
    store.add(
        QuoteObservation(
            symbol="T1",
            source="synthetic",
            event_timestamp_ns=as_of,
            effective_timestamp_ns=as_of,
            bid=0,
            ask=1,
        )
    )
    assert read_spread(store, "T1", as_of, opened, closed, "synthetic", MINUTE).invalid
    store.add(
        QuoteObservation(
            symbol="T2",
            source="synthetic",
            event_timestamp_ns=as_of,
            effective_timestamp_ns=as_of,
            bid=100,
            ask=100,
        )
    )
    zero = read_spread(store, "T2", as_of, opened, closed, "synthetic", MINUTE)
    assert zero.absolute == zero.relative == 0 and not zero.invalid
    assert read_spread(store, "T2", as_of + 2 * MINUTE, opened, closed, "synthetic", MINUTE).stale
    store.add(
        QuoteObservation(
            symbol="T2",
            source="synthetic",
            event_timestamp_ns=as_of + MINUTE,
            effective_timestamp_ns=as_of + MINUTE,
            bid=50,
            ask=150,
        )
    )
    assert read_spread(store, "T2", as_of, opened, closed, "synthetic", MINUTE) == zero
    store.add(
        QuoteObservation(
            symbol="T3",
            source="synthetic",
            event_timestamp_ns=closed + MINUTE,
            effective_timestamp_ns=closed + MINUTE,
            bid=99,
            ask=101,
        )
    )
    assert (
        read_spread(store, "T3", closed + MINUTE, opened, closed, "synthetic", MINUTE).relative
        is None
    )


def test_amihud_price_and_volume_scaling(calendar: MarketSessionService) -> None:
    base, as_of = stress_store(calendar, volumes=(100,) * 6, returns=(0.02,) * 6)
    half_volume, _ = stress_store(calendar, volumes=(50,) * 6, returns=(0.02,) * 6)
    day = calendar.session_for_timestamp(as_of).trading_date
    assert day is not None
    normal = read_session(base, calendar, "T1", day, as_of, "synthetic", MINUTE, MINUTE)
    thin = read_session(half_volume, calendar, "T1", day, as_of, "synthetic", MINUTE, MINUTE)
    assert normal.amihud is not None and thin.amihud is not None
    assert thin.amihud == pytest.approx(2 * normal.amihud)


def test_extreme_prices_do_not_emit_nonfinite_liquidity(
    calendar: MarketSessionService,
) -> None:
    store = CausalDataStore()
    day = date(2024, 6, 28)
    previous = calendar.previous_trading_day(day)
    _, prior_close = calendar.regular_session_bounds(previous)
    opened, _ = calendar.regular_session_bounds(day)
    store.add(
        BarObservation(
            symbol="EXTREME",
            source="synthetic",
            event_timestamp_ns=prior_close,
            effective_timestamp_ns=prior_close,
            bar_start_ns=prior_close - MINUTE,
            bar_end_ns=prior_close,
            open=1e-320,
            high=1e-320,
            low=1e-320,
            close=1e-320,
            volume=1,
        )
    )
    store.add(
        BarObservation(
            symbol="EXTREME",
            source="synthetic",
            event_timestamp_ns=opened + MINUTE,
            effective_timestamp_ns=opened + MINUTE,
            bar_start_ns=opened,
            bar_end_ns=opened + MINUTE,
            open=1e-320,
            high=1e308,
            low=1e-320,
            close=1e308,
            volume=1e-320,
        )
    )
    result = read_session(
        store, calendar, "EXTREME", day, opened + MINUTE, "synthetic", MINUTE, MINUTE
    )
    assert result.session_return is None
    assert result.amihud is None
    assert result.gap_return == pytest.approx(0)


def test_missing_bar_does_not_understate_cumulative_volume(
    calendar: MarketSessionService,
) -> None:
    store = CausalDataStore()
    day = date(2024, 6, 28)
    opened, _ = calendar.regular_session_bounds(day)
    for minute in (1, 3):
        store.add(
            BarObservation(
                symbol="HOLE",
                source="synthetic",
                event_timestamp_ns=opened + minute * MINUTE,
                effective_timestamp_ns=opened + minute * MINUTE,
                bar_start_ns=opened + (minute - 1) * MINUTE,
                bar_end_ns=opened + minute * MINUTE,
                open=100,
                high=100,
                low=100,
                close=100,
                volume=100,
            )
        )
    result = read_session(
        store, calendar, "HOLE", day, opened + 3 * MINUTE, "synthetic", MINUTE, MINUTE
    )
    assert result.volume is None and result.dollar_volume is None
