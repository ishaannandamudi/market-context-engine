from datetime import date

import pytest
from pydantic import ValidationError

from mce.calendar.service import MarketSessionService
from mce.config.settings import (
    EngineSettings,
    FeatureSettings,
    MarketSettings,
    MCEConfig,
    TrendScoreWeights,
    TrendSettings,
    UniverseSettings,
)
from mce.features.engine import TrendFeatureEngine
from mce.features.models import TrendFeatureSet
from mce.features.snapshot import apply_trend_to_snapshot
from mce.features.windows import WindowSpec, WindowUnit
from mce.models.context import MarketContextSnapshot
from mce.models.enums import DataQuality, QualityFlag
from mce.models.market_data import BarObservation
from mce.replay.store import CausalDataStore
from mce.reproducibility import config_fingerprint
from mce.serialization import canonical_json_bytes, canonical_json_str
from tests.fixtures.feature_scenarios import MINUTE, bars_for_day, trend_store


def _engine(
    calendar: MarketSessionService,
    *,
    symbols: tuple[str, ...] = ("SPY", "QQQ", "IWM"),
    trend: TrendSettings | None = None,
) -> TrendFeatureEngine:
    config = MCEConfig(
        _env_file=None,
        market=MarketSettings(universe=UniverseSettings(broad_market=symbols)),
        features=FeatureSettings(trend=trend or TrendSettings()),
    )
    return TrendFeatureEngine(config, calendar)


@pytest.mark.parametrize("direction", [1, -1, 0])
def test_directional_market_scenarios(direction: int, calendar: MarketSessionService) -> None:
    store, as_of = trend_store(calendar, direction=direction)
    result = _engine(calendar).compute(store, as_of)
    assert result.available_symbol_count == 3
    assert result.coverage_ratio == 1
    assert result.market_direction_score is not None
    assert result.market_direction_score * direction >= 0
    assert result.index_agreement == (0 if direction == 0 else 1)
    assert result.symbols["SPY"].sample_count == 31
    if direction == 0:
        assert result.market_direction_score == 0
        assert result.symbols["SPY"].persistence == 0
    else:
        assert result.symbols["SPY"].persistence == 1


def test_complete_history_can_be_good(calendar: MarketSessionService) -> None:
    store, as_of = trend_store(calendar)
    current_day = date(2024, 1, 3)
    prior = calendar.previous_trading_day(current_day)
    fifth = current_day
    for _ in range(5):
        fifth = calendar.previous_trading_day(fifth)
    for day in (prior, fifth):
        _, closing = calendar.regular_session_bounds(day)
        for symbol in ("SPY", "QQQ", "IWM"):
            store.add(
                BarObservation(
                    symbol=symbol,
                    source="synthetic",
                    bar_start_ns=closing - MINUTE,
                    bar_end_ns=closing,
                    event_timestamp_ns=closing - MINUTE,
                    effective_timestamp_ns=closing,
                    open=99.0,
                    high=99.0,
                    low=99.0,
                    close=99.0,
                    volume=0.0,
                )
            )
    result = _engine(calendar).compute(store, as_of)
    assert result.data_quality == DataQuality.GOOD
    assert not result.quality_flags
    assert result.symbols["SPY"].returns["1d"].value is not None
    assert result.symbols["SPY"].returns["5d"].value is not None


def test_mixed_market_disagreement_and_weighting(calendar: MarketSessionService) -> None:
    up, as_of = trend_store(calendar, symbols=("SPY", "QQQ"), direction=1)
    down, _ = trend_store(calendar, symbols=("IWM",), direction=-1)
    for bar in down.bars("IWM", 0, as_of + 1, as_of):
        up.add(bar)
    equal = _engine(calendar).compute(up, as_of)
    weighted = _engine(
        calendar,
        trend=TrendSettings(index_weights={"SPY": 3.0, "QQQ": 1.0, "IWM": 1.0}),
    ).compute(up, as_of)
    assert equal.market_direction_score is not None
    assert weighted.market_direction_score is not None
    assert weighted.market_direction_score > equal.market_direction_score
    huge = _engine(
        calendar,
        trend=TrendSettings(index_weights={"SPY": 1e308, "QQQ": 1e308, "IWM": 1e308}),
    ).compute(up, as_of)
    assert huge.market_direction_score == pytest.approx(equal.market_direction_score)
    assert equal.index_agreement is not None and equal.index_agreement < 1
    assert QualityFlag.INDEX_DISAGREEMENT in equal.quality_flags
    assert equal.trend_score is not None
    assert abs(equal.trend_score) < abs(equal.market_direction_score)
    assert sum(item.contribution or 0 for item in equal.contributors) == pytest.approx(
        equal.market_direction_score
    )


def test_partial_universe_threshold_and_staleness(calendar: MarketSessionService) -> None:
    store, as_of = trend_store(calendar, symbols=("SPY", "QQQ"))
    result = _engine(calendar).compute(store, as_of)
    assert result.coverage_ratio == pytest.approx(2 / 3)
    assert result.market_direction_score is None
    assert result.data_quality == DataQuality.UNUSABLE
    assert QualityFlag.PARTIAL_UNIVERSE in result.quality_flags
    relaxed = _engine(calendar, trend=TrendSettings(minimum_market_coverage=0.6)).compute(
        store, as_of
    )
    assert relaxed.market_direction_score is not None
    assert relaxed.data_quality == DataQuality.DEGRADED
    stale = _engine(calendar, trend=TrendSettings(minimum_market_coverage=0.6)).compute(
        store, as_of + 10 * MINUTE
    )
    assert stale.market_direction_score is not None
    assert QualityFlag.STALE_DATA in stale.quality_flags


def test_insufficient_and_missing_bars(calendar: MarketSessionService) -> None:
    small, as_of = trend_store(calendar, points=2)
    result = _engine(calendar).compute(small, as_of)
    assert result.market_direction_score is None
    assert QualityFlag.INSUFFICIENT_HISTORY in result.quality_flags
    missing, later = trend_store(calendar, missing=frozenset(range(40, 46)))
    reduced = _engine(calendar).compute(missing, later)
    assert reduced.symbols["SPY"].slope_long.sample_count < 31
    assert QualityFlag.INSUFFICIENT_WINDOW_COVERAGE in reduced.quality_flags


def test_future_bars_and_insertion_order_do_not_change_past(calendar: MarketSessionService) -> None:
    store, as_of = trend_store(calendar)
    engine = _engine(calendar)
    baseline = canonical_json_bytes(engine.compute(store, as_of))
    future = bars_for_day(calendar, date(2024, 1, 3), prices=(150,))[0].model_copy(
        update={
            "bar_start_ns": as_of,
            "bar_end_ns": as_of + MINUTE,
            "event_timestamp_ns": as_of,
            "effective_timestamp_ns": as_of + MINUTE,
        }
    )
    store.add(future)
    assert canonical_json_bytes(engine.compute(store, as_of)) == baseline
    reversed_store = CausalDataStore()
    all_bars = [
        bar for symbol in ("SPY", "QQQ", "IWM") for bar in store.bars(symbol, 0, as_of + 1, as_of)
    ]
    for bar in reversed(all_bars):
        reversed_store.add(bar)
    assert canonical_json_bytes(engine.compute(reversed_store, as_of)) == baseline


def test_completed_but_delayed_bar_is_invisible(calendar: MarketSessionService) -> None:
    store, as_of = trend_store(calendar)
    engine = _engine(calendar)
    baseline = canonical_json_bytes(engine.compute(store, as_of))
    for symbol in ("SPY", "QQQ", "IWM"):
        store.add(
            BarObservation(
                symbol=symbol,
                source="synthetic",
                bar_start_ns=as_of,
                bar_end_ns=as_of + MINUTE,
                event_timestamp_ns=as_of,
                effective_timestamp_ns=as_of + 2 * MINUTE,
                open=103.0,
                high=200.0,
                low=103.0,
                close=200.0,
                volume=1.0,
            )
        )
    assert canonical_json_bytes(engine.compute(store, as_of)) == baseline
    assert canonical_json_bytes(engine.compute(store, as_of + 2 * MINUTE)) != baseline


def test_configured_source_isolated_from_other_provider(calendar: MarketSessionService) -> None:
    store, as_of = trend_store(calendar)
    engine = _engine(calendar)
    baseline = canonical_json_bytes(engine.compute(store, as_of))
    for bar in store.bars("SPY", 0, as_of + 1, as_of):
        store.add(
            bar.model_copy(
                update={
                    "source": "another",
                    "close": bar.close * 2,
                    "open": bar.open * 2,
                    "high": bar.high * 2,
                    "low": bar.low * 2,
                }
            )
        )
    assert canonical_json_bytes(engine.compute(store, as_of)) == baseline


def test_positive_price_scaling_preserves_normalized_score(calendar: MarketSessionService) -> None:
    store, as_of = trend_store(calendar)
    scaled = CausalDataStore()
    for symbol in ("SPY", "QQQ", "IWM"):
        for bar in store.bars(symbol, 0, as_of + 1, as_of):
            scaled.add(
                bar.model_copy(
                    update={
                        "open": bar.open * 7,
                        "high": bar.high * 7,
                        "low": bar.low * 7,
                        "close": bar.close * 7,
                    }
                )
            )
    engine = _engine(calendar)
    original_score = engine.compute(store, as_of).market_direction_score
    scaled_score = engine.compute(scaled, as_of).market_direction_score
    assert scaled_score == pytest.approx(original_score, abs=1e-10)


def test_extended_hours_bar_does_not_enter_regular_features(
    calendar: MarketSessionService,
) -> None:
    store, as_of = trend_store(calendar)
    engine = _engine(calendar)
    baseline = engine.compute(store, as_of)
    _, regular_close = calendar.regular_session_bounds(date(2024, 1, 3))
    store.add(
        BarObservation(
            symbol="SPY",
            source="synthetic",
            bar_start_ns=regular_close,
            bar_end_ns=regular_close + MINUTE,
            event_timestamp_ns=regular_close,
            effective_timestamp_ns=regular_close + MINUTE,
            open=200.0,
            high=200.0,
            low=200.0,
            close=200.0,
            volume=1.0,
        )
    )
    later = engine.compute(store, regular_close + MINUTE)
    assert (
        later.symbols["SPY"].returns["session"].value
        == baseline.symbols["SPY"].returns["session"].value
    )
    assert QualityFlag.STALE_DATA in later.symbols["SPY"].quality_flags


def test_short_reversal_with_long_uptrend(calendar: MarketSessionService) -> None:
    store = CausalDataStore()
    prices = tuple(100 + 0.1 * i if i < 55 else 105.4 - 0.3 * (i - 54) for i in range(61))
    bars = bars_for_day(calendar, date(2024, 1, 3), prices=prices)
    for symbol in ("SPY", "QQQ", "IWM"):
        for bar in bars:
            store.add(bar.model_copy(update={"symbol": symbol}))
    result = _engine(calendar).compute(store, bars[-1].bar_end_ns)
    symbol = result.symbols["SPY"]
    assert symbol.returns["1m"].value is not None and symbol.returns["1m"].value < 0
    assert symbol.returns["30m"].value is not None and symbol.returns["30m"].value > 0
    assert QualityFlag.HORIZON_DISAGREEMENT in symbol.quality_flags
    assert symbol.acceleration_per_hour is not None and symbol.acceleration_per_hour < 0


@pytest.mark.parametrize(
    "prices,expected_sign",
    [
        (tuple(100 + 0.002 * i * i for i in range(61)), 1),
        (tuple(100 + 0.1 * i - 0.0007 * i * i for i in range(61)), -1),
        (tuple(100 - 0.002 * i * i for i in range(61)), -1),
        (tuple(100.0 for _ in range(61)), 0),
    ],
)
def test_acceleration_slope_differential(
    prices: tuple[float, ...], expected_sign: int, calendar: MarketSessionService
) -> None:
    bars = bars_for_day(calendar, date(2024, 1, 3), prices=prices)
    store = CausalDataStore()
    for bar in bars:
        store.add(bar)
    result = _engine(calendar, symbols=("SPY",)).compute(store, bars[-1].bar_end_ns)
    acceleration = result.symbols["SPY"].acceleration_per_hour
    assert acceleration is not None
    assert acceleration * expected_sign >= 0
    if expected_sign == 0:
        assert acceleration == 0


def test_snapshot_projection_roundtrip_and_preservation(calendar: MarketSessionService) -> None:
    store, as_of = trend_store(calendar)
    result = _engine(calendar).compute(store, as_of)
    base = MarketContextSnapshot(
        as_of_ns=as_of,
        realized_volatility=0.2,
        feature_values={"existing": 3},
        quality_flags={QualityFlag.REJECTED_DATA},
    )
    snapshot = apply_trend_to_snapshot(result, base)
    assert snapshot.market_direction_score == result.market_direction_score
    assert snapshot.trend_score == result.trend_score
    assert snapshot.index_scores is not None and len(snapshot.index_scores) == 3
    assert snapshot.feature_values is not None
    assert snapshot.feature_values["existing"] == 3
    assert "trend.return.SPY.5m" in snapshot.feature_values
    assert snapshot.realized_volatility == 0.2
    assert snapshot.breadth_score is None
    assert snapshot.market_regime.value == "unknown"
    assert QualityFlag.REJECTED_DATA in snapshot.quality_flags
    assert snapshot.contributors is not None
    assert any(item.name == "SPY.trend.returns" for item in snapshot.contributors)
    assert canonical_json_str(
        MarketContextSnapshot.model_validate_json(canonical_json_str(snapshot))
    ) == canonical_json_str(snapshot)
    assert canonical_json_str(
        TrendFeatureSet.model_validate_json(canonical_json_str(result))
    ) == canonical_json_str(result)
    with pytest.raises(ValueError):
        apply_trend_to_snapshot(result, MarketContextSnapshot(as_of_ns=as_of + 1))
    unusable = apply_trend_to_snapshot(
        result, MarketContextSnapshot(as_of_ns=as_of, data_quality=DataQuality.UNUSABLE)
    )
    assert unusable.data_quality == DataQuality.UNUSABLE
    upgraded = apply_trend_to_snapshot(
        result, MarketContextSnapshot(as_of_ns=as_of, engine_version="0.2.0")
    )
    assert upgraded.engine_version == "0.11.0"


def test_feature_result_rejects_inconsistent_counts_or_clock(
    calendar: MarketSessionService,
) -> None:
    store, as_of = trend_store(calendar)
    result = _engine(calendar).compute(store, as_of)
    payload = result.model_dump(mode="python")
    payload["available_symbol_count"] = 2
    with pytest.raises(ValidationError):
        TrendFeatureSet.model_validate(payload)
    payload = result.model_dump(mode="python")
    payload["as_of_ns"] = as_of + 1
    with pytest.raises(ValidationError):
        TrendFeatureSet.model_validate(payload)


def test_trend_config_and_fingerprint(monkeypatch: pytest.MonkeyPatch) -> None:
    baseline = MCEConfig(_env_file=None)
    changed = MCEConfig(
        _env_file=None, features=FeatureSettings(trend=TrendSettings(return_scale=0.02))
    )
    assert config_fingerprint(baseline) != config_fingerprint(changed)
    changed_horizon = MCEConfig(
        _env_file=None,
        features=FeatureSettings(
            trend=TrendSettings(
                horizons=(WindowSpec(name="three", unit=WindowUnit.OBSERVATIONS, length=3),)
            )
        ),
    )
    assert config_fingerprint(baseline) != config_fingerprint(changed_horizon)
    assert config_fingerprint(baseline) == config_fingerprint(
        MCEConfig(_env_file=None, engine=EngineSettings(log_level="DEBUG"))
    )
    monkeypatch.setenv("MCE_FEATURES__TREND__RETURN_SCALE", "0.02")
    assert MCEConfig(_env_file=None).features.trend.return_scale == 0.02
    with pytest.raises(ValidationError):
        TrendScoreWeights(returns=0.1)
    with pytest.raises(ValidationError):
        TrendSettings(horizons=(WindowSpec(name="x", unit=WindowUnit.ELAPSED, length=MINUTE),) * 2)
    with pytest.raises(ValidationError):
        TrendSettings(short_slope_window_ns=MINUTE, long_slope_window_ns=MINUTE)
    with pytest.raises(ValidationError):
        TrendSettings(index_weights={"SPY": float("inf")})
    assert TrendSettings(index_weights={" spy ": 2.0}).index_weights["SPY"] == 2.0
    with pytest.raises(ValidationError):
        TrendSettings(index_weights={"spy": 1.0, "SPY": 1.0})


def test_empty_market_and_disabled_feature(calendar: MarketSessionService) -> None:
    assert _engine(calendar).compute(CausalDataStore(), 1).data_quality == DataQuality.UNUSABLE
    disabled = _engine(calendar, trend=TrendSettings(enabled=False)).compute(CausalDataStore(), 1)
    assert disabled.data_quality == DataQuality.UNKNOWN
    with pytest.raises(ValueError):
        _engine(calendar, symbols=()).compute(CausalDataStore(), 1)
    with pytest.raises(ValueError):
        _engine(calendar).compute(CausalDataStore(), -1)
    with pytest.raises(ValueError):
        _engine(calendar, trend=TrendSettings(index_weights={"SPY": 1})).compute(
            CausalDataStore(), 1
        )
    incompatible = MarketSessionService(MarketSettings(calendar_start=date(2010, 1, 1)))
    with pytest.raises(ValueError, match="calendar settings"):
        TrendFeatureEngine(MCEConfig(_env_file=None), incompatible)
