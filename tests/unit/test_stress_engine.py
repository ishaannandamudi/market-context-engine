from datetime import date

import pytest

from mce.calendar.service import MarketSessionService
from mce.config.settings import FeatureSettings, LiquiditySettings, MCEConfig, StressSettings
from mce.features.breadth.membership import MembershipTimeline, UniverseMembership
from mce.features.breadth.models import BreadthFeatureSet
from mce.features.correlation.models import CorrelationFeatureSet
from mce.features.dispersion.models import DispersionFeatureSet
from mce.features.sectors.models import SectorContext, SectorFeatureSet
from mce.features.stress.engine import LiquidityStressFeatureEngine
from mce.features.stress.models import LiquidityStressFeatureSet, StressHistoryPoint
from mce.features.stress.snapshot import apply_liquidity_stress_to_snapshot
from mce.features.volatility.models import RealizedVolatilityFeatureSet
from mce.models.context import MarketContextSnapshot
from mce.models.enums import MarketRegime, QualityFlag
from mce.models.market_data import BarObservation, QuoteObservation
from mce.replay.store import CausalDataStore
from mce.reproducibility import config_fingerprint
from mce.serialization import canonical_json_str
from tests.fixtures.breadth_scenarios import MINUTE
from tests.fixtures.stress_scenarios import SYMBOLS, stress_store


def _settings(**updates: object) -> LiquiditySettings:
    return LiquiditySettings(
        universe=SYMBOLS,
        historical_sessions=5,
        minimum_history_samples=3,
        price_adjustment_status="adjusted",
    ).model_copy(update=updates)


def _engine(
    calendar: MarketSessionService,
    *,
    membership: MembershipTimeline | None = None,
    settings: LiquiditySettings | None = None,
    stress_settings: StressSettings | None = None,
) -> LiquidityStressFeatureEngine:
    return LiquidityStressFeatureEngine(
        MCEConfig(
            _env_file=None,
            features=FeatureSettings(
                liquidity=settings or _settings(),
                stress=stress_settings or StressSettings(minimum_history_samples=3),
            ),
        ),
        membership,
        calendar,
    )


def _upstream(
    as_of: int,
    *,
    vol: float = 0.8,
    corr: float = 0.8,
    breadth: float = -0.8,
    dispersion: float = 0.9,
    sectors: tuple[float, ...] = (-0.03, -0.02, -0.01),
) -> dict[str, object]:
    return {
        "volatility_context": RealizedVolatilityFeatureSet.model_construct(
            as_of_ns=as_of,
            volatility_score=vol,
        ),
        "correlation_context": CorrelationFeatureSet.model_construct(
            as_of_ns=as_of,
            systemic_co_movement_score=corr,
        ),
        "breadth_context": BreadthFeatureSet.model_construct(
            as_of_ns=as_of,
            breadth_score=breadth,
        ),
        "dispersion_context": DispersionFeatureSet.model_construct(
            as_of_ns=as_of,
            dispersion_score=dispersion,
        ),
        "sector_context": SectorFeatureSet.model_construct(
            as_of_ns=as_of,
            sectors={
                str(index): SectorContext.model_construct(absolute_returns={"1d": value})
                for index, value in enumerate(sectors)
            },
        ),
    }


def test_liquidity_reference_numbers_snapshot_and_serialization(
    calendar: MarketSessionService,
) -> None:
    store, as_of = stress_store(calendar)
    result = _engine(calendar).compute(store, as_of)
    liquid, stress = result.liquidity, result.stress
    assert liquid.expected_member_count == liquid.valid_member_count == 6
    assert liquid.member_coverage == liquid.quote_coverage == 1
    assert liquid.session_volume == 1800
    assert liquid.relative_volume == pytest.approx(1)
    assert liquid.relative_volume_5d == pytest.approx(1)
    assert liquid.relative_volume_20d is None
    assert liquid.volume_percentile == pytest.approx(0.5)
    assert liquid.illiquidity_percentile == pytest.approx(0.5)
    assert liquid.quoted_spread == pytest.approx(0.02)
    assert liquid.liquidity_score is not None and 0 < liquid.liquidity_score < 1
    assert liquid.price_impact_proxy == liquid.illiquidity_proxy
    assert liquid.turnover is None
    assert liquid.dollar_volume_hhi == pytest.approx(1 / 6)
    assert liquid.effective_active_names == pytest.approx(6)
    assert stress.gap_stress == 0
    assert stress.stress_score is not None and 0 <= stress.stress_score <= 1
    assert sum(item.contribution or 0 for item in liquid.contributors) == pytest.approx(
        liquid.liquidity_score
    )
    assert sum(item.contribution or 0 for item in stress.contributors) == pytest.approx(
        stress.stress_score
    )
    snapshot = apply_liquidity_stress_to_snapshot(result)
    assert snapshot.schema_version == "1.0" and snapshot.engine_version == "1.0.0"
    assert snapshot.liquidity_score == liquid.liquidity_score
    assert snapshot.stress_score == stress.stress_score
    assert snapshot.market_regime == MarketRegime.UNKNOWN
    assert snapshot.feature_values is not None
    assert snapshot.feature_values["liquidity.dollar_volume"] == liquid.session_dollar_volume
    assert canonical_json_str(
        MarketContextSnapshot.model_validate_json(canonical_json_str(snapshot))
    ) == canonical_json_str(snapshot)
    assert canonical_json_str(
        LiquidityStressFeatureSet.model_validate_json(canonical_json_str(result))
    ) == canonical_json_str(result)
    existing = MarketContextSnapshot(as_of_ns=as_of, dispersion_score=0.3)
    assert apply_liquidity_stress_to_snapshot(result, existing).dispersion_score == 0.3
    with pytest.raises(ValueError):
        apply_liquidity_stress_to_snapshot(result, MarketContextSnapshot(as_of_ns=as_of + 1))


def test_volume_surge_collapse_zero_and_missing_quote(calendar: MarketSessionService) -> None:
    standard, as_of = stress_store(calendar)
    surge, _ = stress_store(calendar, volumes=(300,) * 6)
    collapse, _ = stress_store(calendar, volumes=(10,) * 6)
    zero, _ = stress_store(calendar, volumes=(0,) * 6)
    no_quote, _ = stress_store(calendar, quote_width=None)
    engine = _engine(calendar)
    base = engine.compute(standard, as_of).liquidity
    high = engine.compute(surge, as_of).liquidity
    low = engine.compute(collapse, as_of).liquidity
    empty = engine.compute(zero, as_of).liquidity
    absent = engine.compute(no_quote, as_of).liquidity
    assert high.relative_volume == pytest.approx(3)
    assert low.relative_volume == pytest.approx(0.1)
    assert high.liquidity_score is not None and base.liquidity_score is not None
    assert low.liquidity_score is not None
    assert high.liquidity_score > base.liquidity_score > low.liquidity_score
    assert empty.session_volume == 0
    assert empty.per_member["T1"].explicit_zero_volume
    assert empty.per_member["T1"].illiquidity_proxy is None
    assert empty.relative_volume == 0
    assert absent.quoted_spread is None and absent.quote_coverage == 0
    assert absent.liquidity_score is not None
    assert len(absent.active_weights) == 2


def test_gap_direction_breadth_and_liquidity_distinction(calendar: MarketSessionService) -> None:
    down, as_of = stress_store(calendar, gaps=(-0.1,) * 6)
    up, _ = stress_store(calendar, gaps=(0.1,) * 6)
    no_move, _ = stress_store(calendar, gaps=(0,) * 6)
    engine = _engine(calendar)
    stressed = engine.compute(down, as_of)
    positive = engine.compute(up, as_of)
    calm = engine.compute(no_move, as_of)
    assert stressed.stress.gap_down_fraction == 1
    assert stressed.stress.gap_stress is not None and stressed.stress.gap_stress > 0
    assert stressed.stress.stressed_member_fraction == 1
    assert stressed.stress.localized_stress == 0
    assert positive.stress.gap_up_fraction == 1 and positive.stress.gap_stress == 0
    assert calm.stress.gap_stress == 0
    assert stressed.stress.stress_score is not None and positive.stress.stress_score is not None
    assert stressed.stress.stress_score > positive.stress.stress_score


def test_upstream_components_no_double_count_and_systemic_vs_localized(
    calendar: MarketSessionService,
) -> None:
    store, as_of = stress_store(calendar, gaps=(-0.1,) * 6)
    contexts = _upstream(as_of)
    result = _engine(calendar).compute(store, as_of, **contexts)
    stress = result.stress
    assert stress.volatility_stress == 0.8
    assert stress.correlation_stress == 0.8
    assert stress.breadth_stress == pytest.approx(0.9)
    assert stress.dispersion_context_stress == pytest.approx(0.81)
    assert stress.sector_stress == 1
    assert len(stress.active_weights) == 7
    assert sum(weight for _, weight in stress.active_weights) == pytest.approx(1)
    assert stress.stress_score is not None
    assert sum(item.contribution or 0 for item in stress.contributors) == pytest.approx(
        stress.stress_score
    )
    assert stress.systemic_stress == stress.stress_score
    isolated_store, _ = stress_store(calendar, gaps=(-0.1, 0, 0, 0, 0, 0))
    isolated = _engine(calendar).compute(isolated_store, as_of, **contexts).stress
    assert isolated.stressed_member_fraction is not None
    assert isolated.stressed_member_fraction < stress.stressed_member_fraction
    assert isolated.localized_stress is not None and isolated.localized_stress > 0
    positive_context = _upstream(as_of, breadth=0.9, dispersion=0.9)
    optimistic = _engine(calendar).compute(store, as_of, **positive_context).stress
    assert optimistic.dispersion_context_stress == pytest.approx(0.045)
    assert optimistic.stress_score is not None and optimistic.stress_score < stress.stress_score


def test_stress_history_is_explicit_prior_only(calendar: MarketSessionService) -> None:
    store, as_of = stress_store(calendar)
    engine = _engine(calendar)
    prior = tuple(
        StressHistoryPoint(
            as_of_ns=as_of - index * MINUTE,
            available_at_ns=as_of - index * MINUTE,
            stress_score=0.1 * index,
        )
        for index in range(1, 6)
    )
    future = StressHistoryPoint(
        as_of_ns=as_of + MINUTE, available_at_ns=as_of + MINUTE, stress_score=1
    )
    result = engine.compute(store, as_of, stress_history=prior)
    assert result.stress.historical_sample_count == 5
    assert result.stress.stress_percentile is not None
    assert result.stress.stress_change is not None
    assert result.stress.stress_acceleration is not None
    assert result.stress.stress_persistence is not None
    assert canonical_json_str(
        engine.compute(store, as_of, stress_history=prior + (future,))
    ) == canonical_json_str(result)


def test_future_bars_quotes_membership_weights_and_contexts_cannot_revise_past(
    calendar: MarketSessionService,
) -> None:
    store, as_of = stress_store(calendar)
    future = as_of + MINUTE
    past_members = tuple(
        UniverseMembership(symbol=symbol, weight=1, effective_to_ns=future) for symbol in SYMBOLS
    )
    future_members = tuple(
        UniverseMembership(
            symbol=symbol, weight=10 if symbol == "T1" else 1, effective_from_ns=future
        )
        for symbol in SYMBOLS
        if symbol != "T2"
    ) + (UniverseMembership(symbol="NEW", effective_from_ns=future),)
    engine = _engine(calendar, membership=MembershipTimeline(past_members + future_members))
    before = canonical_json_str(engine.compute(store, as_of))
    store.add(
        BarObservation(
            symbol="T1",
            source="synthetic",
            event_timestamp_ns=future,
            effective_timestamp_ns=future,
            bar_start_ns=as_of,
            bar_end_ns=future,
            open=100,
            high=100,
            low=100,
            close=100,
            volume=1_000_000,
        )
    )
    store.add(
        QuoteObservation(
            symbol="T1",
            source="synthetic",
            event_timestamp_ns=future,
            effective_timestamp_ns=future,
            bid=50,
            ask=150,
        )
    )
    assert canonical_json_str(engine.compute(store, as_of)) == before
    same = _engine(
        calendar,
        membership=MembershipTimeline(
            tuple(UniverseMembership(symbol=symbol, weight=1) for symbol in SYMBOLS)
        ),
    ).compute(store, as_of)
    assert canonical_json_str(same) == before
    for name, context in _upstream(as_of).items():
        with pytest.raises(ValueError):
            engine.compute(store, as_of, **{name: context.model_copy(update={"as_of_ns": future})})


def test_stale_missing_quote_and_opening_incomplete(calendar: MarketSessionService) -> None:
    store, as_of = stress_store(
        calendar, missing=frozenset({"T1"}), stale=frozenset({"T2"}), quote_width=None
    )
    day = calendar.session_for_timestamp(as_of).trading_date
    assert day is not None
    opened, _ = calendar.regular_session_bounds(day)
    store.add(
        QuoteObservation(
            symbol="T3",
            source="synthetic",
            event_timestamp_ns=opened + MINUTE,
            effective_timestamp_ns=opened + MINUTE,
            bid=99.99,
            ask=100.01,
        )
    )
    result = _engine(calendar).compute(store, as_of)
    assert result.liquidity.member_coverage == pytest.approx(4 / 6)
    assert result.liquidity.stale_bar_count == 1
    assert result.liquidity.stale_quote_count == 1
    assert result.liquidity.per_member["T1"].session_volume is None
    assert result.liquidity.per_member["T2"].stale_bar
    assert {
        QualityFlag.PARTIAL_UNIVERSE,
        QualityFlag.STALE_DATA,
        QualityFlag.LOW_COVERAGE,
    } <= result.quality_flags
    early_clock = opened + MINUTE // 2
    incomplete = _engine(calendar).compute(store, early_clock)
    assert incomplete.liquidity.valid_member_count == 0
    assert incomplete.liquidity.liquidity_score is None


def test_early_close_same_elapsed_and_fingerprint(calendar: MarketSessionService) -> None:
    store, as_of = stress_store(calendar, day=date(2024, 7, 5), prior_sessions=4)
    engine = _engine(calendar)
    result = engine.compute(store, as_of)
    assert result.liquidity.relative_volume == pytest.approx(1)
    assert config_fingerprint(engine.config) != config_fingerprint(
        _engine(calendar, settings=_settings(spread_scale=0.01)).config
    )


def test_five_and_twenty_session_rvol_horizons(calendar: MarketSessionService) -> None:
    store, as_of = stress_store(calendar, prior_sessions=20, volumes=(200,) * 6)
    settings = _settings(historical_sessions=20)
    result = _engine(calendar, settings=settings).compute(store, as_of)
    assert result.liquidity.relative_volume == pytest.approx(2)
    assert result.liquidity.relative_volume_5d == pytest.approx(2)
    assert result.liquidity.relative_volume_20d == pytest.approx(2)
    snapshot = apply_liquidity_stress_to_snapshot(result)
    assert snapshot.feature_values is not None
    assert snapshot.feature_values["liquidity.relative_volume_5d"] == pytest.approx(2)
    assert snapshot.feature_values["liquidity.relative_volume_20d"] == pytest.approx(2)


def test_large_move_volume_divergence_is_descriptive(calendar: MarketSessionService) -> None:
    weak, as_of = stress_store(calendar, volumes=(30,) * 6, returns=(0.02,) * 6)
    high, _ = stress_store(calendar, volumes=(300,) * 6, returns=(0.02,) * 6)
    calm, _ = stress_store(calendar, volumes=(30,) * 6, returns=(0.001,) * 6)
    engine = _engine(calendar)
    weak_result = engine.compute(weak, as_of).liquidity
    high_result = engine.compute(high, as_of).liquidity
    calm_result = engine.compute(calm, as_of).liquidity
    assert weak_result.large_move_weak_volume_fraction == 1
    assert weak_result.large_move_high_volume_fraction == 0
    assert high_result.large_move_high_volume_fraction == 1
    assert high_result.large_move_weak_volume_fraction == 0
    assert calm_result.large_move_weak_volume_fraction == 0
    assert calm_result.large_move_high_volume_fraction == 0
    assert config_fingerprint(engine.config) != config_fingerprint(
        _engine(calendar, settings=_settings(large_move_threshold=0.03)).config
    )


def test_partial_weights_and_insufficient_coverage(calendar: MarketSessionService) -> None:
    store, as_of = stress_store(calendar, missing=frozenset({"T1", "T2"}))
    weighted = MembershipTimeline(
        tuple(UniverseMembership(symbol=symbol, weight=1) for symbol in SYMBOLS)
    )
    result = _engine(calendar, membership=weighted).compute(store, as_of)
    assert result.liquidity.member_coverage == pytest.approx(4 / 6)
    assert result.liquidity.weight_coverage == pytest.approx(4 / 6)
    assert result.liquidity.weighted_illiquidity_proxy is None
    assert result.liquidity.liquidity_score is None
    assert result.liquidity.data_quality.value == "unusable"
    incomplete_weights = MembershipTimeline(
        tuple(
            UniverseMembership(symbol=symbol, weight=None if symbol == "T1" else 1)
            for symbol in SYMBOLS
        )
    )
    complete_store, _ = stress_store(calendar)
    partial = _engine(calendar, membership=incomplete_weights).compute(complete_store, as_of)
    assert partial.liquidity.weight_coverage is None
    assert partial.liquidity.weighted_illiquidity_proxy is None
    assert QualityFlag.MISSING_DATA in partial.quality_flags


def test_missing_previous_close_does_not_create_zero_gap(
    calendar: MarketSessionService,
) -> None:
    store, as_of = stress_store(calendar, missing=frozenset({"T1"}))
    day = calendar.session_for_timestamp(as_of).trading_date
    assert day is not None
    opened, _ = calendar.regular_session_bounds(day)
    for minute in (1, 2, 3):
        store.add(
            BarObservation(
                symbol="T1",
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
    result = _engine(calendar).compute(store, as_of)
    assert result.liquidity.per_member["T1"].gap_return is None
    assert result.stress.gap_coverage == pytest.approx(5 / 6)


def test_single_upstream_evidence_family_changes_only_its_component(
    calendar: MarketSessionService,
) -> None:
    store, as_of = stress_store(calendar)
    engine = _engine(calendar)
    baseline = engine.compute(store, as_of).stress
    high_vol = engine.compute(
        store,
        as_of,
        volatility_context=RealizedVolatilityFeatureSet.model_construct(
            as_of_ns=as_of, volatility_score=1
        ),
    ).stress
    high_corr = engine.compute(
        store,
        as_of,
        correlation_context=CorrelationFeatureSet.model_construct(
            as_of_ns=as_of, systemic_co_movement_score=1
        ),
    ).stress
    negative_breadth = engine.compute(
        store,
        as_of,
        breadth_context=BreadthFeatureSet.model_construct(as_of_ns=as_of, breadth_score=-1),
    ).stress
    positive_breadth = engine.compute(
        store,
        as_of,
        breadth_context=BreadthFeatureSet.model_construct(as_of_ns=as_of, breadth_score=1),
    ).stress
    assert baseline.stress_score is not None
    assert high_vol.volatility_stress == 1 and high_corr.correlation_stress == 1
    assert high_vol.stress_score is not None and high_vol.stress_score > baseline.stress_score
    assert high_corr.stress_score is not None and high_corr.stress_score > baseline.stress_score
    assert negative_breadth.breadth_stress == 1
    assert positive_breadth.breadth_stress == 0
    assert negative_breadth.stress_score is not None
    assert positive_breadth.stress_score is not None
    assert negative_breadth.stress_score > positive_breadth.stress_score


def test_upstream_stress_remains_available_without_liquidity_universe(
    calendar: MarketSessionService,
) -> None:
    store, as_of = stress_store(calendar)
    engine = LiquidityStressFeatureEngine(MCEConfig(_env_file=None), calendar=calendar)
    result = engine.compute(
        store,
        as_of,
        volatility_context=RealizedVolatilityFeatureSet.model_construct(
            as_of_ns=as_of, volatility_score=0.8
        ),
        correlation_context=CorrelationFeatureSet.model_construct(
            as_of_ns=as_of, systemic_co_movement_score=0.6
        ),
    )
    assert result.liquidity.liquidity_score is None
    assert result.stress.stress_score == pytest.approx((0.22 * 0.8 + 0.17 * 0.6) / 0.39)
    assert result.stress.evidence_quality > 0
    assert result.stress.data_quality.value == "degraded"


def test_aggregate_overflow_yields_missing_instead_of_nonfinite(
    calendar: MarketSessionService,
) -> None:
    store = CausalDataStore()
    day = date(2024, 6, 28)
    opened, _ = calendar.regular_session_bounds(day)
    for symbol in SYMBOLS:
        store.add(
            BarObservation(
                symbol=symbol,
                source="synthetic",
                event_timestamp_ns=opened + MINUTE,
                effective_timestamp_ns=opened + MINUTE,
                bar_start_ns=opened,
                bar_end_ns=opened + MINUTE,
                open=1,
                high=1,
                low=1,
                close=1,
                volume=1e308,
            )
        )
    result = _engine(calendar).compute(store, opened + MINUTE)
    assert result.liquidity.valid_member_count == 6
    assert result.liquidity.session_volume is None
    assert result.liquidity.session_dollar_volume is None
    assert result.liquidity.dollar_volume_hhi is None
