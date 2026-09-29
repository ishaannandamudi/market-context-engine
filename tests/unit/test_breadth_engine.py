from datetime import date

import pytest
from pydantic import ValidationError

from mce.calendar.service import MarketSessionService
from mce.config.settings import (
    BreadthScoreWeights,
    BreadthSettings,
    FeatureSettings,
    MarketSettings,
    MCEConfig,
    UniverseSettings,
)
from mce.features.breadth.engine import BreadthFeatureEngine
from mce.features.breadth.membership import (
    MembershipTimeline,
    UniverseMembership,
    UniverseProvider,
    UniverseSnapshot,
)
from mce.features.breadth.models import BreadthFeatureSet
from mce.features.breadth.prices import BreadthPriceReader
from mce.features.breadth.snapshot import apply_breadth_to_snapshot
from mce.models.context import MarketContextSnapshot
from mce.models.enums import DataQuality, QualityFlag
from mce.models.market_data import BarObservation
from mce.replay.store import CausalDataStore
from mce.reproducibility import config_fingerprint
from mce.serialization import canonical_json_bytes, canonical_json_str
from tests.fixtures.breadth_scenarios import MINUTE, breadth_store, membership

SYMBOLS = ("AAA", "BBB", "CCC", "DDD", "EEE")


def _engine(
    calendar: MarketSessionService,
    provider: UniverseProvider,
    settings: BreadthSettings | None = None,
) -> BreadthFeatureEngine:
    return BreadthFeatureEngine(
        MCEConfig(
            _env_file=None,
            market=MarketSettings(universe=UniverseSettings(breadth=SYMBOLS)),
            features=FeatureSettings(
                breadth=settings or BreadthSettings(price_adjustment_status="adjusted")
            ),
        ),
        provider,
        calendar,
    )


def test_broad_rally_selloff_and_mixed_counts(calendar: MarketSessionService) -> None:
    provider = membership(SYMBOLS)
    engine = _engine(calendar, provider)
    for path, expected_net in (((0.02,) * 5, 1), ((-0.02,) * 5, -1)):
        store, as_of = breadth_store(calendar, SYMBOLS, path)
        result = engine.compute(store, as_of)
        assert result.net_advance_fraction == expected_net
        assert result.participation_score == expected_net
        assert result.breadth_score == expected_net
        assert result.coverage_ratio == 1
        assert result.short_reference.fraction == (1 if expected_net > 0 else 0)
        assert result.medium_reference.fraction == (1 if expected_net > 0 else 0)
        assert result.long_reference.fraction == (1 if expected_net > 0 else 0)
        assert result.new_high_fraction == (1 if expected_net > 0 else 0)
        assert result.new_low_fraction == (0 if expected_net > 0 else 1)
        assert (
            result.advance_decline_ratio is None
            if expected_net > 0
            else result.advance_decline_ratio == 0
        )
        assert result.data_quality == DataQuality.DEGRADED  # no constituent weights supplied
    mixed, as_of = breadth_store(calendar, SYMBOLS, (0.01, -0.01, 0, 0.02, -0.02))
    result = engine.compute(mixed, as_of)
    assert (result.advancers, result.decliners, result.unchanged) == (2, 2, 1)
    assert result.advance_fraction == 0.4
    assert result.decline_fraction == 0.4
    assert result.net_advance_fraction == 0
    assert result.advance_decline_ratio == 1
    assert result.equal_weight_return == pytest.approx(0)
    assert result.median_member_return == 0


def test_narrow_leadership_concentration_and_index_divergence(
    calendar: MarketSessionService,
) -> None:
    provider = membership(SYMBOLS)
    engine = _engine(calendar, provider)
    narrow_store, as_of = breadth_store(
        calendar, SYMBOLS, (0.20, -0.01, -0.01, -0.01, -0.01), index_return=0.01
    )
    narrow = engine.compute(narrow_store, as_of)
    broad_store, _ = breadth_store(calendar, SYMBOLS, (0.02,) * 5, index_return=0.01)
    broad = engine.compute(broad_store, as_of)
    assert narrow.net_advance_fraction == -0.6
    assert narrow.index_proxy_return == pytest.approx(0.01)
    assert narrow.directional_index_divergence == pytest.approx(-1.6)
    assert narrow.equal_weight_return is not None and narrow.equal_weight_return > 0
    assert narrow.contribution_hhi is not None and broad.contribution_hhi is not None
    assert narrow.contribution_hhi > broad.contribution_hhi
    assert narrow.effective_contributors is not None and narrow.effective_contributors < 2
    equal_return = narrow.equal_weight_return
    proxy_return = narrow.index_proxy_return
    assert equal_return is not None and proxy_return is not None
    assert narrow.index_proxy_divergence == pytest.approx(equal_return - proxy_return)
    no_proxy_store, _ = breadth_store(calendar, SYMBOLS, (0.02,) * 5, index_return=None)
    assert engine.compute(no_proxy_store, as_of).index_proxy_divergence is None
    positive_store, _ = breadth_store(calendar, SYMBOLS, (0.01,) * 5, index_return=-0.01)
    positive = engine.compute(positive_store, as_of)
    assert positive.net_advance_fraction == 1
    assert positive.directional_index_divergence == 2
    narrow_selloff_store, _ = breadth_store(
        calendar, SYMBOLS, (-0.20, 0.01, 0.01, 0.01, 0.01), index_return=-0.01
    )
    narrow_selloff = engine.compute(narrow_selloff_store, as_of)
    assert narrow_selloff.net_advance_fraction == 0.6
    assert narrow_selloff.equal_weight_return is not None
    assert narrow_selloff.equal_weight_return < 0
    assert narrow_selloff.directional_index_divergence == pytest.approx(1.6)


def test_weighted_return_coverage_zero_weight_and_missing_weights(
    calendar: MarketSessionService,
) -> None:
    store, as_of = breadth_store(calendar, SYMBOLS, (0.10, -0.02, 0, 0, 0))
    weighted_provider = membership(SYMBOLS, (0.8, 0.2, 0, 0, 0))
    result = _engine(calendar, weighted_provider).compute(store, as_of)
    assert result.weight_coverage == 1
    assert result.weighted_return == pytest.approx(0.076)
    assert result.equal_weight_return == pytest.approx(0.016)
    assert result.equal_weight_divergence == pytest.approx(-0.06)
    assert result.contribution_basis == "weighted"
    no_weights = _engine(calendar, membership(SYMBOLS)).compute(store, as_of)
    assert no_weights.weight_coverage is None
    assert no_weights.weighted_return is None
    assert no_weights.equal_weight_divergence is None
    assert QualityFlag.MISSING_DATA in no_weights.quality_flags
    partially_weighted = _engine(calendar, membership(SYMBOLS, (0.8, None, 0, 0, 0))).compute(
        store, as_of
    )
    assert partially_weighted.weighted_return is None


def test_missing_and_stale_members_lower_count_and_weight_coverage(
    calendar: MarketSessionService,
) -> None:
    store, as_of = breadth_store(
        calendar,
        SYMBOLS,
        (0.02,) * 5,
        missing=frozenset({"DDD"}),
        stale=frozenset({"EEE"}),
    )
    provider = membership(SYMBOLS, (0.2,) * 5)
    result = _engine(calendar, provider).compute(store, as_of)
    assert result.expected_member_count == 5
    assert result.available_member_count == 3
    assert result.stale_member_count == 1
    assert result.coverage_ratio == pytest.approx(0.6)
    assert result.weight_coverage == pytest.approx(0.6)
    assert result.breadth_score is None
    assert result.participation_score == 1
    assert result.weighted_return is None
    assert result.advancers == 3 and result.unchanged == 0
    assert result.per_member["EEE"].stale
    assert result.per_member["DDD"].simple_return is None
    assert {
        QualityFlag.STALE_DATA,
        QualityFlag.PARTIAL_UNIVERSE,
        QualityFlag.LOW_COVERAGE,
    } <= result.quality_flags
    relaxed = BreadthSettings(
        minimum_member_coverage=0.6,
        minimum_weight_coverage=0.6,
        price_adjustment_status="adjusted",
    )
    relaxed_result = _engine(calendar, provider, relaxed).compute(store, as_of)
    assert relaxed_result.breadth_score is not None
    assert relaxed_result.weighted_return is not None


def test_reference_history_equal_boundary_and_early_close(calendar: MarketSessionService) -> None:
    store, as_of = breadth_store(calendar, SYMBOLS, (0,) * 5, history_sessions=4)
    result = _engine(calendar, membership(SYMBOLS)).compute(store, as_of)
    assert result.unchanged == 5
    assert result.short_reference.fraction is None
    assert QualityFlag.INSUFFICIENT_HISTORY in result.quality_flags
    full_store, full_time = breadth_store(calendar, SYMBOLS, (0,) * 5)
    equal = _engine(calendar, membership(SYMBOLS)).compute(full_store, full_time)
    assert equal.short_reference.fraction == 0
    assert equal.new_high_fraction == 0 and equal.new_low_fraction == 0
    assert equal.breadth_score == pytest.approx(-0.5)
    early_store, early_time = breadth_store(
        calendar, SYMBOLS, (0.01,) * 5, day=date(2024, 11, 29), cadence_ns=5 * MINUTE
    )
    early = _engine(calendar, membership(SYMBOLS)).compute(early_store, early_time)
    assert early.advancers == 5 and early.coverage_ratio == 1


def test_opening_future_bar_and_store_order_causality(calendar: MarketSessionService) -> None:
    store, as_of = breadth_store(calendar, SYMBOLS, (0.01,) * 5)
    provider = membership(SYMBOLS, (1,) * 5)
    engine = _engine(calendar, provider)
    opened, _ = calendar.regular_session_bounds(date(2024, 6, 28))
    opening = engine.compute(store, opened + 10_000_000_000)
    assert opening.available_member_count == 0
    assert opening.breadth_score is None
    past = canonical_json_bytes(engine.compute(store, as_of))
    future = BarObservation(
        symbol="AAA",
        source="synthetic",
        bar_start_ns=as_of,
        bar_end_ns=as_of + MINUTE,
        event_timestamp_ns=as_of,
        effective_timestamp_ns=as_of + MINUTE,
        open=101,
        high=150,
        low=101,
        close=150,
        volume=1,
    )
    store.add(future)
    assert canonical_json_bytes(engine.compute(store, as_of)) == past
    reversed_store = CausalDataStore()
    for symbol in (*SYMBOLS, "SPY"):
        for bar in reversed(store.bars(symbol, 0, as_of + 2 * MINUTE, as_of + MINUTE)):
            reversed_store.add(bar)
    assert canonical_json_bytes(engine.compute(reversed_store, as_of)) == past
    assert canonical_json_bytes(engine.compute(store, as_of)) == past


def test_future_membership_and_weight_change_do_not_rewrite_past(
    calendar: MarketSessionService,
) -> None:
    store, as_of = breadth_store(calendar, SYMBOLS, (0.01, -0.01, 0.01, 0.01, 0.01))
    future = as_of + MINUTE
    baseline = MembershipTimeline(
        tuple(UniverseMembership(symbol=symbol, weight=1) for symbol in SYMBOLS)
    )
    changing = MembershipTimeline(
        (
            UniverseMembership(symbol="AAA", weight=1, effective_to_ns=future),
            UniverseMembership(symbol="AAA", weight=9, effective_from_ns=future),
            *(UniverseMembership(symbol=symbol, weight=1) for symbol in SYMBOLS[1:]),
            UniverseMembership(symbol="NEW", weight=2, effective_from_ns=future),
        )
    )
    assert canonical_json_bytes(_engine(calendar, baseline).compute(store, as_of)) == (
        canonical_json_bytes(_engine(calendar, changing).compute(store, as_of))
    )
    after = changing.members(future)
    assert len(after.members) == 6
    assert after.members[0].weight == 9


def test_provider_cannot_return_future_dated_membership(
    calendar: MarketSessionService,
) -> None:
    class FutureProvider:
        def members(self, as_of_ns: int) -> UniverseSnapshot:
            return UniverseSnapshot(
                as_of_ns=as_of_ns + 1,
                members=(UniverseMembership(symbol="NEW", effective_from_ns=as_of_ns + 1),),
                source="synthetic",
                source_quality=DataQuality.GOOD,
            )

    store, as_of = breadth_store(calendar, SYMBOLS, (0.01,) * 5)
    with pytest.raises(ValueError, match="different as_of_ns"):
        _engine(calendar, FutureProvider()).compute(store, as_of)


def test_weight_update_and_removal_apply_only_after_effective_time(
    calendar: MarketSessionService,
) -> None:
    store, as_of = breadth_store(calendar, SYMBOLS, (0.10, -0.02, 0, 0, 0))
    change_at = as_of + MINUTE
    before_only = MembershipTimeline(
        tuple(UniverseMembership(symbol=symbol, weight=1) for symbol in SYMBOLS)
    )
    updated = MembershipTimeline(
        (
            UniverseMembership(symbol="AAA", weight=1, effective_to_ns=change_at),
            UniverseMembership(symbol="AAA", weight=9, effective_from_ns=change_at),
            *(UniverseMembership(symbol=symbol, weight=1) for symbol in SYMBOLS[1:]),
        )
    )
    baseline = _engine(calendar, before_only).compute(store, as_of)
    assert canonical_json_bytes(_engine(calendar, updated).compute(store, as_of)) == (
        canonical_json_bytes(baseline)
    )
    changed = _engine(calendar, updated).compute(store, change_at)
    unchanged = _engine(calendar, before_only).compute(store, change_at)
    assert changed.weighted_return is not None and unchanged.weighted_return is not None
    assert changed.weighted_return > unchanged.weighted_return
    removing = MembershipTimeline(
        (
            UniverseMembership(symbol="AAA", effective_to_ns=change_at),
            *(UniverseMembership(symbol=symbol) for symbol in SYMBOLS[1:]),
        )
    )
    assert _engine(calendar, removing).compute(store, as_of).expected_member_count == 5
    assert _engine(calendar, removing).compute(store, change_at).expected_member_count == 4


def test_future_daily_ohlc_and_late_reference_cannot_change_past(
    calendar: MarketSessionService,
) -> None:
    store, as_of = breadth_store(calendar, SYMBOLS, (0.01,) * 5)
    engine = _engine(calendar, membership(SYMBOLS))
    baseline = canonical_json_bytes(engine.compute(store, as_of))
    day = date(2024, 6, 28)
    opened, closed = calendar.regular_session_bounds(day)
    store.add(
        BarObservation(
            symbol="AAA",
            source="synthetic",
            bar_start_ns=opened,
            bar_end_ns=closed,
            event_timestamp_ns=opened,
            effective_timestamp_ns=closed,
            open=100,
            high=200,
            low=100,
            close=200,
            volume=1,
        )
    )
    assert canonical_json_bytes(engine.compute(store, as_of)) == baseline
    previous = calendar.previous_trading_day(day)
    prior_open, prior_close = calendar.regular_session_bounds(previous)
    store.add(
        BarObservation(
            symbol="BBB",
            source="synthetic",
            bar_start_ns=prior_open,
            bar_end_ns=prior_close,
            event_timestamp_ns=prior_open,
            effective_timestamp_ns=as_of + MINUTE,
            open=100,
            high=150,
            low=100,
            close=150,
            volume=1,
        )
    )
    assert canonical_json_bytes(engine.compute(store, as_of)) == baseline
    assert canonical_json_bytes(engine.compute(store, as_of + MINUTE)) != baseline


def test_holiday_reference_adjacency_and_membership_quality(
    calendar: MarketSessionService,
) -> None:
    day = date(2024, 5, 28)
    store, as_of = breadth_store(calendar, SYMBOLS, (0.01,) * 5, day=day)
    reader = BreadthPriceReader(store, calendar, "synthetic")
    previous = reader.prior_closes("AAA", day, 2, as_of)
    assert previous == (100.0, 100.0)
    assert calendar.previous_trading_day(day) == date(2024, 5, 24)
    degraded = MembershipTimeline(
        tuple(UniverseMembership(symbol=symbol, weight=1) for symbol in SYMBOLS),
        source_quality=DataQuality.DEGRADED,
    )
    healthy = MembershipTimeline(
        tuple(UniverseMembership(symbol=symbol, weight=1) for symbol in SYMBOLS)
    )
    degraded_result = _engine(calendar, degraded).compute(store, as_of)
    healthy_result = _engine(calendar, healthy).compute(store, as_of)
    assert QualityFlag.PROVIDER_DEGRADED in degraded_result.quality_flags
    assert degraded_result.evidence_quality == pytest.approx(healthy_result.evidence_quality / 2)


def test_uniform_price_scale_does_not_change_breadth_metrics(
    calendar: MarketSessionService,
) -> None:
    store, as_of = breadth_store(calendar, SYMBOLS, (0.03, -0.01, 0, 0.02, -0.02))
    scaled = CausalDataStore()
    for symbol in (*SYMBOLS, "SPY"):
        for bar in store.bars(symbol, 0, as_of + 1, as_of):
            scaled.add(
                bar.model_copy(
                    update={
                        name: getattr(bar, name) * 10 for name in ("open", "high", "low", "close")
                    }
                )
            )
    engine = _engine(calendar, membership(SYMBOLS, (1,) * 5))
    before = engine.compute(store, as_of)
    after = engine.compute(scaled, as_of)
    assert after.net_advance_fraction == before.net_advance_fraction
    assert after.breadth_score == before.breadth_score
    assert after.equal_weight_return == pytest.approx(before.equal_weight_return)
    assert after.weighted_return == pytest.approx(before.weighted_return)
    assert after.contribution_hhi == pytest.approx(before.contribution_hhi)


def test_momentum_reversal_persistence_and_future_history_causality(
    calendar: MarketSessionService,
) -> None:
    past_down = (-0.01,) * 65
    store, as_of = breadth_store(calendar, SYMBOLS, (0.01,) * 5, history_changes=past_down)
    engine = _engine(calendar, membership(SYMBOLS))
    rising = engine.compute(store, as_of)
    assert rising.breadth_momentum is not None and rising.breadth_momentum > 0
    assert rising.breadth_acceleration is not None and rising.breadth_acceleration > 0
    assert rising.breadth_thrust is not None and rising.breadth_thrust > 0
    assert rising.participation_persistence is not None
    future = breadth_store(calendar, SYMBOLS, (-0.01,) * 5, history_changes=past_down)[0]
    future_day = calendar.next_trading_day(date(2024, 6, 28))
    next_open, _ = calendar.regular_session_bounds(future_day)
    for bar in future.bars("AAA", 0, as_of + 1, as_of):
        if bar.bar_end_ns == as_of:
            store.add(
                bar.model_copy(
                    update={
                        "bar_start_ns": next_open,
                        "bar_end_ns": next_open + MINUTE,
                        "event_timestamp_ns": next_open,
                        "effective_timestamp_ns": next_open + MINUTE,
                    }
                )
            )
    assert canonical_json_bytes(engine.compute(store, as_of)) == canonical_json_bytes(rising)
    improving_store, _ = breadth_store(
        calendar,
        SYMBOLS,
        (0.01,) * 5,
        history_changes=(-0.01,) * 55 + (0.01,) * 10,
    )
    improving = engine.compute(improving_store, as_of)
    assert improving.breadth_momentum == pytest.approx(0)
    falling_store, _ = breadth_store(
        calendar,
        SYMBOLS,
        (-0.01,) * 5,
        history_changes=(0.01,) * 65,
    )
    falling = engine.compute(falling_store, as_of)
    assert falling.breadth_momentum is not None and falling.breadth_momentum < 0


def test_snapshot_projection_serialization_settings_and_fingerprint(
    calendar: MarketSessionService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, as_of = breadth_store(calendar, SYMBOLS, (0.01,) * 5)
    result = _engine(calendar, membership(SYMBOLS)).compute(store, as_of)
    assert canonical_json_str(
        BreadthFeatureSet.model_validate_json(canonical_json_str(result))
    ) == canonical_json_str(result)
    base = MarketContextSnapshot(
        as_of_ns=as_of,
        trend_score=0.3,
        realized_volatility=0.2,
        feature_values={"trend.existing": 0.3},
    )
    snapshot = apply_breadth_to_snapshot(result, base)
    assert snapshot.breadth_score == result.breadth_score
    assert snapshot.trend_score == 0.3 and snapshot.realized_volatility == 0.2
    assert snapshot.breadth_regime.value == "unknown"
    assert snapshot.market_regime.value == "unknown"
    assert snapshot.feature_values is not None
    assert snapshot.feature_values["trend.existing"] == 0.3
    assert snapshot.feature_values["breadth.above_reference.20d"] == 1
    assert snapshot.feature_values["breadth.new_high_fraction.20d"] == 1
    assert canonical_json_str(
        MarketContextSnapshot.model_validate_json(canonical_json_str(snapshot))
    ) == canonical_json_str(snapshot)
    with pytest.raises(ValueError):
        apply_breadth_to_snapshot(result, MarketContextSnapshot(as_of_ns=as_of + 1))
    base_config = MCEConfig(_env_file=None)
    changed = MCEConfig(
        _env_file=None,
        features=FeatureSettings(breadth=BreadthSettings(minimum_member_coverage=0.9)),
    )
    assert config_fingerprint(base_config) != config_fingerprint(changed)
    monkeypatch.setenv("MCE_FEATURES__BREADTH__MINIMUM_MEMBER_COVERAGE", "0.9")
    assert MCEConfig(_env_file=None).features.breadth.minimum_member_coverage == 0.9
    with pytest.raises(ValidationError):
        BreadthSettings(reference_windows_sessions=(20, 5, 60))
    with pytest.raises(ValidationError):
        BreadthScoreWeights(net_advance=0.9)


def test_disabled_empty_membership_and_no_proxy(calendar: MarketSessionService) -> None:
    store, as_of = breadth_store(calendar, SYMBOLS, (0.01,) * 5)
    disabled = _engine(calendar, membership(SYMBOLS), BreadthSettings(enabled=False)).compute(
        store, as_of
    )
    assert disabled.data_quality == DataQuality.UNKNOWN
    empty = _engine(calendar, membership(())).compute(store, as_of)
    assert empty.breadth_score is None
    assert QualityFlag.MISSING_BREADTH in empty.quality_flags
    with pytest.raises(ValueError):
        _engine(calendar, membership(SYMBOLS)).compute(store, -1)
    no_proxy = _engine(
        calendar,
        membership(SYMBOLS),
        BreadthSettings(index_proxy_symbol=None, price_adjustment_status="adjusted"),
    ).compute(store, as_of)
    assert no_proxy.index_proxy_return is None


def test_static_config_universe_and_wrong_source_are_handled(
    calendar: MarketSessionService,
) -> None:
    store, as_of = breadth_store(calendar, SYMBOLS, (0.01,) * 5)
    config = MCEConfig(
        _env_file=None,
        market=MarketSettings(universe=UniverseSettings(breadth=SYMBOLS)),
        features=FeatureSettings(breadth=BreadthSettings(price_adjustment_status="adjusted")),
    )
    engine = BreadthFeatureEngine(config, calendar=calendar)
    baseline = canonical_json_bytes(engine.compute(store, as_of))
    opened, _ = calendar.regular_session_bounds(date(2024, 6, 28))
    store.add(
        BarObservation(
            symbol="AAA",
            source="other",
            bar_start_ns=as_of - MINUTE,
            bar_end_ns=as_of,
            event_timestamp_ns=as_of - MINUTE,
            effective_timestamp_ns=as_of,
            open=100,
            high=200,
            low=100,
            close=200,
            volume=1,
        )
    )
    assert opened < as_of
    assert canonical_json_bytes(engine.compute(store, as_of)) == baseline


def test_stale_price_age_boundary_excludes_only_after_limit(
    calendar: MarketSessionService,
) -> None:
    original, as_of = breadth_store(calendar, SYMBOLS, (0.01,) * 5)
    store = CausalDataStore()
    for symbol in (*SYMBOLS, "SPY"):
        for bar in original.bars(symbol, 0, as_of + 1, as_of):
            if symbol == "AAA" and bar.bar_end_ns == as_of:
                continue
            store.add(bar)
    store.add(
        BarObservation(
            symbol="AAA",
            source="synthetic",
            bar_start_ns=as_of - 3 * MINUTE,
            bar_end_ns=as_of - 2 * MINUTE,
            event_timestamp_ns=as_of - 3 * MINUTE,
            effective_timestamp_ns=as_of - 2 * MINUTE,
            open=100,
            high=101,
            low=100,
            close=101,
            volume=1,
        )
    )
    engine = _engine(calendar, membership(SYMBOLS))
    boundary = engine.compute(store, as_of)
    later = engine.compute(store, as_of + 1)
    assert boundary.per_member["AAA"].simple_return is not None
    assert not boundary.per_member["AAA"].stale
    assert later.per_member["AAA"].stale
    assert later.available_member_count == boundary.available_member_count - 1
