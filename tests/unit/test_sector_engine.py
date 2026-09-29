from datetime import date

import pytest

from mce.calendar.service import MarketSessionService
from mce.config.settings import FeatureSettings, MCEConfig, SectorSettings
from mce.features.sectors.engine import SectorFeatureEngine
from mce.features.sectors.membership import (
    SectorMembership,
    SectorMembershipTimeline,
    SectorUniverseSnapshot,
)
from mce.features.sectors.models import SectorFeatureSet
from mce.features.sectors.snapshot import apply_sectors_to_snapshot
from mce.models.context import MarketContextSnapshot
from mce.models.enums import DataQuality, QualityFlag
from mce.models.market_data import BarObservation
from mce.replay.store import CausalDataStore
from mce.reproducibility import config_fingerprint
from mce.serialization import canonical_json_bytes, canonical_json_str
from tests.fixtures.breadth_scenarios import MINUTE
from tests.fixtures.sector_scenarios import (
    GROUPS,
    PROXIES,
    sector_membership,
    sector_store,
)


def _engine(
    calendar: MarketSessionService,
    membership: SectorMembershipTimeline | None = None,
    settings: SectorSettings | None = None,
) -> SectorFeatureEngine:
    return SectorFeatureEngine(
        MCEConfig(
            _env_file=None,
            features=FeatureSettings(
                sectors=settings
                or SectorSettings(
                    groups=GROUPS,
                    proxy_mapping=PROXIES,
                    price_adjustment_status="adjusted",
                )
            ),
        ),
        membership,
        calendar,
    )


def test_technology_energy_defensive_leadership_and_multi_horizon_strength(
    calendar: MarketSessionService,
) -> None:
    for leading, movement in (
        ("technology", {"T1": 0.02, "T2": 0.02, "T3": 0.02}),
        ("energy", {"E1": 0.02, "E2": 0.02, "E3": 0.02}),
        ("utilities", {"U1": 0.02, "U2": 0.02, "U3": 0.02}),
    ):
        store, as_of = sector_store(calendar, current=movement, market_current=0.005)
        result = _engine(calendar).compute(store, as_of)
        assert result.leader_sector == leading
        assert result.market_coverage == 1
        sector = result.sectors[leading]
        assert sector.absolute_returns["session"] == pytest.approx(0.02)
        assert sector.relative_returns["session"] == pytest.approx(0.015)
        assert sector.relative_returns["5m"] is not None
        assert sector.relative_returns["30m"] == pytest.approx(0.015)
        assert sector.breadth_score == 1
        assert sector.cross_horizon_agreement == "positive"
        assert sector.rank == 1 and sector.percentile == 1
        assert sector.relative_strength_score is not None
        assert -1 <= sector.relative_strength_score <= 1
        assert sum(term.contribution or 0 for term in sector.contributors) == pytest.approx(
            sector.relative_strength_score
        )


def test_absolute_and_relative_strength_can_disagree(calendar: MarketSessionService) -> None:
    store, as_of = sector_store(
        calendar,
        current=dict.fromkeys(GROUPS["technology"], 0.01),
        market_current=0.02,
    )
    technology = _engine(calendar).compute(store, as_of).sectors["technology"]
    assert technology.absolute_returns["session"] == pytest.approx(0.01)
    assert technology.relative_returns["session"] == pytest.approx(-0.01)
    assert technology.absolute_strength_score is not None and technology.absolute_strength_score > 0
    assert technology.relative_strength_score is not None and technology.relative_strength_score < 0


def test_synchronized_market_has_zero_relative_strength(calendar: MarketSessionService) -> None:
    store, as_of = sector_store(
        calendar,
        current=dict.fromkeys(tuple(symbol for group in GROUPS.values() for symbol in group), 0.01),
        market_current=0.01,
    )
    result = _engine(calendar).compute(store, as_of)
    for item in result.sectors.values():
        assert item.absolute_returns["1d"] == pytest.approx(0.01)
        assert item.relative_returns["1d"] == pytest.approx(0)
        assert item.breadth_score == 1
        assert item.relative_strength_score == pytest.approx(0)


def test_sector_breadth_weights_proxy_and_concentration(calendar: MarketSessionService) -> None:
    store, as_of = sector_store(
        calendar,
        current={"T1": 0.1, "T2": -0.01, "T3": -0.01},
        proxy_current={"XLK": 0.02},
    )
    weights = {"T1": 8.0, "T2": 1.0, "T3": 1.0}
    result = _engine(calendar, sector_membership(weights=weights)).compute(store, as_of)
    tech = result.sectors["technology"]
    assert (tech.advancers, tech.decliners, tech.unchanged) == (1, 2, 0)
    assert tech.net_advance_fraction == pytest.approx(-1 / 3)
    assert tech.breadth_proxy_divergence == pytest.approx(-4 / 3)
    assert tech.weighted_returns["1d"] == pytest.approx(0.078)
    assert tech.absolute_returns["1d"] == pytest.approx(0.08 / 3)
    assert tech.weight_coverage == 1
    assert tech.proxy_returns["1d"] == pytest.approx(0.02)
    assert tech.proxy_divergence["1d"] == pytest.approx(0.08 / 3 - 0.02)
    assert result.leadership_concentration is not None
    assert 0 < result.leadership_concentration <= 1
    unweighted = _engine(calendar).compute(store, as_of).sectors["technology"]
    assert unweighted.weighted_returns["1d"] is None
    assert unweighted.weight_coverage is None
    incomplete_store, incomplete_time = sector_store(
        calendar,
        current={"T1": 0.1, "T2": -0.01},
        missing=frozenset({"T3"}),
    )
    relaxed = SectorSettings(
        groups=GROUPS,
        minimum_member_coverage=0.6,
        minimum_weight_coverage=0.8,
        price_adjustment_status="adjusted",
    )
    incomplete = (
        _engine(calendar, sector_membership(weights=weights), relaxed)
        .compute(incomplete_store, incomplete_time)
        .sectors["technology"]
    )
    assert incomplete.member_coverage == pytest.approx(2 / 3)
    assert incomplete.weight_coverage == pytest.approx(0.9)
    assert incomplete.weighted_returns["1d"] is not None
    heavy_missing_store, heavy_time = sector_store(calendar, missing=frozenset({"T1"}))
    heavy_missing = (
        _engine(calendar, sector_membership(weights=weights), relaxed)
        .compute(heavy_missing_store, heavy_time)
        .sectors["technology"]
    )
    assert heavy_missing.weight_coverage == pytest.approx(0.2)
    assert heavy_missing.weighted_returns["1d"] is None


def test_partial_stale_low_coverage_and_missing_reference(
    calendar: MarketSessionService,
) -> None:
    store, as_of = sector_store(
        calendar,
        current={"T3": 0.02},
        missing=frozenset({"T1"}),
        stale=frozenset({"T2"}),
    )
    result = _engine(calendar).compute(store, as_of)
    tech = result.sectors["technology"]
    assert tech.expected_member_count == 3 and tech.valid_member_count == 1
    assert tech.stale_member_count == 1
    assert tech.advancers == 1 and tech.unchanged == 0
    assert tech.member_coverage == pytest.approx(1 / 3)
    assert tech.relative_strength_score is None
    assert tech.absolute_returns["1d"] is None
    assert {
        QualityFlag.STALE_DATA,
        QualityFlag.PARTIAL_UNIVERSE,
        QualityFlag.LOW_COVERAGE,
    } <= tech.quality_flags
    assert "technology" not in rankable_scores(result)
    without_market = CausalDataStore(max_observations=100_000)
    for symbol in (
        *GROUPS["technology"],
        *GROUPS["energy"],
        *GROUPS["utilities"],
        *PROXIES.values(),
    ):
        for bar in store.bars(symbol, 0, as_of + 1, as_of):
            without_market.add(bar)
    unavailable = _engine(calendar).compute(without_market, as_of)
    assert unavailable.sectors["energy"].relative_strength_score is None
    assert unavailable.sectors["energy"].market_volatility is None
    assert unavailable.sectors["energy"].relative_volatility is None
    assert QualityFlag.MISSING_VOLATILITY_DATA in unavailable.quality_flags
    assert unavailable.leader_sector is None
    assert QualityFlag.MISSING_DATA in unavailable.quality_flags


def rankable_scores(result: SectorFeatureSet) -> dict[str, float]:
    return {
        name: item.relative_strength_score
        for name, item in result.sectors.items()
        if item.relative_strength_score is not None
    }


def test_volatility_above_below_equal_market_and_missing_history(
    calendar: MarketSessionService,
) -> None:
    alternating = tuple((1 if index % 2 else -1) for index in range(70))
    store, as_of = sector_store(
        calendar,
        history_changes={
            "technology": tuple(sign * 0.02 for sign in alternating),
            "energy": tuple(sign * 0.005 for sign in alternating),
            "utilities": tuple(sign * 0.01 for sign in alternating),
        },
        market_history=tuple(sign * 0.01 for sign in alternating),
    )
    result = _engine(calendar).compute(store, as_of)
    assert result.sectors["technology"].relative_volatility is not None
    assert result.sectors["technology"].relative_volatility > 1
    assert result.sectors["energy"].relative_volatility is not None
    assert result.sectors["energy"].relative_volatility < 1
    assert result.sectors["utilities"].relative_volatility == pytest.approx(1)
    assert result.sectors["technology"].volatility_percentile is not None
    short_store, short_time = sector_store(calendar, history_sessions=4)
    short = _engine(calendar).compute(short_store, short_time)
    assert short.sectors["technology"].realized_volatility is None
    assert short.sectors["technology"].breadth_references["5d"].fraction is None
    assert QualityFlag.INSUFFICIENT_HISTORY in short.sectors["technology"].quality_flags


def test_rotation_persistent_leader_switch_and_no_rotation(
    calendar: MarketSessionService,
) -> None:
    history = {
        "technology": (0.01,) * 70,
        "energy": (-0.01,) * 70,
        "utilities": (0.0,) * 70,
    }
    steady_store, as_of = sector_store(
        calendar,
        history_changes=history,
        current=dict.fromkeys(GROUPS["technology"], 0.01),
    )
    steady = _engine(calendar).compute(steady_store, as_of)
    assert steady.leader_sector == "technology"
    assert steady.previous_leader_sector == "technology"
    assert steady.leader_changed is False
    assert steady.rotation_velocity == pytest.approx(0)
    assert steady.sectors["technology"].leadership_persistence == 1
    switched_store, _ = sector_store(
        calendar,
        history_changes=history,
        current={
            **dict.fromkeys(GROUPS["technology"], -0.02),
            **dict.fromkeys(GROUPS["energy"], 0.2),
        },
    )
    switched = _engine(calendar).compute(switched_store, as_of)
    assert switched.leader_sector == "energy"
    assert switched.previous_leader_sector == "technology"
    assert switched.leader_changed is True
    assert switched.rotation_velocity is not None and switched.rotation_velocity > 0
    assert switched.sectors["energy"].rank_change is not None
    assert switched.sectors["energy"].rank_change > 0
    assert switched.sectors["technology"].leadership_persistence is not None
    assert switched.sectors["technology"].leadership_persistence < 1


def test_future_sector_bar_membership_weight_and_history_cannot_change_past(
    calendar: MarketSessionService,
) -> None:
    store, as_of = sector_store(calendar, current={"T1": 0.03})
    future = as_of + MINUTE
    all_symbols = tuple(symbol for symbols in GROUPS.values() for symbol in symbols)
    baseline_provider = sector_membership(weights=dict.fromkeys(all_symbols, 1.0))
    records = tuple(item for item in baseline_provider.records if item.symbol != "T1")
    changing = SectorMembershipTimeline(
        (
            *records,
            SectorMembership(sector="technology", symbol="T1", weight=1, effective_to_ns=future),
            SectorMembership(sector="energy", symbol="T1", weight=9, effective_from_ns=future),
            SectorMembership(sector="energy", symbol="NEW", effective_from_ns=future),
        )
    )
    past_result = _engine(calendar, baseline_provider).compute(store, as_of)
    assert past_result.sectors["technology"].weighted_returns["1d"] is not None
    baseline = canonical_json_bytes(past_result)
    assert canonical_json_bytes(_engine(calendar, changing).compute(store, as_of)) == baseline
    store.add(
        BarObservation(
            symbol="T1",
            source="synthetic",
            bar_start_ns=future,
            bar_end_ns=future + MINUTE,
            event_timestamp_ns=future,
            effective_timestamp_ns=future + MINUTE,
            open=100,
            high=150,
            low=100,
            close=150,
            volume=1,
        )
    )
    assert canonical_json_bytes(_engine(calendar, changing).compute(store, as_of)) == baseline
    changed = _engine(calendar, changing).compute(store, future)
    assert changed.sectors["technology"].expected_member_count == 2
    assert changed.sectors["energy"].expected_member_count == 5
    assert changed.sectors["energy"].members["T1"].weight == 9
    assert canonical_json_bytes(_engine(calendar, changing).compute(store, as_of)) == baseline
    late = SectorMembershipTimeline(
        (
            *baseline_provider.records,
            SectorMembership(
                sector="energy", symbol="LATE", effective_from_ns=0, available_at_ns=future
            ),
        )
    )
    assert canonical_json_bytes(_engine(calendar, late).compute(store, as_of)) == baseline
    assert any(item.symbol == "LATE" for item in late.members(future).members)
    weighted_change = SectorMembershipTimeline(
        (
            *(item for item in baseline_provider.records if item.symbol != "T1"),
            SectorMembership(sector="technology", symbol="T1", weight=1, effective_to_ns=future),
            SectorMembership(sector="technology", symbol="T1", weight=9, effective_from_ns=future),
        )
    )
    assert (
        canonical_json_bytes(_engine(calendar, weighted_change).compute(store, as_of)) == baseline
    )
    before_weight = (
        _engine(calendar, baseline_provider)
        .compute(store, future)
        .sectors["technology"]
        .weighted_returns["1d"]
    )
    after_weight = (
        _engine(calendar, weighted_change)
        .compute(store, future)
        .sectors["technology"]
        .weighted_returns["1d"]
    )
    assert before_weight is not None and after_weight is not None and after_weight > before_weight
    previous = calendar.previous_trading_day(date(2024, 6, 28))
    opened, closed = calendar.regular_session_bounds(previous)
    store.add(
        BarObservation(
            symbol="T2",
            source="synthetic",
            bar_start_ns=opened,
            bar_end_ns=closed,
            event_timestamp_ns=opened,
            effective_timestamp_ns=future + MINUTE,
            open=100,
            high=200,
            low=100,
            close=200,
            volume=1,
        )
    )
    assert canonical_json_bytes(_engine(calendar, changing).compute(store, as_of)) == baseline


def test_provider_clock_guard_snapshot_and_config_fingerprint(
    calendar: MarketSessionService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FutureProvider:
        def members(self, as_of_ns: int) -> SectorUniverseSnapshot:
            return SectorUniverseSnapshot(
                as_of_ns=as_of_ns + 1,
                members=(
                    SectorMembership(
                        sector="technology", symbol="T1", effective_from_ns=as_of_ns + 1
                    ),
                ),
                source="synthetic",
                source_quality=DataQuality.GOOD,
            )

    store, as_of = sector_store(calendar, current={"T1": 0.02})
    with pytest.raises(ValueError, match="different as_of_ns"):
        SectorFeatureEngine(_engine(calendar).config, FutureProvider(), calendar).compute(
            store, as_of
        )
    result = _engine(calendar).compute(store, as_of)
    assert canonical_json_str(
        SectorFeatureSet.model_validate_json(canonical_json_str(result))
    ) == canonical_json_str(result)
    base = MarketContextSnapshot(
        as_of_ns=as_of, breadth_score=0.4, trend_score=0.2, feature_values={"breadth.existing": 1}
    )
    snapshot = apply_sectors_to_snapshot(result, base)
    assert snapshot.breadth_score == 0.4 and snapshot.trend_score == 0.2
    assert snapshot.sector_scores is not None
    assert (
        snapshot.sector_scores["technology"] == result.sectors["technology"].relative_strength_score
    )
    assert snapshot.feature_values is not None
    assert (
        snapshot.feature_values["sector.technology.relative_return.1d"]
        == result.sectors["technology"].relative_returns["1d"]
    )
    assert snapshot.market_regime.value == "unknown"
    assert canonical_json_str(
        MarketContextSnapshot.model_validate_json(canonical_json_str(snapshot))
    ) == canonical_json_str(snapshot)
    with pytest.raises(ValueError):
        apply_sectors_to_snapshot(result, MarketContextSnapshot(as_of_ns=as_of + 1))
    original = MCEConfig(_env_file=None)
    changed = MCEConfig(
        _env_file=None, features=FeatureSettings(sectors=SectorSettings(return_scale=0.02))
    )
    assert config_fingerprint(original) != config_fingerprint(changed)
    monkeypatch.setenv("MCE_FEATURES__SECTORS__MINIMUM_MEMBER_COVERAGE", "0.9")
    assert MCEConfig(_env_file=None).features.sectors.minimum_member_coverage == 0.9


def test_static_groups_disabled_and_missing_proxy(calendar: MarketSessionService) -> None:
    store, as_of = sector_store(calendar)
    standard = _engine(calendar).compute(store, as_of)
    assert set(standard.sectors) == set(GROUPS)
    no_proxy = _engine(
        calendar, settings=SectorSettings(groups=GROUPS, price_adjustment_status="adjusted")
    ).compute(store, as_of)
    assert no_proxy.sectors["technology"].proxy_returns["1d"] is None
    removed_proxy = CausalDataStore(max_observations=100_000)
    for symbol in (
        *GROUPS["technology"],
        *GROUPS["energy"],
        *GROUPS["utilities"],
        "SPY",
        "XLE",
        "XLU",
    ):
        for bar in store.bars(symbol, 0, as_of + 1, as_of):
            removed_proxy.add(bar)
    missing_proxy = _engine(calendar).compute(removed_proxy, as_of).sectors["technology"]
    assert missing_proxy.proxy_returns["1d"] is None
    assert missing_proxy.relative_strength_score is not None
    assert QualityFlag.MISSING_DATA in missing_proxy.quality_flags
    disabled = _engine(calendar, settings=SectorSettings(enabled=False, groups=GROUPS)).compute(
        store, as_of
    )
    assert disabled.sectors == {} and disabled.data_quality == DataQuality.UNKNOWN
    assert standard.leader_sector == "energy"  # lexical tie break on flat scores
    opened, _ = calendar.regular_session_bounds(date(2024, 6, 28))
    opening = _engine(calendar).compute(store, opened + 10_000_000_000)
    assert opening.sectors["technology"].valid_member_count == 0
    assert opening.sectors["technology"].relative_strength_score is None
    too_few = _engine(
        calendar,
        settings=SectorSettings(
            groups=GROUPS,
            minimum_valid_sectors=4,
            price_adjustment_status="adjusted",
        ),
    ).compute(store, as_of)
    assert too_few.leader_sector is None and too_few.laggard_sector is None
    assert QualityFlag.LOW_COVERAGE in too_few.quality_flags
    tiny = _engine(
        calendar,
        settings=SectorSettings(
            groups=GROUPS,
            minimum_member_count=4,
            price_adjustment_status="adjusted",
        ),
    ).compute(store, as_of)
    assert tiny.sectors["technology"].advancers == 0
    assert tiny.sectors["technology"].relative_strength_score is None


def test_scale_order_early_close_and_degraded_membership(
    calendar: MarketSessionService,
) -> None:
    store, as_of = sector_store(calendar, current={"T1": 0.03, "E1": -0.02})
    base_provider = sector_membership()
    engine = _engine(calendar, base_provider)
    baseline = engine.compute(store, as_of)
    reversed_provider = SectorMembershipTimeline(tuple(reversed(base_provider.records)))
    assert canonical_json_bytes(
        _engine(calendar, reversed_provider).compute(store, as_of)
    ) == canonical_json_bytes(baseline)
    scaled = CausalDataStore(max_observations=100_000)
    for symbol in (
        *GROUPS["technology"],
        *GROUPS["energy"],
        *GROUPS["utilities"],
        *PROXIES.values(),
        "SPY",
    ):
        for bar in store.bars(symbol, 0, as_of + 1, as_of):
            scaled.add(
                bar.model_copy(
                    update={
                        name: getattr(bar, name) * 10 for name in ("open", "high", "low", "close")
                    }
                )
            )
    changed = engine.compute(scaled, as_of)
    for sector in GROUPS:
        assert changed.sectors[sector].relative_strength_score == pytest.approx(
            baseline.sectors[sector].relative_strength_score
        )
        assert changed.sectors[sector].realized_volatility == pytest.approx(
            baseline.sectors[sector].realized_volatility
        )
    degraded = SectorMembershipTimeline(base_provider.records, source_quality=DataQuality.DEGRADED)
    low_quality = _engine(calendar, degraded).compute(store, as_of)
    assert QualityFlag.PROVIDER_DEGRADED in low_quality.quality_flags
    assert (
        low_quality.sectors["technology"].evidence_quality
        < baseline.sectors["technology"].evidence_quality
    )
    early_store, early_as_of = sector_store(calendar, day=date(2024, 11, 29))
    early = _engine(calendar).compute(early_store, early_as_of)
    assert early.sectors["technology"].member_coverage == 1
