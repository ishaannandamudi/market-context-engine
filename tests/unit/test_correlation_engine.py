from datetime import date

import numpy as np
import pytest
from pydantic import ValidationError

from mce.calendar.service import MarketSessionService
from mce.config.settings import CorrelationSettings, FeatureSettings, MCEConfig, SectorSettings
from mce.features.breadth.membership import MembershipTimeline, UniverseMembership
from mce.features.correlation.engine import CorrelationFeatureEngine
from mce.features.correlation.snapshot import apply_correlation_to_snapshot
from mce.features.sectors.membership import SectorMembership, SectorMembershipTimeline
from mce.models.context import MarketContextSnapshot
from mce.models.enums import CorrelationRegime, DataQuality, QualityFlag
from mce.models.market_data import BarObservation
from mce.replay.store import CausalDataStore
from mce.reproducibility import config_fingerprint
from mce.serialization import canonical_json_bytes
from tests.fixtures.correlation_scenarios import DAY, MINUTE, add_panel, dependence_moves

SYMBOLS = ("AAA", "BBB", "CCC", "DDD")


def _settings(**updates: object) -> CorrelationSettings:
    values: dict[str, object] = {
        "universe": SYMBOLS,
        "short_window_bars": 10,
        "medium_window_bars": 20,
        "long_window_bars": 30,
        "minimum_aligned_samples": 8,
        "minimum_symbol_coverage": 0.5,
        "minimum_time_coverage": 0.7,
        "historical_baseline_sessions": 0,
    }
    values.update(updates)
    return CorrelationSettings.model_validate(values)


def _engine(
    calendar: MarketSessionService,
    settings: CorrelationSettings | None = None,
    provider: MembershipTimeline | None = None,
    sectors: SectorMembershipTimeline | None = None,
) -> CorrelationFeatureEngine:
    return CorrelationFeatureEngine(
        MCEConfig(_env_file=None, features=FeatureSettings(correlation=settings or _settings())),
        provider,
        sectors,
        calendar,
    )


def test_matrix_invariants_score_snapshot_and_replay(calendar: MarketSessionService) -> None:
    store = CausalDataStore()
    asof = add_panel(store, calendar, minutes=40)
    engine = _engine(calendar)
    result = engine.compute(store, asof)
    assert result.expected_symbol_count == result.valid_symbol_count == 4
    assert result.aligned_sample_count == 20
    assert result.time_coverage == 1
    assert result.data_quality in (DataQuality.GOOD, DataQuality.DEGRADED)
    assert result.correlation_matrix is not None
    matrix = np.asarray(result.correlation_matrix.values)
    assert np.allclose(matrix, matrix.T)
    assert np.allclose(np.diag(matrix), 1)
    assert np.linalg.eigvalsh(matrix).min() >= -1e-10
    assert result.first_eigenvalue_share is not None
    assert 1 / 4 <= result.first_eigenvalue_share <= 1
    assert result.effective_dimension is not None and 1 <= result.effective_dimension <= 4
    assert result.spearman_matrix is not None and result.shrunk_matrix is not None
    assert result.systemic_co_movement_score is not None
    assert sum(item.contribution or 0 for item in result.contributors) == pytest.approx(
        result.systemic_co_movement_score
    )
    assert canonical_json_bytes(engine.compute(store, asof)) == canonical_json_bytes(result)
    projected = apply_correlation_to_snapshot(result, MarketContextSnapshot(as_of_ns=asof))
    assert projected.correlation_score == result.systemic_co_movement_score
    assert projected.correlation_regime == CorrelationRegime.UNKNOWN
    assert projected.feature_values is not None
    assert projected.feature_values["correlation.average"] == result.average_pairwise_correlation
    assert projected.schema_version == "1.0" and projected.engine_version == "0.9.0"
    with pytest.raises(ValueError):
        apply_correlation_to_snapshot(result, MarketContextSnapshot(as_of_ns=asof + 1))


def test_future_bar_membership_weight_sector_do_not_change_past(
    calendar: MarketSessionService,
) -> None:
    store = CausalDataStore()
    asof = add_panel(store, calendar, minutes=40)
    future = asof + 5 * MINUTE
    baseline_members = MembershipTimeline(
        tuple(UniverseMembership(symbol=s, weight=1) for s in SYMBOLS)
    )
    changed_members = MembershipTimeline(
        tuple(UniverseMembership(symbol=s, weight=1, effective_to_ns=future) for s in SYMBOLS)
        + tuple(UniverseMembership(symbol=s, weight=10, effective_from_ns=future) for s in SYMBOLS)
        + (UniverseMembership(symbol="EEE", available_at_ns=future),)
    )
    sectors = SectorMembershipTimeline(
        tuple(
            SectorMembership(
                symbol=s, sector="first" if i < 2 else "second", effective_to_ns=future
            )
            for i, s in enumerate(SYMBOLS)
        )
        + tuple(SectorMembership(symbol=s, sector="new", effective_from_ns=future) for s in SYMBOLS)
    )
    original_sectors = SectorMembershipTimeline(
        tuple(
            SectorMembership(symbol=s, sector="first" if i < 2 else "second")
            for i, s in enumerate(SYMBOLS)
        )
    )
    first = _engine(calendar, provider=baseline_members).compute(store, asof)
    changed = _engine(calendar, provider=changed_members, sectors=sectors).compute(store, asof)
    assert changed.correlation_matrix == first.correlation_matrix
    assert changed.weighted_average_correlation == first.weighted_average_correlation
    original_sector_result = _engine(
        calendar, provider=baseline_members, sectors=original_sectors
    ).compute(store, asof)
    assert changed.sector_pairs == original_sector_result.sector_pairs
    before = canonical_json_bytes(
        _engine(calendar, provider=changed_members, sectors=sectors).compute(store, asof)
    )
    add_panel(store, calendar, minutes=80)
    after = canonical_json_bytes(
        _engine(calendar, provider=changed_members, sectors=sectors).compute(store, asof)
    )
    assert before == after
    late = BarObservation(
        symbol="AAA",
        source="synthetic",
        event_timestamp_ns=asof,
        effective_timestamp_ns=future,
        bar_start_ns=asof - MINUTE,
        bar_end_ns=asof,
        open=100,
        high=101,
        low=100,
        close=101,
        volume=1,
    )
    store.add(late)
    assert (
        canonical_json_bytes(
            _engine(calendar, provider=changed_members, sectors=sectors).compute(store, asof)
        )
        == before
    )


def test_partial_missing_constant_stale_and_low_coverage(calendar: MarketSessionService) -> None:
    store = CausalDataStore()
    asof = add_panel(store, calendar, minutes=40, missing=frozenset({("DDD", 40), ("CCC", 27)}))
    result = _engine(calendar).compute(store, asof)
    assert "DDD" in result.stale_symbols
    assert result.valid_symbol_count == 3
    assert result.aligned_sample_count == 18  # one absent close removes two adjacent returns
    assert QualityFlag.PARTIAL_UNIVERSE in result.quality_flags
    assert QualityFlag.STALE_DATA in result.quality_flags
    missing = CausalDataStore()
    add_panel(missing, calendar, minutes=40, symbols=("AAA", "BBB"))
    poor = _engine(calendar).compute(missing, asof)
    assert poor.systemic_co_movement_score is None
    assert QualityFlag.LOW_COVERAGE in poor.quality_flags
    moves = tuple((0.001 * (i % 2), 0.001 * (i % 3), 0.0, 0.001 * (i % 5)) for i in range(40))
    constant = CausalDataStore()
    add_panel(constant, calendar, minutes=40, moves=moves)
    degenerate = _engine(calendar).compute(constant, asof)
    assert "CCC" in degenerate.constant_symbols
    assert "CCC" not in degenerate.valid_symbols


def test_sector_pairs_and_weighted_correlation(calendar: MarketSessionService) -> None:
    store = CausalDataStore()
    asof = add_panel(store, calendar, minutes=40)
    provider = MembershipTimeline(
        tuple(
            UniverseMembership(symbol=s, weight=weight)
            for s, weight in zip(SYMBOLS, (4, 2, 1, 1), strict=True)
        )
    )
    sectors = SectorMembershipTimeline(
        tuple(
            SectorMembership(symbol=s, sector="one" if i < 2 else "two")
            for i, s in enumerate(SYMBOLS)
        )
    )
    result = _engine(calendar, provider=provider, sectors=sectors).compute(store, asof)
    assert result.weight_coverage == 1
    assert result.weighted_average_correlation is not None
    assert result.within_sector_correlation is not None
    assert result.cross_sector_correlation is not None
    assert result.within_cross_spread == pytest.approx(
        result.within_sector_correlation - result.cross_sector_correlation
    )
    assert sum(item.pair_count for item in result.sector_pairs) == 6
    configured = CorrelationFeatureEngine(
        MCEConfig(
            _env_file=None,
            features=FeatureSettings(
                correlation=_settings(),
                sectors=SectorSettings(groups={"one": ("AAA", "BBB"), "two": ("CCC", "DDD")}),
            ),
        ),
        calendar=calendar,
    ).compute(store, asof)
    assert configured.sector_pairs == result.sector_pairs


def test_prior_only_baseline_and_future_snapshot_invariance(calendar: MarketSessionService) -> None:
    store = CausalDataStore(max_observations=20000)
    prior_days = (date(2024, 6, 25), date(2024, 6, 26), date(2024, 6, 27))
    for day in prior_days:
        add_panel(store, calendar, day, minutes=390)
    asof = add_panel(store, calendar, minutes=40)
    settings = _settings(historical_baseline_sessions=3, minimum_baseline_samples=2)
    engine = _engine(calendar, settings)
    first = engine.compute(store, asof)
    assert first.correlation_percentile is not None
    assert first.correlation_robust_z is None  # zero-MAD prior, distinct current value
    assert first.session_change is not None
    assert first.five_session_change is None
    past = canonical_json_bytes(first)
    add_panel(store, calendar, date(2024, 7, 1), minutes=390)
    assert canonical_json_bytes(engine.compute(store, asof)) == past
    additions = MembershipTimeline(
        tuple(UniverseMembership(symbol=s) for s in SYMBOLS)
        + (UniverseMembership(symbol="EEE", effective_from_ns=asof + MINUTE),)
    )
    assert (
        _engine(calendar, settings, provider=additions).compute(store, asof).correlation_percentile
        == first.correlation_percentile
    )


def test_daily_horizon_config_fingerprint_and_disabled(calendar: MarketSessionService) -> None:
    store = CausalDataStore(max_observations=20000)
    cursor = DAY
    for index in range(10):
        add_panel(store, calendar, cursor, minutes=390, price_scale=1 + 0.01 * (index % 3))
        cursor = calendar.previous_trading_day(cursor)
    asof = calendar.regular_session_bounds(DAY)[1]
    settings = _settings(daily_horizons_sessions=(8,))
    result = _engine(calendar, settings).compute(store, asof)
    assert result.daily_horizons[0].name == "8d"
    assert result.daily_horizons[0].aligned_sample_count == 8
    changed = _settings(shrinkage_strength=0.3)
    assert config_fingerprint(
        MCEConfig(_env_file=None, features=FeatureSettings(correlation=settings))
    ) != config_fingerprint(
        MCEConfig(_env_file=None, features=FeatureSettings(correlation=changed))
    )
    disabled = _engine(calendar, _settings(enabled=False)).compute(store, asof)
    assert disabled.correlation_matrix is None
    assert disabled.systemic_co_movement_score is None
    with pytest.raises(ValidationError):
        _settings(short_window_bars=30)


def test_price_scale_does_not_change_correlation(calendar: MarketSessionService) -> None:
    first, second = CausalDataStore(), CausalDataStore()
    asof = add_panel(first, calendar, minutes=40)
    add_panel(second, calendar, minutes=40, price_scale=100)
    engine = _engine(calendar)
    left = engine.compute(first, asof)
    right = engine.compute(second, asof)
    assert left.average_pairwise_correlation == pytest.approx(right.average_pairwise_correlation)


def test_negative_average_cannot_imply_high_systemic_score(
    calendar: MarketSessionService,
) -> None:
    from math import sin

    moves = tuple(
        (value, value, -value, -value)
        for value in (0.01 * sin(index * 0.31) for index in range(1, 41))
    )
    store = CausalDataStore()
    asof = add_panel(store, calendar, minutes=40, moves=moves)
    result = _engine(calendar).compute(store, asof)
    assert result.average_pairwise_correlation is not None
    assert result.average_pairwise_correlation < 0
    assert result.systemic_co_movement_score == 0
    assert result.first_eigenvalue_share is not None and result.first_eigenvalue_share > 0.9


def test_correlation_spike_collapse_and_volatility_separation(
    calendar: MarketSessionService,
) -> None:
    opened, _ = calendar.regular_session_bounds(DAY)
    spike = CausalDataStore()
    add_panel(
        spike,
        calendar,
        minutes=80,
        moves=dependence_moves(80, first="independent", second="common"),
    )
    collapse = CausalDataStore()
    add_panel(
        collapse,
        calendar,
        minutes=80,
        moves=dependence_moves(80, first="common", second="independent"),
    )
    amplified = CausalDataStore()
    add_panel(
        amplified,
        calendar,
        minutes=80,
        moves=dependence_moves(80, first="independent", second="common", scale=5),
    )
    engine = _engine(calendar)
    early, late = opened + 40 * MINUTE, opened + 80 * MINUTE
    early_spike, late_spike = engine.compute(spike, early), engine.compute(spike, late)
    early_collapse, late_collapse = engine.compute(collapse, early), engine.compute(collapse, late)
    assert early_spike.average_pairwise_correlation is not None
    assert late_spike.average_pairwise_correlation is not None
    assert early_spike.average_pairwise_correlation < 0.2
    assert late_spike.average_pairwise_correlation > 0.9
    assert early_collapse.average_pairwise_correlation is not None
    assert late_collapse.average_pairwise_correlation is not None
    assert early_collapse.average_pairwise_correlation > 0.9
    assert late_collapse.average_pairwise_correlation < 0.2
    assert engine.compute(amplified, late).average_pairwise_correlation == pytest.approx(
        late_spike.average_pairwise_correlation
    )


def test_fifth_session_change_preserves_missing_session_slot(
    calendar: MarketSessionService,
) -> None:
    store = CausalDataStore(max_observations=20000)
    day = DAY
    for index in range(1, 7):
        day = calendar.previous_trading_day(day)
        if index != 5:
            add_panel(store, calendar, day, minutes=390)
    asof = add_panel(store, calendar, minutes=40)
    result = _engine(
        calendar, _settings(historical_baseline_sessions=6, minimum_baseline_samples=3)
    ).compute(store, asof)
    assert result.correlation_percentile is not None
    assert result.session_change is not None
    assert result.five_session_change is None
