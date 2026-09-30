from datetime import date
from math import sqrt

import pytest

from mce.calendar.service import MarketSessionService
from mce.config.settings import DispersionSettings, FeatureSettings, MCEConfig
from mce.features.breadth.membership import MembershipTimeline, UniverseMembership
from mce.features.correlation.models import CorrelationFeatureSet
from mce.features.dispersion.engine import DispersionFeatureEngine
from mce.features.dispersion.models import DispersionFeatureSet
from mce.features.dispersion.snapshot import apply_dispersion_to_snapshot
from mce.features.sectors.membership import SectorMembership, SectorMembershipTimeline
from mce.features.sectors.models import SectorContext, SectorFeatureSet
from mce.features.volatility.models import RealizedVolatilityFeatureSet, SymbolVolatilityFeatures
from mce.features.windows import WindowSpec, WindowUnit
from mce.models.context import MarketContextSnapshot
from mce.models.enums import DataQuality, DispersionRegime, QualityFlag
from mce.models.market_data import BarObservation
from mce.reproducibility import config_fingerprint
from mce.serialization import canonical_json_str
from tests.fixtures.breadth_scenarios import MINUTE
from tests.fixtures.dispersion_scenarios import GROUPS, SYMBOLS, dispersion_store


def _settings(**updates: object) -> DispersionSettings:
    return DispersionSettings(
        universe=SYMBOLS,
        short_horizon=WindowSpec(name="1m", unit=WindowUnit.ELAPSED, length=MINUTE),
        medium_horizon=WindowSpec(name="2m", unit=WindowUnit.ELAPSED, length=2 * MINUTE),
        long_horizon=WindowSpec(name="1d", unit=WindowUnit.TRADING_DAYS, length=1),
        additional_horizons=(WindowSpec(name="session", unit=WindowUnit.SESSION, length=1),),
        historical_baseline_sessions=10,
        minimum_baseline_samples=5,
        persistence_observations=3,
        price_adjustment_status="adjusted",
    ).model_copy(update=updates)


def _engine(
    calendar: MarketSessionService,
    membership: MembershipTimeline | None = None,
    sector_membership: SectorMembershipTimeline | None = None,
    settings: DispersionSettings | None = None,
) -> DispersionFeatureEngine:
    return DispersionFeatureEngine(
        MCEConfig(_env_file=None, features=FeatureSettings(dispersion=settings or _settings())),
        membership,
        sector_membership,
        calendar,
    )


def _membership(weights: tuple[float | None, ...]) -> MembershipTimeline:
    return MembershipTimeline(
        tuple(
            UniverseMembership(symbol=symbol, weight=weight)
            for symbol, weight in zip(SYMBOLS, weights, strict=True)
        )
    )


def _sectors() -> SectorMembershipTimeline:
    return SectorMembershipTimeline.from_groups(GROUPS)


def test_return_moments_horizons_history_and_snapshot(calendar: MarketSessionService) -> None:
    returns = (-0.06, -0.02, 0.0, 0.01, 0.02, 0.08)
    store, as_of = dispersion_store(calendar, returns)
    result = _engine(calendar, _membership((1, 1, 1, 1, 1, 5)), _sectors()).compute(store, as_of)
    mean = sum(returns) / len(returns)
    expected_std = sqrt(sum((value - mean) ** 2 for value in returns) / len(returns))
    assert result.return_dispersion == pytest.approx(expected_std)
    assert result.weighted_return_dispersion is not None
    assert result.weighted_return_dispersion != pytest.approx(expected_std)
    assert result.short_dispersion == pytest.approx(expected_std)
    assert result.medium_dispersion == pytest.approx(expected_std)
    assert result.dispersion_acceleration == pytest.approx(0)
    assert result.dispersion_change == pytest.approx(expected_std)
    assert result.dispersion_percentile == result.dispersion_score == 1
    assert result.dispersion_robust_z is not None and result.dispersion_robust_z > 0
    assert result.dispersion_shock_score is not None and result.dispersion_shock_score > 0
    assert result.movement_hhi is not None and result.effective_movers is not None
    assert result.weighted_movement_hhi is not None
    assert result.between_sector_variance is not None
    assert result.within_sector_variance is not None
    assert result.decomposition_total_variance == pytest.approx(
        result.within_sector_variance + result.between_sector_variance
    )
    assert result.within_sector_share is not None and result.between_sector_share is not None
    assert result.within_sector_share + result.between_sector_share == pytest.approx(1)
    assert sum(item.contribution or 0 for item in result.contributors) == result.dispersion_score
    projected = apply_dispersion_to_snapshot(result)
    assert projected.schema_version == "1.0" and projected.engine_version == "1.0.0"
    assert projected.dispersion_score == 1
    assert projected.dispersion_regime == DispersionRegime.UNKNOWN
    assert projected.feature_values is not None
    assert projected.feature_values["dispersion.return.std"] == pytest.approx(expected_std)
    assert projected.feature_values["dispersion.between_sector_share"] == pytest.approx(
        result.between_sector_share
    )
    assert canonical_json_str(projected) == canonical_json_str(apply_dispersion_to_snapshot(result))
    assert canonical_json_str(
        MarketContextSnapshot.model_validate_json(canonical_json_str(projected))
    ) == canonical_json_str(projected)
    assert canonical_json_str(
        DispersionFeatureSet.model_validate_json(canonical_json_str(result))
    ) == canonical_json_str(result)
    existing = MarketContextSnapshot(as_of_ns=as_of, correlation_score=0.75)
    combined = apply_dispersion_to_snapshot(result, existing)
    assert combined.correlation_score == 0.75 and combined.dispersion_score == 1
    with pytest.raises(ValueError):
        apply_dispersion_to_snapshot(result, MarketContextSnapshot(as_of_ns=as_of + 1))


def test_missing_stale_partial_and_weight_coverage(calendar: MarketSessionService) -> None:
    store, as_of = dispersion_store(
        calendar, (0.01,) * 6, missing=frozenset({"T1"}), stale=frozenset({"T2"})
    )
    result = _engine(calendar, _membership((4, 1, 1, 1, 1, 1)), _sectors()).compute(store, as_of)
    assert result.member_coverage == pytest.approx(4 / 6)
    assert result.weight_coverage == pytest.approx(4 / 9)
    assert result.weighted_return_dispersion is None
    assert result.dispersion_score is None
    assert result.missing_symbols == ("T1",) and result.stale_symbols == ("T2",)
    assert {
        QualityFlag.LOW_COVERAGE,
        QualityFlag.STALE_DATA,
        QualityFlag.PARTIAL_UNIVERSE,
    } <= result.quality_flags
    assert result.data_quality == DataQuality.POOR
    unweighted = _engine(calendar, sector_membership=_sectors()).compute(store, as_of)
    assert unweighted.weight_coverage is None


def test_future_bars_membership_weights_and_sector_changes_cannot_revise_past(
    calendar: MarketSessionService,
) -> None:
    store, as_of = dispersion_store(calendar, (-0.02, -0.01, 0, 0.01, 0.02, 0.03))
    future = as_of + 4 * MINUTE
    baseline_members = tuple(
        UniverseMembership(symbol=symbol, weight=1.0, effective_to_ns=future) for symbol in SYMBOLS
    )
    future_members = tuple(
        UniverseMembership(
            symbol=symbol, weight=10.0 if symbol == "T1" else 1.0, effective_from_ns=future
        )
        for symbol in SYMBOLS
        if symbol != "T2"
    ) + (UniverseMembership(symbol="NEW", effective_from_ns=future),)
    baseline_sectors = tuple(
        SectorMembership(symbol=symbol, sector=sector, effective_to_ns=future)
        for sector, symbols in GROUPS.items()
        for symbol in symbols
    )
    future_sectors = tuple(
        SectorMembership(
            symbol=symbol,
            sector="energy" if sector == "technology" else "technology",
            effective_from_ns=future,
        )
        for sector, symbols in GROUPS.items()
        for symbol in symbols
    )
    engine = _engine(
        calendar,
        MembershipTimeline(baseline_members + future_members),
        SectorMembershipTimeline(baseline_sectors + future_sectors),
    )
    initial = canonical_json_str(engine.compute(store, as_of))
    opened, _ = calendar.regular_session_bounds(date(2024, 6, 28))
    store.add(
        BarObservation(
            symbol="T1",
            source="synthetic",
            event_timestamp_ns=future - MINUTE,
            effective_timestamp_ns=future,
            bar_start_ns=future - MINUTE,
            bar_end_ns=future,
            open=100,
            high=1000,
            low=100,
            close=1000,
            volume=1,
        )
    )
    assert opened < as_of < future
    assert canonical_json_str(engine.compute(store, as_of)) == initial
    without_future = _engine(calendar, _membership((1,) * 6), _sectors()).compute(store, as_of)
    assert canonical_json_str(engine.compute(store, as_of)) == canonical_json_str(without_future)


def test_nominal_price_scale_invariance_and_config_fingerprint(
    calendar: MarketSessionService,
) -> None:
    returns = (-0.02, -0.01, 0, 0.01, 0.02, 0.03)
    low, as_of = dispersion_store(calendar, returns, scale=1)
    high, _ = dispersion_store(calendar, returns, scale=100)
    engine = _engine(calendar, sector_membership=_sectors())
    left = engine.compute(low, as_of)
    right = engine.compute(high, as_of)
    assert left.return_dispersion == pytest.approx(right.return_dispersion)
    assert left.return_mad == pytest.approx(right.return_mad)
    assert left.movement_hhi == pytest.approx(right.movement_hhi)
    assert left.dispersion_score == right.dispersion_score
    assert left.data_quality == DataQuality.GOOD
    assert config_fingerprint(engine.config) != config_fingerprint(
        MCEConfig(
            _env_file=None,
            features=FeatureSettings(dispersion=_settings(tail_quantiles=(0.05, 0.95))),
        )
    )


def test_optional_sector_volatility_and_correlation_contexts(
    calendar: MarketSessionService,
) -> None:
    store, as_of = dispersion_store(calendar, (-0.06, -0.02, 0, 0.01, 0.02, 0.08))
    sectors = SectorFeatureSet.model_construct(
        as_of_ns=as_of,
        sectors={
            "technology": SectorContext.model_construct(
                absolute_returns={"2m": -0.02}, relative_returns={"2m": -0.03}
            ),
            "energy": SectorContext.model_construct(
                absolute_returns={"2m": 0.04}, relative_returns={"2m": 0.03}
            ),
        },
    )
    vols = RealizedVolatilityFeatureSet.model_construct(
        as_of_ns=as_of,
        per_symbol={
            symbol: SymbolVolatilityFeatures.model_construct(rv_medium=0.1 + index * 0.02)
            for index, symbol in enumerate(SYMBOLS)
        },
        market_realized_volatility=0.2,
    )
    correlation = CorrelationFeatureSet.model_construct(
        as_of_ns=as_of,
        average_pairwise_correlation=0.8,
        systemic_co_movement_score=0.9,
        correlation_percentile=0.9,
        quality_flags=frozenset(),
    )
    result = _engine(calendar, sector_membership=_sectors()).compute(
        store,
        as_of,
        sector_context=sectors,
        volatility_context=vols,
        correlation_context=correlation,
    )
    assert result.sector_dispersion == pytest.approx(0.03)
    assert result.sector_relative_dispersion == pytest.approx(0.03)
    assert result.sector_relative_mad == pytest.approx(0.03)
    assert result.sector_spread == pytest.approx(0.06)
    assert result.volatility_dispersion is not None and result.volatility_dispersion > 0
    assert result.relative_volatility_dispersion == pytest.approx(
        result.volatility_dispersion / 0.2
    )
    assert result.volatility_coverage == 1
    assert result.correlation_context is not None
    assert result.correlation_context.differentiation_score == pytest.approx(0.1)
    assert result.correlation_context.return_dispersion == result.return_dispersion
    projected = apply_dispersion_to_snapshot(result)
    assert projected.feature_values is not None
    assert projected.feature_values["dispersion.correlation.differentiation"] == pytest.approx(0.1)
    for keyword, context in (
        ("sector_context", sectors),
        ("volatility_context", vols),
        ("correlation_context", correlation),
    ):
        with pytest.raises(ValueError):
            _engine(calendar).compute(
                store, as_of, **{keyword: context.model_copy(update={"as_of_ns": as_of + 1})}
            )


@pytest.mark.parametrize(
    ("returns", "co_movement", "expected_percentile", "expected_differentiation"),
    [
        ((0.0,) * 6, 0.9, 0.0, 0.0),
        ((-0.06, -0.02, 0, 0.01, 0.02, 0.08), 0.9, 1.0, 0.1),
        ((-0.06, -0.02, 0, 0.01, 0.02, 0.08), 0.1, 1.0, 0.9),
        ((0.0,) * 6, 0.1, 0.0, 0.0),
    ],
)
def test_four_continuous_correlation_dispersion_cases(
    calendar: MarketSessionService,
    returns: tuple[float, ...],
    co_movement: float,
    expected_percentile: float,
    expected_differentiation: float,
) -> None:
    store, as_of = dispersion_store(calendar, returns)
    correlation = CorrelationFeatureSet.model_construct(
        as_of_ns=as_of,
        average_pairwise_correlation=2 * co_movement - 1,
        systemic_co_movement_score=co_movement,
        correlation_percentile=co_movement,
        quality_flags=frozenset(),
    )
    result = _engine(calendar).compute(store, as_of, correlation_context=correlation)
    assert result.dispersion_percentile == expected_percentile
    assert result.correlation_context is not None
    assert result.correlation_context.systemic_correlation_score == co_movement
    assert result.correlation_context.differentiation_score == pytest.approx(
        expected_differentiation
    )
    assert result.correlation_context.dispersion_minus_correlation == pytest.approx(
        expected_percentile - co_movement
    )
    assert result.dispersion_score == expected_percentile
    assert apply_dispersion_to_snapshot(result).dispersion_regime == DispersionRegime.UNKNOWN


def test_unavailable_optional_inputs_and_adjustment_quality(
    calendar: MarketSessionService,
) -> None:
    store, as_of = dispersion_store(calendar, (0.0,) * 6, days=2)
    missing_sectors = SectorFeatureSet.model_construct(
        as_of_ns=as_of, sectors={}, quality_flags=frozenset({QualityFlag.LOW_COVERAGE})
    )
    missing_vol = RealizedVolatilityFeatureSet.model_construct(
        as_of_ns=as_of,
        per_symbol={},
        market_realized_volatility=None,
        quality_flags=frozenset(),
    )
    missing_corr = CorrelationFeatureSet.model_construct(
        as_of_ns=as_of,
        average_pairwise_correlation=None,
        systemic_co_movement_score=None,
        correlation_percentile=None,
        quality_flags=frozenset(),
    )
    result = _engine(calendar, settings=_settings(price_adjustment_status="unknown")).compute(
        store,
        as_of,
        sector_context=missing_sectors,
        volatility_context=missing_vol,
        correlation_context=missing_corr,
    )
    assert result.return_dispersion == 0
    assert result.movement_hhi is None
    assert result.dispersion_score is None
    assert result.correlation_context is not None
    assert result.correlation_context.differentiation_score is None
    assert result.volatility_coverage == 0
    assert {
        QualityFlag.PROVIDER_DEGRADED,
        QualityFlag.MISSING_SECTOR_DATA,
        QualityFlag.MISSING_VOLATILITY_DATA,
        QualityFlag.LOW_COVERAGE,
    } <= result.quality_flags
    assert result.data_quality == DataQuality.POOR
    disabled = _engine(calendar, settings=_settings(enabled=False)).compute(store, as_of)
    assert disabled.return_dispersion is None and disabled.dispersion_score is None


@pytest.mark.parametrize(
    ("returns", "expected_zero"),
    [
        ((0.02,) * 6, True),
        ((-0.02,) * 6, True),
        ((0.01, 0.02, 0.03, 0.04, 0.05, 0.06), False),
        ((-0.01, -0.02, -0.03, -0.04, -0.05, -0.06), False),
        ((0, 0, 0, 0, 0, 0.5), False),
        ((-0.03, -0.03, -0.03, 0.03, 0.03, 0.03), False),
        ((-0.03, 0, 0.03, -0.03, 0, 0.03), False),
    ],
)
def test_deterministic_market_shape_scenarios(
    calendar: MarketSessionService, returns: tuple[float, ...], expected_zero: bool
) -> None:
    store, as_of = dispersion_store(calendar, returns)
    result = _engine(calendar, sector_membership=_sectors()).compute(store, as_of)
    assert result.return_dispersion is not None
    assert (result.return_dispersion == 0) is expected_zero
    assert result.return_mad is not None and result.return_mad >= 0
    assert result.return_iqr is not None and result.return_iqr >= 0
    assert result.dispersion_percentile is not None and 0 <= result.dispersion_percentile <= 1
    if result.movement_hhi is not None:
        assert 1 / len(SYMBOLS) <= result.movement_hhi <= 1
        assert result.effective_movers is not None and result.effective_movers >= 1
