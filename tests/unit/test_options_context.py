"""Options contract, arithmetic, quality, and causal replay regression tests."""

from itertools import permutations
from json import dumps

import pytest
from pydantic import ValidationError

from mce.config.settings import OptionsContextSettings
from mce.features.options_context import (
    OptionsContextAdapter,
    OptionsContextFeatureEngine,
    OptionsContextHistory,
    OptionsHistoricalPoint,
    OptionsMarketContextInput,
    apply_options_context_to_snapshot,
)
from mce.features.options_context.contracts import DAY_NS, ATMIVPoint, SurfaceIVPoint
from mce.features.options_context.history import compare_history
from mce.features.options_context.realized_implied import compare_implied_realized
from mce.features.options_context.skew import skew_context
from mce.features.options_context.stress import options_stress
from mce.features.options_context.term_structure import term_structure
from mce.models.context import MarketContextSnapshot
from mce.models.enums import DataQuality, MarketRegime, QualityFlag
from tests.fixtures.options_scenarios import (
    BASE,
    options_config,
    options_history,
    options_input,
    rv_context,
)


def _atm(days: int, vol: float) -> ATMIVPoint:
    return ATMIVPoint(
        dte_days=days,
        expiration_ns=BASE + days * DAY_NS,
        implied_volatility=vol,
        combination="combined",
        source_method="fixture",
        data_as_of_ns=BASE,
        available_at_ns=BASE,
    )


def _compute(source: OptionsMarketContextInput | None = None, *, rv: float = 0.20):
    return OptionsContextFeatureEngine(options_config()).compute(
        (source or options_input(),),
        BASE,
        realized_volatility_context=rv_context(rv),
        historical_options_context=options_history(),
    )


def test_contract_adapter_version_units_metadata_and_forward_fields() -> None:
    source = options_input()
    payload = source.model_dump(mode="json")
    payload["future_additive_field"] = {"unused": True}
    parsed = OptionsContextAdapter().translate(payload)
    assert parsed == source
    assert parsed.options_context_schema_version == "1.0"
    assert parsed.iv_units == "annualized_decimal"
    assert parsed.source_engine == "synthetic-options"
    assert parsed.source_engine_version == "1.2.3"
    assert parsed.source_fingerprint == "synthetic-v1"
    assert parsed.calculation_mode == "surface_fit"
    assert parsed.market_data_mode == "synthetic"
    for patch in (
        {"options_context_schema_version": "2.0"},
        {"iv_units": "percent"},
        {"source_engine_version": "bad"},
        {"underlying_price": -1},
        {"data_as_of_ns": BASE + 1},
        {"available_at_ns": BASE - 1},
    ):
        with pytest.raises(ValidationError):
            OptionsContextAdapter().translate({**payload, **patch})
    for required in ("options_context_schema_version", "iv_units"):
        incomplete = {key: value for key, value in payload.items() if key != required}
        with pytest.raises(ValidationError):
            OptionsContextAdapter().translate(incomplete)
    optional = {
        key: value
        for key, value in payload.items()
        if key not in ("source_fingerprint", "underlying_price")
    }
    assert OptionsContextAdapter().translate(optional).source_fingerprint is None


def test_contract_rejects_bad_points_duplicates_and_future_component() -> None:
    source = options_input()
    payload = source.model_dump(mode="json")
    atm = payload["atm_points"][0]
    wing = payload["surface_points"][0]
    for patch in (
        {"implied_volatility": -0.01},
        {"implied_volatility": float("nan")},
        {"dte_days": 0},
        {"expiration_ns": BASE - 1},
        {"data_as_of_ns": BASE + 1},
    ):
        with pytest.raises(ValidationError):
            ATMIVPoint.model_validate_json(dumps({**atm, **patch}))
    for patch in (
        {"coordinate": 0.25},
        {"coordinate_system": "unknown"},
        {"implied_volatility": float("inf")},
        {"available_at_ns": BASE - 1},
    ):
        with pytest.raises(ValidationError):
            SurfaceIVPoint.model_validate_json(dumps({**wing, **patch}))
    for name in ("atm_points", "surface_points"):
        with pytest.raises(ValidationError):
            OptionsContextAdapter().translate({**payload, name: payload[name] + [payload[name][0]]})
    later_point = dict(atm, data_as_of_ns=BASE + 1, available_at_ns=BASE + 1)
    later_point["expiration_ns"] += 1
    with pytest.raises(ValidationError):
        OptionsContextAdapter().translate({**payload, "atm_points": [later_point]})


@pytest.mark.parametrize(
    ("iv", "rv", "spread", "ratio"),
    [(0.24, 0.20, 0.04, 1.2), (0.20, 0.20, 0.0, 1.0), (0.16, 0.20, -0.04, 0.8)],
)
def test_iv_rv_comparable_and_vrp_proxy(iv: float, rv: float, spread: float, ratio: float) -> None:
    point = _atm(30, iv)
    result = compare_implied_realized(
        point, "SPY", BASE, rv_context(rv), options_config().features.options_context
    )
    assert result.comparable and result.rv_horizon_sessions == 20
    assert result.rv_equivalent_calendar_days == pytest.approx(20 * 365 / 252)
    assert result.spread == pytest.approx(spread)
    assert result.ratio == pytest.approx(ratio)
    assert result.vrp_proxy == pytest.approx(spread)


def test_iv_rv_zero_missing_future_and_horizon_mismatch() -> None:
    settings = options_config().features.options_context
    zero = compare_implied_realized(_atm(30, 0.2), "SPY", BASE, rv_context(0), settings)
    assert zero.spread == 0.2 and zero.ratio is None
    tiny = compare_implied_realized(_atm(30, 0.2), "SPY", BASE, rv_context(1e-15), settings)
    assert tiny.ratio is None
    for realized in (None, rv_context(0.2, clock=BASE + 1), rv_context(0.2, symbol="QQQ")):
        missing = compare_implied_realized(_atm(30, 0.2), "SPY", BASE, realized, settings)
        assert missing.spread is None and missing.vrp_proxy is None
    mismatch = compare_implied_realized(
        _atm(7, 0.2), "SPY", BASE, rv_context(0.2, sessions=60), settings
    )
    assert not mismatch.comparable
    assert mismatch.horizon_mismatch_days is not None
    assert mismatch.spread is None and mismatch.vrp_proxy is None


@pytest.mark.parametrize(
    ("vols", "spread", "curvature"),
    [
        ((0.1, 0.2, 0.3), -0.2, 0.044578313253),
        ((0.2, 0.2, 0.2), 0, 0),
        ((0.3, 0.2, 0.1), 0.2, -0.044578313253),
    ],
)
def test_term_structure_shapes(
    vols: tuple[float, float, float], spread: float, curvature: float
) -> None:
    points = tuple(_atm(dte, vol) for dte, vol in zip((7, 30, 90), vols, strict=True))
    term = term_structure(points, 30, 2)
    assert term.short_long_spread == pytest.approx(spread)
    assert term.curvature == pytest.approx(curvature, abs=0.002)
    assert term.slope_per_day is not None
    assert term.short is not None and term.short.dte_days == 7
    assert term.medium is not None and term.medium.dte_days == 30
    assert term.long is not None and term.long.dte_days == 90


def test_term_irregular_dte_exact_slope_and_insufficient_tenors() -> None:
    points = tuple(_atm(day, 0.1 + 0.002 * day) for day in (7, 13, 47, 90))
    term = term_structure(points, 30, 2)
    assert term.slope_per_day == pytest.approx(0.002)
    assert term.curvature == pytest.approx(0)
    assert term_structure(points[:1], 30, 2).short_long_spread is None
    assert term_structure((_atm(10, 0.2), _atm(20, 0.2)), 30, 2).short_long_spread == 0
    convex = term_structure((_atm(7, 0.1), _atm(30, 0.25), _atm(90, 0.3)), 30, 2)
    assert convex.curvature is not None and convex.curvature > 0
    concave = term_structure((_atm(7, 0.3), _atm(30, 0.15), _atm(90, 0.1)), 30, 2)
    assert concave.curvature is not None and concave.curvature < 0


def test_skew_conventions_missing_wings_and_incompatible_coordinate() -> None:
    source = options_input()
    settings = options_config().features.options_context
    skew = skew_context(source.atm_points[1], source.atm_points, source.surface_points, settings)
    assert skew.downside == pytest.approx(0.05)
    assert skew.upside == pytest.approx(-0.01)
    assert skew.risk_reversal_call_minus_put == pytest.approx(-0.06)
    assert skew.butterfly == pytest.approx(0.02)
    symmetric = options_input(put_iv=0.25, call_iv=0.25)
    smile = skew_context(
        symmetric.atm_points[1], symmetric.atm_points, symmetric.surface_points, settings
    )
    assert smile.risk_reversal_call_minus_put == 0
    assert smile.butterfly == pytest.approx(0.05)
    flat = options_input(put_iv=0.2, call_iv=0.2)
    flat_skew = skew_context(flat.atm_points[1], flat.atm_points, flat.surface_points, settings)
    assert flat_skew.downside == 0 and flat_skew.upside == 0
    upside = options_input(put_iv=0.21, call_iv=0.3)
    assert skew_context(
        upside.atm_points[1], upside.atm_points, upside.surface_points, settings
    ).risk_reversal_call_minus_put == pytest.approx(0.09)
    missing = skew_context(
        source.atm_points[1], source.atm_points, source.surface_points[:1], settings
    )
    assert missing.downside is None and missing.risk_reversal_call_minus_put is None
    other = OptionsContextSettings(skew_coordinate="relative_moneyness")
    incompatible = skew_context(
        source.atm_points[1], source.atm_points, source.surface_points, other
    )
    assert incompatible.downside is None and incompatible.upside is None


def test_history_prior_only_midrank_and_bounds() -> None:
    current = BASE
    history = options_history(count=5)
    future = OptionsHistoricalPoint(
        underlying_symbol="SPY", as_of_ns=current + 1, available_at_ns=current + 1, atm_iv=100
    )
    late = OptionsHistoricalPoint(
        underlying_symbol="SPY", as_of_ns=current - 1, available_at_ns=current + 1, atm_iv=100
    )
    mixed = OptionsContextHistory(points=(future, late, *reversed(history.points)))
    assert mixed.prior("SPY", current, 3) == history.points[-3:]
    assert compare_history(2, (1, 2, 3), 3).percentile == 0.5
    assert compare_history(4, (1, 2, 3), 3).percentile == 1
    assert compare_history(4, (1, 2, 3), 4).percentile is None
    assert compare_history(1, (1, 1, 1), 3).robust_z == 0
    assert compare_history(2, (1, 1, 1), 3).robust_z is None
    with pytest.raises(ValidationError):
        OptionsContextHistory(points=(history.points[0], history.points[0]))
    with pytest.raises(ValidationError):
        OptionsHistoricalPoint(underlying_symbol="SPY", as_of_ns=BASE, available_at_ns=BASE - 1)


def test_stress_components_and_scenarios() -> None:
    settings = options_config().features.options_context
    history = options_history()
    normal = _compute(
        options_input(atm_ivs=(0.16, 0.17, 0.18), put_iv=0.18, call_iv=0.17), rv=0.17
    ).per_underlying[0]
    elevated = _compute(options_input(atm_ivs=(0.18, 0.35, 0.22))).per_underlying[0]
    downside = _compute(options_input(put_iv=0.40)).per_underlying[0]
    inverted = _compute(options_input(atm_ivs=(0.30, 0.20, 0.15))).per_underlying[0]
    dislocated = _compute(options_input(), rv=0.10).per_underlying[0]
    assert elevated.stress is not None and elevated.stress.iv == 1
    assert downside.stress is not None and downside.stress.skew == 1
    assert (
        inverted.stress is not None
        and inverted.stress.term is not None
        and inverted.stress.term > 0
    )
    assert dislocated.stress is not None and dislocated.stress.dislocation == 1
    assert normal.stress is not None and normal.stress.score is not None
    assert 0 <= normal.stress.score <= 1
    assert _compute().options_stress_score is not None
    assert options_stress(None, None, None, None, settings).score is None
    assert len(history.points) == 12


def test_quality_modes_sparse_circular_stale_and_missing_rv() -> None:
    for source, flag in (
        (options_input(mode="eod"), QualityFlag.OPTIONS_MODE_LIMITED),
        (options_input(mode="delayed"), QualityFlag.OPTIONS_MODE_LIMITED),
        (options_input(surface=False), QualityFlag.OPTIONS_SURFACE_SPARSE),
        (options_input(own_fallback=True), QualityFlag.OPTIONS_CIRCULAR_EVIDENCE),
        (options_input(clock=BASE - DAY_NS), QualityFlag.STALE_DATA),
    ):
        result = OptionsContextFeatureEngine(options_config()).compute(
            (source,),
            BASE,
            realized_volatility_context=rv_context(0.2),
            historical_options_context=options_history(),
        )
        item = result.per_underlying[0]
        assert flag in item.quality_flags
        assert item.atm_iv is not None
        assert not item.headline_eligible
        assert result.options_stress_score is None
        assert item.iv_rv is not None and item.iv_rv.vrp_proxy is None
    missing_rv = OptionsContextFeatureEngine(options_config()).compute(
        (options_input(),), BASE, historical_options_context=options_history()
    )
    assert QualityFlag.MISSING_VOLATILITY_DATA in missing_rv.quality_flags
    assert missing_rv.per_underlying[0].iv_rv is not None
    assert missing_rv.per_underlying[0].iv_rv.spread is None


def test_future_inputs_revisions_history_and_order_cannot_change_past() -> None:
    source = options_input()
    future = options_input(clock=BASE + DAY_NS, atm_ivs=(0.7, 0.8, 0.9))
    revised = source.model_copy(
        update={
            "available_at_ns": BASE + 1,
            "atm_points": tuple(
                point.model_copy(update={"implied_volatility": 0.99}) for point in source.atm_points
            ),
        }
    )
    history = options_history()
    future_history = OptionsContextHistory(
        points=(
            *history.points,
            OptionsHistoricalPoint(
                underlying_symbol="SPY", as_of_ns=BASE + 1, available_at_ns=BASE + 1, atm_iv=0.99
            ),
        )
    )
    engine = OptionsContextFeatureEngine(options_config())
    baseline = engine.compute(
        (source,),
        BASE,
        realized_volatility_context=rv_context(0.2),
        historical_options_context=history,
    )
    for inputs in permutations((source, future, revised)):
        result = engine.compute(
            inputs,
            BASE,
            realized_volatility_context=rv_context(0.2),
            historical_options_context=future_history,
        )
        assert result.fingerprint == baseline.fingerprint
    no_future_rv = engine.compute(
        (source,),
        BASE,
        realized_volatility_context=rv_context(0.2, clock=BASE + 1),
        historical_options_context=history,
    )
    assert no_future_rv.per_underlying[0].iv_rv is not None
    assert no_future_rv.per_underlying[0].iv_rv.spread is None
    assert (
        engine.compute(
            (source,),
            BASE,
            realized_volatility_context=rv_context(0.2),
            historical_options_context=history,
        ).fingerprint
        == baseline.fingerprint
    )
    assert engine.compute((future,), BASE).per_underlying == ()


def test_surface_tenor_future_component_contract_and_point_freshness() -> None:
    source = options_input()
    payload = source.model_dump(mode="json")
    future_wing = dict(
        payload["surface_points"][0],
        data_as_of_ns=BASE + 1,
        available_at_ns=BASE + 1,
        expiration_ns=payload["surface_points"][0]["expiration_ns"] + 1,
    )
    with pytest.raises(ValidationError):
        OptionsContextAdapter().translate({**payload, "surface_points": [future_wing]})
    future_tenor = dict(
        payload["atm_points"][0],
        data_as_of_ns=BASE + 1,
        available_at_ns=BASE + 1,
        expiration_ns=payload["atm_points"][0]["expiration_ns"] + 1,
    )
    with pytest.raises(ValidationError):
        OptionsContextAdapter().translate({**payload, "atm_points": [future_tenor]})
    stale_wing = source.surface_points[0].model_copy(
        update={
            "data_as_of_ns": BASE - DAY_NS,
            "available_at_ns": BASE - DAY_NS,
            "expiration_ns": source.surface_points[0].expiration_ns - DAY_NS,
        }
    )
    stale_source = source.model_copy(
        update={"surface_points": (stale_wing, *source.surface_points[1:])}
    )
    result = _compute(stale_source)
    assert QualityFlag.STALE_DATA in result.quality_flags
    assert result.per_underlying[0].atm_iv == 0.2
    assert result.options_stress_score is None


def test_multi_underlying_sector_dispersion_snapshot_and_config_fingerprint() -> None:
    config = options_config(broad=("SPY", "QQQ"), sectors=("XLK",))
    sources = (
        options_input(symbol="SPY"),
        options_input(symbol="QQQ", atm_ivs=(0.2, 0.25, 0.3)),
        options_input(symbol="XLK", scope="sector", atm_ivs=(0.22, 0.3, 0.34)),
    )
    history = OptionsContextHistory(
        points=(
            *options_history(symbol="SPY").points,
            *options_history(symbol="QQQ").points,
            *options_history(symbol="XLK").points,
        )
    )
    engine = OptionsContextFeatureEngine(config)
    result = engine.compute(
        sources,
        BASE,
        realized_volatility_context=rv_context(0.2),
        historical_options_context=history,
    )
    assert result.coverage == 1
    assert result.atm_iv_dispersion == pytest.approx(0.040824829, abs=1e-8)
    sector = result.per_underlying[2]
    assert sector.relative_to_broad_iv == pytest.approx(0.1)
    assert sector.relative_to_broad_skew is not None
    assert sector.relative_to_broad_term is not None
    assert result.options_stress_score is not None
    snapshot = apply_options_context_to_snapshot(
        result,
        MarketContextSnapshot(
            as_of_ns=BASE,
            market_regime=MarketRegime.RISK_OFF,
            stress_score=0.77,
            realized_volatility=0.2,
            feature_values={"trend.score": 0.3, "options.old": 9},
        ),
    )
    assert snapshot.schema_version == "1.0" and snapshot.engine_version == "0.17.0"
    assert snapshot.market_regime == MarketRegime.RISK_OFF and snapshot.stress_score == 0.77
    assert snapshot.realized_volatility == 0.2
    assert snapshot.feature_values is not None
    assert "options.old" not in snapshot.feature_values
    assert snapshot.feature_values["options.iv.atm"] == 0.2
    assert snapshot.feature_values["options.skew.risk_reversal_call_minus_put"] == pytest.approx(
        -0.06
    )
    assert snapshot.feature_values["options.underlying.SPY.source_as_of_ns"] == BASE
    assert snapshot.implied_volatility_context is not None
    assert snapshot.volatility_skew_context is not None
    assert snapshot.volatility_term_structure is not None
    assert snapshot.source_freshness is not None and len(snapshot.source_freshness) == 3
    assert MarketContextSnapshot.model_validate_json(snapshot.model_dump_json()) == snapshot
    altered = options_config(broad=("SPY", "QQQ"), sectors=("XLK",), minimum_history=4)
    assert (
        OptionsContextFeatureEngine(altered).compute(sources, BASE).config_fingerprint
        != result.config_fingerprint
    )
    with pytest.raises(ValueError):
        apply_options_context_to_snapshot(result, MarketContextSnapshot(as_of_ns=BASE + 1))


def test_partial_universe_disabled_and_ambiguous_revisions() -> None:
    config = options_config(broad=("SPY", "QQQ"))
    engine = OptionsContextFeatureEngine(config)
    source = options_input()
    result = engine.compute((source,), BASE)
    assert result.coverage == 0.5
    assert QualityFlag.MISSING_OPTIONS_CONTEXT in result.quality_flags
    assert QualityFlag.PARTIAL_UNIVERSE in result.quality_flags
    assert result.options_stress_score is None
    changed = source.model_copy(update={"source_fingerprint": "other"})
    with pytest.raises(ValueError, match="ambiguous"):
        engine.compute((source, changed), BASE)
    disabled = config.features.options_context.model_copy(update={"enabled": False})
    disabled_config = config.model_copy(
        update={"features": config.features.model_copy(update={"options_context": disabled})}
    )
    disabled_result = OptionsContextFeatureEngine(disabled_config).compute((source,), BASE)
    assert disabled_result.per_underlying == ()
    assert disabled_result.data_quality == DataQuality.UNKNOWN
    assert not disabled_result.quality_flags
    assert _compute().data_quality in (DataQuality.GOOD, DataQuality.DEGRADED)


def test_zero_extreme_iv_asymmetry_and_same_expiration_skew() -> None:
    zero = _atm(30, 0)
    extreme = _atm(90, 24)
    assert zero.implied_volatility == 0
    assert extreme.implied_volatility == 24  # Explicit decimal units; raw is not clipped.
    source = options_input()
    huge_wings = tuple(
        point.model_copy(update={"implied_volatility": 1e308}) for point in source.surface_points
    )
    extreme_skew = skew_context(
        source.atm_points[1],
        source.atm_points,
        huge_wings,
        options_config().features.options_context,
    )
    assert extreme_skew.butterfly == pytest.approx(1e308)
    call = source.atm_points[1].model_copy(
        update={"combination": "call", "implied_volatility": 0.205}
    )
    put = source.atm_points[1].model_copy(
        update={"combination": "put", "implied_volatility": 0.198}
    )
    asymmetry = skew_context(
        source.atm_points[1],
        (*source.atm_points, call, put),
        source.surface_points,
        options_config().features.options_context,
    )
    assert asymmetry.atm_call_minus_put == pytest.approx(0.007)
    shifted = tuple(
        point.model_copy(update={"expiration_ns": point.expiration_ns + DAY_NS})
        for point in source.surface_points
    )
    incompatible = skew_context(
        source.atm_points[1], source.atm_points, shifted, options_config().features.options_context
    )
    assert incompatible.risk_reversal_call_minus_put is None


def test_quality_threshold_crossed_extrapolated_bad_wings_and_scope() -> None:
    source = options_input()
    engine = OptionsContextFeatureEngine(options_config())
    crossed_quality = source.surface_quality.model_copy(update={"crossed_quote_count": 1})
    crossed = engine.compute(
        (source.model_copy(update={"surface_quality": crossed_quality}),),
        BASE,
        historical_options_context=options_history(),
    )
    assert QualityFlag.REJECTED_DATA in crossed.quality_flags
    assert crossed.options_stress_score is None
    extrapolated_quality = source.surface_quality.model_copy(update={"extrapolated": True})
    extrapolated = engine.compute(
        (source.model_copy(update={"surface_quality": extrapolated_quality}),),
        BASE,
        historical_options_context=options_history(),
    )
    assert extrapolated.options_stress_score is None
    bad_wings = tuple(
        point.model_copy(update={"quality": DataQuality.POOR}) for point in source.surface_points
    )
    bad = engine.compute((source.model_copy(update={"surface_points": bad_wings}),), BASE)
    assert QualityFlag.OPTIONS_SURFACE_SPARSE in bad.quality_flags
    assert bad.per_underlying[0].skew is not None
    assert bad.per_underlying[0].skew.downside is None
    no_target_call = tuple(
        point
        for point in source.surface_points
        if not (point.dte_days == 30 and point.side == "call")
    )
    missing_wing_result = engine.compute(
        (source.model_copy(update={"surface_points": no_target_call}),), BASE
    )
    missing_wing_snapshot = apply_options_context_to_snapshot(missing_wing_result)
    assert missing_wing_result.per_underlying[0].skew is not None
    assert missing_wing_snapshot.volatility_skew_context is None
    assert missing_wing_snapshot.feature_values is not None
    assert "options.skew.downside" not in missing_wing_snapshot.feature_values
    poor = engine.compute((source.model_copy(update={"quality": DataQuality.POOR}),), BASE)
    assert QualityFlag.PROVIDER_DEGRADED in poor.quality_flags
    strict_config = options_config()
    strict_settings = strict_config.features.options_context.model_copy(
        update={"minimum_quality": "good"}
    )
    strict_config = strict_config.model_copy(
        update={
            "features": strict_config.features.model_copy(
                update={"options_context": strict_settings}
            )
        }
    )
    degraded_atm = tuple(
        point.model_copy(update={"quality": DataQuality.DEGRADED}) for point in source.atm_points
    )
    strict_result = OptionsContextFeatureEngine(strict_config).compute(
        (source.model_copy(update={"atm_points": degraded_atm}),), BASE
    )
    assert not strict_result.per_underlying[0].headline_eligible
    no_price = engine.compute((source.model_copy(update={"underlying_price": None}),), BASE)
    assert QualityFlag.OPTIONS_UNDERLYING_PRICE_MISSING in no_price.quality_flags
    with pytest.raises(ValueError, match="scope"):
        engine.compute((source.model_copy(update={"scope": "sector"}),), BASE)
    with pytest.raises(ValueError, match="nonnegative"):
        engine.compute((source,), -1)


def test_engine_horizon_mismatch_and_reliable_primary_fallback() -> None:
    engine = OptionsContextFeatureEngine(options_config())
    mismatch = engine.compute(
        (options_input(),),
        BASE,
        realized_volatility_context=rv_context(0.2, sessions=60),
        historical_options_context=options_history(),
    )
    assert QualityFlag.OPTIONS_HORIZON_MISMATCH in mismatch.quality_flags
    assert mismatch.per_underlying[0].iv_rv is not None
    assert mismatch.per_underlying[0].iv_rv.vrp_proxy is None
    config = options_config(broad=("SPY", "QQQ"))
    result = OptionsContextFeatureEngine(config).compute(
        (options_input(mode="eod"), options_input(symbol="QQQ")), BASE
    )
    assert result.primary_symbol == "QQQ"
    assert result.atm_iv_dispersion is None
