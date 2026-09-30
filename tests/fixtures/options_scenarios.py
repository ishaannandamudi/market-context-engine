"""Deterministic, provider-neutral options evidence for MCE-17 tests."""

from mce.config.settings import MCEConfig, OptionsContextSettings
from mce.features.options_context.contracts import (
    DAY_NS,
    ATMIVPoint,
    OptionsMarketContextInput,
    SurfaceIVPoint,
    SurfaceQuality,
)
from mce.features.options_context.models import OptionsContextHistory, OptionsHistoricalPoint
from mce.features.volatility.models import (
    DailyVolatilityWindow,
    EstimatorValue,
    RealizedVolatilityFeatureSet,
    SymbolVolatilityFeatures,
)
from mce.models.enums import DataQuality

BASE = 1_800_000_000_000_000_000


def options_config(
    *,
    broad: tuple[str, ...] = ("SPY",),
    sectors: tuple[str, ...] = (),
    minimum_history: int = 3,
    maximum_market_age_ns: int = 900_000_000_000,
) -> MCEConfig:
    base = MCEConfig(_env_file=None)
    settings = OptionsContextSettings(
        broad_underlyings=broad,
        sector_underlyings=sectors,
        minimum_history=minimum_history,
        maximum_market_age_ns=maximum_market_age_ns,
    )
    return base.model_copy(
        update={"features": base.features.model_copy(update={"options_context": settings})}
    )


def options_input(
    *,
    symbol: str = "SPY",
    scope: str = "broad",
    clock: int = BASE,
    atm_ivs: tuple[float, float, float] = (0.18, 0.20, 0.22),
    put_iv: float = 0.25,
    call_iv: float = 0.19,
    mode: str = "synthetic",
    surface: bool = True,
    own_fallback: bool = False,
) -> OptionsMarketContextInput:
    tenors = (7, 30, 90)
    atm = tuple(
        ATMIVPoint(
            dte_days=days,
            expiration_ns=clock + days * DAY_NS,
            implied_volatility=vol,
            combination="combined",
            source_method="synthetic_surface",
            data_as_of_ns=clock,
            available_at_ns=clock,
        )
        for days, vol in zip(tenors, atm_ivs, strict=True)
    )
    wings = (
        tuple(
            SurfaceIVPoint(
                dte_days=days,
                expiration_ns=clock + days * DAY_NS,
                side=side,  # type: ignore[arg-type]
                coordinate_system="delta",
                coordinate=-0.25 if side == "put" else 0.25,
                implied_volatility=vol,
                source_method="synthetic_surface",
                evidence_origin="own_contract_fallback"
                if own_fallback
                else "independent_neighbors",
                data_as_of_ns=clock,
                available_at_ns=clock,
            )
            for days in tenors
            for side, vol in (("put", put_iv), ("call", call_iv))
        )
        if surface
        else ()
    )
    return OptionsMarketContextInput(
        options_context_schema_version="1.0",
        iv_units="annualized_decimal",
        underlying_symbol=symbol,
        scope=scope,  # type: ignore[arg-type]
        source_engine="synthetic-options",
        source_engine_version="1.2.3",
        source_fingerprint="synthetic-v1",
        calculation_mode="surface_fit",
        market_data_mode=mode,  # type: ignore[arg-type]
        as_of_ns=clock,
        data_as_of_ns=clock,
        available_at_ns=clock,
        underlying_price=500,
        atm_points=atm,
        surface_points=wings,
        surface_quality=SurfaceQuality(
            tenor_coverage=1,
            strike_coverage=1 if surface else 0,
            independent_neighbor_count=len(wings) if not own_fallback else 0,
            own_contract_fallback=own_fallback,
        ),
        quality=DataQuality.GOOD,
    )


def rv_context(
    rv: float, *, symbol: str = "SPY", clock: int = BASE, sessions: int = 20
) -> RealizedVolatilityFeatureSet:
    estimator = EstimatorValue(
        annualized_volatility=rv,
        variance_per_session=rv * rv / 252,
        sample_count=sessions,
    )
    daily = DailyVolatilityWindow(
        name=f"{sessions}d",
        sessions=sessions,
        close_to_close=estimator,
        observed_sessions=sessions,
        expected_sessions=sessions,
        coverage_ratio=1,
    )
    per_symbol = SymbolVolatilityFeatures(
        symbol=symbol,
        as_of_ns=clock,
        intraday={},
        daily={daily.name: daily},
        baseline_sample_count=0,
        sample_count=sessions,
        expected_count=sessions,
        coverage_ratio=1,
    )
    return RealizedVolatilityFeatureSet(
        as_of_ns=clock,
        per_symbol={symbol: per_symbol},
        index_percentiles={},
        available_symbol_count=1,
        expected_symbol_count=1,
        coverage_ratio=1,
        data_quality=DataQuality.GOOD,
    )


def options_history(
    *, symbol: str = "SPY", count: int = 12, clock: int = BASE
) -> OptionsContextHistory:
    return OptionsContextHistory(
        points=tuple(
            OptionsHistoricalPoint(
                underlying_symbol=symbol,
                as_of_ns=clock - (count - i) * DAY_NS,
                available_at_ns=clock - (count - i) * DAY_NS,
                atm_iv=0.15 + i * 0.002,
                downside_skew=0.02 + i * 0.001,
                term_short_long_spread=-0.04 + i * 0.001,
                iv_rv_spread=-0.01 + i * 0.001,
            )
            for i in range(count)
        )
    )
