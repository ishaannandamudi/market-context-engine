"""Deterministic precomputed context vectors for MCE-19 rules."""

from mce.config.settings import MCEConfig
from mce.features.ensemble import EnsembleContextFeatureEngine, EnsembleContextInput
from mce.features.network.models import NetworkContext
from mce.features.options_context.models import (
    OptionsContextFeatureSet,
    OptionsStressContext,
    UnderlyingOptionsContext,
)
from mce.models.context import MarketContextSnapshot
from mce.models.enums import DataQuality
from tests.fixtures.options_scenarios import BASE, options_input

VECTORS: dict[str, dict[str, float]] = {
    "broad_risk_on": {
        "market_direction_score": 0.8,
        "breadth_score": 0.8,
        "volatility_score": 0.1,
        "stress_score": 0.1,
        "liquidity_score": 0.9,
        "correlation_score": 0.2,
        "dispersion_score": 0.2,
    },
    "broad_risk_off": {
        "market_direction_score": -0.8,
        "breadth_score": -0.8,
        "volatility_score": 0.6,
        "stress_score": 0.6,
        "liquidity_score": 0.5,
        "correlation_score": 0.5,
        "dispersion_score": 0.5,
    },
    "calm_range": {
        "market_direction_score": 0,
        "breadth_score": 0,
        "volatility_score": 0.1,
        "stress_score": 0.1,
        "liquidity_score": 0.9,
        "correlation_score": 0.2,
        "dispersion_score": 0.2,
    },
    "rotation": {
        "market_direction_score": 0.1,
        "breadth_score": 0.1,
        "volatility_score": 0.3,
        "stress_score": 0.3,
        "liquidity_score": 0.8,
        "correlation_score": 0.1,
        "dispersion_score": 0.8,
    },
    "systemic_stress": {
        "market_direction_score": 0.7,
        "breadth_score": 0.7,
        "volatility_score": 0.95,
        "stress_score": 0.95,
        "liquidity_score": 0.1,
        "correlation_score": 0.9,
        "dispersion_score": 0.1,
    },
    "liquidity_stress": {
        "market_direction_score": 0.6,
        "breadth_score": 0.6,
        "volatility_score": 0.2,
        "stress_score": 0.2,
        "liquidity_score": 0.1,
        "correlation_score": 0.3,
        "dispersion_score": 0.4,
    },
    "volatility_stress": {
        "market_direction_score": 0.4,
        "breadth_score": 0.4,
        "volatility_score": 0.9,
        "stress_score": 0.9,
        "liquidity_score": 0.8,
        "correlation_score": 0.4,
        "dispersion_score": 0.4,
    },
    "mixed": {
        "market_direction_score": 0.25,
        "breadth_score": 0.25,
        "volatility_score": 0.5,
        "stress_score": 0.5,
        "liquidity_score": 0.5,
        "correlation_score": 0.5,
        "dispersion_score": 0.5,
    },
    "external_event_risk": {
        "market_direction_score": 0,
        "breadth_score": 0,
        "volatility_score": 0.5,
        "stress_score": 0.5,
        "liquidity_score": 0.4,
        "correlation_score": 0.5,
        "dispersion_score": 0.5,
    },
}


def snapshot(name: str, *, clock: int = BASE) -> MarketContextSnapshot:
    return MarketContextSnapshot(
        as_of_ns=clock,
        data_quality=DataQuality.GOOD,
        **VECTORS[name],
    )


def result(name: str):
    config = MCEConfig(_env_file=None)
    return EnsembleContextFeatureEngine(config).compute(
        EnsembleContextInput(as_of_ns=BASE, snapshot=snapshot(name))
    )


def network(
    score: float,
    *,
    quality: DataQuality = DataQuality.GOOD,
    coverage: float = 1.0,
    clock: int = BASE,
) -> NetworkContext:
    return NetworkContext(
        as_of_ns=clock,
        input_fingerprint="a" * 64,
        config_fingerprint="b" * 64,
        expected_nodes=10,
        valid_nodes=round(10 * coverage),
        coverage=coverage,
        graph=None,
        topology=None,
        centrality=None,
        communities=None,
        sector_alignment=None,
        mst=None,
        dynamics=None,
        connectivity_score=score,
        residual_density=None,
        raw_minus_residual_density=None,
        quality_flags=frozenset(),
        data_quality=quality,
    )


def options(
    score: float, *, quality: DataQuality = DataQuality.GOOD, mode: str = "synthetic"
) -> OptionsContextFeatureSet:
    source = options_input(mode=mode)
    underlying = UnderlyingOptionsContext(
        source=source,
        stress=OptionsStressContext(score=score),
        market_data_age_ns=0,
        headline_eligible=True,
        data_quality=quality,
    )
    return OptionsContextFeatureSet(
        as_of_ns=BASE,
        config_fingerprint="c" * 64,
        expected_underlyings=("SPY",),
        available_underlyings=("SPY",),
        coverage=1,
        per_underlying=(underlying,),
        primary_symbol="SPY",
        options_stress_score=score,
        data_quality=quality,
    )
