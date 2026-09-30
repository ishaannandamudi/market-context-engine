"""Manual MCE-19 benchmark over precomputed synthetic upstream contexts."""

import argparse
import platform
from time import perf_counter

from mce.config.settings import MCEConfig
from mce.features.ensemble import EnsembleContextFeatureEngine, EnsembleContextInput
from mce.features.ensemble.contracts import FilteredHMMContext
from mce.features.ensemble.disagreement import detect_conflicts
from mce.features.ensemble.evidence import collect_candidates
from mce.features.ensemble.grouping import aggregate
from mce.features.ensemble.snapshot import apply_ensemble_to_snapshot
from mce.features.ensemble.taxonomy import classify
from mce.features.factors.engine import FactorFeatureEngine
from mce.features.macro.models import (
    MacroContextFeatureSet,
    MacroEventContext,
    MacroStressContext,
)
from mce.features.options_context.models import (
    OptionsContextFeatureSet,
    OptionsStressContext,
    UnderlyingOptionsContext,
)
from mce.features.regime import MarketRegimeEngine, RegimeEvidence
from mce.features.regime_transition.models import RegimeTransitionFeatureSet
from mce.features.seasonality.models import IntradaySeasonalityFeatureSet
from mce.models.enums import DataQuality
from mce.statistical.models import LatentStateInference, LatentStateProfile
from mce.statistical.regime.hmm import HMMQuality
from tests.fixtures.ensemble_scenarios import network, snapshot
from tests.fixtures.factor_scenarios import factor_config, factor_store
from tests.fixtures.options_scenarios import options_input


def _input() -> EnsembleContextInput:
    config = MCEConfig(_env_file=None)
    factor_cfg = factor_config(window=20)
    store, times = factor_store(rows=22)
    clock = times[-1]
    factor = FactorFeatureEngine(factor_cfg).compute(store, clock)
    source = options_input(clock=clock)
    underlying = UnderlyingOptionsContext(
        source=source,
        stress=OptionsStressContext(score=0.5),
        market_data_age_ns=0,
        headline_eligible=True,
        data_quality=DataQuality.GOOD,
    )
    options = OptionsContextFeatureSet(
        as_of_ns=clock,
        config_fingerprint="a" * 64,
        expected_underlyings=("SPY",),
        available_underlyings=("SPY",),
        coverage=1,
        per_underlying=(underlying,),
        primary_symbol="SPY",
        options_stress_score=0.5,
        data_quality=DataQuality.GOOD,
    )
    macro = MacroContextFeatureSet(
        as_of_ns=clock,
        config_fingerprint="b" * 64,
        event=MacroEventContext(proximity_score=0.5, upcoming_count_24h=1),
        stress=MacroStressContext(score=0.3),
        data_quality=DataQuality.GOOD,
    )
    regime = MarketRegimeEngine(config).compute(
        RegimeEvidence(
            as_of_ns=clock,
            market_direction_score=0.7,
            trend_score=0.7,
            breadth_score=0.7,
            breadth_participation_score=0.7,
            volatility_score=0.2,
            liquidity_score=0.8,
            correlation_score=0.2,
            stress_score=0.2,
        )
    )
    transition = RegimeTransitionFeatureSet(
        as_of_ns=clock,
        config_fingerprint="c" * 64,
        primary_window_sessions=5,
        windows=(),
        stability_score=0.8,
        instability_score=0.2,
        data_quality=DataQuality.GOOD,
    )
    inference = LatentStateInference(
        as_of_ns=clock,
        state_id=0,
        state_probabilities=(0.9, 0.1),
        mode="filtered",
        retrospective=False,
        training_end_ns=clock - 1,
        model_version="fixture/1",
        feature_schema_version="1.0",
        posterior_entropy=0.2,
    )
    profile = LatentStateProfile(
        state_id=0,
        observations=100,
        occupancy_fraction=0.5,
        centroid=(0.7,),
        median=(0.7,),
        dispersion=(0.1,),
        average_dwell_observations=5,
        deterministic_overlap=(),
    )
    quality = HMMQuality(True, -10, 30, 40, 0.5, 0.2, False, ())
    hmm = FilteredHMMContext(
        inference=inference, profile=profile, quality=quality, direction=0.7, stress=0.2
    )
    seasonal = IntradaySeasonalityFeatureSet(
        as_of_ns=clock,
        session_time=None,
        history_fingerprint="d" * 64,
        config_fingerprint="e" * 64,
        expected_symbols=("SPY",),
        available_symbols=("SPY",),
        member_coverage=1,
        per_symbol=(),
        market_metrics=(),
        correlation=None,
        quality_flags=frozenset(),
        data_quality=DataQuality.GOOD,
    )
    return EnsembleContextInput(
        as_of_ns=clock,
        snapshot=snapshot("broad_risk_on", clock=clock),
        factors=factor,
        network=network(0.7, clock=clock),
        options=options,
        macro=macro,
        deterministic_regime=regime,
        regime_transition=transition,
        hmm=hmm,
        seasonality=seasonal,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=10_000)
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("iterations must be positive")
    source = _input()
    config = MCEConfig(_env_file=None)
    settings = config.features.ensemble
    engine = EnsembleContextFeatureEngine(config)
    observations = collect_candidates(source, settings)
    evidence, groups, _, _ = aggregate(observations, source.as_of_ns, settings)
    baseline = engine.compute(source)
    scores = {
        name: value
        for name, value in baseline.scorecard.model_dump().items()
        if name != "quality_flags"
    }
    context_weights = {item.family: item.reliability for item in baseline.context_reliability}

    def measure(operation) -> float:
        started = perf_counter()
        for _ in range(args.iterations):
            operation()
        return (perf_counter() - started) / args.iterations

    reliability_seconds = measure(
        lambda: tuple(item.reliability(source.as_of_ns, settings) for item in observations)
    )
    grouping_seconds = measure(lambda: aggregate(observations, source.as_of_ns, settings))
    conflict_seconds = measure(
        lambda: detect_conflicts(
            source,
            evidence,
            groups,
            settings.minimum_reliability,
            hmm_weight=context_weights.get("hmm", 0),
            regime_weight=context_weights.get("deterministic_regime", 0),
        )
    )
    rule_seconds = measure(lambda: classify(scores, settings, eligible=True))
    projection_seconds = measure(lambda: apply_ensemble_to_snapshot(baseline, source.snapshot))
    total_seconds = measure(lambda: engine.compute(source))
    print(f"Python {platform.python_version()} on {platform.platform()}")
    print(
        f"iterations={args.iterations} candidates={len(observations)} "
        f"groups={len(groups)} headline={baseline.headline.value}"
    )
    print(
        f"reliability_us={reliability_seconds * 1e6:.1f} "
        f"grouping_us={grouping_seconds * 1e6:.1f} "
        f"conflicts_us={conflict_seconds * 1e6:.1f} "
        f"rules_us={rule_seconds * 1e6:.1f} "
        f"projection_us={projection_seconds * 1e6:.1f} "
        f"engine_us={total_seconds * 1e6:.1f}"
    )


if __name__ == "__main__":
    main()
