"""MCE-19 grouping, reliability, conflicts, headlines, causality, and projection."""

from dataclasses import replace

import pytest
from pydantic import ValidationError

from mce.config.settings import EnsembleSettings, MCEConfig
from mce.features.ensemble import (
    EnsembleContextFeatureEngine,
    EnsembleContextInput,
    EnsembleHeadline,
    FilteredHMMContext,
    apply_ensemble_to_snapshot,
)
from mce.features.factors.engine import FactorFeatureEngine
from mce.features.macro.contracts import MacroEventType
from mce.features.macro.engine import MacroContextFeatureEngine
from mce.features.models import TrendFeatureSet
from mce.features.regime import MarketRegimeEngine, RegimeEvidence
from mce.features.regime.taxonomy import HeadlineRegime
from mce.features.regime_transition.models import RegimeTransitionFeatureSet
from mce.features.seasonality.models import (
    IntradaySeasonalityFeatureSet,
    MarketSeasonalMetric,
)
from mce.models.context import MarketContextSnapshot, SourceFreshness
from mce.models.enums import DataQuality, MarketRegime, QualityFlag
from mce.serialization import canonical_json_str
from mce.statistical.models import LatentStateInference, LatentStateProfile
from mce.statistical.regime.hmm import HMMQuality
from tests.fixtures.ensemble_scenarios import BASE, network, options, result, snapshot
from tests.fixtures.factor_scenarios import factor_config, factor_store
from tests.fixtures.macro_scenarios import MINUTE, batch, event, full_context, macro_config


def _engine(settings: EnsembleSettings | None = None) -> EnsembleContextFeatureEngine:
    config = MCEConfig(_env_file=None)
    if settings is not None:
        config = config.model_copy(
            update={"features": config.features.model_copy(update={"ensemble": settings})}
        )
    return EnsembleContextFeatureEngine(config)


def _input(name: str = "broad_risk_on", **updates: object) -> EnsembleContextInput:
    return EnsembleContextInput(as_of_ns=BASE, snapshot=snapshot(name), **updates)


def _macro_event() -> object:
    return MacroContextFeatureEngine(macro_config()).compute(
        (batch(events=(event("cpi", MacroEventType.CPI, BASE + MINUTE),)),), BASE
    )


def _hmm(*, entropy: float = 0.2, converged: bool = True) -> FilteredHMMContext:
    inference = LatentStateInference(
        as_of_ns=BASE,
        state_id=0,
        state_probabilities=(0.9, 0.1),
        mode="filtered",
        retrospective=False,
        training_end_ns=BASE - 1,
        model_version="fixture/1",
        feature_schema_version="1.0",
        posterior_entropy=entropy,
    )
    profile = LatentStateProfile(
        state_id=0,
        observations=100,
        occupancy_fraction=0.5,
        centroid=(0.8, -0.8),
        median=(0.8, -0.8),
        dispersion=(0.1, 0.1),
        average_dwell_observations=5,
        deterministic_overlap=(),
    )
    quality = HMMQuality(
        converged=converged,
        log_likelihood=-10,
        aic=30,
        bic=40,
        minimum_occupancy=0.5,
        mean_posterior_entropy=entropy,
        excessive_switching=False,
        degenerate_states=(),
    )
    return FilteredHMMContext(
        inference=inference, profile=profile, quality=quality, direction=-0.8, stress=0.8
    )


@pytest.mark.parametrize(
    ("name", "headline"),
    [
        ("broad_risk_on", EnsembleHeadline.BROAD_RISK_ON),
        ("broad_risk_off", EnsembleHeadline.BROAD_RISK_OFF),
        ("calm_range", EnsembleHeadline.CALM_RANGE),
        ("rotation", EnsembleHeadline.ROTATION),
        ("systemic_stress", EnsembleHeadline.SYSTEMIC_STRESS),
        ("liquidity_stress", EnsembleHeadline.LIQUIDITY_STRESS),
        ("volatility_stress", EnsembleHeadline.VOLATILITY_STRESS),
        ("mixed", EnsembleHeadline.MIXED),
    ],
)
def test_headline_taxonomy_and_rule_trace(name: str, headline: EnsembleHeadline) -> None:
    context = result(name)
    assert context.headline == headline
    assert [item.rule_id for item in context.rule_trace] == list(_engine().settings.precedence)
    assert next(item for item in context.rule_trace if item.matched).rule_id == headline.value
    assert context.scorecard.evidence_strength is not None
    assert 0 <= context.scorecard.evidence_strength <= 1


def test_grouping_active_weights_and_exact_contributor_reconciliation() -> None:
    context = result("broad_risk_on")
    groups = {item.name: item for item in context.groups}
    assert groups["volatility_stress"].active_families == ("realized_volatility", "market_stress")
    assert groups["systemic_structure"].active_families == ("correlation", "dispersion")
    assert groups["external_risk"].score is None
    assert sum(item.normalized_active_weight for item in context.groups) == pytest.approx(1)
    assert sum(item.contribution for item in context.contributors) == pytest.approx(
        context.scorecard.core_market_score
    )
    assert all(0 <= item.reliability <= 1 for item in context.evidence)
    assert all(0 <= item.normalized_active_weight <= 1 for item in context.evidence)
    assert all(
        0 <= item.score <= 1
        for item in context.groups
        if item.score is not None and item.name not in ("direction", "participation")
    )
    assert context.scorecard.consensus == pytest.approx(1 - context.scorecard.disagreement)


def test_sector_participation_snapshot_uses_breadth_not_relative_strength() -> None:
    relative_only = snapshot("broad_risk_on").model_copy(update={"sector_scores": {"Tech": -0.9}})
    without_breadth = _engine().compute(EnsembleContextInput(as_of_ns=BASE, snapshot=relative_only))
    assert "sectors" not in {item.family for item in without_breadth.evidence}
    projected = relative_only.model_copy(update={"feature_values": {"sector.Tech.breadth": -0.6}})
    with_breadth = _engine().compute(EnsembleContextInput(as_of_ns=BASE, snapshot=projected))
    sector_vote = next(item for item in with_breadth.evidence if item.family == "sectors")
    assert sector_vote.value == -0.6
    assert sector_vote.source_feature == "snapshot.sector.*.breadth"


def test_structure_family_double_counting_with_factor_network_and_correlation() -> None:
    config = factor_config(window=20)
    store, times = factor_store(rows=22)
    factor = FactorFeatureEngine(config).compute(store, times[-1])
    snap = snapshot("broad_risk_on", clock=times[-1])
    context = _engine().compute(
        EnsembleContextInput(
            as_of_ns=times[-1],
            snapshot=snap,
            factors=factor,
            network=network(0.8, clock=times[-1]),
        )
    )
    group = next(item for item in context.groups if item.name == "systemic_structure")
    assert "correlation" in group.active_families
    assert "factors" in group.active_families
    assert "network" in group.active_families
    assert len([item for item in context.contributors if item.group == "systemic_structure"]) == 1
    assert group.normalized_active_weight <= 0.11
    assert sum(
        item.normalized_active_weight
        for item in context.evidence
        if item.group == "systemic_structure"
    ) == pytest.approx(1)


def test_reliability_reduces_or_withholds_poor_external_network_and_hmm() -> None:
    good = _engine().compute(_input(options=options(0.9), network=network(0.8)))
    poor = _engine().compute(
        _input(
            options=options(0.9, quality=DataQuality.POOR),
            network=network(0.8, quality=DataQuality.POOR, coverage=0.2),
        )
    )
    weights = {item.family: item for item in good.evidence}
    poor_weights = {item.family: item for item in poor.evidence}
    assert weights["options"].adjusted_weight > 0
    assert poor_weights["options"].adjusted_weight == 0
    assert poor_weights["network"].adjusted_weight == 0
    assert QualityFlag.ENSEMBLE_SOURCE_DEGRADED in poor.quality_flags
    eod = _engine().compute(_input(options=options(0.9, mode="eod")))
    assert (
        next(item.reliability for item in eod.evidence if item.family == "options")
        < weights["options"].reliability
    )
    low_entropy = _engine().compute(_input(hmm=_hmm(entropy=0.1)))
    high_entropy = _engine().compute(_input(hmm=_hmm(entropy=0.6, converged=False)))

    def hmm_weight(context):
        return next(
            item.reliability for item in context.context_reliability if item.family == "hmm"
        )

    assert hmm_weight(high_entropy) < hmm_weight(low_entropy)


def test_direction_options_network_liquidity_and_macro_conflicts() -> None:
    snap = snapshot("broad_risk_on").model_copy(
        update={
            "breadth_score": -0.8,
            "volatility_score": 0.1,
            "stress_score": 0.8,
            "liquidity_score": 0.9,
            "correlation_score": 0.1,
        }
    )
    context = _engine().compute(
        EnsembleContextInput(
            as_of_ns=BASE,
            snapshot=snap,
            options=options(0.9),
            network=network(0.9),
        )
    )
    ids = {item.conflict_id for item in context.conflicts}
    assert {
        "direction_vs_breadth",
        "realized_vs_options",
        "correlation_vs_network",
        "liquidity_vs_stress",
    } <= ids
    assert context.scorecard.disagreement is not None and context.scorecard.disagreement > 0
    assert context.scorecard.uncertainty is not None and context.scorecard.uncertainty > 0
    macro = MacroContextFeatureEngine(macro_config()).compute((batch(),), BASE)
    stressed = macro.model_copy(update={"stress": macro.stress.model_copy(update={"score": 0.9})})
    calm = snapshot("calm_range")
    context = _engine().compute(EnsembleContextInput(as_of_ns=BASE, snapshot=calm, macro=stressed))
    assert "macro_vs_market" in {item.conflict_id for item in context.conflicts}


def test_external_event_context_default_precedence_and_explicit_rule() -> None:
    macro = _macro_event()
    calm = _engine().compute(_input("broad_risk_on", macro=macro))
    assert calm.headline == EnsembleHeadline.BROAD_RISK_ON
    assert calm.scorecard.external_risk is not None and calm.scorecard.external_risk > 0.9
    neutral = _engine(EnsembleSettings(external_event_headline_enabled=True)).compute(
        _input("external_event_risk", macro=macro)
    )
    assert neutral.headline == EnsembleHeadline.EXTERNAL_EVENT_RISK
    assert neutral.scorecard.volatility_stress == 0.5


def test_transition_stability_separate_from_evidence_strength() -> None:
    transition = RegimeTransitionFeatureSet(
        as_of_ns=BASE,
        config_fingerprint="a" * 64,
        primary_window_sessions=5,
        windows=(),
        stability_score=0.2,
        instability_score=0.8,
        data_quality=DataQuality.GOOD,
    )
    context = _engine().compute(_input("mixed", regime_transition=transition))
    assert context.headline == EnsembleHeadline.TRANSITIONAL
    assert context.scorecard.temporal_stability == 0.2
    assert context.scorecard.evidence_strength is not None
    assert context.scorecard.evidence_strength != context.scorecard.temporal_stability
    poor = _engine().compute(
        _input(
            "mixed",
            regime_transition=transition.model_copy(update={"data_quality": DataQuality.POOR}),
        )
    )
    assert poor.scorecard.temporal_stability is None
    assert poor.headline == EnsembleHeadline.MIXED


def test_hmm_filtered_profile_conflict_and_retroactive_modes_rejected() -> None:
    config = MCEConfig(_env_file=None)
    regime = MarketRegimeEngine(config).compute(
        RegimeEvidence(
            as_of_ns=BASE,
            market_direction_score=0.8,
            trend_score=0.8,
            breadth_score=0.8,
            breadth_participation_score=0.8,
            volatility_score=0.1,
            liquidity_score=0.9,
            correlation_score=0.2,
            stress_score=0.1,
        )
    )
    assert regime.headline_regime == HeadlineRegime.RISK_ON
    context = _engine().compute(_input(deterministic_regime=regime, hmm=_hmm()))
    assert context.deterministic_regime == "risk_on"
    assert context.hmm_state_id == 0
    assert "deterministic_vs_hmm" in {item.conflict_id for item in context.conflicts}
    assert context.headline != EnsembleHeadline.UNKNOWN
    weak = _engine().compute(
        _input(deterministic_regime=regime, hmm=_hmm(entropy=0.6, converged=False))
    )
    assert "deterministic_vs_hmm" not in {item.conflict_id for item in weak.conflicts}
    inference = _hmm().inference
    for mode in ("smoothed", "viterbi"):
        with pytest.raises(ValidationError, match="filtered"):
            FilteredHMMContext(inference=replace(inference, mode=mode, retrospective=True))
    with pytest.raises(ValidationError, match="profile"):
        FilteredHMMContext(inference=inference, direction=0.8)
    with pytest.raises(ValidationError, match="schema"):
        FilteredHMMContext(inference=replace(inference, feature_schema_version="9.0"))


def test_critical_missingness_gates_headline_without_zero_imputation() -> None:
    snap = MarketContextSnapshot(
        as_of_ns=BASE, data_quality=DataQuality.GOOD, market_direction_score=0.8
    )
    context = _engine().compute(EnsembleContextInput(as_of_ns=BASE, snapshot=snap))
    assert context.headline == EnsembleHeadline.UNKNOWN
    assert context.scorecard.participation is None
    assert context.scorecard.core_market_score is None
    assert context.scorecard.evidence_strength is None
    assert QualityFlag.ENSEMBLE_INSUFFICIENT_EVIDENCE in context.quality_flags


def test_snapshot_projection_preserves_upstream_regime_and_serialization() -> None:
    base = snapshot("broad_risk_on").model_copy(
        update={
            "market_regime": MarketRegime.RISK_OFF,
            "feature_values": {"macro.stress.score": 0.3, "ensemble.old": 7},
        }
    )
    context = _engine().compute(EnsembleContextInput(as_of_ns=BASE, snapshot=base))
    projected = apply_ensemble_to_snapshot(context, base)
    assert projected.schema_version == "1.0" and projected.engine_version == "1.0.0"
    assert projected.market_regime == MarketRegime.RISK_OFF
    assert projected.feature_values is not None
    assert projected.feature_values["ensemble.headline"] == "broad_risk_on"
    assert projected.feature_values["macro.stress.score"] == 0.3
    assert "ensemble.old" not in projected.feature_values
    assert MarketContextSnapshot.model_validate_json(projected.model_dump_json()) == projected
    assert canonical_json_str(projected) == canonical_json_str(
        apply_ensemble_to_snapshot(context, projected)
    )
    with pytest.raises(ValueError):
        apply_ensemble_to_snapshot(context, MarketContextSnapshot(as_of_ns=BASE + 1))


def test_config_fingerprint_replay_and_precedence_validation() -> None:
    first = result("broad_risk_on")
    assert first.fingerprint == result("broad_risk_on").fingerprint
    assert canonical_json_str(first) == canonical_json_str(result("broad_risk_on"))
    settings = EnsembleSettings(minimum_reliability=0.7)
    other = _engine(settings).compute(_input())
    assert other.config_fingerprint != first.config_fingerprint
    with pytest.raises(ValidationError):
        EnsembleSettings(precedence=("mixed",))
    with pytest.raises(ValidationError):
        EnsembleSettings(minimum_active_groups=6)


def test_all_future_source_families_are_rejected() -> None:
    snap = snapshot("broad_risk_on")
    with pytest.raises(ValidationError, match="snapshot"):
        EnsembleContextInput(as_of_ns=BASE - 1, snapshot=snap)
    trend = TrendFeatureSet(
        as_of_ns=BASE + 1,
        symbols={},
        index_scores={},
        market_direction_score=0.7,
        available_symbol_count=0,
        expected_symbol_count=0,
        coverage_ratio=0,
        data_quality=DataQuality.GOOD,
    )
    with pytest.raises(ValidationError, match="trend"):
        EnsembleContextInput(as_of_ns=BASE, snapshot=snap, trend=trend)
    with pytest.raises(ValidationError, match="HMM"):
        EnsembleContextInput(
            as_of_ns=BASE,
            snapshot=snap,
            hmm=FilteredHMMContext(inference=replace(_hmm().inference, as_of_ns=BASE + 1)),
        )
    with pytest.raises(ValidationError, match="options"):
        EnsembleContextInput(
            as_of_ns=BASE,
            snapshot=snap,
            options=options(0.8).model_copy(update={"as_of_ns": BASE + 1}),
        )
    options_context = options(0.8)
    underlying = options_context.per_underlying[0]
    future_source = underlying.source.model_copy(update={"available_at_ns": BASE + 1})
    with pytest.raises(ValidationError, match="options source"):
        EnsembleContextInput(
            as_of_ns=BASE,
            snapshot=snap,
            options=options_context.model_copy(
                update={
                    "per_underlying": (underlying.model_copy(update={"source": future_source}),)
                }
            ),
        )
    with pytest.raises(ValidationError, match="macro"):
        EnsembleContextInput(
            as_of_ns=BASE,
            snapshot=snap,
            macro=_macro_event().model_copy(update={"as_of_ns": BASE + 1}),
        )
    macro_context = _macro_event()
    metadata = macro_context.sources[0].model_copy(update={"available_at_ns": BASE + 1})
    with pytest.raises(ValidationError, match="macro source"):
        EnsembleContextInput(
            as_of_ns=BASE,
            snapshot=snap,
            macro=macro_context.model_copy(update={"sources": (metadata,)}),
        )
    assert macro_context.event is not None and macro_context.event.next_event is not None
    future_event = macro_context.event.next_event.model_copy(
        update={"calendar_available_at_ns": BASE + 1}
    )
    with pytest.raises(ValidationError, match="calendar"):
        EnsembleContextInput(
            as_of_ns=BASE,
            snapshot=snap,
            macro=macro_context.model_copy(
                update={
                    "event": macro_context.event.model_copy(update={"next_event": future_event})
                }
            ),
        )
    macro_full = MacroContextFeatureEngine(macro_config()).compute((full_context(),), BASE)
    series = macro_full.per_series[0]
    assert series.observation is not None
    revision = series.observation.model_copy(
        update={"available_at_ns": BASE + 1, "vintage_at_ns": BASE + 1}
    )
    with pytest.raises(ValidationError, match="macro observation"):
        EnsembleContextInput(
            as_of_ns=BASE,
            snapshot=snap,
            macro=macro_full.model_copy(
                update={
                    "per_series": (
                        series.model_copy(update={"observation": revision}),
                        *macro_full.per_series[1:],
                    )
                }
            ),
        )
    with pytest.raises(ValidationError, match="network"):
        EnsembleContextInput(as_of_ns=BASE, snapshot=snap, network=network(0.8, clock=BASE + 1))
    config = factor_config(window=20)
    store, times = factor_store(rows=22)
    factor = FactorFeatureEngine(config).compute(store, times[-1])
    with pytest.raises(ValidationError, match="factors"):
        EnsembleContextInput(
            as_of_ns=times[-1] - 1,
            snapshot=snapshot("broad_risk_on", clock=times[-1] - 1),
            factors=factor,
        )
    assert factor.artifact is not None
    future_artifact = replace(factor.artifact, fit_end_ns=times[-1] + 1)
    with pytest.raises(ValidationError, match="future factor fit"):
        EnsembleContextInput(
            as_of_ns=times[-1],
            snapshot=snapshot("broad_risk_on", clock=times[-1]),
            factors=replace(factor, artifact=future_artifact),
        )
    seasonal = IntradaySeasonalityFeatureSet(
        as_of_ns=BASE + 1,
        session_time=None,
        history_fingerprint=None,
        config_fingerprint="a" * 64,
        expected_symbols=(),
        available_symbols=(),
        member_coverage=0,
        per_symbol=(),
        market_metrics=(),
        correlation=None,
        quality_flags=frozenset(),
        data_quality=DataQuality.UNKNOWN,
    )
    with pytest.raises(ValidationError, match="seasonality"):
        EnsembleContextInput(as_of_ns=BASE, snapshot=snap, seasonality=seasonal)


def test_stale_snapshot_source_reduces_fallback_reliability() -> None:
    fresh = _engine().compute(_input())
    stale_snapshot = snapshot("broad_risk_on").model_copy(
        update={
            "source_freshness": (
                SourceFreshness(
                    source="market-data",
                    latest_effective_timestamp_ns=BASE - 1,
                    age_ns=1,
                    is_stale=True,
                ),
            )
        }
    )
    stale = _engine().compute(EnsembleContextInput(as_of_ns=BASE, snapshot=stale_snapshot))

    def trend_reliability(context):
        return next(item.reliability for item in context.evidence if item.family == "trend")

    assert trend_reliability(stale) == pytest.approx(0.5 * trend_reliability(fresh))
    assert stale.scorecard.evidence_strength < fresh.scorecard.evidence_strength


def test_seasonal_normality_reduces_elevated_volatility_weight() -> None:
    seasonal = IntradaySeasonalityFeatureSet(
        as_of_ns=BASE,
        session_time=None,
        history_fingerprint="a" * 64,
        config_fingerprint="b" * 64,
        expected_symbols=("SPY",),
        available_symbols=("SPY",),
        member_coverage=1,
        per_symbol=(),
        market_metrics=(
            MarketSeasonalMetric(
                metric="volatility",
                ratio=1.0,
                percentile=0.5,
                robust_z=0,
                normalized_symbols=1,
                coverage=1,
            ),
        ),
        correlation=None,
        quality_flags=frozenset(),
        data_quality=DataQuality.GOOD,
    )
    snap = snapshot("volatility_stress")
    base = _engine().compute(EnsembleContextInput(as_of_ns=BASE, snapshot=snap))
    adjusted = _engine().compute(
        EnsembleContextInput(as_of_ns=BASE, snapshot=snap, seasonality=seasonal)
    )

    def vol_weight(context):
        return next(
            item.reliability for item in context.evidence if item.family == "realized_volatility"
        )

    assert vol_weight(adjusted) < vol_weight(base)
    assert adjusted.scorecard.volatility_stress == pytest.approx(base.scorecard.volatility_stress)
