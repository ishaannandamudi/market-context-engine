"""Point-in-time panel, rolling context, projection and walk-forward invariants."""

from dataclasses import replace

import pytest

from mce.calendar.service import MarketSessionService
from mce.config.settings import FactorSettings, MCEConfig, StatisticalSplitSettings
from mce.features.breadth.membership import MembershipTimeline, UniverseMembership
from mce.features.factors.engine import FactorFeatureEngine
from mce.features.factors.exposures import compare_factor_contexts, compare_with_latent_states
from mce.features.factors.models import ReturnPanel
from mce.features.factors.panel import FactorPanelBuilder
from mce.features.factors.rolling import factor_history, walk_forward_factors
from mce.features.factors.snapshot import apply_factors_to_snapshot
from mce.features.sectors.membership import SectorMembershipTimeline
from mce.models.context import MarketContextSnapshot
from mce.models.enums import DataQuality, MarketRegime, QualityFlag
from mce.models.market_data import BarObservation
from mce.reproducibility import config_fingerprint
from mce.statistical.models import LatentStateInference
from tests.fixtures.factor_scenarios import factor_config, factor_panel, factor_store


def test_daily_panel_and_future_bars_are_point_in_time() -> None:
    config = factor_config(window=20)
    store, timestamps = factor_store(rows=22)
    builder = FactorPanelBuilder(config)
    past = builder.build(store, timestamps[-2])
    assert past.symbols == ("AAA", "BBB", "CCC", "DDD")
    assert len(past.returns) == 20 and past.timestamps_ns[-1] == timestamps[-2]
    assert past.row_available_at_ns == past.timestamps_ns
    assert past.symbol_coverage == past.time_coverage == 1
    truncated, _ = factor_store(rows=21)
    assert builder.build(truncated, timestamps[-2]) == past
    assert builder.build(store, timestamps[-1]).fingerprint != past.fingerprint
    with pytest.raises(ValueError, match="future"):
        replace(past, as_of_ns=past.timestamps_ns[-1] - 1)


def test_future_membership_and_partial_stale_universe() -> None:
    config = factor_config(window=20)
    store, timestamps = factor_store(rows=21)
    as_of = timestamps[-1]
    static = FactorPanelBuilder(config).build(store, as_of)
    records = tuple(
        UniverseMembership(symbol=symbol) for symbol in reversed(static.expected_symbols)
    )
    future = UniverseMembership(
        symbol="EEE",
        effective_from_ns=as_of + 1,
        available_at_ns=as_of + 1,
    )
    timeline = MembershipTimeline(records + (future,))
    assert FactorPanelBuilder(config, timeline).build(store, as_of) == static
    missing_store, _ = factor_store(rows=21, missing=frozenset({("AAA", 20)}))
    partial = FactorPanelBuilder(config).build(missing_store, as_of)
    assert "AAA" in partial.stale_symbols and "AAA" not in partial.symbols
    assert all(len(row) == 3 for row in partial.returns)
    context = FactorFeatureEngine(config).compute(missing_store, as_of)
    assert context.artifact is None and QualityFlag.STALE_DATA in context.quality_flags
    assert QualityFlag.LOW_COVERAGE in context.quality_flags
    projected = apply_factors_to_snapshot(context)
    assert projected.feature_values is not None
    assert "factor.first.explained_variance" not in projected.feature_values
    previous = MarketContextSnapshot(
        as_of_ns=as_of,
        feature_values={"factor.first.explained_variance": 0.9, "trend.custom": 1.0},
    )
    replaced = apply_factors_to_snapshot(context, previous)
    assert replaced.feature_values is not None
    assert "factor.first.explained_variance" not in replaced.feature_values
    assert replaced.feature_values["trend.custom"] == 1.0


def test_factor_context_sector_exposure_alignment_and_snapshot() -> None:
    config = factor_config(window=20)
    store, timestamps = factor_store(rows=22)
    sectors = SectorMembershipTimeline.from_groups(
        {
            "tech": ("AAA", "BBB"),
            "other": ("CCC", "DDD"),
        }
    )
    engine = FactorFeatureEngine(config, sector_provider=sectors)
    first, second = factor_history(engine, store, (timestamps[-2], timestamps[-1]))
    assert first.causal and second.causal
    assert first.data_quality in (DataQuality.GOOD, DataQuality.DEGRADED)
    assert first.artifact is not None and second.artifact is not None
    assert first.artifact.fit_end_ns == timestamps[-2]
    assert first.first_share is not None and first.first_share > 0.99
    assert second.alignment is not None and second.alignment.mean_similarity > 0.9
    assert len(second.sector_exposures) == 2
    assert second.decomposition is not None
    assert second.decomposition.residual_variance < second.decomposition.total_variance
    original = MarketContextSnapshot(as_of_ns=second.as_of_ns, market_regime=MarketRegime.RISK_ON)
    projected = apply_factors_to_snapshot(second, original)
    assert projected.market_regime == MarketRegime.RISK_ON
    assert projected.schema_version == "1.0" and projected.engine_version == "0.17.0"
    assert projected.feature_values is not None
    assert projected.feature_values["factor.first.explained_variance"] == second.first_share
    assert (
        projected.feature_values["factor.residual_variance"]
        == second.decomposition.residual_variance
    )
    assert "factor.sector.tech.loading.1" in projected.feature_values
    assert projected.feature_values["factor.loading_stability"] == second.alignment.mean_similarity
    with pytest.raises(ValueError, match="clocks"):
        apply_factors_to_snapshot(second, MarketContextSnapshot(as_of_ns=second.as_of_ns + 1))


def test_rolling_future_fit_cannot_rewrite_past_and_context_comparison_is_separate() -> None:
    config = factor_config(window=20)
    store, times = factor_store(rows=23)
    engine = FactorFeatureEngine(config)
    before = factor_history(engine, store, (times[-3], times[-2]))
    after = factor_history(engine, store, (times[-3], times[-2], times[-1]))
    assert before == after[:2]
    assert before[0].artifact is not None and after[2].artifact is not None
    assert before[0].artifact.fingerprint != after[2].artifact.fingerprint
    grouped = compare_factor_contexts(before, {times[-3]: "calm", times[-2]: "stress"})
    assert tuple(item.label for item in grouped) == ("calm", "stress")
    assert compare_factor_contexts(before, {times[-3]: 0, times[-2]: 1})[0].observations == 1
    inference = LatentStateInference(
        as_of_ns=times[-3],
        state_id=0,
        state_probabilities=(0.8, 0.2),
        mode="filtered",
        retrospective=False,
        training_end_ns=times[-4],
        model_version="diagonal-gaussian-hmm/1",
        feature_schema_version="1.0",
    )
    assert compare_with_latent_states(before, (inference,))[0].label == "0"
    with pytest.raises(ValueError, match="causal filtered"):
        compare_with_latent_states(
            before, (replace(inference, mode="smoothed", retrospective=True),)
        )
    with pytest.raises(ValueError, match="no factor context"):
        compare_with_latent_states(before, (replace(inference, as_of_ns=times[-1]),))
    assert before == after[:2]  # later labels and factor refits do not mutate contexts
    with pytest.raises(ValueError, match="precede"):
        engine.compute(store, times[-3], previous=before[1])


def test_walk_forward_frozen_artifact_and_future_append() -> None:
    full = factor_panel("dominant", 60)
    config = factor_config(window=60)
    factor = config.features.factor.model_copy(update={"minimum_samples": 10})
    config = config.model_copy(
        update={"features": config.features.model_copy(update={"factor": factor})}
    )
    calendar = MarketSessionService(config.market)
    split = StatisticalSplitSettings(
        train_sessions=20,
        validation_sessions=5,
        test_sessions=5,
        embargo_sessions=1,
        step_sessions=10,
    )
    early = walk_forward_factors(full.take(tuple(range(50))), config, calendar, split)
    complete = walk_forward_factors(full, config, calendar, split)
    assert len(complete) > len(early)
    assert early[0] == complete[0]
    assert early[0].artifact.fit_end_ns == early[0].train_end_ns
    assert early[0].validation_scores and early[0].test_scores
    assert complete[1].train_end_ns > complete[0].train_end_ns
    assert complete[1].artifact.fingerprint != complete[0].artifact.fingerprint


def test_late_bar_revision_cannot_enter_historical_walk_forward_fit() -> None:
    config = factor_config(window=20)
    store, timestamps = factor_store(rows=21)
    calendar = MarketSessionService(config.market)
    day = calendar.session_for_timestamp(timestamps[5]).trading_date
    assert day is not None
    opened, closed = calendar.regular_session_bounds(day)
    baseline = FactorPanelBuilder(config).build(store, timestamps[7])
    store.add(
        BarObservation(
            symbol="AAA",
            source="synthetic",
            event_timestamp_ns=closed,
            effective_timestamp_ns=timestamps[10],
            bar_start_ns=opened,
            bar_end_ns=closed,
            open=100,
            high=101,
            low=100,
            close=101,
            volume=1,
        )
    )
    assert FactorPanelBuilder(config).build(store, timestamps[7]) == baseline
    revised = FactorPanelBuilder(config).build(store, timestamps[-1])
    assert any(
        available > timestamp
        for timestamp, available in zip(
            revised.timestamps_ns, revised.row_available_at_ns, strict=True
        )
    )
    split = StatisticalSplitSettings(
        train_sessions=10, validation_sessions=2, test_sessions=2, embargo_sessions=0
    )
    with pytest.raises(ValueError, match="late historical revisions"):
        walk_forward_factors(revised, config, calendar, split)


def test_weak_identification_constant_data_and_config_fingerprint() -> None:
    config = factor_config(window=20)
    panel = factor_panel("dominant", 20)
    constant = replace(panel, returns=tuple((0.01,) * 4 for _ in panel.returns))
    result = FactorFeatureEngine(config).from_panel(constant)
    assert result.artifact is None and result.data_quality == DataQuality.POOR
    assert QualityFlag.FACTOR_IDENTIFICATION_WEAK in result.quality_flags
    changed = config.model_copy(
        update={
            "features": config.features.model_copy(
                update={
                    "factor": config.features.factor.model_copy(
                        update={"matrix_type": "covariance"}
                    )
                }
            )
        }
    )
    assert config_fingerprint(config) != config_fingerprint(changed)
    weak_panel = ReturnPanel(
        4,
        (1, 2, 3, 4),
        ("AAA", "BBB"),
        ("AAA", "BBB"),
        ((1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)),
        4,
    )
    weak_settings = FactorSettings(
        window_sessions=5,
        minimum_samples=3,
        minimum_symbols=2,
        minimum_time_coverage=0.8,
        factor_count=2,
        matrix_type="covariance",
    )
    weak_config = MCEConfig(_env_file=None)
    weak_config = weak_config.model_copy(
        update={"features": weak_config.features.model_copy(update={"factor": weak_settings})}
    )
    weak = FactorFeatureEngine(weak_config).from_panel(weak_panel)
    assert weak.artifact is not None and weak.artifact.near_degenerate
    assert QualityFlag.FACTOR_IDENTIFICATION_WEAK in weak.quality_flags
