"""Causal input, quality, rolling dynamics, residual and snapshot tests."""

from dataclasses import replace

import numpy as np
import pytest
from pydantic import ValidationError

from mce.config.settings import FactorSettings, MCEConfig, NetworkSettings
from mce.features.breadth.membership import MembershipTimeline, UniverseMembership
from mce.features.correlation.models import CorrelationFeatureSet
from mce.features.factors.engine import FactorFeatureEngine
from mce.features.factors.models import ReturnPanel
from mce.features.network.dynamics import compare_networks
from mce.features.network.engine import (
    NetworkFeatureEngine,
    dependence_from_correlation,
    network_history,
    residual_dependence,
)
from mce.features.network.models import DependenceInput
from mce.features.network.snapshot import apply_network_to_snapshot
from mce.features.sectors.membership import SectorMembership, SectorMembershipTimeline
from mce.models.context import MarketContextSnapshot
from mce.models.enums import DataQuality, MarketRegime, QualityFlag
from mce.reproducibility import config_fingerprint
from tests.fixtures.network_scenarios import blocks, dependence, equicorrelation, star


def _config(**network: object) -> MCEConfig:
    base = MCEConfig(_env_file=None)
    settings = NetworkSettings(**network)
    return base.model_copy(
        update={"features": base.features.model_copy(update={"network": settings})}
    )


def _correlation(data: DependenceInput) -> CorrelationFeatureSet:
    return CorrelationFeatureSet(
        as_of_ns=data.as_of_ns,
        expected_symbols=data.expected_symbols,
        valid_symbols=data.valid_symbols,
        expected_symbol_count=len(data.expected_symbols),
        valid_symbol_count=len(data.valid_symbols),
        symbol_coverage=data.coverage,
        expected_observations=40,
        aligned_sample_count=40,
        time_coverage=1,
        matrix_dimension=len(data.valid_symbols),
        correlation_matrix=data.matrix,
        shrunk_matrix=data.matrix,
        evidence_quality=1,
        data_quality=data.data_quality,
    )


def test_typed_correlation_adapter_config_fingerprint_and_snapshot() -> None:
    source = dependence(blocks())
    correlation = _correlation(source)
    assert dependence_from_correlation(correlation).matrix == source.matrix
    assert dependence_from_correlation(correlation, source="shrunk").source == "shrunk"
    with pytest.raises(ValueError, match="source"):
        dependence_from_correlation(correlation, source="unknown")  # type: ignore[arg-type]
    config = _config(correlation_threshold=0.6)
    result = NetworkFeatureEngine(config).compute(correlation)
    assert result.graph is not None and result.topology is not None
    assert result.topology.largest_component_fraction == 0.5
    assert QualityFlag.NETWORK_DISCONNECTED in result.quality_flags
    assert result.config_fingerprint == config_fingerprint(config)
    assert result.fingerprint == NetworkFeatureEngine(config).compute(correlation).fingerprint
    snapshot = MarketContextSnapshot(
        as_of_ns=source.as_of_ns,
        market_regime=MarketRegime.RISK_OFF,
        feature_values={"trend.custom": 0.3, "network.density": 0.99},
    )
    projected = apply_network_to_snapshot(result, snapshot)
    assert projected.market_regime == MarketRegime.RISK_OFF
    assert projected.schema_version == "1.0" and projected.engine_version == "0.17.0"
    assert projected.feature_values is not None
    assert projected.feature_values["network.density"] == pytest.approx(1 / 3)
    assert projected.feature_values["trend.custom"] == 0.3
    with pytest.raises(ValueError, match="clocks"):
        apply_network_to_snapshot(result, MarketContextSnapshot(as_of_ns=101))
    changed = _config(correlation_threshold=0.8)
    assert config_fingerprint(config) != config_fingerprint(changed)
    shrunk = _correlation(dependence(equicorrelation()))
    alternate = correlation.model_copy(update={"shrunk_matrix": shrunk.correlation_matrix})
    shrunk_result = NetworkFeatureEngine(_config(source_matrix="shrunk")).compute(alternate)
    assert shrunk_result.topology is not None and shrunk_result.topology.density == 1
    mst_only = NetworkFeatureEngine(_config(graph_mode="mst")).compute(correlation)
    assert mst_only.graph is not None and mst_only.connectivity_score is None


def test_coverage_provider_quality_disabled_and_stale_projection() -> None:
    source = dependence(equicorrelation(), expected=("AAA", "BBB", "CCC", "DDD", "EEE"))
    engine = NetworkFeatureEngine(_config(minimum_coverage=0.8))
    partial = engine.from_input(source)
    assert partial.graph is not None and partial.coverage == 0.8
    assert QualityFlag.PARTIAL_UNIVERSE in partial.quality_flags
    low = engine.from_input(replace(source, expected_symbols=source.expected_symbols + ("FFF",)))
    assert low.graph is None
    assert QualityFlag.INSUFFICIENT_NETWORK_COVERAGE in low.quality_flags
    absent = engine.from_input(replace(source, matrix=None))
    assert absent.graph is None and absent.valid_nodes == 0
    old = apply_network_to_snapshot(partial)
    withheld = apply_network_to_snapshot(low, old)
    assert withheld.feature_values is not None
    assert "network.density" not in withheld.feature_values
    poor = engine.from_input(replace(source, data_quality=DataQuality.UNUSABLE))
    assert poor.graph is None and poor.data_quality == DataQuality.UNUSABLE
    degraded = engine.from_input(replace(source, data_quality=DataQuality.DEGRADED))
    assert degraded.graph is not None and degraded.data_quality == DataQuality.DEGRADED
    assert QualityFlag.PROVIDER_DEGRADED in degraded.quality_flags
    disabled = NetworkFeatureEngine(_config(enabled=False)).from_input(source)
    assert disabled.graph is None
    with pytest.raises(ValidationError, match="less than or equal"):
        NetworkSettings(correlation_threshold=2)


def test_point_in_time_membership_sector_and_future_matrices() -> None:
    members = MembershipTimeline(
        tuple(UniverseMembership(symbol=symbol) for symbol in ("AAA", "BBB", "CCC"))
        + (UniverseMembership(symbol="DDD", effective_from_ns=200, available_at_ns=200),)
    )
    sectors = SectorMembershipTimeline(
        (
            SectorMembership(symbol="AAA", sector="tech", effective_to_ns=200),
            SectorMembership(
                symbol="AAA", sector="banks", effective_from_ns=200, available_at_ns=200
            ),
            SectorMembership(symbol="BBB", sector="tech"),
            SectorMembership(symbol="CCC", sector="banks"),
            SectorMembership(
                symbol="DDD", sector="banks", effective_from_ns=200, available_at_ns=200
            ),
        )
    )
    early_symbols = tuple(member.symbol for member in members.members(100).members)
    late_symbols = tuple(member.symbol for member in members.members(200).members)
    assert early_symbols == ("AAA", "BBB", "CCC") and late_symbols == ("AAA", "BBB", "CCC", "DDD")
    engine = NetworkFeatureEngine(_config(correlation_threshold=0.6), sectors)
    early = dependence(equicorrelation(3), as_of_ns=100, symbols=early_symbols)
    before = engine.from_input(early)
    assert before.graph is not None
    assert tuple(node.sector for node in before.graph.nodes) == ("tech", "tech", "banks")
    late = dependence(blocks(), as_of_ns=200, symbols=late_symbols)
    after = engine.from_input(late, previous=before)
    assert after.graph is not None
    assert tuple(node.sector for node in after.graph.nodes) == ("banks", "tech", "banks", "banks")
    assert engine.from_input(early) == before
    assert before.fingerprint == engine.from_input(early).fingerprint
    assert before.graph.nodes == engine.from_input(early).graph.nodes  # type: ignore[union-attr]
    assert after.dynamics is not None
    with pytest.raises(ValueError, match="precede"):
        engine.from_input(early, previous=after)


def test_network_dynamics_persistence_rewire_hub_shift_merge_split() -> None:
    engine = NetworkFeatureEngine(_config(correlation_threshold=0.6, community_threshold=0.6))
    first = engine.from_input(dependence(blocks(), as_of_ns=100))
    same = engine.from_input(dependence(blocks(), as_of_ns=200), previous=first)
    assert same.dynamics is not None
    assert same.dynamics.edge_persistence == 1
    assert same.dynamics.edge_turnover == same.dynamics.rewiring == 0
    merged = engine.from_input(dependence(equicorrelation(), as_of_ns=300), previous=same)
    assert merged.dynamics is not None
    assert merged.dynamics.edge_persistence == pytest.approx(1 / 3)
    assert merged.dynamics.community_stability == 0
    assert merged.dynamics.component_count_change == -1
    assert merged.dynamics.rewiring > 0
    split = engine.from_input(dependence(blocks(), as_of_ns=400), previous=merged)
    assert split.dynamics is not None and split.dynamics.component_count_change == 1
    shifted = engine.from_input(dependence(star(center=1), as_of_ns=500), previous=split)
    assert shifted.dynamics is not None and shifted.dynamics.edge_turnover > 0
    first_hub = engine.from_input(dependence(star(center=0), as_of_ns=600))
    second_hub = engine.from_input(dependence(star(center=1), as_of_ns=700), previous=first_hub)
    assert second_hub.dynamics is not None
    assert second_hub.dynamics.centrality_drift > 0
    assert second_hub.dynamics.node_centrality_changes[0][0] == "AAA"
    assert second_hub.dynamics.node_centrality_changes[0][1] < 0
    assert second_hub.dynamics.mst_edge_turnover is not None
    assert engine.from_input(dependence(blocks(), as_of_ns=100)) == first
    with pytest.raises(ValueError, match="precede"):
        compare_networks(first, same)


def test_history_future_append_and_incompatible_comparison() -> None:
    base = dependence(blocks())
    records = tuple(_correlation(replace(base, as_of_ns=clock)) for clock in (100, 200, 300))
    engine = NetworkFeatureEngine(_config(correlation_threshold=0.6))
    early = network_history(engine, records[:2])
    full = network_history(engine, records)
    assert early == full[:2]
    assert full[-1].dynamics is not None
    with pytest.raises(ValueError, match="increase"):
        network_history(engine, records[::-1])
    changed_engine = NetworkFeatureEngine(_config(correlation_threshold=0.8))
    changed = changed_engine.compute(records[-1])
    assert compare_networks(changed, early[-1]) is None


def test_factor_residual_network_is_separate_and_causal() -> None:
    rng = np.random.default_rng(15)
    rows, size = 120, 8
    common = rng.normal(size=(rows, 1))
    returns = 0.02 * common + rng.normal(scale=0.004, size=(rows, size))
    symbols = tuple(f"S{index:02d}" for index in range(size))
    panel = ReturnPanel(
        rows,
        tuple(range(1, rows + 1)),
        symbols,
        symbols,
        tuple(tuple(float(value) for value in row) for row in returns),
        rows,
    )
    base = _config(correlation_threshold=0.6)
    factor = FactorSettings(
        window_sessions=rows,
        minimum_samples=20,
        minimum_symbols=3,
        factor_count=1,
    )
    config = base.model_copy(
        update={"features": base.features.model_copy(update={"factor": factor})}
    )
    fitted = FactorFeatureEngine(config).from_panel(panel)
    residual = residual_dependence(panel, fitted)
    assert residual.matrix is not None and residual.source == "factor_residual"
    matrix = np.corrcoef(returns, rowvar=False)
    raw = dependence(matrix, as_of_ns=rows, symbols=symbols)
    result = NetworkFeatureEngine(config).from_input(raw, residual=residual)
    assert result.topology is not None and result.topology.density == 1
    assert result.residual_density is not None and result.residual_density < 0.2
    assert result.raw_minus_residual_density is not None
    assert result.raw_minus_residual_density > 0.8
    future_panel = replace(panel, as_of_ns=rows + 1)
    with pytest.raises(ValueError, match="same-clock"):
        residual_dependence(future_panel, fitted)
    with pytest.raises(ValueError, match="current clock"):
        NetworkFeatureEngine(config).from_input(raw, residual=replace(residual, as_of_ns=rows + 1))
    unavailable = NetworkFeatureEngine(config).from_input(
        raw, residual=replace(residual, matrix=None, data_quality=DataQuality.POOR)
    )
    assert unavailable.residual_density is None
    assert QualityFlag.NETWORK_TOPOLOGY_DEGRADED in unavailable.quality_flags
