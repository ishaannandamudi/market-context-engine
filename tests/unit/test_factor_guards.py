"""Factor input, artifact, coverage and provider-quality boundary tests."""

import json
from dataclasses import replace

import numpy as np
import pytest
from pydantic import ValidationError

from mce.config.settings import FactorSettings
from mce.features.breadth.membership import MembershipTimeline, UniverseMembership
from mce.features.factors.dynamics import _assignment, factor_shock_and_volatility
from mce.features.factors.engine import FactorFeatureEngine
from mce.features.factors.models import FactorArtifact
from mce.features.factors.panel import FactorPanelBuilder
from mce.features.factors.pca import fit_pca
from mce.features.factors.rolling import factor_history
from mce.features.sectors.membership import SectorMembership, SectorMembershipTimeline
from mce.models.enums import DataQuality, QualityFlag
from tests.fixtures.factor_scenarios import factor_config, factor_panel, factor_store


def test_panel_validation_and_stale_source_quality() -> None:
    with pytest.raises(ValidationError, match="unique"):
        FactorSettings(universe=("AAA", "AAA"))
    with pytest.raises(ValidationError, match="exceed"):
        FactorSettings(window_sessions=5, minimum_samples=6)
    panel = factor_panel("dominant", 20)
    with pytest.raises(ValueError, match="clock"):
        replace(panel, as_of_ns=-1)
    with pytest.raises(ValueError, match="sorted"):
        replace(panel, expected_symbols=("BBB", "AAA", "CCC", "DDD"))
    with pytest.raises(ValueError, match="sorted members"):
        replace(panel, symbols=("AAA", "BBB", "CCC", "EEE"))
    with pytest.raises(ValueError, match="timestamps differ"):
        replace(panel, timestamps_ns=panel.timestamps_ns[:-1])
    with pytest.raises(ValueError, match="increase"):
        replace(
            panel,
            timestamps_ns=(panel.timestamps_ns[1], panel.timestamps_ns[0])
            + panel.timestamps_ns[2:],
        )
    with pytest.raises(ValueError, match="finite aligned"):
        replace(panel, returns=((float("nan"),) + panel.returns[0][1:],) + panel.returns[1:])
    with pytest.raises(ValueError, match="selection"):
        panel.take(())
    config = factor_config(window=20)
    store, times = factor_store(rows=21)
    records = tuple(UniverseMembership(symbol=symbol) for symbol in panel.symbols)
    degraded = MembershipTimeline(records, source_quality=DataQuality.UNUSABLE)
    result = FactorFeatureEngine(config, universe_provider=degraded).compute(store, times[-1])
    assert result.artifact is None and result.data_quality == DataQuality.UNUSABLE
    assert QualityFlag.PROVIDER_DEGRADED in result.quality_flags


def test_historical_membership_and_missing_history_are_excluded() -> None:
    config = factor_config(window=20)
    store, times = factor_store(rows=21)
    records = (UniverseMembership(symbol="AAA", effective_from_ns=times[10]),) + tuple(
        UniverseMembership(symbol=symbol) for symbol in ("BBB", "CCC", "DDD")
    )
    joined = FactorPanelBuilder(config, MembershipTimeline(records)).build(store, times[-1])
    assert "AAA" in joined.missing_symbols and "AAA" not in joined.symbols
    assert len(joined.returns) == 20
    missing_store, _ = factor_store(
        rows=21, missing=frozenset(("AAA", index) for index in range(5, 10))
    )
    short_history = FactorPanelBuilder(config).build(missing_store, times[-1])
    assert "AAA" in short_history.missing_symbols and "AAA" not in short_history.symbols
    before_close = FactorPanelBuilder(config).build(store, times[-1] - 1)
    assert before_close.timestamps_ns[-1] == times[-2]


def test_withheld_context_for_time_coverage_and_disabled_model() -> None:
    config = factor_config(window=20)
    panel = factor_panel("dominant", 20)
    short = replace(panel, timestamps_ns=panel.timestamps_ns[-10:], returns=panel.returns[-10:])
    withheld = FactorFeatureEngine(config).from_panel(short)
    assert withheld.artifact is None
    assert QualityFlag.INSUFFICIENT_WINDOW_COVERAGE in withheld.quality_flags
    disabled_factor = config.features.factor.model_copy(update={"enabled": False})
    disabled = config.model_copy(
        update={"features": config.features.model_copy(update={"factor": disabled_factor})}
    )
    assert FactorFeatureEngine(disabled).from_panel(panel).artifact is None


def test_artifact_corruption_and_settings_identity() -> None:
    panel = factor_panel("blocks", 40)
    settings = FactorSettings(
        window_sessions=40,
        minimum_samples=5,
        minimum_symbols=2,
        minimum_symbol_coverage=1,
        minimum_time_coverage=1,
    )
    artifact, _ = fit_pca(panel, settings, config_fingerprint="a" * 64)
    with pytest.raises(ValueError, match="settings mismatch"):
        FactorArtifact.from_json(artifact.serialize(), settings_fingerprint="b" * 64)
    mutations = (
        ("model_version", "bad", "model"),
        ("input_schema_version", "return-panel/2", "model"),
        ("symbols", ["AAA"], "symbols"),
        ("fit_start_ns", artifact.fit_end_ns + 1, "interval"),
        ("means", [0.0], "dimensions"),
        ("eigenvalues", [1.0], "spectrum dimensions"),
        ("scales", [0.0] * len(artifact.symbols), "scales"),
        ("explained_variance", [0.5] * len(artifact.symbols), "sum to one"),
        ("settings_fingerprint", "bad", "fingerprint"),
    )
    for key, value, reason in mutations:
        data = json.loads(artifact.serialize())
        data[key] = value
        with pytest.raises(ValueError, match=reason):
            FactorArtifact.from_json(json.dumps(data))
    with pytest.raises(ValueError, match="symbol schema"):
        FactorArtifact.from_json(artifact.serialize(), expected_symbols=("AAA", "BBB"))
    with pytest.raises(ValueError, match="configuration mismatch"):
        FactorArtifact.from_json(artifact.serialize(), config_fingerprint="b" * 64)
    data = json.loads(artifact.serialize())
    data["loadings"][0][0] *= -1
    with pytest.raises(ValueError, match="orthonormal|canonical"):
        FactorArtifact.from_json(json.dumps(data))


def test_alignment_and_history_reject_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="square"):
        _assignment(np.ones((2, 3)))
    with pytest.raises(ValueError, match="finite"):
        factor_shock_and_volatility(np.array([[float("nan")]]), 3)
    config = factor_config(window=20)
    store, times = factor_store(rows=22)
    engine = FactorFeatureEngine(config)
    with pytest.raises(ValueError, match="increase"):
        factor_history(engine, store, (times[-1], times[-2]))
    first = engine.compute(store, times[-2])
    assert first.artifact is not None
    with pytest.raises(ValueError, match="precede"):
        engine.from_panel(factor_panel("dominant", 20), previous=first)


def test_sector_membership_changes_causally() -> None:
    config = factor_config(window=20)
    store, times = factor_store(rows=21)
    baseline = SectorMembershipTimeline.from_groups(
        {"tech": ("AAA", "BBB"), "other": ("CCC", "DDD")}
    )
    change = times[-1] + 1
    timeline = SectorMembershipTimeline(
        (
            SectorMembership(symbol="AAA", sector="tech", effective_to_ns=change),
            SectorMembership(
                symbol="AAA", sector="other", effective_from_ns=change, available_at_ns=change
            ),
            SectorMembership(symbol="BBB", sector="tech"),
            SectorMembership(symbol="CCC", sector="other"),
            SectorMembership(symbol="DDD", sector="other"),
        )
    )
    ordinary = FactorFeatureEngine(config, sector_provider=baseline).compute(store, times[-1])
    future_change = FactorFeatureEngine(config, sector_provider=timeline).compute(store, times[-1])
    assert ordinary.sector_exposures == future_change.sector_exposures
