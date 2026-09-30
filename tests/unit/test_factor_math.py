"""Independent PCA, sign, residual, assignment and artifact references."""

import json
from dataclasses import replace
from math import sqrt
from types import SimpleNamespace

import numpy as np
import pytest

from mce.config.settings import FactorSettings
from mce.features.factors.dynamics import (
    align_factors,
    factor_shock_and_volatility,
    spectrum_diagnostics,
)
from mce.features.factors.models import FactorArtifact, ReturnPanel
from mce.features.factors.pca import canonical_signs, fit_pca, pca_spectrum, transform
from mce.features.factors.residuals import decompose, reconstruct
from tests.fixtures.factor_scenarios import factor_panel


def _settings(**updates: object) -> FactorSettings:
    return FactorSettings(
        window_sessions=40,
        minimum_samples=5,
        minimum_symbols=2,
        minimum_symbol_coverage=1,
        minimum_time_coverage=1,
        factor_count=3,
        **updates,
    )


def test_independent_two_by_two_covariance_and_identity_reference() -> None:
    x = np.array(
        [
            [sqrt(3), sqrt(3) / 2],
            [-sqrt(3), -sqrt(3) / 2],
            [0, 1.5],
            [0, -1.5],
        ]
    )
    values, vectors, method = pca_spectrum(x)
    assert method == "eigh"
    assert values == pytest.approx([3, 1])
    assert vectors[:, 0] == pytest.approx([sqrt(0.5), sqrt(0.5)])
    identity = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]], dtype=float)
    equal, _, _ = pca_spectrum(identity)
    assert equal == pytest.approx([2 / 3, 2 / 3])
    first, _, effective, _ = spectrum_diagnostics(tuple(equal))
    assert first == pytest.approx(0.5) and effective == pytest.approx(2)


def test_perfect_common_factor_sign_and_residual_identity() -> None:
    panel = factor_panel("dominant", 40)
    artifact, scores = fit_pca(panel, _settings(), config_fingerprint="a" * 64)
    assert artifact.explained_variance[0] > 0.99
    assert all(row[0] > 0 for row in artifact.loadings)
    assert artifact.method == "eigh"
    residual = decompose(panel, artifact, scores)
    assert residual.mean_common_share is not None and residual.mean_common_share > 0.99
    assert residual.total_variance == pytest.approx(
        residual.explained_variance + residual.residual_variance, abs=1e-12
    )
    assert np.asarray(reconstruct(artifact, scores)) + (
        np.asarray(panel.returns) - reconstruct(artifact, scores)
    ) == pytest.approx(np.asarray(panel.returns))
    assert transform(artifact, panel) == pytest.approx(scores)
    assert canonical_signs(-np.asarray(artifact.loadings)) == pytest.approx(
        np.asarray(artifact.loadings)
    )


def test_block_independent_and_high_dimensional_svd() -> None:
    block = factor_panel("blocks", 40)
    independent = factor_panel("independent", 40)
    block_artifact, _ = fit_pca(block, _settings(), config_fingerprint="a" * 64)
    independent_artifact, _ = fit_pca(independent, _settings(), config_fingerprint="a" * 64)
    assert sum(block_artifact.explained_variance[:2]) > 0.98
    assert independent_artifact.explained_variance[0] < block_artifact.explained_variance[0]
    x = np.random.default_rng(2).normal(size=(12, 30))
    values, vectors, method = pca_spectrum(x - x.mean(axis=0))
    assert method == "svd" and len(values) == 30 and vectors.shape == (30, 12)
    assert np.all(values >= 0)
    reference = np.linalg.eigvalsh(np.cov(x, rowvar=False))[::-1]
    assert values[:12] == pytest.approx(np.clip(reference[:12], 0, None))
    symbols = tuple(f"S{index:03d}" for index in range(30))
    wide = ReturnPanel(
        12,
        tuple(range(1, 13)),
        symbols,
        symbols,
        tuple(tuple(float(value) for value in row) for row in x),
        12,
    )
    wide_settings = FactorSettings(
        window_sessions=12, minimum_samples=5, minimum_symbols=2, factor_count=5
    )
    wide_artifact, wide_scores = fit_pca(wide, wide_settings, config_fingerprint="a" * 64)
    assert wide_artifact.method == "svd" and wide_scores.shape == (12, 5)
    assert sum(wide_artifact.explained_variance) == pytest.approx(1)


def test_covariance_and_correlation_pca_have_distinct_scale_semantics() -> None:
    panel = ReturnPanel(
        4,
        (1, 2, 3, 4),
        ("AAA", "BBB"),
        ("AAA", "BBB"),
        ((10.0, 1.0), (-10.0, 1.0), (10.0, -1.0), (-10.0, -1.0)),
        4,
    )
    base = FactorSettings(
        window_sessions=5,
        minimum_samples=3,
        minimum_symbols=2,
        minimum_time_coverage=0.8,
        factor_count=1,
    )
    covariance, _ = fit_pca(
        panel, base.model_copy(update={"matrix_type": "covariance"}), config_fingerprint="a" * 64
    )
    correlation, _ = fit_pca(panel, base, config_fingerprint="a" * 64)
    assert covariance.explained_variance[0] > 0.98
    assert correlation.explained_variance[0] == pytest.approx(0.5)
    assert correlation.near_degenerate


def test_threshold_factor_count_and_degenerate_eigenvalues() -> None:
    panel = factor_panel("blocks", 40)
    settings = _settings(variance_threshold=0.9)
    artifact, _ = fit_pca(panel, settings, config_fingerprint="a" * 64)
    assert 1 <= artifact.selected_count <= 3
    assert sum(artifact.explained_variance[: artifact.selected_count]) >= 0.9
    identity_panel = ReturnPanel(
        4,
        (1, 2, 3, 4),
        ("AAA", "BBB"),
        ("AAA", "BBB"),
        ((1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)),
        4,
    )
    identity_settings = FactorSettings(
        window_sessions=5,
        minimum_samples=3,
        minimum_symbols=2,
        minimum_symbol_coverage=1,
        minimum_time_coverage=0.8,
        factor_count=2,
        matrix_type="covariance",
    )
    identity_artifact, _ = fit_pca(identity_panel, identity_settings, config_fingerprint="a" * 64)
    assert identity_artifact.near_degenerate
    assert identity_artifact.explained_variance == pytest.approx((0.5, 0.5))


def test_alignment_handles_sign_swap_drift_and_subspace() -> None:
    symbols = ("AAA", "BBB", "CCC")
    previous = SimpleNamespace(
        symbols=symbols, selected_count=2, loadings=((1.0, 0.0), (0.0, 1.0), (0.0, 0.0))
    )
    current = SimpleNamespace(
        symbols=symbols, selected_count=2, loadings=((0.0, -1.0), (1.0, 0.0), (0.0, 0.0))
    )
    aligned, shared, drift = align_factors(current, previous)
    assert aligned is not None
    assert aligned.current_to_previous == (1, 0)
    assert aligned.signs == (1, -1)
    assert aligned.similarities == pytest.approx((1, 1))
    assert aligned.subspace_similarity == pytest.approx(1)
    assert shared == symbols and np.asarray(drift) == pytest.approx(np.zeros((2, 3)))
    assert align_factors(current, previous) == (aligned, shared, drift)
    assert (
        align_factors(
            SimpleNamespace(symbols=("X", "Y"), selected_count=2, loadings=((1, 0), (0, 1))),
            previous,
        )[0]
        is None
    )


def test_artifact_roundtrip_corruption_and_input_validation() -> None:
    panel = factor_panel("dominant", 40)
    artifact, scores = fit_pca(panel, _settings(), config_fingerprint="a" * 64)
    assert FactorArtifact.from_json(artifact.serialize(), expected_fingerprint=artifact.fingerprint)
    assert FactorArtifact.from_json(artifact.serialize()) == artifact
    data = json.loads(artifact.serialize())
    data["eigenvalues"][0] *= 0.5
    with pytest.raises(ValueError, match="eigenvalues|spectrum"):
        FactorArtifact.from_json(json.dumps(data))
    data = json.loads(artifact.serialize())
    data["loadings"][0].pop()
    with pytest.raises(ValueError, match="dimensions"):
        FactorArtifact.from_json(json.dumps(data))
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        FactorArtifact.from_json(artifact.serialize(), expected_fingerprint="b" * 64)
    with pytest.raises(ValueError, match="same ordered symbols"):
        transform(
            artifact,
            replace(
                panel,
                expected_symbols=("AAA", "BBB", "CCC", "ZZZ"),
                symbols=("AAA", "BBB", "CCC", "ZZZ"),
            ),
        )
    with pytest.raises(ValueError, match="dimensions"):
        reconstruct(artifact, scores[:, :0])
    with pytest.raises(ValueError, match="finite"):
        canonical_signs(np.array([[np.nan]]))
    with pytest.raises(ValueError, match="finite"):
        pca_spectrum(np.array([[1.0, np.inf], [2.0, 3.0]]))
    with pytest.raises(ValueError, match="threshold"):
        fit_pca(panel.take((0, 1, 2, 3)), _settings(), config_fingerprint="a" * 64)
    with pytest.raises(ValueError, match="configuration fingerprint"):
        fit_pca(panel, _settings(), config_fingerprint="bad")
    empty = ReturnPanel(1, (), panel.expected_symbols, panel.symbols, (), 1)
    with pytest.raises(ValueError, match="transform dimensions"):
        transform(artifact, empty)
    with pytest.raises(ValueError, match="aligned fitted returns"):
        decompose(panel.take((0,)), artifact, scores[:1])
    with pytest.raises(ValueError, match="reconstruction dimensions"):
        decompose(panel, artifact, scores[:-1])
    with pytest.raises(ValueError, match="identity failed"):
        decompose(panel, artifact, scores * 0.5)
    constant = replace(panel, returns=tuple(panel.returns[0] for _ in panel.returns))
    frozen_scores = transform(artifact, constant)
    decomposition = decompose(constant, artifact, frozen_scores)
    assert decomposition.total_variance == pytest.approx(0)
    assert all(value is None for value in decomposition.residual_variance_shares)


def test_shock_and_spectrum_diagnostics() -> None:
    shock, volatility = factor_shock_and_volatility(
        np.array([[0.0], [1.0], [-1.0], [0.5], [-0.5], [3.0]]), 5
    )
    assert shock is not None and shock > 3 and volatility is not None
    assert spectrum_diagnostics((4.0, 1.0, 0.0)) == pytest.approx(
        (0.8, 1.0, 25 / 17, 0.6 * 0.8 + 0.4 * (17 / 25))
    )
