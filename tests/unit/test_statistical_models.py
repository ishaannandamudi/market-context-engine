"""Independent HMM math references and MCE-13 model/causality tests."""

import json
from dataclasses import replace
from math import log, pi

import numpy as np
import pytest

from mce.calendar.service import MarketSessionService
from mce.config.settings import (
    ChangePointSettings,
    HMMSettings,
    KMeansSettings,
    MCEConfig,
    StatisticalSplitSettings,
)
from mce.reproducibility import config_fingerprint
from mce.statistical.normalization import fit_normalizer
from mce.statistical.regime.changepoint import MeanShiftDetector
from mce.statistical.regime.clustering import KMeansRegimeModel, fit_kmeans_array
from mce.statistical.regime.hmm import (
    GaussianHMMRegimeModel,
    HMMArtifact,
    _fit_restart,
    information_criteria,
    parameter_count,
)
from mce.statistical.regime.hmm_math import (
    backward_log,
    diagonal_gaussian_log_density,
    forward_log,
    logsumexp,
    posteriors,
    viterbi_path,
)
from mce.statistical.regime.selection import (
    adjusted_rand_index,
    compare_assignments,
    compare_hmm_state_counts,
    compare_kmeans_cluster_counts,
    normalized_mutual_information,
)
from mce.statistical.regime.walkforward import run_walk_forward_hmm
from tests.fixtures.statistical_scenarios import synthetic_dataset


def _fit(kind: str = "two_state", count: int = 60, k: int = 2):
    dataset, labels = synthetic_dataset(kind, count)
    normalizer = fit_normalizer(dataset, tuple(range(count)))
    settings = HMMSettings(state_count=k, max_iterations=25, restarts=2)
    model = GaussianHMMRegimeModel(settings)
    artifact = model.fit(dataset, normalizer, random_seed=13)
    return dataset, labels, normalizer, model, artifact


def test_hmm_math_against_independent_two_step_reference() -> None:
    initial = np.array([0.6, 0.4])
    transition = np.array([[0.7, 0.3], [0.4, 0.6]])
    emissions = np.log(np.array([[0.5, 0.1], [0.4, 0.8]]))
    alpha, likelihood = forward_log(emissions, initial, transition)
    assert likelihood == pytest.approx(log(0.1816), abs=1e-12)
    assert np.exp(alpha[0]) == pytest.approx([0.3, 0.04])
    beta = backward_log(emissions, transition)
    assert np.exp(beta[0]) == pytest.approx([0.52, 0.64])
    gamma, xi = posteriors(emissions, alpha, beta, transition, likelihood)
    assert gamma[0] == pytest.approx([0.156 / 0.1816, 0.0256 / 0.1816])
    assert gamma[1] == pytest.approx([0.0904 / 0.1816, 0.0912 / 0.1816])
    assert xi[0] == pytest.approx(np.array([[0.084, 0.072], [0.0064, 0.0192]]) / 0.1816)
    assert np.allclose(xi[0].sum(axis=1), gamma[0])
    assert np.allclose(xi[0].sum(axis=0), gamma[1])
    assert viterbi_path(emissions, initial, transition) == (0, 0)
    assert float(logsumexp(np.array([-1000.0, -1000.0]))) == pytest.approx(-1000 + log(2))
    assert np.isneginf(float(logsumexp(np.array([-np.inf, -np.inf]))))


def test_gaussian_log_density_and_validation() -> None:
    density = diagonal_gaussian_log_density(
        np.array([[0.0], [1.0]]), np.array([[0.0]]), np.array([[1.0]]), 1e-4
    )
    assert density[:, 0] == pytest.approx([-0.5 * log(2 * pi), -0.5 * log(2 * pi) - 0.5])
    floor = diagonal_gaussian_log_density(
        np.array([[0.0]]), np.array([[0.0]]), np.array([[1e-10]]), 1e-4
    )
    assert floor[0, 0] == pytest.approx(-0.5 * log(2 * pi * 1e-4))
    with pytest.raises(ValueError, match="dimensions"):
        diagonal_gaussian_log_density(
            np.array([[0.0]]), np.array([[0.0, 1.0]]), np.array([[1.0, 1.0]]), 1e-4
        )
    with pytest.raises(ValueError, match="parameters"):
        diagonal_gaussian_log_density(
            np.array([[0.0]]), np.array([[0.0]]), np.array([[-1.0]]), 1e-4
        )


def test_hmm_math_rejects_invalid_probabilities_and_impossible_paths() -> None:
    emissions = np.log(np.array([[0.5, 0.5], [0.5, 0.5]]))
    initial = np.array([0.5, 0.5])
    transition = np.array([[0.8, 0.2], [0.2, 0.8]])
    with pytest.raises(ValueError, match="probabilities"):
        forward_log(emissions, np.array([0.7, 0.7]), transition)
    with pytest.raises(ValueError, match="emission"):
        forward_log(np.array([[np.nan, 0.0]]), initial, transition)
    with pytest.raises(ValueError, match="emission"):
        backward_log(np.array([[np.inf, 0.0]]), transition)
    with pytest.raises(ValueError, match="shapes"):
        posteriors(emissions, emissions[:1], emissions, transition, -1.0)
    with pytest.raises(ValueError, match="emission"):
        viterbi_path(np.array([[np.nan, 0.0]]), initial, transition)
    with pytest.raises(ValueError, match="finite Viterbi"):
        viterbi_path(np.full((2, 2), -np.inf), initial, transition)
    alpha, _ = forward_log(emissions, initial, transition)
    carried, _ = forward_log(
        emissions[1:],
        initial,
        transition,
        previous_filtered=(np.exp(alpha[0] - logsumexp(alpha[0]))),
    )
    assert carried[0] == pytest.approx(alpha[1] - logsumexp(alpha[0]))


def test_hmm_two_state_em_reproducibility_and_quality() -> None:
    dataset, _, normalizer, model, artifact = _fit()
    assert artifact.restarts[artifact.selected_restart].converged
    for restart in artifact.restarts:
        if restart.error is None:
            assert all(
                b >= a - 1e-6
                for a, b in zip(
                    restart.likelihood_history, restart.likelihood_history[1:], strict=False
                )
            )
    assert np.asarray(artifact.transition_matrix).sum(axis=1) == pytest.approx([1, 1])
    assert sum(artifact.initial_probabilities) == pytest.approx(1)
    assert np.isfinite(artifact.variances).all()
    assert np.min(artifact.variances) >= artifact.covariance_floor
    raw_means = np.asarray(artifact.means) * np.asarray(normalizer.scales) + np.asarray(
        normalizer.centers
    )
    assert abs(raw_means[0, 6] - raw_means[1, 6]) > 0.3
    quality = model.quality(dataset)
    assert quality.converged and quality.minimum_occupancy > 0.1
    assert len(model.state_profiles(dataset)) == 2
    second = GaussianHMMRegimeModel(model.settings)
    assert second.fit(dataset, normalizer, random_seed=13).fingerprint == artifact.fingerprint
    assert model.serialize_artifact() == second.serialize_artifact()


def test_em_iteration_cap_reports_nonconvergence_without_hiding_progress() -> None:
    matrix = np.random.default_rng(4).normal(size=(100, 3))
    settings = HMMSettings(state_count=2, max_iterations=1, tolerance=1e-12, restarts=1)
    _, diagnostic = _fit_restart(matrix, 2, 7, settings)
    assert not diagnostic.converged and diagnostic.iterations == 1
    assert diagnostic.log_likelihood == diagnostic.likelihood_history[-1]
    assert len(diagnostic.likelihood_history) == 2
    assert diagnostic.likelihood_history[1] >= diagnostic.likelihood_history[0] - 1e-6


def test_hmm_three_state_canonicalization_and_size_report() -> None:
    dataset, _, normalizer, model, artifact = _fit("three_state", 60, 3)
    raw_means = np.asarray(artifact.means) * np.asarray(normalizer.scales) + np.asarray(
        normalizer.centers
    )
    assert np.all(np.diff(raw_means[:, 6]) > 0)
    assert set(artifact.state_mapping) == {0, 1, 2}
    report = compare_hmm_state_counts(
        dataset,
        normalizer,
        (2, 3),
        settings=HMMSettings(max_iterations=20, restarts=1),
        random_seed=3,
    )
    assert tuple(item.state_count for item in report) == (2, 3)
    assert all(item.aic < float("inf") and item.minimum_centroid_separation >= 0 for item in report)
    assert model.quality(dataset).bic == pytest.approx(report[1].bic, rel=0.2)
    with pytest.raises(ValueError, match="state counts"):
        compare_hmm_state_counts(dataset, normalizer, (3, 2))


def test_hmm_parameter_count_and_aic_bic() -> None:
    assert parameter_count(2, 1) == 7
    assert parameter_count(3, 18) == (3 - 1) + 3 * 2 + 2 * 3 * 18
    aic, bic = information_criteria(-100.0, 100, 2, 1)
    assert aic == pytest.approx(214)
    assert bic == pytest.approx(7 * log(100) + 200)
    with pytest.raises(ValueError, match="dimensions"):
        parameter_count(1, 1)


def test_filtered_future_invariance_and_retrospective_modes() -> None:
    dataset, _ = synthetic_dataset("two_state", 60)
    training = dataset.take(tuple(range(30)))
    normalizer = fit_normalizer(dataset, tuple(range(30)))
    model = GaussianHMMRegimeModel(HMMSettings(state_count=2, restarts=1, max_iterations=15))
    artifact = model.fit(training, normalizer, random_seed=5)
    first = model.infer(dataset.take((30,)), mode="filtered")
    longer = model.infer(dataset.take(tuple(range(30, 45))), mode="filtered")
    assert first[0] == longer[0]
    assert first[0].causal and first[0].model_fingerprint == artifact.fingerprint
    assert first[0].posterior_entropy is not None
    smooth = model.infer(dataset.take(tuple(range(30, 45))), mode="smoothed")
    path = model.infer(dataset.take(tuple(range(30, 45))), mode="viterbi")
    assert all(item.retrospective and not item.causal for item in smooth + path)
    assert all(item.state_probabilities is None for item in path)
    assert all(item.state_probabilities is not None for item in smooth)
    assert model.infer(training, mode="filtered")[0].retrospective
    assert model.artifact == artifact


def test_future_changes_smoothing_but_not_filtering_reference() -> None:
    initial = np.array([0.5, 0.5])
    transition = np.array([[0.95, 0.05], [0.05, 0.95]])
    early = np.log(np.array([[0.5, 0.5]]))
    future = np.log(np.array([[0.01, 0.99]]))
    first_alpha, _ = forward_log(early, initial, transition)
    full = np.vstack((early, future))
    full_alpha, full_likelihood = forward_log(full, initial, transition)
    beta = backward_log(full, transition)
    gamma, _ = posteriors(full, full_alpha, beta, transition, full_likelihood)
    assert np.exp(first_alpha[0] - logsumexp(first_alpha[0])) == pytest.approx(
        np.exp(full_alpha[0] - logsumexp(full_alpha[0]))
    )
    assert gamma[0, 1] > 0.8


def test_hmm_artifact_round_trip_corruption_and_schema_guards() -> None:
    dataset, _, normalizer, model, artifact = _fit()
    payload = artifact.serialize()
    loaded = HMMArtifact.from_json(
        payload,
        schema_fingerprint=dataset.schema.fingerprint,
        normalizer_fingerprint=normalizer.fingerprint,
    )
    assert loaded == artifact and loaded.fingerprint == artifact.fingerprint
    assert model.load_artifact(payload, normalizer) == artifact
    data = json.loads(payload)
    data["initial_probabilities"] = [-1.0, 2.0]
    with pytest.raises(ValueError, match="probabilities"):
        HMMArtifact.from_json(
            json.dumps(data),
            schema_fingerprint=dataset.schema.fingerprint,
            normalizer_fingerprint=normalizer.fingerprint,
        )
    data = json.loads(payload)
    data["variances"][0][0] = -1.0
    with pytest.raises(ValueError, match="emissions"):
        HMMArtifact.from_json(
            json.dumps(data),
            schema_fingerprint=dataset.schema.fingerprint,
            normalizer_fingerprint=normalizer.fingerprint,
        )
    data = json.loads(payload)
    data["means"][0][0] += 0.1
    with pytest.raises(ValueError, match="parameter hash"):
        HMMArtifact.from_json(
            json.dumps(data),
            schema_fingerprint=dataset.schema.fingerprint,
            normalizer_fingerprint=normalizer.fingerprint,
        )
    with pytest.raises(ValueError, match="mismatch"):
        HMMArtifact.from_json(
            payload, schema_fingerprint="f" * 64, normalizer_fingerprint=normalizer.fingerprint
        )
    data = json.loads(payload)
    data["metadata"]["feature_schema_version"] = "2.0"
    with pytest.raises(ValueError, match="version or dimensions"):
        HMMArtifact.from_json(
            json.dumps(data),
            schema_fingerprint=dataset.schema.fingerprint,
            normalizer_fingerprint=normalizer.fingerprint,
        )
    data = json.loads(payload)
    data["transition_matrix"][0].pop()
    with pytest.raises(ValueError, match="dimensions"):
        HMMArtifact.from_json(
            json.dumps(data),
            schema_fingerprint=dataset.schema.fingerprint,
            normalizer_fingerprint=normalizer.fingerprint,
        )
    wrong_settings = GaussianHMMRegimeModel(HMMSettings(state_count=2, restarts=1))
    with pytest.raises(ValueError, match="settings mismatch"):
        wrong_settings.load_artifact(payload, normalizer)


def test_kmeans_two_three_clusters_empty_and_restarts() -> None:
    dataset, _ = synthetic_dataset("two_state", 45)
    normalizer = fit_normalizer(dataset, tuple(range(45)))
    model = KMeansRegimeModel(KMeansSettings(cluster_count=2, restarts=3))
    artifact = model.fit(dataset, normalizer, random_seed=11)
    states = model.infer(dataset)
    assert len(set(states)) == 2 and len(artifact.restarts) == 3
    assert model.diagnostics()["inertia"] >= 0
    assert KMeansRegimeModel(model.settings).fit(dataset, normalizer, random_seed=11) == artifact
    assert model.load_artifact(artifact.serialize(), normalizer) == artifact
    corrupted = json.loads(artifact.serialize())
    corrupted["centers"][0][0] += 0.1
    with pytest.raises(ValueError, match="parameter hash"):
        model.load_artifact(json.dumps(corrupted), normalizer)
    with pytest.raises(ValueError, match="settings mismatch"):
        KMeansRegimeModel(KMeansSettings(cluster_count=2, restarts=1)).load_artifact(
            artifact.serialize(), normalizer
        )
    three, _ = synthetic_dataset("three_state", 45)
    three_norm = fit_normalizer(three, tuple(range(45)))
    assert (
        len(
            set(
                KMeansRegimeModel(KMeansSettings(cluster_count=3))
                .fit(three, three_norm, random_seed=2)
                .mapping
            )
        )
        == 3
    )
    duplicate = fit_kmeans_array(np.ones((8, 2)), 3, seed=3)
    assert duplicate.diagnostic.empty_clusters >= 1
    assert np.isfinite(duplicate.centers).all()
    with pytest.raises(ValueError, match="invalid"):
        fit_kmeans_array(np.ones((1, 2)), 2, seed=3)


def test_kmeans_size_comparison_is_deterministic_and_descriptive() -> None:
    dataset, _ = synthetic_dataset("three_state", 45)
    normalizer = fit_normalizer(dataset, tuple(range(45)))
    settings = KMeansSettings(max_iterations=20, restarts=2)
    first = compare_kmeans_cluster_counts(
        dataset, normalizer, (2, 3), settings=settings, random_seed=4
    )
    assert first == compare_kmeans_cluster_counts(
        dataset, normalizer, (2, 3), settings=settings, random_seed=4
    )
    assert tuple(entry.cluster_count for entry in first) == (2, 3)
    assert all(entry.inertia >= 0 and 0 <= entry.occupancy_balance <= 1 for entry in first)
    assert first[1].inertia <= first[0].inertia + 1e-8
    with pytest.raises(ValueError, match="cluster counts"):
        compare_kmeans_cluster_counts(dataset, normalizer, (3, 2))


def test_change_points_offline_future_dependence_and_causal_scores() -> None:
    dataset, _ = synthetic_dataset("mean_shift", 60)
    normalizer = fit_normalizer(dataset, tuple(range(60)))
    detector = MeanShiftDetector(
        ChangePointSettings(left_window=5, right_window=5, threshold=0.25, minimum_separation=8)
    )
    points = detector.offline(dataset, normalizer)
    assert points and min(abs(point.index - 30) for point in points) <= 3
    assert all(not point.causal and point.right_window_end_ns >= point.as_of_ns for point in points)
    assert all(b.index - a.index >= 8 for a, b in zip(points, points[1:], strict=False))
    unchanged = replace(
        dataset,
        rows=tuple(
            replace(
                row,
                feature_vector=dataset.rows[0].feature_vector,
                feature_mask=dataset.rows[0].feature_mask,
            )
            for row in dataset.rows
        ),
    )
    assert detector.offline(unchanged, normalizer) == ()
    prior_normalizer = fit_normalizer(dataset, tuple(range(20)))
    tail = dataset.take(tuple(range(20, 60)))
    scores = detector.causal_scores(tail, prior_normalizer)
    prefix = detector.causal_scores(tail.take(tuple(range(30))), prior_normalizer)
    assert scores[: len(prefix)] == prefix
    assert all(item.causal for item in scores)
    with pytest.raises(ValueError, match="prior-fitted"):
        detector.causal_scores(dataset, normalizer)


def test_multiple_change_points_and_degenerate_hmm_quality() -> None:
    source, _ = synthetic_dataset("three_state", 90)
    prototypes = (source.rows[1], source.rows[6], source.rows[11])
    segmented = replace(
        source,
        rows=tuple(
            replace(
                row,
                feature_vector=prototypes[index // 30].feature_vector,
                feature_mask=prototypes[index // 30].feature_mask,
            )
            for index, row in enumerate(source.rows)
        ),
    )
    normalizer = fit_normalizer(segmented, tuple(range(90)))
    detector = MeanShiftDetector(
        ChangePointSettings(left_window=5, right_window=5, threshold=0.2, minimum_separation=15)
    )
    points = detector.offline(segmented, normalizer)
    assert any(abs(point.index - 30) <= 2 for point in points)
    assert any(abs(point.index - 60) <= 2 for point in points)
    constant = replace(
        segmented,
        rows=tuple(
            replace(
                row,
                feature_vector=prototypes[0].feature_vector,
                feature_mask=prototypes[0].feature_mask,
            )
            for row in segmented.rows
        ),
    )
    constant_normalizer = fit_normalizer(constant, tuple(range(90)))
    model = GaussianHMMRegimeModel(HMMSettings(state_count=2, max_iterations=10, restarts=1))
    model.fit(constant, constant_normalizer, random_seed=2)
    assert model.quality(constant).degenerate_states
    assert detector.offline(constant, constant_normalizer) == ()


def test_comparison_metrics_and_deterministic_contingency() -> None:
    a = (0, 0, 1, 1, 2, 2)
    permuted = (2, 2, 0, 0, 1, 1)
    mixed = (0, 1, 0, 1, 0, 1)
    assert adjusted_rand_index(a, permuted) == pytest.approx(1)
    assert normalized_mutual_information(a, permuted) == pytest.approx(1)
    assert adjusted_rand_index(a, mixed) < 1
    assert normalized_mutual_information(a, mixed) < 1
    assert adjusted_rand_index((0, 0), (1, 1)) == 1
    assert normalized_mutual_information((0, 0), (1, 1)) == 1
    assert adjusted_rand_index((0, 0, 0, 0), (0, 0, 1, 1)) == 0
    assert normalized_mutual_information((0, 0, 0, 0), (0, 0, 1, 1)) == 0
    dataset, states = synthetic_dataset("two_state", 30)
    comparison = compare_assignments(dataset, states, model_type="fixture")
    assert comparison.evaluation.profiles
    assert comparison.adjusted_rand is None  # no MCE-10 labels in this fixture
    assert comparison.normalized_mutual_information is None
    labeled = replace(
        dataset,
        rows=tuple(
            replace(row, deterministic_regime="UP" if state == 0 else "DOWN")
            for row, state in zip(dataset.rows, states, strict=True)
        ),
    )
    matched = compare_assignments(labeled, states, model_type="fixture")
    assert matched.adjusted_rand == pytest.approx(1)
    assert matched.normalized_mutual_information == pytest.approx(1)
    assert matched.evaluation.deterministic_contingency


def test_walk_forward_frozen_artifact_and_future_append() -> None:
    dataset, _ = synthetic_dataset("two_state", 60)
    calendar = MarketSessionService(MCEConfig(_env_file=None).market)
    split = StatisticalSplitSettings(
        train_sessions=20,
        validation_sessions=5,
        test_sessions=5,
        embargo_sessions=1,
        step_sessions=10,
    )
    settings = HMMSettings(state_count=2, max_iterations=12, restarts=1)
    early = run_walk_forward_hmm(
        dataset.take(tuple(range(50))), calendar, split, settings, random_seed=9
    )
    full = run_walk_forward_hmm(dataset, calendar, split, settings, random_seed=9)
    assert len(full) > len(early)
    assert early[0].artifact_fingerprint == full[0].artifact_fingerprint
    assert early[0].normalizer_fingerprint == full[0].normalizer_fingerprint
    assert early[0].validation == full[0].validation
    assert early[0].test == full[0].test
    assert all(item.causal for fold in full for item in fold.validation + fold.test)
    assert full[1].train_end_ns > full[0].train_end_ns


def test_model_settings_affect_configuration_fingerprint() -> None:
    config = MCEConfig(_env_file=None)
    changed = config.model_copy(
        update={
            "statistical": config.statistical.model_copy(update={"hmm": HMMSettings(state_count=4)})
        }
    )
    assert config_fingerprint(config) != config_fingerprint(changed)
