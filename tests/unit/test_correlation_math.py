from math import isfinite

import numpy as np
import pytest

from mce.features.correlation.diagnostics import (
    complete_link_groups,
    correlation_distance,
    eigen_diagnostics,
    shrink_identity,
    summary,
    weighted_average,
)
from mce.features.correlation.matrix import (
    average_ranks,
    covariance_to_correlation,
    estimate,
    sample_covariance,
)
from mce.features.correlation.models import CorrelationMatrix


def test_pearson_spearman_and_ordering() -> None:
    data = {
        "CCC": {i: -float(i * i) for i in range(10)},
        "AAA": {i: float(i) for i in range(10)},
        "BBB": {i: float(i * i) for i in range(10)},
    }
    result = estimate(data, 10, 8)
    assert result is not None
    assert result.symbols == ("AAA", "BBB", "CCC")
    assert result.matrix[1][2] == pytest.approx(-1)
    assert result.spearman_matrix is not None
    assert result.spearman_matrix[0][1] == pytest.approx(1)
    assert result.spearman_matrix[0][2] == pytest.approx(-1)
    assert result.matrix[0][1] < 1
    assert estimate({"AAA": dict.fromkeys(range(10), 1.0), "BBB": data["BBB"]}, 10, 8) is None
    assert estimate(data, 10, 11) is None


def test_covariance_psd_shrinkage_and_eigen_known_cases() -> None:
    identity = np.eye(3)
    assert eigen_diagnostics(identity) == pytest.approx((1, 1 / 3, 3))
    common = np.ones((3, 3))
    assert eigen_diagnostics(common) == pytest.approx((3, 1, 1))
    block = np.asarray([[1, 0.8, 0], [0.8, 1, 0], [0, 0, 1]])
    assert eigen_diagnostics(block) is not None
    shrunk = shrink_identity(common, 0.2)
    assert np.linalg.eigvalsh(shrunk).min() >= 0
    assert np.allclose(np.diag(shrunk), 1)
    assert shrink_identity(identity, 0.2).tolist() == identity.tolist()
    assert sample_covariance(np.ones((1, 3))) is None
    assert covariance_to_correlation(np.diag([1.0, 0.0])) is None
    assert covariance_to_correlation(np.diag([1.0, 2.0])) is not None
    assert covariance_to_correlation(np.asarray([[1.0, 2.0], [2.0, 1.0]])) is None
    assert eigen_diagnostics(np.asarray([[1.0, 2.0], [2.0, 1.0]])) is None
    with pytest.raises(ValueError):
        shrink_identity(identity, 1.1)


def test_alignment_missingness_orthogonality_and_ties() -> None:
    x = [1.0, -1.0, 1.0, -1.0] * 3
    y = [1.0, 1.0, -1.0, -1.0] * 3
    result = estimate({"BBB": dict(enumerate(y)), "AAA": dict(enumerate(x))}, 12, 8)
    assert result is not None
    assert result.matrix[0][1] == pytest.approx(0, abs=1e-12)
    missing = estimate(
        {"AAA": dict(enumerate(x)), "BBB": {i: value for i, value in enumerate(y) if i != 3}},
        12,
        8,
    )
    assert missing is not None and missing.sample_count == 11
    rank = average_ranks(np.asarray([[1.0], [1.0], [3.0], [4.0]]))
    assert rank[:, 0].tolist() == [1.5, 1.5, 3, 4]


def test_pair_summaries_weighting_and_grouping() -> None:
    matrix = np.asarray([[1, 0.9, 0], [0.9, 1, 0.1], [0, 0.1, 1]])
    assert summary(matrix)[0] == pytest.approx(1 / 3)
    assert weighted_average(matrix, (2, 2, 1)) == pytest.approx((4 * 0.9 + 2 * 0 + 2 * 0.1) / 8)
    assert weighted_average(matrix, (0, 0, 1)) is None
    assert weighted_average(matrix, (1, float("nan"), 1)) is None
    assert correlation_distance(1) == 0
    assert correlation_distance(-1) == 2
    with pytest.raises(ValueError):
        correlation_distance(float("nan"))
    assert complete_link_groups(matrix, ("AAA", "BBB", "CCC"), 0.8) == (("AAA", "BBB"), ("CCC",))
    assert complete_link_groups(np.ones((3, 3)), ("AAA", "BBB", "CCC"), 0.8) == (
        ("AAA", "BBB", "CCC"),
    )
    model = CorrelationMatrix(symbols=("AAA", "BBB"), values=((1.0, 0.2), (0.2, 1.0)))
    assert isfinite(model.values[0][1])
    with pytest.raises(ValueError):
        CorrelationMatrix(symbols=("BBB", "AAA"), values=((1.0, 0.2), (0.2, 1.0)))
