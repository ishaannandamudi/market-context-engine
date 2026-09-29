from math import sqrt

import pytest

from mce.config.settings import DispersionSettings
from mce.features.dispersion.concentration import movement_concentration
from mce.features.dispersion.cross_section import cross_section, weighted_std
from mce.features.dispersion.decomposition import decompose
from mce.features.dispersion.history import summarize_history
from mce.features.windows import WindowSpec, WindowUnit


def test_known_population_moments_and_quantiles() -> None:
    result = cross_section((-2.0, -1.0, 0.0, 1.0, 2.0))
    assert result is not None
    assert result.std == pytest.approx(sqrt(2))
    assert result.mad == 1
    assert result.scaled_mad == pytest.approx(1.4826)
    assert result.iqr == 2
    assert result.tail_spread == pytest.approx(3.2)
    assert result.full_range == 4
    assert result.skewness == pytest.approx(0)
    assert result.excess_kurtosis == pytest.approx(-1.3)
    assert result.upside_std == 0.5
    assert result.downside_std == 0.5
    assert result.downside_upside_spread == 0


def test_identical_tiny_and_outlier_cross_sections() -> None:
    assert cross_section((1.0,)) is None
    identical = cross_section((2.0, 2.0, 2.0, 2.0))
    assert identical is not None
    assert identical.std == identical.mad == identical.iqr == 0
    assert identical.skewness is None and identical.excess_kurtosis is None
    assert identical.downside_std is None
    outlier = cross_section((0.0,) * 10 + (100.0,))
    assert outlier is not None
    assert outlier.std > outlier.scaled_mad == 0
    assert outlier.tail_spread < outlier.full_range
    assert cross_section((1.0, 2.0, float("inf"))) is None


def test_upside_downside_higher_moments_and_weights() -> None:
    mixed = cross_section((-5.0, -2.0, -1.0, 1.0, 2.0))
    assert mixed is not None
    assert mixed.downside_std is not None and mixed.upside_std is not None
    assert mixed.downside_std > mixed.upside_std
    right = cross_section((0.0, 0.0, 0.0, 10.0))
    left = cross_section((-10.0, 0.0, 0.0, 0.0))
    assert right is not None and left is not None
    assert right.skewness is not None and right.skewness > 0
    assert left.skewness is not None and left.skewness < 0
    assert cross_section((1.0, 2.0)) is not None
    assert weighted_std((0.0, 2.0), (1.0, 3.0)) == pytest.approx(sqrt(0.75))
    assert weighted_std((0.0, 2.0), (0.0, 0.0)) is None
    assert weighted_std((0.0, 2.0), (1.0,)) is None


def test_concentration_and_exact_sector_decomposition() -> None:
    assert movement_concentration((0.0, 0.0)) is None
    assert movement_concentration((1.0, 1.0)) == pytest.approx((0.5, 2.0))
    assert movement_concentration((1.0, 1.0), (3.0, 1.0)) == pytest.approx((0.625, 1.6))
    dominant = movement_concentration((100.0, 1.0, 1.0))
    assert dominant is not None and dominant[0] > 0.9 and 1 <= dominant[1] < 2
    values = {"A": -2.0, "B": 0.0, "C": 1.0, "D": 3.0}
    groups = {"A": "one", "B": "one", "C": "two", "D": "two"}
    result = decompose(values, groups, minimum_sectors=2, minimum_coverage=1.0)
    assert result is not None
    assert result.member_count == 4
    assert result.total_variance == pytest.approx(3.25)
    assert result.within_variance == pytest.approx(1.0)
    assert result.between_variance == pytest.approx(2.25)
    assert result.within_share + result.between_share == pytest.approx(1.0)
    assert result.sector_counts == (("one", 2), ("two", 2))
    assert decompose(values, {"A": "one"}, minimum_sectors=2, minimum_coverage=0.5) is None
    same_means = decompose(
        {"A": -1.0, "B": 1.0, "C": -1.0, "D": 1.0},
        groups,
        minimum_sectors=2,
        minimum_coverage=1.0,
    )
    assert same_means is not None and same_means.between_variance == 0
    flat_groups = decompose(
        {"A": -1.0, "B": -1.0, "C": 1.0, "D": 1.0},
        groups,
        minimum_sectors=2,
        minimum_coverage=1.0,
    )
    assert flat_groups is not None and flat_groups.within_variance == 0
    assert flat_groups.between_share == 1


def test_prior_only_midrank_robust_z_shock_and_persistence() -> None:
    baseline = (1.0, 2.0, 3.0, 4.0, 5.0)
    high = summarize_history(
        6.0, baseline, baseline_sessions=5, minimum_samples=5, persistence_observations=3
    )
    assert high.percentile == 1
    assert high.robust_z == pytest.approx(3 / 1.4826)
    assert high.shock == pytest.approx(1 / 3)
    assert high.persistence == pytest.approx(1 / 3)
    tied = summarize_history(
        3.0, baseline + (100.0,), baseline_sessions=5, minimum_samples=5, persistence_observations=3
    )
    assert tied.percentile == pytest.approx(0.5)
    assert tied.shock == 0
    zero_mad = summarize_history(
        2.0, (1.0,) * 5, baseline_sessions=5, minimum_samples=5, persistence_observations=3
    )
    assert zero_mad.robust_z is None
    assert zero_mad.percentile == 1
    insufficient = summarize_history(
        2.0, (1.0, None), baseline_sessions=2, minimum_samples=2, persistence_observations=2
    )
    assert insufficient.percentile is None


def test_typed_dispersion_configuration_rejects_ambiguous_settings() -> None:
    with pytest.raises(ValueError):
        DispersionSettings(tail_quantiles=(0.2, 0.9))
    with pytest.raises(ValueError):
        DispersionSettings(universe=("A", "A"))
    with pytest.raises(ValueError):
        DispersionSettings(historical_baseline_sessions=3, minimum_baseline_samples=4)
    with pytest.raises(ValueError):
        DispersionSettings(
            short_horizon=WindowSpec(name="30m", unit=WindowUnit.ELAPSED, length=60_000_000_000)
        )
