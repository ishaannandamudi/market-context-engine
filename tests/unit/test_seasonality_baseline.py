"""Reference calculations and immutable prior-session baseline checks."""

from dataclasses import replace
from datetime import date

import pytest

from mce.features.seasonality.baseline import baseline
from mce.features.seasonality.models import SeasonalHistoryIndex, SeasonalObservation
from mce.features.seasonality.normalization import normalize
from mce.models.enums import QualityFlag
from mce.reproducibility import config_fingerprint
from tests.fixtures.seasonal_scenarios import seasonal_config


def _index() -> SeasonalHistoryIndex:
    days = (date(2024, 1, 3), date(2024, 1, 4), date(2024, 1, 5))
    values = (10.0, 20.0, 30.0)
    return SeasonalHistoryIndex(
        date(2024, 1, 8),
        days,
        100,
        config_fingerprint(seasonal_config()),
        tuple(
            SeasonalObservation("volume", "AAA", day, 0, 1, 1, value)
            for day, value in zip(days, values, strict=True)
        ),
    )


def test_robust_reference_and_midrank_percentile() -> None:
    history = _index()
    stats = baseline(history, "volume", "AAA", 0)
    assert (stats.sample_count, stats.expected_sessions, stats.coverage) == (3, 3, 1)
    assert (stats.median, stats.mean, stats.mad, stats.q25, stats.q75) == (
        20,
        20,
        10,
        15,
        25,
    )
    assert stats.std == pytest.approx(10)
    current = normalize(history, "volume", "AAA", 0, 20, 2)
    assert current.raw_value == 20 and current.ratio == 1
    assert current.percentile == 0.5 and current.robust_z == 0
    high = normalize(history, "volume", "AAA", 0, 40, 2)
    assert high.ratio == 2 and high.percentile == 1
    assert high.robust_z == pytest.approx(20 / (1.4826 * 10))
    assert normalize(history, "volume", "AAA", 0, 5, 2).percentile == 0
    assert history.fingerprint == _index().fingerprint


def test_sparse_history_zero_denominators_and_raw_preservation() -> None:
    history = _index()
    sparse = normalize(history, "volume", "BBB", 0, 100, 2)
    assert sparse.raw_value == 100 and sparse.ratio is None
    assert sparse.baseline is not None and sparse.baseline.coverage == 0
    assert QualityFlag.INSUFFICIENT_SEASONAL_HISTORY in sparse.quality_flags
    missing = normalize(history, "volume", "AAA", 0, None, 2)
    assert QualityFlag.MISSING_DATA in missing.quality_flags
    zeros = replace(
        history,
        observations=tuple(replace(row, value=0) for row in history.observations),
    )
    flat = normalize(zeros, "volume", "AAA", 0, 0, 2)
    assert flat.ratio is None and flat.robust_z == 0
    assert normalize(zeros, "volume", "AAA", 0, 1, 2).robust_z is None
    with pytest.raises(ValueError, match="finite"):
        normalize(history, "volume", "AAA", 0, float("nan"), 2)


def test_future_duplicate_or_current_session_sample_rejected() -> None:
    history = _index()
    first = history.observations[0]
    with pytest.raises(ValueError, match="future"):
        replace(history, observations=history.observations + (first,))
    with pytest.raises(ValueError, match="future"):
        replace(
            history,
            observations=history.observations + (replace(first, trading_date=history.target_date),),
        )
    with pytest.raises(ValueError, match="future"):
        replace(history, observations=history.observations + (replace(first, available_at_ns=101),))


def test_negative_correlation_uses_percentile_without_undefined_ratio() -> None:
    history = _index()
    negative = replace(
        history,
        observations=tuple(
            SeasonalObservation("correlation", None, day, 0, 1, 1, value)
            for day, value in zip(history.prior_dates, (-0.4, -0.2, -0.1), strict=True)
        ),
    )
    current = normalize(negative, "correlation", None, 0, -0.3, 2)
    assert current.ratio is None
    assert current.percentile == pytest.approx(1 / 3)
    assert current.robust_z is not None


def test_extreme_finite_history_keeps_expected_value_finite() -> None:
    history = _index()
    extreme = replace(
        history,
        observations=tuple(replace(row, value=1e308) for row in history.observations[:2]),
    )
    reference = baseline(extreme, "volume", "AAA", 0)
    assert reference.median == 1e308
    assert reference.mean is None and reference.std is None
    result = normalize(extreme, "volume", "AAA", 0, 1e308, 2)
    assert result.ratio == 1 and result.robust_z == 0
