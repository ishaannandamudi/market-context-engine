from datetime import date, timedelta
from math import log, pi, sqrt

import pytest

from mce.features.returns import log_return
from mce.features.volatility.estimators import (
    SessionOHLC,
    close_to_close,
    garman_klass,
    intraday_variation,
    overnight_context,
    parkinson,
    rogers_satchell,
    sample_variance,
    yang_zhang,
)
from mce.features.volatility.rolling import estimator_consensus, prior_distribution


def _bar(index: int, opening: float, high: float, low: float, closing: float) -> SessionOHLC:
    return SessionOHLC(date(2024, 1, 2) + timedelta(days=index), opening, high, low, closing, index)


def test_range_reference_vectors_and_annualization() -> None:
    bar = _bar(0, 100, 110, 90, 105)
    pk = parkinson((bar,), 252)
    gk = garman_klass((bar,), 252)
    rs = rogers_satchell((bar,), 252)
    assert pk is not None and gk is not None and rs is not None
    assert pk.variance_per_session == pytest.approx(0.014523873553353097)
    assert gk.variance_per_session == pytest.approx(0.019214797961641272)
    assert rs.variance_per_session == pytest.approx(0.02067522617026606)
    assert pk.annualized_volatility == pytest.approx(sqrt(252 * 0.014523873553353097))
    assert parkinson((bar,), 100).annualized_volatility == pytest.approx(
        sqrt(100 * 0.014523873553353097)
    )  # type: ignore[union-attr]


def test_flat_invalid_and_minimum_range_samples() -> None:
    flat = _bar(0, 100, 100, 100, 100)
    for estimator in (parkinson, garman_klass, rogers_satchell):
        result = estimator((flat,), 252)
        assert result is not None and result.annualized_volatility == 0
        assert estimator((), 252) is None
        assert estimator((_bar(0, 100, 110, 0, 105),), 252) is None
        assert estimator((_bar(0, 100, 101, 99, 110),), 252) is None


def test_daily_close_sample_variance_distinct_from_squared_sum() -> None:
    closes = (100.0, 100.0 * 1.01, 100.0 * 1.01 * 0.99)
    bars = tuple(_bar(i, price, price, price, price) for i, price in enumerate(closes))
    result = close_to_close(bars, 252)
    assert result is not None
    r1, r2 = log(1.01), log(0.99)
    expected_sample_variance = (r1 - (r1 + r2) / 2) ** 2 + (r2 - (r1 + r2) / 2) ** 2
    assert result.variance_per_session == pytest.approx(expected_sample_variance)
    assert result.sum_squared_returns == pytest.approx(r1 * r1 + r2 * r2)
    assert result.annualized_volatility == pytest.approx(sqrt(252 * expected_sample_variance))
    assert close_to_close(bars[:2], 252) is None
    assert (
        close_to_close(
            tuple(_bar(i, 100, 100, 100, 100) for i in range(4)), 252
        ).annualized_volatility
        == 0
    )  # type: ignore[union-attr]


def test_yang_zhang_independent_decimal_reference_vector() -> None:
    bars = (
        _bar(0, 100, 100, 100, 100),
        _bar(1, 102, 106, 98, 104),
        _bar(2, 103, 107, 101, 102),
        _bar(3, 105, 109, 100, 108),
        _bar(4, 107, 111, 104, 110),
    )
    result = yang_zhang(bars, 252)
    assert result is not None
    assert result.sample_count == 4
    assert result.variance_per_session == pytest.approx(0.0029057574922814002, rel=1e-12)
    assert result.annualized_volatility == pytest.approx(sqrt(252 * 0.0029057574922814002))
    assert yang_zhang(bars[:3], 252) is None
    shares = overnight_context(bars)
    assert shares is not None and shares[0] + shares[1] == pytest.approx(1)
    assert shares[2] == pytest.approx(abs(log(107 / 108)))


def test_bipower_semivariance_and_jump_reference() -> None:
    vector = (0.01, -0.02, 0.03)
    result = intraday_variation(vector)
    assert result is not None
    rv, bipower, downside, upside = result
    assert rv == pytest.approx(0.0014)
    assert downside == pytest.approx(0.0004)
    assert upside == pytest.approx(0.001)
    assert downside + upside == pytest.approx(rv)
    assert bipower == pytest.approx((pi / 2) * (3 / 2) * (0.01 * 0.02 + 0.02 * 0.03))
    jump = intraday_variation((0.001, 0.001, 0.1, 0.001, 0.001))
    assert jump is not None and jump[1] is not None
    assert (jump[0] - jump[1]) / jump[0] > 0.9
    assert intraday_variation((0, 0))[0] == 0  # type: ignore[index]
    assert intraday_variation(()) is None
    assert intraday_variation((float("inf"),)) is None


def test_prior_distribution_ties_zero_mad_and_consensus() -> None:
    assert prior_distribution(2, (1, 2, 3), 3) == (0.5, 0.0, 0.0)
    assert prior_distribution(4, (1, 2, 3), 3) == pytest.approx((1, 2 / 1.4826, 1))
    assert prior_distribution(2, (2, 2, 2), 3) == (0.5, 0.0, 0.0)
    assert prior_distribution(3, (2, 2, 2), 3) == (1, None, 1)
    assert prior_distribution(1, (1,), 2) is None
    assert estimator_consensus((0.2, 0.2, 0.2)) == pytest.approx((0.2, 0, 0, 1))
    divergent = estimator_consensus((0.2, 0.31, 0.36))
    assert divergent is not None and divergent[1] == pytest.approx(0.16)
    assert divergent[3] < 1
    assert estimator_consensus(()) is None


def test_sample_variance_and_invalid_numerics() -> None:
    assert sample_variance((1.0, 3.0)) == pytest.approx(2.0)
    assert sample_variance((1.0,)) is None
    assert sample_variance((1.0, float("nan"))) is None
    assert prior_distribution(1, (float("inf"),), 1) is None
    assert estimator_consensus((-1.0,)) is None


@pytest.mark.parametrize("scale", [0.1, 1.0, 1000.0])
def test_log_return_and_estimator_price_scale_invariance(scale: float) -> None:
    assert log_return(100 * scale, 110 * scale) == pytest.approx(log(1.1))
    assert log_return(100 * scale, 90 * scale) == pytest.approx(log(0.9))
    assert log_return(100 * scale, 100 * scale) == 0
    assert log_return(0, 100 * scale) is None
    assert log_return(-1, 100 * scale) is None
    bars = (
        _bar(0, 100 * scale, 102 * scale, 99 * scale, 101 * scale),
        _bar(1, 103 * scale, 106 * scale, 101 * scale, 105 * scale),
        _bar(2, 102 * scale, 104 * scale, 98 * scale, 99 * scale),
        _bar(3, 101 * scale, 107 * scale, 100 * scale, 106 * scale),
    )
    reference = (
        _bar(0, 100, 102, 99, 101),
        _bar(1, 103, 106, 101, 105),
        _bar(2, 102, 104, 98, 99),
        _bar(3, 101, 107, 100, 106),
    )
    for estimator in (close_to_close, parkinson, garman_klass, rogers_satchell, yang_zhang):
        actual = estimator(bars, 252)
        expected = estimator(reference, 252)
        assert actual is not None and expected is not None
        assert actual.annualized_volatility == pytest.approx(expected.annualized_volatility)


def test_semivariance_single_direction_and_huge_finite_input() -> None:
    positive = intraday_variation((0.01, 0.02))
    negative = intraday_variation((-0.01, -0.02))
    assert positive is not None and negative is not None
    assert positive[2] == 0 and positive[3] == pytest.approx(positive[0])
    assert negative[3] == 0 and negative[2] == pytest.approx(negative[0])
    downside_dominated = intraday_variation((0.001, -0.05, 0.001))
    assert downside_dominated is not None
    assert downside_dominated[2] / downside_dominated[0] > 0.99
    assert intraday_variation((1e308,)) is None
    assert sample_variance((1e308, -1e308)) is None
