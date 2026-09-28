"""Pure, explicit realized-variance estimator mathematics."""

from dataclasses import dataclass
from datetime import date
from math import isfinite, log, pi, sqrt

from mce.features.returns import log_return


@dataclass(frozen=True)
class SessionOHLC:
    day: date
    open: float
    high: float
    low: float
    close: float
    effective_timestamp_ns: int


@dataclass(frozen=True)
class Estimate:
    annualized_volatility: float
    variance_per_session: float
    sample_count: int
    sum_squared_returns: float | None = None


def _valid_bar(bar: SessionOHLC) -> bool:
    values = (bar.open, bar.high, bar.low, bar.close)
    return (
        all(isfinite(value) and value > 0 for value in values)
        and bar.low <= min(bar.open, bar.close)
        and bar.high >= max(bar.open, bar.close)
    )


def _finish(
    variance: float,
    count: int,
    trading_sessions_per_year: int,
    sum_squared_returns: float | None = None,
) -> Estimate | None:
    if count <= 0 or trading_sessions_per_year <= 0 or not isfinite(variance):
        return None
    annualized = sqrt(max(variance, 0.0) * trading_sessions_per_year)
    if not isfinite(annualized):
        return None
    return Estimate(annualized, max(variance, 0.0), count, sum_squared_returns)


def sample_variance(values: tuple[float, ...]) -> float | None:
    """Unbiased sample variance with denominator n-1; at least two values."""
    if len(values) < 2 or any(not isfinite(value) for value in values):
        return None
    mean = sum(value / len(values) for value in values)
    result = sum((value - mean) * (value - mean) for value in values) / (len(values) - 1)
    return result if isfinite(result) else None


def close_to_close(sessions: tuple[SessionOHLC, ...], annual_sessions: int) -> Estimate | None:
    """Annualized sample standard deviation of adjacent daily log close returns."""
    if len(sessions) < 3 or any(not _valid_bar(bar) for bar in sessions):
        return None
    returns = tuple(
        log_return(previous.close, current.close)
        for previous, current in zip(sessions, sessions[1:], strict=False)
    )
    if any(value is None for value in returns):
        return None
    numbers = tuple(value for value in returns if value is not None)
    variance = sample_variance(numbers)
    squared_sum = sum(value * value for value in numbers)
    if variance is None or not isfinite(squared_sum):
        return None
    return _finish(variance, len(numbers), annual_sessions, squared_sum)


def parkinson(sessions: tuple[SessionOHLC, ...], annual_sessions: int) -> Estimate | None:
    if not sessions or any(not _valid_bar(bar) for bar in sessions):
        return None
    variance = sum((log(bar.high) - log(bar.low)) ** 2 for bar in sessions)
    return _finish(variance / (4 * log(2) * len(sessions)), len(sessions), annual_sessions)


def garman_klass(sessions: tuple[SessionOHLC, ...], annual_sessions: int) -> Estimate | None:
    if not sessions or any(not _valid_bar(bar) for bar in sessions):
        return None
    variance = sum(
        0.5 * (log(bar.high) - log(bar.low)) ** 2
        - (2 * log(2) - 1) * (log(bar.close) - log(bar.open)) ** 2
        for bar in sessions
    ) / len(sessions)
    return _finish(variance, len(sessions), annual_sessions)


def rogers_satchell(sessions: tuple[SessionOHLC, ...], annual_sessions: int) -> Estimate | None:
    if not sessions or any(not _valid_bar(bar) for bar in sessions):
        return None
    variance = sum(
        (log(bar.high) - log(bar.open)) * (log(bar.high) - log(bar.close))
        + (log(bar.low) - log(bar.open)) * (log(bar.low) - log(bar.close))
        for bar in sessions
    ) / len(sessions)
    return _finish(variance, len(sessions), annual_sessions)


def yang_zhang(sessions: tuple[SessionOHLC, ...], annual_sessions: int) -> Estimate | None:
    """Standard YZ: sample overnight/open-close variances plus mean RS, n>=3 pairs."""
    if len(sessions) < 4 or any(not _valid_bar(bar) for bar in sessions):
        return None
    pairs = tuple(zip(sessions, sessions[1:], strict=False))
    overnight = tuple(log(current.open) - log(previous.close) for previous, current in pairs)
    open_close = tuple(log(current.close) - log(current.open) for _, current in pairs)
    overnight_variance = sample_variance(overnight)
    open_close_variance = sample_variance(open_close)
    rs = rogers_satchell(sessions[1:], 1)
    if overnight_variance is None or open_close_variance is None or rs is None:
        return None
    n = len(pairs)
    k = 0.34 / (1.34 + (n + 1) / (n - 1))
    variance = overnight_variance + k * open_close_variance + (1 - k) * rs.variance_per_session
    return _finish(variance, n, annual_sessions)


def overnight_context(sessions: tuple[SessionOHLC, ...]) -> tuple[float, float, float] | None:
    """Squared-log-return shares and latest absolute gap; previous close is required."""
    if len(sessions) < 2 or any(not _valid_bar(bar) for bar in sessions):
        return None
    overnight = [
        log(current.open) - log(previous.close)
        for previous, current in zip(sessions, sessions[1:], strict=False)
    ]
    intraday = [log(current.close) - log(current.open) for current in sessions[1:]]
    overnight_sum = sum(value * value for value in overnight)
    intraday_sum = sum(value * value for value in intraday)
    total = overnight_sum + intraday_sum
    if not isfinite(total):
        return None
    return (
        overnight_sum / total if total else 0.0,
        intraday_sum / total if total else 0.0,
        abs(overnight[-1]),
    )


def intraday_variation(
    returns: tuple[float, ...],
) -> tuple[float, float | None, float, float] | None:
    """RV, finite-sample-corrected bipower, downside and upside semivariance."""
    if not returns or any(not isfinite(value) for value in returns):
        return None
    rv = sum(value * value for value in returns)
    downside = sum(value * value for value in returns if value < 0)
    upside = sum(value * value for value in returns if value > 0)
    bipower = (
        (pi / 2)
        * len(returns)
        / (len(returns) - 1)
        * sum(abs(a) * abs(b) for a, b in zip(returns, returns[1:], strict=False))
        if len(returns) >= 2
        else None
    )
    if any(not isfinite(value) for value in (rv, downside, upside)) or (
        bipower is not None and not isfinite(bipower)
    ):
        return None
    return rv, bipower, downside, upside
