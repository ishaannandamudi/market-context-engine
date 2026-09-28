"""Transparent trend diagnostics on causally selected completed closes."""

from dataclasses import dataclass
from math import isfinite, log

from mce.models.market_data import BarObservation

NS_PER_HOUR = 3_600_000_000_000


@dataclass(frozen=True)
class Regression:
    slope_per_hour: float
    intercept: float
    r_squared: float
    sample_count: int


def log_price_regression(bars: tuple[BarObservation, ...]) -> Regression | None:
    """OLS of log(close) against elapsed hours from the first bar end."""
    if len(bars) < 2 or any(bar.close <= 0 or bar.bar_end_ns is None for bar in bars):
        return None
    first = bars[0].bar_end_ns
    assert first is not None
    x = [(bar.bar_end_ns - first) / NS_PER_HOUR for bar in bars if bar.bar_end_ns is not None]
    if len(x) != len(bars) or len(set(x)) != len(x):
        return None
    y = [log(bar.close) for bar in bars]
    if all(value == y[0] for value in y):
        return Regression(0.0, y[0], 1.0, len(bars))
    x_mean, y_mean = sum(x) / len(x), sum(y) / len(y)
    xx = sum((v - x_mean) ** 2 for v in x)
    yy = sum((v - y_mean) ** 2 for v in y)
    if xx <= 0:
        return None
    slope = sum((a - x_mean) * (b - y_mean) for a, b in zip(x, y, strict=True)) / xx
    intercept = y_mean - slope * x_mean
    if yy == 0:
        r_squared = 1.0
    else:
        residual = sum((b - intercept - slope * a) ** 2 for a, b in zip(x, y, strict=True))
        r_squared = max(0.0, min(1.0, 1.0 - residual / yy))
    if not all(isfinite(v) for v in (slope, intercept, r_squared)):
        return None
    return Regression(slope, intercept, r_squared, len(bars))


def persistence(bars: tuple[BarObservation, ...]) -> float | None:
    """Fraction of nonzero adjacent log returns aligned with net move; flat is zero."""
    if len(bars) < 2 or any(bar.close <= 0 for bar in bars):
        return None
    changes = [log(b.close) - log(a.close) for a, b in zip(bars, bars[1:], strict=False)]
    nonzero = [change for change in changes if change != 0]
    net = log(bars[-1].close) - log(bars[0].close)
    if not nonzero or net == 0:
        return 0.0
    return sum(change * net > 0 for change in nonzero) / len(nonzero)


def path_efficiency(bars: tuple[BarObservation, ...]) -> float | None:
    """Absolute net log move divided by total absolute stepwise log move."""
    if len(bars) < 2 or any(bar.close <= 0 for bar in bars):
        return None
    path = sum(abs(log(b.close) - log(a.close)) for a, b in zip(bars, bars[1:], strict=False))
    return min(1.0, abs(log(bars[-1].close) - log(bars[0].close)) / path) if path else 0.0


def distance_from_sma(bars: tuple[BarObservation, ...]) -> float | None:
    if not bars:
        return None
    if all(bar.close == bars[0].close for bar in bars):
        return 0.0 if bars[0].close > 0 else None
    reference = sum(bar.close / len(bars) for bar in bars)
    if reference <= 0:
        return None
    result = bars[-1].close / reference - 1.0
    return result if isfinite(result) else None


def sign_agreement(values: tuple[float, ...]) -> tuple[float | None, bool]:
    """Absolute majority of nonzero signs, with explicit opposing-sign detection."""
    if any(not isfinite(value) for value in values):
        raise ValueError("agreement inputs must be finite")
    signs = [1 if value > 0 else -1 for value in values if value != 0]
    if not signs:
        return (0.0 if values else None), False
    return abs(sum(signs)) / len(signs), min(signs) < max(signs)
