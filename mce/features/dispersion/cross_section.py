"""Population cross-sectional moments and explicitly interpolated robust spreads."""

from dataclasses import dataclass
from math import isfinite

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class CrossSection:
    count: int
    mean: float
    std: float
    median: float
    mad: float
    scaled_mad: float
    iqr: float
    tail_spread: float
    full_range: float
    upside_std: float | None
    downside_std: float | None
    downside_upside_spread: float | None
    skewness: float | None
    excess_kurtosis: float | None


def _finite(value: np.floating | float) -> float | None:
    native = float(value)
    return native if isfinite(native) else None


def cross_section(
    values: tuple[float, ...], *, minimum: int = 2, tail_quantiles: tuple[float, float] = (0.1, 0.9)
) -> CrossSection | None:
    if len(values) < max(2, minimum) or any(not isfinite(value) for value in values):
        return None
    lower, upper = tail_quantiles
    if not 0 < lower < 0.5 < upper < 1:
        raise ValueError("tail quantiles must straddle the median")
    data: NDArray[np.float64] = np.asarray(values, dtype=np.float64)
    mean = _finite(np.mean(data))
    std = _finite(np.std(data, ddof=0))
    middle = _finite(np.median(data))
    if mean is None or std is None or middle is None:
        return None
    mad = _finite(np.median(np.abs(data - middle)))
    q25, q75 = np.quantile(data, (0.25, 0.75), method="linear")
    qlow, qhigh = np.quantile(data, (lower, upper), method="linear")
    iqr = _finite(q75 - q25)
    tail = _finite(qhigh - qlow)
    full_range = _finite(np.max(data) - np.min(data))
    if mad is None or iqr is None or tail is None or full_range is None:
        return None
    upside = data[data > 0]
    downside = data[data < 0]
    up = _finite(np.std(upside, ddof=0)) if len(upside) >= 2 else None
    down = _finite(np.std(downside, ddof=0)) if len(downside) >= 2 else None
    skew = kurtosis = None
    if std > 1e-15:
        standardized = (data - mean) / std
        skew = _finite(np.mean(standardized**3)) if len(data) >= 3 else None
        kurtosis = _finite(np.mean(standardized**4) - 3) if len(data) >= 4 else None
    return CrossSection(
        count=len(data),
        mean=mean,
        std=std,
        median=middle,
        mad=mad,
        scaled_mad=1.4826 * mad,
        iqr=iqr,
        tail_spread=tail,
        full_range=full_range,
        upside_std=up,
        downside_std=down,
        downside_upside_spread=down - up if down is not None and up is not None else None,
        skewness=skew,
        excess_kurtosis=kurtosis,
    )


def weighted_std(values: tuple[float, ...], weights: tuple[float, ...]) -> float | None:
    """Population-style variance after normalizing nonnegative point-in-time weights."""
    if (
        len(values) < 2
        or len(values) != len(weights)
        or any(not isfinite(item) for item in values)
        or any(not isfinite(item) or item < 0 for item in weights)
    ):
        return None
    total = sum(weights)
    if total <= 0:
        return None
    data = np.asarray(values)
    normalized = np.asarray(weights) / total
    mean = float(np.dot(normalized, data))
    variance = float(np.dot(normalized, (data - mean) ** 2))
    return _finite(np.sqrt(variance))
