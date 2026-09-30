"""Finite median-relative, midrank and robust MAD normalization."""

from math import isfinite

from mce.features.seasonality.baseline import baseline
from mce.features.seasonality.models import (
    MetricName,
    SeasonalHistoryIndex,
    SeasonallyAdjustedMetric,
)
from mce.models.enums import QualityFlag


def normalize(
    history: SeasonalHistoryIndex,
    metric: MetricName,
    symbol: str | None,
    bucket_index: int,
    raw_value: float | None,
    minimum_history_sessions: int,
) -> SeasonallyAdjustedMetric:
    if raw_value is not None and not isfinite(raw_value):
        raise ValueError("seasonal current value must be finite")
    reference = baseline(history, metric, symbol, bucket_index)
    if raw_value is None:
        return SeasonallyAdjustedMetric(
            metric, None, reference, None, None, None, frozenset({QualityFlag.MISSING_DATA})
        )
    if reference.sample_count < minimum_history_sessions or reference.median is None:
        return SeasonallyAdjustedMetric(
            metric,
            raw_value,
            reference,
            None,
            None,
            None,
            frozenset({QualityFlag.INSUFFICIENT_SEASONAL_HISTORY}),
        )
    values = tuple(item.value for item in history.samples(metric, symbol, bucket_index))
    percentile = (
        sum(value < raw_value for value in values)
        + 0.5 * sum(value == raw_value for value in values)
    ) / len(values)
    ratio = raw_value / reference.median if reference.median > 1e-24 else None
    mad = reference.mad
    scale = 1.4826 * mad if mad is not None and mad > 1e-24 else None
    robust_z = (
        (raw_value - reference.median) / scale
        if scale is not None and isfinite(scale)
        else 0.0
        if raw_value == reference.median
        else None
    )
    if robust_z is not None and not isfinite(robust_z):
        robust_z = None
    if ratio is not None and not isfinite(ratio):
        ratio = None
    return SeasonallyAdjustedMetric(metric, raw_value, reference, ratio, percentile, robust_z)
