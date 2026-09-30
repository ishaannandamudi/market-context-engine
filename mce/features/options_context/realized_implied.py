"""Explicit calendar-day option tenor to trading-session trailing RV comparison."""

from math import isfinite

from mce.config.settings import OptionsContextSettings
from mce.features.options_context.contracts import ATMIVPoint
from mce.features.options_context.models import IVRVContext
from mce.features.volatility.models import RealizedVolatilityFeatureSet
from mce.models.enums import DataQuality


def compare_implied_realized(
    point: ATMIVPoint,
    symbol: str,
    as_of_ns: int,
    realized: RealizedVolatilityFeatureSet | None,
    settings: OptionsContextSettings,
) -> IVRVContext:
    empty = IVRVContext(iv_tenor_days=point.dte_days)
    if (
        realized is None
        or realized.as_of_ns != as_of_ns
        or realized.data_quality
        in (
            DataQuality.POOR,
            DataQuality.UNUSABLE,
        )
    ):
        return empty
    features = realized.per_symbol.get(symbol)
    if features is None:
        return empty
    candidates = tuple(
        (window.sessions, window.close_to_close.annualized_volatility)
        for window in features.daily.values()
        if window.close_to_close is not None and window.coverage_ratio >= 0.8
    )
    if not candidates:
        return empty
    sessions, rv = min(
        candidates,
        key=lambda item: (abs(item[0] * 365 / 252 - point.dte_days), item[0]),
    )
    equivalent = sessions * 365 / 252
    mismatch = abs(equivalent - point.dte_days)
    comparable = mismatch <= settings.tenor_match_tolerance_days
    spread = point.implied_volatility - rv if comparable else None
    ratio = point.implied_volatility / rv if comparable and rv > 1e-12 else None
    return IVRVContext(
        iv_tenor_days=point.dte_days,
        rv_horizon_sessions=sessions,
        rv_equivalent_calendar_days=equivalent,
        horizon_mismatch_days=mismatch,
        exact_match=mismatch <= 0.5,
        comparable=comparable,
        trailing_realized_volatility=rv,
        spread=spread if spread is None or isfinite(spread) else None,
        ratio=ratio if ratio is None or isfinite(ratio) else None,
        vrp_proxy=spread if spread is None or isfinite(spread) else None,
    )
