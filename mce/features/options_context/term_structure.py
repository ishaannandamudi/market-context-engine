"""Descriptive IV term shape using actual supplied maturity in calendar days."""

from math import isfinite

from mce.features.options_context.contracts import ATMIVPoint
from mce.features.options_context.models import TenorIV, TermStructureContext


def term_structure(
    points: tuple[ATMIVPoint, ...], target_days: int, minimum_tenors: int
) -> TermStructureContext:
    combined = sorted(
        (point for point in points if point.combination == "combined"),
        key=lambda point: point.dte_days,
    )
    tenors = tuple(
        TenorIV(dte_days=point.dte_days, implied_volatility=point.implied_volatility)
        for point in combined
    )
    if len(tenors) < minimum_tenors:
        return TermStructureContext(tenors=tenors)
    short, long = tenors[0], tenors[-1]
    middle = (
        min(tenors[1:-1], key=lambda item: (abs(item.dte_days - target_days), item.dte_days))
        if len(tenors) > 2
        else None
    )
    spread = short.implied_volatility - long.implied_volatility
    ratio = (
        short.implied_volatility / long.implied_volatility
        if long.implied_volatility > 1e-12
        else None
    )
    xbar = sum(item.dte_days for item in tenors) / len(tenors)
    ybar = sum(item.implied_volatility / len(tenors) for item in tenors)
    denominator = sum((item.dte_days - xbar) ** 2 for item in tenors)
    numerator = sum((item.dte_days - xbar) * (item.implied_volatility - ybar) for item in tenors)
    slope = numerator / denominator if denominator > 0 else None
    curvature = (
        middle.implied_volatility
        - (
            short.implied_volatility
            + (long.implied_volatility - short.implied_volatility)
            * (middle.dte_days - short.dte_days)
            / (long.dte_days - short.dte_days)
        )
        if middle is not None and long.dte_days > short.dte_days
        else None
    )
    return TermStructureContext(
        tenors=tenors,
        short=short,
        medium=middle,
        long=long,
        short_long_spread=spread if isfinite(spread) else None,
        short_long_ratio=ratio if ratio is None or isfinite(ratio) else None,
        slope_per_day=slope if slope is None or isfinite(slope) else None,
        curvature=curvature if curvature is None or isfinite(curvature) else None,
    )
