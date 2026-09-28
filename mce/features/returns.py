"""Finite, explicit bar-close return formulas."""

from math import isfinite, log


def simple_return(start_price: float | None, end_price: float | None) -> float | None:
    """(end/start)-1; a missing or nonpositive reference has no return."""
    if start_price is None or end_price is None:
        return None
    if start_price <= 0 or end_price < 0:
        return None
    result = end_price / start_price - 1.0
    return result if isfinite(result) else None


def log_return(start_price: float | None, end_price: float | None) -> float | None:
    """Natural log(end/start); both prices must be strictly positive."""
    if start_price is None or end_price is None or start_price <= 0 or end_price <= 0:
        return None
    result = log(end_price) - log(start_price)
    return result if isfinite(result) else None
