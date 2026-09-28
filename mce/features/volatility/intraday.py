"""Causal intraday squared-return variation on actual exchange sessions."""

from dataclasses import dataclass
from datetime import date
from math import isfinite, sqrt

from mce.calendar.service import MarketSessionService
from mce.features.returns import log_return
from mce.features.volatility.estimators import intraday_variation
from mce.models.market_data import BarObservation
from mce.replay.store import CausalDataStore


@dataclass(frozen=True)
class IntradayMeasure:
    annualized_volatility: float | None
    realized_variance: float | None
    bipower_variation: float | None
    jump_variation: float | None
    jump_share: float | None
    downside_semivariance: float | None
    upside_semivariance: float | None
    downside_share: float | None
    observed_return_count: int
    expected_return_count: int
    coverage_ratio: float
    duplicate_interval: bool


def completed_intraday_bars(
    store: CausalDataStore,
    calendar: MarketSessionService,
    source: str,
    symbol: str,
    day: date,
    as_of_ns: int,
    cadence_ns: int,
) -> tuple[tuple[BarObservation, ...], bool]:
    opened, closed = calendar.regular_session_bounds(day)
    selected: dict[int, BarObservation] = {}
    duplicate = False
    for bar in store.bars(symbol, opened, min(closed, as_of_ns) + 1, as_of_ns):
        if (
            bar.source != source
            or bar.bar_start_ns is None
            or bar.bar_end_ns is None
            or bar.bar_start_ns < opened
            or bar.bar_end_ns > closed
            or bar.bar_end_ns - bar.bar_start_ns != cadence_ns
        ):
            continue
        if bar.bar_start_ns in selected:
            duplicate = True
            prior = selected[bar.bar_start_ns]
            if (bar.effective_timestamp_ns, bar.close) > (
                prior.effective_timestamp_ns,
                prior.close,
            ):
                selected[bar.bar_start_ns] = bar
        else:
            selected[bar.bar_start_ns] = bar
    return tuple(selected[key] for key in sorted(selected)), duplicate


def measure_intraday(
    bars: tuple[BarObservation, ...],
    regular_open_ns: int,
    regular_close_ns: int,
    as_of_ns: int,
    window_ns: int | None,
    cadence_ns: int,
    trading_sessions_per_year: int,
    duplicate_interval: bool = False,
) -> IntradayMeasure:
    if (
        cadence_ns <= 0
        or trading_sessions_per_year <= 0
        or (window_ns is not None and window_ns <= 0)
    ):
        raise ValueError("cadence, annual sessions and window must be positive")
    completed_cutoff = min(as_of_ns, regular_close_ns)
    boundary = regular_open_ns if window_ns is None else max(regular_open_ns, as_of_ns - window_ns)
    expected = max(0, (completed_cutoff - boundary) // cadence_ns)
    eligible = tuple(
        bar
        for bar in bars
        if bar.bar_end_ns is not None
        and bar.bar_end_ns <= completed_cutoff
        and bar.effective_timestamp_ns <= as_of_ns
    )
    if not eligible:
        return IntradayMeasure(
            None, None, None, None, None, None, None, None, 0, expected, 0.0, duplicate_interval
        )
    end = eligible[-1].bar_end_ns
    assert end is not None
    if boundary >= end or expected == 0:
        return IntradayMeasure(
            None, None, None, None, None, None, None, None, 0, expected, 0.0, duplicate_interval
        )
    current = tuple(
        bar for bar in eligible if bar.bar_end_ns is not None and bar.bar_end_ns > boundary
    )
    anchor = next(
        (
            bar
            for bar in reversed(eligible)
            if bar.bar_end_ns is not None
            and bar.bar_end_ns <= boundary
            and bar.bar_end_ns >= boundary - cadence_ns
        ),
        None,
    )
    returns: list[float] = []
    previous_close = anchor.close if anchor else None
    if boundary == regular_open_ns and current:
        previous_close = current[0].open
    for bar in current:
        value = log_return(previous_close, bar.close)
        if value is None:
            returns = []
            break
        returns.append(value)
        previous_close = bar.close
    duration = expected * cadence_ns
    coverage = min(1.0, len(returns) / expected) if expected else 0.0
    variation = intraday_variation(tuple(returns))
    if variation is None:
        return IntradayMeasure(
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            len(returns),
            expected,
            coverage,
            duplicate_interval,
        )
    rv, bipower, downside, upside = variation
    annual_variance = (
        rv * trading_sessions_per_year * (regular_close_ns - regular_open_ns) / duration
    )
    annualized = sqrt(annual_variance) if isfinite(annual_variance) else None
    if annualized is not None and not isfinite(annualized):
        annualized = None
    jump = max(rv - bipower, 0.0) if bipower is not None else None
    return IntradayMeasure(
        annualized,
        rv,
        bipower,
        jump,
        jump / rv if jump is not None and rv > 0 else None,
        downside,
        upside,
        downside / rv if rv > 0 else None,
        len(returns),
        expected,
        coverage,
        duplicate_interval,
    )
