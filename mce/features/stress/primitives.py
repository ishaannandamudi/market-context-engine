"""Causal completed-session bar and actual quoted-spread measurements."""

from dataclasses import dataclass
from datetime import date
from math import isfinite, log

from mce.calendar.service import MarketSessionService
from mce.errors import CalendarRangeError
from mce.models.market_data import BarObservation
from mce.replay.store import CausalDataStore


@dataclass(frozen=True)
class SessionMeasurement:
    volume: float | None
    dollar_volume: float | None
    session_return: float | None
    amihud: float | None
    gap_return: float | None
    stale: bool
    zero_volume: bool


def _completed_regular(
    store: CausalDataStore,
    symbol: str,
    opened: int,
    closed: int,
    cutoff: int,
    source: str,
) -> tuple[BarObservation, ...]:
    return tuple(
        bar
        for bar in store.bars(symbol, opened, cutoff + 1, cutoff)
        if bar.source == source
        and bar.bar_start_ns is not None
        and bar.bar_end_ns is not None
        and opened <= bar.bar_start_ns < bar.bar_end_ns <= min(closed, cutoff)
    )


def _previous_close(
    store: CausalDataStore,
    calendar: MarketSessionService,
    symbol: str,
    day: date,
    as_of_ns: int,
    source: str,
) -> float | None:
    try:
        previous = calendar.previous_trading_day(day)
    except CalendarRangeError:
        return None
    opened, closed = calendar.regular_session_bounds(previous)
    bars = _completed_regular(store, symbol, opened, closed, min(closed, as_of_ns), source)
    ending = tuple(bar.close for bar in bars if bar.bar_end_ns == closed and bar.close > 0)
    return ending[-1] if ending else None


def read_session(
    store: CausalDataStore,
    calendar: MarketSessionService,
    symbol: str,
    day: date,
    cutoff_ns: int,
    source: str,
    freshness_ns: int,
    cadence_ns: int,
) -> SessionMeasurement:
    opened, closed = calendar.regular_session_bounds(day)
    if cutoff_ns <= opened:
        return SessionMeasurement(None, None, None, None, None, False, False)
    bars = _completed_regular(store, symbol, opened, closed, min(cutoff_ns, closed), source)
    if not bars:
        latest = store.latest_bar(symbol, cutoff_ns)
        return SessionMeasurement(None, None, None, None, None, latest is not None, False)
    expected_end = opened + ((min(cutoff_ns, closed) - opened) // cadence_ns) * cadence_ns
    if min(cutoff_ns, closed) == closed:
        expected_end = closed
    if (
        bars[0].bar_start_ns != opened
        or bars[-1].bar_end_ns != expected_end
        or any(
            previous.bar_end_ns != following.bar_start_ns
            for previous, following in zip(bars, bars[1:], strict=False)
        )
        or any(
            bar.bar_end_ns is None
            or bar.bar_start_ns is None
            or bar.bar_end_ns - bar.bar_start_ns > cadence_ns
            for bar in bars
        )
    ):
        return SessionMeasurement(None, None, None, None, None, False, False)
    latest = bars[-1]
    assert latest.bar_end_ns is not None
    stale = cutoff_ns - latest.bar_end_ns > freshness_ns or (
        cutoff_ns - latest.effective_timestamp_ns > freshness_ns
    )
    if stale:
        return SessionMeasurement(None, None, None, None, None, True, False)
    volume = sum(bar.volume for bar in bars)
    dollar_volume = sum(bar.close * bar.volume for bar in bars)
    if not isfinite(volume) or not isfinite(dollar_volume):
        return SessionMeasurement(None, None, None, None, None, False, False)
    first = bars[0]
    opening_price = first.open if first.bar_start_ns == opened and first.open > 0 else None
    session_return = latest.close / opening_price - 1 if opening_price else None
    if session_return is not None and not isfinite(session_return):
        session_return = None
    amihud = (
        abs(session_return) / dollar_volume
        if session_return is not None and dollar_volume > 0
        else None
    )
    if amihud is not None and not isfinite(amihud):
        amihud = None
    prior = _previous_close(store, calendar, symbol, day, cutoff_ns, source)
    gap = log(opening_price) - log(prior) if opening_price and prior else None
    return SessionMeasurement(
        volume, dollar_volume, session_return, amihud, gap, False, volume == 0
    )


@dataclass(frozen=True)
class QuotedSpread:
    absolute: float | None
    relative: float | None
    stale: bool
    invalid: bool


def read_spread(
    store: CausalDataStore,
    symbol: str,
    as_of_ns: int,
    opened_ns: int,
    closed_ns: int,
    source: str,
    freshness_ns: int,
) -> QuotedSpread:
    quote = store.latest_quote(symbol, as_of_ns)
    if (
        quote is None
        or quote.source != source
        or not opened_ns <= quote.event_timestamp_ns <= min(as_of_ns, closed_ns)
    ):
        return QuotedSpread(None, None, False, False)
    if as_of_ns - quote.effective_timestamp_ns > freshness_ns:
        return QuotedSpread(None, None, True, False)
    if quote.bid <= 0 or quote.ask < quote.bid:
        return QuotedSpread(None, None, False, True)
    midpoint = (quote.ask + quote.bid) / 2
    if midpoint <= 0 or not isfinite(midpoint):
        return QuotedSpread(None, None, False, True)
    absolute = quote.ask - quote.bid
    relative = absolute / midpoint
    return QuotedSpread(absolute, relative, False, False)
