"""Deterministic completed-bar panels for correlation and causality tests."""

from datetime import date
from math import exp, sin

from mce.calendar.service import MarketSessionService
from mce.models.market_data import BarObservation
from mce.replay.store import CausalDataStore

MINUTE = 60_000_000_000
DAY = date(2024, 6, 28)


def synthetic_returns(index: int) -> tuple[float, float, float, float]:
    common = 0.001 * sin(index * 0.37)
    noise = 0.0003 * sin(index * 1.19)
    return common + noise, common - noise, -common + 0.0002 * sin(index * 0.79), common


def dependence_moves(
    length: int, *, first: str, second: str, scale: float = 1.0
) -> tuple[tuple[float, float, float, float], ...]:
    """Independent Walsh patterns and a one-factor block for spike/collapse cases."""
    choices = {"independent", "common"}
    if first not in choices or second not in choices or length % 2:
        raise ValueError("need an even length and two known dependence states")
    moves = []
    for index in range(length):
        state = first if index < length // 2 else second
        sign = -1.0 if index % 2 else 1.0
        independent = (
            sign,
            -1.0 if (index // 2) % 2 else 1.0,
            -1.0 if (index // 4) % 2 else 1.0,
            (-1.0 if (index // 2) % 2 else 1.0) * sign,
        )
        values = (sign,) * 4 if state == "common" else independent
        moves.append(tuple(0.001 * scale * value for value in values))
    return tuple(moves)


def add_panel(
    store: CausalDataStore,
    calendar: MarketSessionService,
    day: date = DAY,
    *,
    minutes: int = 125,
    missing: frozenset[tuple[str, int]] = frozenset(),
    moves: tuple[tuple[float, ...], ...] | None = None,
    symbols: tuple[str, ...] = ("AAA", "BBB", "CCC", "DDD"),
    price_scale: float = 1.0,
) -> int:
    opened, _ = calendar.regular_session_bounds(day)
    prices = dict.fromkeys(symbols, 100.0 * price_scale)
    for minute in range(1, minutes + 1):
        end = opened + minute * MINUTE
        changes = moves[minute - 1] if moves is not None else synthetic_returns(minute)
        for index, symbol in enumerate(symbols):
            previous = prices[symbol]
            current = previous * exp(changes[index])
            if (symbol, minute) not in missing:
                store.add(
                    BarObservation(
                        symbol=symbol,
                        source="synthetic",
                        event_timestamp_ns=end,
                        effective_timestamp_ns=end,
                        bar_start_ns=end - MINUTE,
                        bar_end_ns=end,
                        open=previous,
                        high=max(previous, current),
                        low=min(previous, current),
                        close=current,
                        volume=1,
                    )
                )
            prices[symbol] = current
    return opened + minutes * MINUTE
