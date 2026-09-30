"""Deterministic common, block, rotating and independent return panels."""

from datetime import date
from math import exp, sin

from mce.calendar.service import MarketSessionService
from mce.config.settings import MCEConfig
from mce.features.factors.models import ReturnPanel
from mce.models.market_data import BarObservation
from mce.replay.store import CausalDataStore


def factor_config(
    *, symbols: tuple[str, ...] = ("AAA", "BBB", "CCC", "DDD"), window: int = 20
) -> MCEConfig:
    config = MCEConfig(_env_file=None)
    factor = config.features.factor.model_copy(
        update={
            "universe": symbols,
            "window_sessions": window,
            "minimum_samples": max(5, window // 2),
            "minimum_symbols": min(3, len(symbols)),
            "factor_count": min(3, len(symbols)),
        }
    )
    return config.model_copy(
        update={"features": config.features.model_copy(update={"factor": factor})}
    )


def factor_moves(kind: str, rows: int, symbols: int) -> tuple[tuple[float, ...], ...]:
    if kind not in {
        "dominant",
        "blocks",
        "independent",
        "weak",
        "changing_dominance",
        "rotation",
        "high_residual",
        "systemic",
        "sector_specific",
    }:
        raise ValueError("unknown factor scenario")
    values = []
    for t in range(rows):
        common = sin(t * 0.49) + 0.5 * sin(t * 0.17)
        block = sin(t * 0.31 + 1.3)
        independent = tuple(sin(t * (0.13 + i * 0.071) + i) for i in range(symbols))
        if kind in ("dominant", "systemic"):
            row = tuple(0.01 * (common + 0.03 * independent[i]) for i in range(symbols))
        elif kind in ("blocks", "sector_specific"):
            row = tuple(
                0.01 * ((common if i < symbols // 2 else block) + 0.02 * independent[i])
                for i in range(symbols)
            )
        elif kind == "changing_dominance":
            strength = 0.1 if t < rows // 2 else 1.0
            row = tuple(0.01 * (strength * common + 0.5 * independent[i]) for i in range(symbols))
        elif kind == "rotation":
            row = tuple(
                0.01 * ((common if (i + (t >= rows // 2)) % 2 else block) + 0.02 * independent[i])
                for i in range(symbols)
            )
        else:
            strength = 0.05 if kind in ("independent", "high_residual") else 0.2
            row = tuple(0.01 * (strength * common + independent[i]) for i in range(symbols))
        values.append(row)
    return tuple(values)


def factor_panel(
    kind: str = "dominant", rows: int = 40, symbols: tuple[str, ...] = ("AAA", "BBB", "CCC", "DDD")
) -> ReturnPanel:
    config = factor_config(symbols=symbols, window=max(rows, 5))
    calendar = MarketSessionService(config.market)
    day = date(2024, 1, 2)
    timestamps = []
    for index in range(rows):
        timestamps.append(calendar.regular_session_bounds(day)[1])
        if index + 1 < rows:
            day = calendar.next_trading_day(day)
    return ReturnPanel(
        timestamps[-1],
        tuple(timestamps),
        tuple(sorted(symbols)),
        tuple(sorted(symbols)),
        factor_moves(kind, rows, len(symbols)),
        rows,
    )


def factor_store(
    kind: str = "dominant",
    rows: int = 21,
    symbols: tuple[str, ...] = ("AAA", "BBB", "CCC", "DDD"),
    missing: frozenset[tuple[str, int]] = frozenset(),
) -> tuple[CausalDataStore, tuple[int, ...]]:
    config = factor_config(symbols=symbols, window=max(5, rows - 1))
    calendar = MarketSessionService(config.market)
    store = CausalDataStore()
    prices = dict.fromkeys(symbols, 100.0)
    day = date(2024, 1, 2)
    timestamps = []
    moves = factor_moves(kind, rows, len(symbols))
    for index in range(rows):
        opened, closed = calendar.regular_session_bounds(day)
        timestamps.append(closed)
        for column, symbol in enumerate(symbols):
            earlier = prices[symbol]
            current = earlier * exp(moves[index][column])
            if (symbol, index) not in missing:
                store.add(
                    BarObservation(
                        symbol=symbol,
                        source="synthetic",
                        event_timestamp_ns=closed,
                        effective_timestamp_ns=closed,
                        bar_start_ns=opened,
                        bar_end_ns=closed,
                        open=earlier,
                        high=max(earlier, current),
                        low=min(earlier, current),
                        close=current,
                        volume=1,
                    )
                )
            prices[symbol] = current
        if index + 1 < rows:
            day = calendar.next_trading_day(day)
    return store, tuple(timestamps)
