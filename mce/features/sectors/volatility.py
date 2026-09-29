"""Point-in-time daily sector composites and reused MCE-4 variance statistics."""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from math import isfinite, log1p, sqrt

from mce.calendar.service import MarketSessionService
from mce.config.settings import SectorSettings
from mce.errors import CalendarRangeError
from mce.features.breadth.prices import BreadthPriceReader
from mce.features.returns import simple_return
from mce.features.sectors.membership import SectorUniverseSnapshot
from mce.features.volatility.estimators import sample_variance
from mce.features.volatility.rolling import prior_distribution


@dataclass(frozen=True)
class SectorVolatility:
    short: float | None
    medium: float | None
    long: float | None
    percentile: float | None
    market: float | None
    relative: float | None


def _annualized(returns: tuple[float | None, ...], sessions: int, annual: int) -> float | None:
    tail = returns[-sessions:]
    if len(tail) != sessions or any(value is None or value <= -1 for value in tail):
        return None
    logs = tuple(log1p(value) for value in tail if value is not None)
    variance = sample_variance(logs)
    if variance is None:
        return None
    result = sqrt(variance * annual)
    return result if isfinite(result) else None


def daily_composites(
    reader: BreadthPriceReader,
    calendar: MarketSessionService,
    members_at: Callable[[int], SectorUniverseSnapshot],
    end_day: date | None,
    days: int,
    settings: SectorSettings,
) -> tuple[dict[str, tuple[float | None, ...]], tuple[float | None, ...]]:
    if end_day is None:
        return {}, ()
    endpoints: list[date] = []
    cursor = end_day
    for _ in range(days):
        endpoints.append(cursor)
        try:
            cursor = calendar.previous_trading_day(cursor)
        except CalendarRangeError:
            break
    endpoints.reverse()
    by_sector: dict[str, list[float | None]] = {}
    market: list[float | None] = []
    close_cache: dict[tuple[str, date, int], float | None] = {}

    def close(symbol: str, day: date, as_of_ns: int) -> float | None:
        key = (symbol, day, as_of_ns)
        if key not in close_cache:
            close_cache[key] = reader.session_close(symbol, day, as_of_ns)
        return close_cache[key]

    for day in endpoints:
        _, as_of_ns = calendar.regular_session_bounds(day)
        try:
            previous = calendar.previous_trading_day(day)
        except CalendarRangeError:
            previous = None
        snapshot = members_at(as_of_ns)
        grouped: dict[str, list[float | None]] = {}
        for item in snapshot.members:
            current = close(item.symbol, day, as_of_ns)
            prior = close(item.symbol, previous, as_of_ns) if previous is not None else None
            grouped.setdefault(item.sector, []).append(
                simple_return(prior, current) if prior is not None and current is not None else None
            )
        for sector in by_sector.keys() | grouped.keys():
            values = grouped.get(sector, [])
            valid = tuple(value for value in values if value is not None)
            usable = (
                (
                    len(values) >= settings.minimum_member_count
                    and len(valid) / len(values) >= settings.minimum_member_coverage
                )
                if values
                else False
            )
            by_sector.setdefault(sector, [None] * (len(market))).append(
                sum(value / len(valid) for value in valid) if usable and valid else None
            )
        current_market = close(settings.reference_symbol, day, as_of_ns)
        prior_market = (
            close(settings.reference_symbol, previous, as_of_ns) if previous is not None else None
        )
        market.append(
            simple_return(prior_market, current_market)
            if current_market is not None and prior_market is not None
            else None
        )
    return {sector: tuple(values) for sector, values in by_sector.items()}, tuple(market)


def volatility_context(
    sector_returns: tuple[float | None, ...],
    market_returns: tuple[float | None, ...],
    settings: SectorSettings,
    annual: int,
) -> SectorVolatility:
    short_n, medium_n, long_n = settings.volatility_windows_sessions
    short = _annualized(sector_returns, short_n, annual)
    medium = _annualized(sector_returns, medium_n, annual)
    long = _annualized(sector_returns, long_n, annual)
    market = _annualized(market_returns, medium_n, annual)
    prior = tuple(
        value
        for end in range(
            max(short_n, len(sector_returns) - settings.volatility_baseline_sessions),
            len(sector_returns),
        )
        if (value := _annualized(sector_returns[:end], short_n, annual)) is not None
    )
    distribution = (
        prior_distribution(short, prior, settings.minimum_volatility_baseline_samples)
        if short is not None
        else None
    )
    relative = medium / market if medium is not None and market is not None and market > 0 else None
    return SectorVolatility(
        short, medium, long, distribution[0] if distribution else None, market, relative
    )
