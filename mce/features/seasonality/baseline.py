"""Explicit prior-session index and robust same-bucket historical baselines."""

from datetime import date
from math import isfinite

import numpy as np

from mce.calendar.service import MarketSessionService
from mce.config.settings import MCEConfig
from mce.errors import CalendarRangeError
from mce.features.breadth.membership import MembershipTimeline, UniverseProvider
from mce.features.correlation.models import CorrelationFeatureSet
from mce.features.seasonality.clock import bucket_grid, session_relative_time
from mce.features.seasonality.models import (
    MetricName,
    SeasonalBaseline,
    SeasonalHistoryIndex,
    SeasonalObservation,
)
from mce.features.seasonality.readers import read_session_metrics
from mce.models.enums import DataQuality
from mce.replay.store import CausalDataStore
from mce.reproducibility import config_fingerprint


def prior_session_dates(calendar: MarketSessionService, day: date, count: int) -> tuple[date, ...]:
    cursor = day
    dates: list[date] = []
    for _ in range(count):
        try:
            cursor = calendar.previous_trading_day(cursor)
        except CalendarRangeError:
            break
        dates.append(cursor)
    return tuple(sorted(dates))


def baseline(
    history: SeasonalHistoryIndex, metric: MetricName, symbol: str | None, bucket_index: int
) -> SeasonalBaseline:
    rows = history.samples(metric, symbol, bucket_index)
    values = np.asarray([item.value for item in rows], dtype=np.float64)
    count = len(values)
    expected = len(history.prior_dates)
    if not count:
        return SeasonalBaseline(
            metric, symbol, bucket_index, 0, expected, 0.0, None, None, None, None, None, None
        )
    ordered = sorted(float(value) for value in values)
    center = finite_median(ordered)
    with np.errstate(over="ignore", invalid="ignore"):
        mean = _finite_or_none(float(np.mean(values)))
        std = _finite_or_none(float(np.std(values, ddof=1))) if count > 1 else 0.0
        q25 = _finite_or_none(float(np.quantile(values, 0.25, method="linear")))
        q75 = _finite_or_none(float(np.quantile(values, 0.75, method="linear")))
    return SeasonalBaseline(
        metric,
        symbol,
        bucket_index,
        count,
        expected,
        count / expected if expected else 0.0,
        center,
        mean,
        std,
        _finite_or_none(finite_median(sorted(float(abs(value - center)) for value in values))),
        q25,
        q75,
    )


def finite_median(values: list[float]) -> float:
    """Median of already sorted finite values without overflowing an even pair."""
    midpoint = len(values) // 2
    return values[midpoint] if len(values) % 2 else values[midpoint - 1] / 2 + values[midpoint] / 2


def _finite_or_none(value: float) -> float | None:
    return value if isfinite(value) else None


class SeasonalityHistoryBuilder:
    def __init__(
        self,
        config: MCEConfig,
        universe_provider: UniverseProvider | None = None,
        calendar: MarketSessionService | None = None,
    ) -> None:
        self.config = config
        self.settings = config.features.seasonality
        if calendar is not None and calendar.settings.model_dump(
            mode="python"
        ) != config.market.model_dump(mode="python"):
            raise ValueError("calendar settings must match result-relevant market configuration")
        self.calendar = calendar or MarketSessionService(config.market)
        self.universe_provider = universe_provider or MembershipTimeline.from_symbols(
            self.settings.universe
        )

    def build(
        self,
        store: CausalDataStore,
        as_of_ns: int,
        *,
        correlation_history: tuple[CorrelationFeatureSet, ...] = (),
    ) -> SeasonalHistoryIndex:
        info = self.calendar.session_for_timestamp(as_of_ns)
        if not info.is_trading_day or info.trading_date is None:
            raise ValueError("seasonal history target must be an XNYS trading day")
        day = info.trading_date
        prior = prior_session_dates(self.calendar, day, self.settings.baseline_sessions)
        current = self.universe_provider.members(as_of_ns)
        if current.as_of_ns != as_of_ns:
            raise ValueError("seasonal universe provider returned a different clock")
        symbols = {member.symbol for member in current.members}
        rows: list[SeasonalObservation] = []
        for historical_day in prior:
            _, close = self.calendar.regular_session_bounds(historical_day)
            grid = bucket_grid(self.calendar, historical_day, self.settings)
            historical = self.universe_provider.members(close)
            if historical.as_of_ns != close:
                raise ValueError("seasonal universe provider returned a different historical clock")
            members = symbols & {member.symbol for member in historical.members}
            for symbol in sorted(members):
                rows.extend(
                    read_session_metrics(
                        store,
                        self.calendar,
                        self.config,
                        symbol,
                        historical_day,
                        as_of_ns,
                        current=False,
                        buckets=grid,
                    ).observations
                )
        for context in correlation_history:
            if context.as_of_ns > as_of_ns or context.data_quality in (
                DataQuality.POOR,
                DataQuality.UNUSABLE,
            ):
                continue
            clock = session_relative_time(self.calendar, context.as_of_ns, self.settings)
            if (
                clock is None
                or clock.trading_date not in prior
                or clock.completed_bucket is None
                or clock.completed_bucket.end_ns != context.as_of_ns
                or context.average_pairwise_correlation is None
            ):
                continue
            rows.append(
                SeasonalObservation(
                    "correlation",
                    None,
                    clock.trading_date,
                    clock.completed_bucket.index,
                    context.as_of_ns,
                    context.as_of_ns,
                    context.average_pairwise_correlation,
                )
            )
        return SeasonalHistoryIndex(
            day,
            prior,
            as_of_ns,
            config_fingerprint(self.config),
            tuple(rows),
        )
