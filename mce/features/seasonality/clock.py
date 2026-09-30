"""XNYS session-relative clock and completed-bucket selection."""

from bisect import bisect_right
from datetime import date

from mce.calendar.service import MarketSessionService
from mce.config.settings import SeasonalitySettings
from mce.features.seasonality.models import SessionBucket, SessionRelativeTime


def bucket_grid(
    calendar: MarketSessionService, day: date, settings: SeasonalitySettings
) -> tuple[SessionBucket, ...]:
    opened, closed = calendar.regular_session_bounds(day)
    length = closed - opened
    if settings.bucket_mode == "wall_clock":
        assert settings.bucket_minutes is not None
        step = settings.bucket_minutes * 60_000_000_000
        boundaries = tuple(range(opened, closed, step)) + (closed,)
    else:
        assert settings.session_fraction_bins is not None
        bins = settings.session_fraction_bins
        # Fraction boundaries are snapped to complete bar slots. A boundary inside
        # a one-minute bar would otherwise include an incomplete bar in a bucket.
        slots = length // settings.bar_cadence_ns
        if bins > slots or length % settings.bar_cadence_ns:
            raise ValueError("fraction buckets require whole, nonempty bar slots")
        boundaries = tuple(
            opened + (slots * index // bins) * settings.bar_cadence_ns for index in range(bins)
        ) + (closed,)
    return tuple(
        SessionBucket(index, start, end)
        for index, (start, end) in enumerate(zip(boundaries, boundaries[1:], strict=False))
    )


def session_relative_time(
    calendar: MarketSessionService, as_of_ns: int, settings: SeasonalitySettings
) -> SessionRelativeTime | None:
    info = calendar.session_for_timestamp(as_of_ns)
    if (
        not info.is_trading_day
        or info.trading_date is None
        or info.regular_open_ns is None
        or info.regular_close_ns is None
        or not info.regular_open_ns <= as_of_ns <= info.regular_close_ns
    ):
        return None
    opened, closed = info.regular_open_ns, info.regular_close_ns
    buckets = bucket_grid(calendar, info.trading_date, settings)
    completed = bisect_right(tuple(bucket.end_ns for bucket in buckets), as_of_ns) - 1
    elapsed = as_of_ns - opened
    return SessionRelativeTime(
        info.trading_date,
        as_of_ns,
        opened,
        closed,
        elapsed,
        closed - as_of_ns,
        elapsed / (closed - opened),
        closed - opened,
        buckets[completed] if completed >= 0 else None,
        info.is_early_close,
        settings.bucket_mode,
    )
