"""Calendar and bucket edge cases for the intraday seasonal clock."""

from datetime import date, datetime
from zoneinfo import ZoneInfo

import pytest
from pydantic import ValidationError

from mce.calendar.service import MarketSessionService
from mce.config.settings import SeasonalitySettings
from mce.features.seasonality.clock import bucket_grid, session_relative_time
from mce.reproducibility import config_fingerprint
from tests.fixtures.seasonal_scenarios import MINUTE, seasonal_config


def test_wall_clock_completed_buckets_and_early_close() -> None:
    calendar = MarketSessionService()
    settings = SeasonalitySettings()
    normal = date(2024, 11, 27)
    early = date(2024, 11, 29)
    opened, closed = calendar.regular_session_bounds(normal)
    grid = bucket_grid(calendar, normal, settings)
    assert len(grid) == 78 and grid[0].start_ns == opened and grid[-1].end_ns == closed
    assert session_relative_time(calendar, opened, settings).completed_bucket is None  # type: ignore[union-attr]
    at_boundary = session_relative_time(calendar, opened + 5 * MINUTE, settings)
    assert at_boundary is not None and at_boundary.completed_bucket == grid[0]
    inside = session_relative_time(calendar, opened + 8 * MINUTE, settings)
    assert inside is not None and inside.completed_bucket == grid[0]
    assert inside.elapsed_fraction == pytest.approx(8 / 390)
    assert session_relative_time(calendar, opened - MINUTE, settings) is None
    assert session_relative_time(calendar, closed + MINUTE, settings) is None
    early_open, early_close = calendar.regular_session_bounds(early)
    early_grid = bucket_grid(calendar, early, settings)
    assert len(early_grid) == 42 and early_close - early_open == 210 * MINUTE
    early_time = session_relative_time(calendar, early_close, settings)
    assert early_time is not None and early_time.is_early_close
    assert early_time.completed_bucket == early_grid[-1]
    assert early_time.elapsed_fraction == 1
    assert (
        session_relative_time(
            calendar,
            int(datetime(2024, 11, 28, 10, tzinfo=ZoneInfo("America/New_York")).timestamp() * 1e9),
            settings,
        )
        is None
    )


def test_fraction_buckets_align_to_complete_bars_and_dst() -> None:
    calendar = MarketSessionService()
    settings = SeasonalitySettings(
        bucket_mode="session_fraction", bucket_minutes=None, session_fraction_bins=7
    )
    normal = bucket_grid(calendar, date(2024, 3, 8), settings)
    after_dst = bucket_grid(calendar, date(2024, 3, 11), settings)
    early = bucket_grid(calendar, date(2024, 11, 29), settings)
    assert len(normal) == len(after_dst) == len(early) == 7
    assert all((bucket.end_ns - bucket.start_ns) % MINUTE == 0 for bucket in normal + early)
    assert [bucket.end_ns - bucket.start_ns for bucket in normal] == [
        bucket.end_ns - bucket.start_ns for bucket in after_dst
    ]
    assert normal[0].start_ns == int(
        datetime(2024, 3, 8, 9, 30, tzinfo=ZoneInfo("America/New_York")).timestamp() * 1e9
    )
    assert after_dst[0].start_ns == int(
        datetime(2024, 3, 11, 9, 30, tzinfo=ZoneInfo("America/New_York")).timestamp() * 1e9
    )
    assert after_dst[0].start_ns - normal[0].start_ns == 71 * 60 * MINUTE
    assert (normal[0].end_ns - normal[0].start_ns) != (early[0].end_ns - early[0].start_ns)
    assert sum(bucket.end_ns - bucket.start_ns for bucket in early) == 210 * MINUTE


def test_bucket_config_validation_and_fingerprint() -> None:
    with pytest.raises(ValidationError, match="unique"):
        SeasonalitySettings(universe=("AAA", "AAA"))
    with pytest.raises(ValidationError, match="minimum history"):
        SeasonalitySettings(baseline_sessions=2, minimum_history_sessions=3)
    with pytest.raises(ValidationError, match="fraction buckets"):
        SeasonalitySettings(bucket_mode="session_fraction", session_fraction_bins=10)
    with pytest.raises(ValidationError):
        SeasonalitySettings(bucket_minutes=0)
    config = seasonal_config()
    changed = seasonal_config(
        bucket_mode="session_fraction", bucket_minutes=None, session_fraction_bins=10
    )
    assert config_fingerprint(config) != config_fingerprint(changed)


def test_one_and_fifteen_minute_buckets_and_weekend() -> None:
    calendar = MarketSessionService()
    day = date(2024, 1, 8)
    opened, closed = calendar.regular_session_bounds(day)
    one = bucket_grid(calendar, day, SeasonalitySettings(bucket_minutes=1))
    fifteen = bucket_grid(calendar, day, SeasonalitySettings(bucket_minutes=15))
    assert len(one) == 390 and len(fifteen) == 26
    assert one[0].end_ns == opened + MINUTE
    assert fifteen[-1].end_ns == closed
    almost_close = session_relative_time(calendar, closed - 1, SeasonalitySettings())
    assert almost_close is not None and almost_close.completed_bucket is not None
    assert almost_close.completed_bucket.index == 76
    weekend = int(datetime(2024, 1, 6, 10, tzinfo=ZoneInfo("America/New_York")).timestamp() * 1e9)
    assert session_relative_time(calendar, weekend, SeasonalitySettings()) is None
