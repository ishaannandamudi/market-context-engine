"""Manual 100,000-point, 100-query transition-analytics benchmark."""

import argparse
import platform
from datetime import date
from time import perf_counter

from mce.calendar.service import MarketSessionService
from mce.config.settings import MCEConfig
from mce.features.regime.taxonomy import HeadlineRegime
from mce.features.regime_transition import (
    RegimeHistory,
    RegimeHistoryPoint,
    RegimeTransitionAnalyticsEngine,
)
from mce.models.enums import DataQuality

MINUTE = 60_000_000_000
FINGERPRINT = "a" * 64


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--observations", type=int, default=100_000)
    parser.add_argument("--queries", type=int, default=100)
    args = parser.parse_args()
    if not 1_000 <= args.observations <= 200_000 or not 1 <= args.queries <= 1_000:
        parser.error("observations must be 1,000–200,000 and queries 1–1,000")
    config = MCEConfig(_env_file=None)
    calendar = MarketSessionService(config.market)
    started = perf_counter()
    points = []
    day = date(2020, 1, 2)
    regimes = (
        HeadlineRegime.RISK_ON,
        HeadlineRegime.RANGE_BOUND,
        HeadlineRegime.RISK_OFF,
        HeadlineRegime.SYSTEMIC_STRESS,
    )
    while len(points) < args.observations:
        opened, closed = calendar.regular_session_bounds(day)
        for clock in range(opened + 4 * MINUTE, closed + 1, 4 * MINUTE):
            if len(points) == args.observations:
                break
            mode = (len(points) // 175) % len(regimes)
            points.append(
                RegimeHistoryPoint(
                    as_of_ns=clock,
                    available_at_ns=clock,
                    headline_regime=regimes[mode],
                    dimensions=(
                        ("positive", "neutral", "negative", "negative")[mode],
                        ("up", "sideways", "down", "strong_down")[mode],
                        ("normal", "low", "high", "extreme")[mode],
                        ("healthy", "normal", "thin", "stressed")[mode],
                        ("normal", "low", "high", "extreme")[mode],
                        ("broad_positive", "mixed", "narrow_negative", "broad_negative")[mode],
                        ("normal", "high", "normal", "extreme")[mode],
                        ("calm", "calm", "elevated", "systemic")[mode],
                        ("stable_leadership", "active_rotation", "mixed", "fragmented")[mode],
                    ),
                    evidence_strength=0.8,
                    data_quality=DataQuality.GOOD,
                    config_fingerprint=FINGERPRINT,
                )
            )
        day = calendar.next_trading_day(day)
    history = RegimeHistory.build(points)
    construction_seconds = perf_counter() - started
    engine = RegimeTransitionAnalyticsEngine(config, calendar)
    indices = tuple(
        args.observations - 1 - (args.queries - 1 - index) * 10 for index in range(args.queries)
    )
    if indices[0] < 0:
        parser.error("queries are too numerous for the selected observation count")
    began = perf_counter()
    results = [engine.compute(history, history.points[index].as_of_ns) for index in indices]
    compute_seconds = perf_counter() - began
    print(f"Python {platform.python_version()} on {platform.platform()}")
    print(
        f"observations={len(history.points)} queries={len(results)} "
        f"first={history.points[0].as_of_ns} last={history.points[-1].as_of_ns}"
    )
    print(f"construction_seconds={construction_seconds:.3f}")
    print(
        f"compute_seconds={compute_seconds:.3f} "
        f"seconds_per_snapshot={compute_seconds / len(results):.6f}"
    )


if __name__ == "__main__":
    main()
