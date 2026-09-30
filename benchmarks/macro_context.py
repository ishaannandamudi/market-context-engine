"""Manual MCE-18 indexed macro/vintage/event benchmark; no external provider."""

import argparse
import platform
from time import perf_counter

from mce.features.macro import (
    MacroCategory,
    MacroContextFeatureEngine,
    MacroHistoryIndex,
    SeriesCadence,
)
from mce.features.macro.contracts import DAY_NS, EventImportance, MacroEventType
from mce.features.macro.events import event_context
from mce.features.macro.rates import yield_curve_context
from mce.features.macro.surprises import surprise_context
from tests.fixtures.macro_scenarios import (
    BASE,
    HOUR,
    MINUTE,
    batch,
    event,
    full_context,
    macro_config,
    observation,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshots", type=int, default=120)
    args = parser.parse_args()
    if args.snapshots < 100:
        parser.error("benchmark requires at least 100 snapshots")
    baseline = full_context()
    historic = tuple(
        observation(
            series_id,
            MacroCategory.TREASURY_YIELD,
            4.0 + tenor / 1000 + day / 10000,
            period_end=BASE - (day + 22) * DAY_NS,
            released=BASE - (day + 22) * DAY_NS,
            cadence=SeriesCadence.DAILY,
            tenor_months=tenor,
        )
        for series_id, tenor in (
            ("UST3M", 3),
            ("UST2Y", 24),
            ("UST5Y", 60),
            ("UST10Y", 120),
            ("UST30Y", 360),
        )
        for day in range(200)
    )
    calendar = tuple(
        event(
            f"economic-{number}",
            MacroEventType.CPI if number % 3 == 0 else MacroEventType.GDP,
            BASE + (number + 1) * HOUR,
            importance=EventImportance.HIGH if number % 5 == 0 else EventImportance.MEDIUM,
        )
        for number in range(300)
    )
    first = batch(
        observations=(*baseline.observations, *historic),
        releases=baseline.releases,
        events=(*baseline.scheduled_events, *calendar),
    )
    revisions = tuple(
        batch(
            clock=BASE + (number + 1) * MINUTE,
            observations=(
                observation(
                    "CPI_YOY",
                    MacroCategory.INFLATION,
                    3.5 + number / 1000,
                    period_end=BASE - 30 * DAY_NS,
                    released=BASE - DAY_NS,
                    vintage=BASE + (number + 1) * MINUTE,
                ),
            ),
            events=(
                event(
                    "economic-299",
                    MacroEventType.GDP,
                    BASE + (300 + number + 1) * HOUR,
                    announced=BASE + (number + 1) * MINUTE,
                    previous=BASE + (300 + number) * HOUR,
                    importance=EventImportance.MEDIUM,
                ),
            ),
        )
        for number in range(20)
    )
    config = macro_config()
    started = perf_counter()
    index = MacroHistoryIndex((first, *revisions))
    indexing_seconds = perf_counter() - started
    engine = MacroContextFeatureEngine(config)
    clocks = tuple(BASE + number * MINUTE for number in range(args.snapshots))
    started = perf_counter()
    results = tuple(engine.compute(index, clock) for clock in clocks)
    total_seconds = perf_counter() - started
    final = results[-1]
    started = perf_counter()
    observed = index.observation((first.source, "CPI_YOY"), clocks[-1])
    vintage_seconds = perf_counter() - started
    started = perf_counter()
    calendar_state = index.calendar(clocks[-1])
    assert calendar_state is not None
    upcoming = calendar_state.upcoming(clocks[-1], DAY_NS)
    event_lookup_seconds = perf_counter() - started
    started = perf_counter()
    events_context = event_context(
        calendar_state, (), clocks[-1], config.features.macro, frozenset((first.source,))
    )
    event_seconds = perf_counter() - started
    started = perf_counter()
    curve = yield_curve_context(final.per_series, index, clocks[-1], config.features.macro)
    curve_seconds = perf_counter() - started
    started = perf_counter()
    current_release = index.release((first.source, "CPI_YOY"), clocks[-1])
    assert current_release is not None
    prior = index.prior_releases(
        (first.source, "CPI_YOY"), current_release.period_end_ns, clocks[-1], 60
    )
    surprise = surprise_context(current_release, prior, 3)
    surprise_seconds = perf_counter() - started
    print(f"Python {platform.python_version()} on {platform.platform()}")
    print(
        f"observations={len(first.observations) + len(revisions)} "
        f"events={len(first.scheduled_events)} schedule_revisions={len(revisions)} "
        f"releases={len(first.releases)} snapshots={len(results)}"
    )
    print(
        f"indexing_seconds={indexing_seconds:.6f} vintage_lookup_seconds={vintage_seconds:.6f} "
        f"event_lookup_seconds={event_lookup_seconds:.6f} "
        f"event_context_seconds={event_seconds:.6f} curve_seconds={curve_seconds:.6f} "
        f"surprise_seconds={surprise_seconds:.6f}"
    )
    print(
        f"engine_seconds_per_snapshot={total_seconds / len(results):.6f} "
        f"upcoming_24h={len(upcoming)} "
        f"current_vintage={observed.vintage_at_ns if observed else None} "
        f"curve_2s10s={curve.spread_2s10s_pp} surprise={surprise.raw} "
        f"proximity={events_context.proximity_score} "
        f"macro_stress={final.stress.score if final.stress else None}"
    )


if __name__ == "__main__":
    main()
