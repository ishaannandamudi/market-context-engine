"""Deterministic macro vintages, releases, yields, and event-calendar scenarios."""

from mce.config.settings import MacroSettings, MCEConfig
from mce.features.macro.contracts import (
    DAY_NS,
    EventImportance,
    MacroCategory,
    MacroContextInput,
    MacroEventType,
    MacroObservation,
    MacroRelease,
    MacroUnit,
    RevisionStatus,
    ScheduledMacroEvent,
    SeriesCadence,
)

BASE = 1_800_000_000_000_000_000
HOUR = 3_600_000_000_000
MINUTE = 60_000_000_000
SOURCE = "synthetic-macro"


def macro_config(*, minimum_history: int = 3) -> MCEConfig:
    base = MCEConfig(_env_file=None)
    settings = MacroSettings(minimum_history=minimum_history)
    return base.model_copy(
        update={"features": base.features.model_copy(update={"macro": settings})}
    )


def observation(
    series_id: str,
    category: MacroCategory,
    value: float,
    *,
    period_end: int,
    released: int,
    unit: MacroUnit = MacroUnit.PERCENT,
    cadence: SeriesCadence = SeriesCadence.MONTHLY,
    tenor_months: int | None = None,
    vintage: int | None = None,
    status: RevisionStatus = RevisionStatus.FINAL,
) -> MacroObservation:
    return MacroObservation(
        source=SOURCE,
        series_id=series_id,
        category=category,
        cadence=cadence,
        value=value,
        unit=unit,
        period_end_ns=period_end,
        release_at_ns=released,
        available_at_ns=vintage or released,
        vintage_at_ns=vintage or released,
        revision_status=status,
        tenor_months=tenor_months,
    )


def release(
    series_id: str,
    category: MacroCategory,
    actual: float,
    consensus: float | None,
    *,
    period_end: int,
    released: int,
    unit: MacroUnit = MacroUnit.PERCENT,
    cadence: SeriesCadence = SeriesCadence.MONTHLY,
    event_id: str | None = None,
    vintage: int | None = None,
    status: RevisionStatus = RevisionStatus.FINAL,
) -> MacroRelease:
    return MacroRelease(
        source=SOURCE,
        event_id=event_id or f"{series_id}-{period_end}",
        series_id=series_id,
        category=category,
        cadence=cadence,
        actual=actual,
        consensus=consensus,
        consensus_available_at_ns=released - HOUR if consensus is not None else None,
        unit=unit,
        period_end_ns=period_end,
        release_at_ns=released,
        available_at_ns=vintage or released,
        vintage_at_ns=vintage or released,
        revision_status=status,
    )


def event(
    event_id: str,
    event_type: MacroEventType,
    scheduled: int,
    *,
    announced: int = BASE - DAY_NS,
    importance: EventImportance = EventImportance.HIGH,
    previous: int | None = None,
) -> ScheduledMacroEvent:
    return ScheduledMacroEvent(
        source=SOURCE,
        event_id=event_id,
        event_type=event_type,
        scheduled_at_ns=scheduled,
        calendar_available_at_ns=announced,
        importance=importance,
        previous_scheduled_at_ns=previous,
    )


def batch(
    *,
    clock: int = BASE,
    observations: tuple[MacroObservation, ...] = (),
    releases: tuple[MacroRelease, ...] = (),
    events: tuple[ScheduledMacroEvent, ...] = (),
    calendar_known: bool = True,
) -> MacroContextInput:
    return MacroContextInput(
        macro_context_schema_version="1.0",
        source=SOURCE,
        source_version="1.2.3",
        source_fingerprint=f"synthetic-{clock}",
        as_of_ns=clock,
        available_at_ns=clock,
        observations=observations,
        releases=releases,
        scheduled_events=events,
        calendar_coverage_declared=calendar_known,
    )


def full_context() -> MacroContextInput:
    observations: list[MacroObservation] = []
    releases: list[MacroRelease] = []
    macro_series = (
        ("CPI_YOY", MacroCategory.INFLATION, 3.5, 3.2, MacroUnit.PERCENT),
        ("GDP_GROWTH", MacroCategory.GROWTH, 2.1, 2.5, MacroUnit.ANNUALIZED_RATE),
        ("NFP", MacroCategory.EMPLOYMENT, 250, 200, MacroUnit.THOUSANDS),
        ("UNEMPLOYMENT", MacroCategory.UNEMPLOYMENT, 4.2, 4.0, MacroUnit.PERCENT),
    )
    for series_id, category, actual, consensus, unit in macro_series:
        for i in range(6, -1, -1):
            period = BASE - (i + 1) * 30 * DAY_NS
            released = BASE - i * 30 * DAY_NS - DAY_NS
            prior_actual = actual if i == 0 else actual - i * 0.05
            prior_consensus = consensus if i == 0 else prior_actual - ((i % 3) - 1) * 0.08
            observations.append(
                observation(
                    series_id,
                    category,
                    prior_actual,
                    period_end=period,
                    released=released,
                    unit=unit,
                    cadence=SeriesCadence.QUARTERLY
                    if category == MacroCategory.GROWTH
                    else SeriesCadence.MONTHLY,
                )
            )
            releases.append(
                release(
                    series_id,
                    category,
                    prior_actual,
                    prior_consensus,
                    period_end=period,
                    released=released,
                    unit=unit,
                    cadence=SeriesCadence.QUARTERLY
                    if category == MacroCategory.GROWTH
                    else SeriesCadence.MONTHLY,
                    event_id=f"{series_id}-{i}",
                )
            )
    for series_id, tenor, current in (
        ("UST3M", 3, 4.5),
        ("UST2Y", 24, 4.0),
        ("UST5Y", 60, 4.2),
        ("UST10Y", 120, 4.4),
        ("UST30Y", 360, 4.8),
    ):
        for days, value in (
            (21, current - 0.30),
            (6, current - 0.20),
            (2, current - 0.10),
            (1, current),
        ):
            period = BASE - days * DAY_NS
            observations.append(
                observation(
                    series_id,
                    MacroCategory.TREASURY_YIELD,
                    value,
                    period_end=period,
                    released=period,
                    cadence=SeriesCadence.DAILY,
                    tenor_months=tenor,
                )
            )
    for days, value in ((60, 4.0), (30, 4.0), (1, 4.25)):
        period = BASE - days * DAY_NS
        observations.append(
            observation(
                "FED_TARGET_MID",
                MacroCategory.POLICY_RATE,
                value,
                period_end=period,
                released=period,
                cadence=SeriesCadence.EVENT_DRIVEN,
            )
        )
    events = (
        event("cpi-next", MacroEventType.CPI, BASE + 5 * MINUTE),
        event("fomc-next", MacroEventType.FOMC_RATE_DECISION, BASE + 30 * MINUTE),
        event(
            "auction-next",
            MacroEventType.TREASURY_AUCTION,
            BASE + 30 * HOUR,
            importance=EventImportance.MEDIUM,
        ),
    )
    return batch(observations=tuple(observations), releases=tuple(releases), events=events)
