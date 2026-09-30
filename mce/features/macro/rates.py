"""Supplied policy and Treasury rates, converted to percentage-point units."""

from math import isfinite
from statistics import mean

from mce.config.settings import MacroSettings
from mce.features.macro.contracts import DAY_NS, MacroCategory, MacroObservation, MacroUnit
from mce.features.macro.history import MacroHistoryIndex, SeriesKey
from mce.features.macro.models import MacroSeriesContext, PolicyRateContext, YieldCurveContext


def rate_percent(value: float, unit: MacroUnit) -> float:
    if unit == MacroUnit.PERCENT:
        return value
    if unit == MacroUnit.DECIMAL_RATE:
        return value * 100
    if unit == MacroUnit.BASIS_POINTS:
        return value / 100
    raise ValueError("rate unit is not convertible to percent")


def _finite_difference(left: float, right: float) -> float | None:
    value = left - right
    return value if isfinite(value) else None


def policy_context(
    series: tuple[MacroSeriesContext, ...],
    index: MacroHistoryIndex,
    as_of_ns: int,
    settings: MacroSettings,
) -> PolicyRateContext:
    eligible = tuple(
        item
        for item in series
        if item.category == MacroCategory.POLICY_RATE
        and item.observation is not None
        and not item.stale
        and (settings.policy_series_id is None or item.series_id == settings.policy_series_id)
    )
    if not eligible:
        return PolicyRateContext()
    selected = max(
        eligible,
        key=lambda item: (
            item.observation.period_end_ns if item.observation else -1,
            item.source,
            item.series_id,
        ),
    )
    current = selected.observation
    assert current is not None
    key = (selected.source, selected.series_id)
    state = index.observation_state(key, as_of_ns)
    values = tuple((item, rate_percent(item.value, item.unit)) for item in state)
    previous = values[-2][1] if len(values) > 1 else None
    current_value = rate_percent(current.value, current.unit)
    recent = tuple(
        pair for pair in values if pair[0].available_at_ns >= as_of_ns - settings.policy_lookback_ns
    )
    changes = tuple(
        pair
        for before, pair in zip(values, values[1:], strict=False)
        if pair[1] != before[1]
        and pair[0].available_at_ns >= as_of_ns - settings.policy_lookback_ns
    )
    last_change = next(
        (
            pair[0]
            for before, pair in reversed(tuple(zip(values, values[1:], strict=False)))
            if pair[1] != before[1]
        ),
        None,
    )
    return PolicyRateContext(
        series_id=selected.series_id,
        current_rate_percent=current_value,
        latest_change_pp=_finite_difference(current_value, previous)
        if previous is not None
        else None,
        trailing_change_count=len(changes),
        cumulative_change_pp=_finite_difference(current_value, recent[0][1])
        if len(recent) >= 2
        else None,
        seconds_since_last_change=(as_of_ns - last_change.available_at_ns) // 1_000_000_000
        if last_change is not None
        else None,
    )


def _reference(
    index: MacroHistoryIndex,
    key: SeriesKey,
    as_of_ns: int,
    cutoff_ns: int,
    tolerance_ns: int,
) -> MacroObservation | None:
    records = index.observation_state(key, as_of_ns)
    return next(
        (
            item
            for item in reversed(records)
            if item.period_end_ns <= cutoff_ns and cutoff_ns - item.period_end_ns <= tolerance_ns
        ),
        None,
    )


def yield_curve_context(
    series: tuple[MacroSeriesContext, ...],
    index: MacroHistoryIndex,
    as_of_ns: int,
    settings: MacroSettings,
) -> YieldCurveContext:
    candidates = tuple(
        item
        for item in series
        if item.category == MacroCategory.TREASURY_YIELD
        and item.observation is not None
        and not item.stale
    )
    if not candidates:
        return YieldCurveContext()
    anchor = max(
        item.observation.period_end_ns for item in candidates if item.observation is not None
    )
    chosen: dict[int, MacroSeriesContext] = {}
    for item in candidates:
        observation = item.observation
        assert observation is not None and observation.tenor_months is not None
        if anchor - observation.period_end_ns > settings.curve_max_observation_skew_ns:
            continue
        tenor = observation.tenor_months
        previous = chosen.get(tenor)
        if previous is None or (observation.period_end_ns, item.source, item.series_id) > (
            previous.observation.period_end_ns if previous.observation else -1,
            previous.source,
            previous.series_id,
        ):
            chosen[tenor] = item
    yields = {
        tenor: rate_percent(item.observation.value, item.observation.unit)
        for tenor, item in chosen.items()
        if item.observation is not None
    }

    def spread(long: int, short: int) -> float | None:
        return (
            _finite_difference(yields[long], yields[short])
            if long in yields and short in yields
            else None
        )

    two_ten = spread(120, 24)
    three_month_ten = spread(120, 3)
    thirty_five = spread(360, 60)
    curve = None
    if 24 in yields and 60 in yields and 120 in yields:
        weight = (60 - 24) / (120 - 24)
        baseline = yields[24] * (1 - weight) + yields[120] * weight
        curve = _finite_difference(yields[60], baseline)
    inversion_values = tuple(
        -value for value in (two_ten, three_month_ten) if value is not None and value < 0
    )
    inversion = (
        max(inversion_values, default=0.0)
        if two_ten is not None or three_month_ten is not None
        else None
    )
    changes: dict[tuple[int, int], float] = {}
    for tenor, item in chosen.items():
        observation = item.observation
        assert observation is not None
        key = (item.source, item.series_id)
        for days in (1, 5, 20):
            cutoff = observation.period_end_ns - days * DAY_NS
            reference = _reference(index, key, as_of_ns, cutoff, settings.cadence_age.daily_ns)
            if reference is not None:
                change = _finite_difference(
                    yields[tenor], rate_percent(reference.value, reference.unit)
                )
                if change is not None:
                    changes[(tenor, days)] = change
    five_day = tuple(
        value for (tenor, days), value in changes.items() if days == 5 and tenor in chosen
    )
    return YieldCurveContext(
        yields_percent={str(tenor): value for tenor, value in yields.items()},
        observed_period_end_ns=anchor,
        spread_2s10s_pp=two_ten,
        spread_3m10y_pp=three_month_ten,
        spread_30y5y_pp=thirty_five,
        level_10y_percent=yields.get(120),
        mean_level_percent=mean(yields.values()) if yields else None,
        curvature_2y5y10y_pp=curve,
        inversion_magnitude_pp=inversion,
        inverted=inversion > 0 if inversion is not None else None,
        change_10y_1d_pp=changes.get((120, 1)),
        change_10y_5d_pp=changes.get((120, 5)),
        change_10y_20d_pp=changes.get((120, 20)),
        parallel_shift_5d_pp=mean(five_day) if len(five_day) >= 2 else None,
        steepening_2s10s_5d_pp=_finite_difference(changes[(120, 5)], changes[(24, 5)])
        if (120, 5) in changes and (24, 5) in changes
        else None,
    )
