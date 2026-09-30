"""Transparent options-specific evidence; no reuse of MCE-9 stress score."""

from mce.config.settings import OptionsContextSettings
from mce.features.options_context.models import (
    HistoricalComparison,
    OptionsStressContext,
    TermStructureContext,
)


def _elevated(history: HistoricalComparison | None) -> float | None:
    if history is None or history.percentile is None:
        return None
    return max(0.0, 2 * history.percentile - 1)


def options_stress(
    iv_history: HistoricalComparison | None,
    skew_history: HistoricalComparison | None,
    term: TermStructureContext | None,
    dislocation_history: HistoricalComparison | None,
    settings: OptionsContextSettings,
) -> OptionsStressContext:
    iv = _elevated(iv_history)
    skew = _elevated(skew_history)
    spread = term.short_long_spread if term is not None else None
    inversion = (
        min(1.0, max(0.0, spread) / settings.term_inversion_scale) if spread is not None else None
    )
    dislocation = _elevated(dislocation_history)
    candidates = (
        ("iv", iv, settings.stress_weights.iv),
        ("skew", skew, settings.stress_weights.skew),
        ("term", inversion, settings.stress_weights.term),
        ("dislocation", dislocation, settings.stress_weights.dislocation),
    )
    active = tuple(
        (name, value, weight)
        for name, value, weight in candidates
        if value is not None and weight > 0
    )
    total = sum(weight for _, _, weight in active)
    score = (
        sum(value * weight for _, value, weight in active) / total
        if len(active) >= settings.minimum_stress_components and total > 0
        else None
    )
    return OptionsStressContext(
        iv=iv,
        skew=skew,
        term=inversion,
        dislocation=dislocation,
        score=score,
        active_weights=tuple((name, weight / total) for name, _, weight in active) if total else (),
    )
