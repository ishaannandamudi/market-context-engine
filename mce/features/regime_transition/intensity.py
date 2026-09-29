"""Observed ordered-pair transition intensity from simultaneous dimension movement."""

from collections import defaultdict
from statistics import mean

from mce.features.regime.taxonomy import HeadlineRegime
from mce.features.regime_transition.history import DIMENSION_NAMES, HistoryStep
from mce.features.regime_transition.models import DirectedTransitionIntensity


def directed_intensity(steps: tuple[HistoryStep, ...]) -> tuple[DirectedTransitionIntensity, ...]:
    grouped: dict[tuple[HeadlineRegime, HeadlineRegime], list[float]] = defaultdict(list)
    for step in steps:
        if step.headline_changed:
            grouped[(step.from_regime, step.to_regime)].append(
                len(step.changed_dimensions) / len(DIMENSION_NAMES)
            )
    return tuple(
        DirectedTransitionIntensity(
            from_regime=source,
            to_regime=target,
            observed_count=len(values),
            mean_changed_dimension_fraction=mean(values),
        )
        for (source, target), values in sorted(
            grouped.items(), key=lambda item: (item[0][0].value, item[0][1].value)
        )
    )
