"""Retrospective exit counts, conditional observed frequencies, and reversals."""

from collections import Counter, defaultdict

from mce.config.settings import RegimeTransitionSettings
from mce.features.regime.taxonomy import HeadlineRegime
from mce.features.regime_transition.history import HistoryStep
from mce.features.regime_transition.models import (
    DominantTransitionPair,
    ObservedTransitionRow,
    ObservedTransitionTarget,
)
from mce.features.regime_transition.time_axis import SessionAxis


def headline_steps(steps: tuple[HistoryStep, ...]) -> tuple[HistoryStep, ...]:
    return tuple(step for step in steps if step.headline_changed)


def observed_frequencies(steps: tuple[HistoryStep, ...]) -> tuple[ObservedTransitionRow, ...]:
    counts: dict[HeadlineRegime, Counter[HeadlineRegime]] = defaultdict(Counter)
    for step in headline_steps(steps):
        counts[step.from_regime][step.to_regime] += 1
    return tuple(
        ObservedTransitionRow(
            from_regime=source,
            observed_exits=sum(targets.values()),
            targets=tuple(
                ObservedTransitionTarget(
                    to_regime=target,
                    count=count,
                    observed_frequency=count / sum(targets.values()),
                )
                for target, count in sorted(targets.items(), key=lambda item: item[0].value)
            ),
        )
        for source, targets in sorted(counts.items(), key=lambda item: item[0].value)
    )


def dominant_pair(steps: tuple[HistoryStep, ...]) -> DominantTransitionPair | None:
    counts: Counter[tuple[HeadlineRegime, HeadlineRegime]] = Counter()
    for step in headline_steps(steps):
        pair = tuple(sorted((step.from_regime, step.to_regime), key=lambda item: item.value))
        counts[(pair[0], pair[1])] += 1
    if not counts:
        return None
    pair, count = min(
        counts.items(), key=lambda item: (-item[1], item[0][0].value, item[0][1].value)
    )
    return DominantTransitionPair(first=pair[0], second=pair[1], count=count)


def reversal_metrics(
    steps: tuple[HistoryStep, ...], axis: SessionAxis, config: RegimeTransitionSettings
) -> tuple[int, float | None]:
    transitions = headline_steps(steps)
    if len(transitions) < 2:
        return 0, None
    count = sum(
        previous.from_regime == current.to_regime
        and previous.to_regime == current.from_regime
        and axis.session_equivalents_between(previous.at_ns, current.at_ns)
        <= config.reversal_window_sessions
        for previous, current in zip(transitions, transitions[1:], strict=False)
    )
    return count, count / (len(transitions) - 1)
