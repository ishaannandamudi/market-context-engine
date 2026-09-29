"""Causal rolling breadth history with no hidden mutable accumulation."""

from dataclasses import dataclass


@dataclass(frozen=True)
class BreadthHistoryPoint:
    net_advance_fraction: float
    advance_fraction: float


@dataclass(frozen=True)
class BreadthHistorySummary:
    momentum: float | None
    acceleration: float | None
    thrust: float | None
    persistence: float | None


def summarize_history(
    current: BreadthHistoryPoint | None,
    prior_newest_first: tuple[BreadthHistoryPoint, ...],
    short_window: int,
    long_window: int,
) -> BreadthHistorySummary:
    if short_window <= 0 or long_window <= short_window:
        raise ValueError("breadth history windows must satisfy 0 < short < long")
    if current is None or len(prior_newest_first) < long_window - 1:
        return BreadthHistorySummary(None, None, None, None)
    points = (current, *prior_newest_first)
    net = tuple(item.net_advance_fraction for item in points)
    advance = tuple(item.advance_fraction for item in points)
    short_net = sum(net[:short_window]) / short_window
    long_net = sum(net[:long_window]) / long_window
    momentum = short_net - long_net
    thrust = sum(advance[:short_window]) / short_window - sum(advance[:long_window]) / long_window
    if len(prior_newest_first) >= long_window:
        previous_net = net[1 : long_window + 1]
        previous_momentum = (
            sum(previous_net[:short_window]) / short_window - sum(previous_net) / long_window
        )
        acceleration = momentum - previous_momentum
    else:
        acceleration = None
    current_sign = 1 if net[0] > 0 else -1 if net[0] < 0 else 0
    persistence = (
        sum((value > 0) - (value < 0) == current_sign for value in net[:long_window]) / long_window
        if current_sign
        else 0.0
    )
    return BreadthHistorySummary(momentum, acceleration, thrust, persistence)
