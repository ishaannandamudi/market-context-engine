"""Point-in-time same-elapsed-session liquidity baselines and prior stress ranks."""

from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from math import isfinite
from statistics import median

from mce.calendar.service import MarketSessionService
from mce.errors import CalendarRangeError
from mce.features.breadth.membership import UniverseProvider
from mce.features.stress.models import StressHistoryPoint
from mce.features.stress.primitives import read_session
from mce.replay.store import CausalDataStore


@dataclass(frozen=True)
class LiquidityHistory:
    market_volume: tuple[float, ...]
    market_illiquidity: tuple[float, ...]
    market_gap_magnitude: tuple[float, ...]
    member_volume: dict[str, tuple[float, ...]]
    member_illiquidity: dict[str, tuple[float, ...]]


def collect_liquidity_history(
    store: CausalDataStore,
    calendar: MarketSessionService,
    provider: UniverseProvider,
    day: date,
    elapsed_ns: int,
    sessions: int,
    source: str,
    freshness_ns: int,
    cadence_ns: int,
) -> LiquidityHistory:
    market_volume: list[float] = []
    market_illiquidity: list[float] = []
    market_gap: list[float] = []
    member_volume: dict[str, list[float]] = defaultdict(list)
    member_illiquidity: dict[str, list[float]] = defaultdict(list)
    cursor = day
    for _ in range(sessions):
        try:
            cursor = calendar.previous_trading_day(cursor)
        except CalendarRangeError:
            break
        opened, closed = calendar.regular_session_bounds(cursor)
        cutoff = opened + elapsed_ns
        if cutoff > closed:
            continue
        membership = provider.members(cutoff)
        if membership.as_of_ns != cutoff:
            raise ValueError("universe provider returned a different historical as_of_ns")
        volumes: list[float] = []
        illiquidities: list[float] = []
        gaps: list[float] = []
        for member in membership.members:
            panel = read_session(
                store, calendar, member.symbol, cursor, cutoff, source, freshness_ns, cadence_ns
            )
            if panel.volume is not None:
                volumes.append(panel.volume)
                member_volume[member.symbol].append(panel.volume)
            if panel.amihud is not None:
                illiquidities.append(panel.amihud)
                member_illiquidity[member.symbol].append(panel.amihud)
            if panel.gap_return is not None:
                gaps.append(abs(panel.gap_return))
        if volumes and len(volumes) == len(membership.members):
            market_volume.append(sum(volumes))
        if illiquidities:
            market_illiquidity.append(median(illiquidities))
        if gaps:
            market_gap.append(median(gaps))
    return LiquidityHistory(
        tuple(market_volume),
        tuple(market_illiquidity),
        tuple(market_gap),
        {key: tuple(value) for key, value in member_volume.items()},
        {key: tuple(value) for key, value in member_illiquidity.items()},
    )


@dataclass(frozen=True)
class PriorRank:
    sample_count: int
    percentile: float | None
    robust_z: float | None
    median: float | None


def prior_rank(current: float | None, values: tuple[float, ...], minimum: int) -> PriorRank:
    if (
        current is None
        or not isfinite(current)
        or len(values) < minimum
        or any(not isfinite(value) for value in values)
    ):
        return PriorRank(len(values), None, None, None)
    percentile = (
        sum(value < current for value in values) + 0.5 * sum(value == current for value in values)
    ) / len(values)
    center = median(values)
    if not isfinite(center):
        return PriorRank(len(values), percentile, None, None)
    mad = median(abs(value - center) for value in values)
    if not isfinite(mad):
        return PriorRank(len(values), percentile, None, center)
    z = (current - center) / (1.4826 * mad) if mad > 0 else (0.0 if current == center else None)
    return PriorRank(len(values), percentile, z if z is None or isfinite(z) else None, center)


@dataclass(frozen=True)
class StressHistorySummary:
    sample_count: int
    percentile: float | None
    change: float | None
    acceleration: float | None
    persistence: float | None
    shock: float | None


def summarize_stress_history(
    current: float | None,
    history: tuple[StressHistoryPoint, ...],
    as_of_ns: int,
    baseline: int,
    minimum: int,
    persistence_observations: int,
) -> StressHistorySummary:
    valid = sorted(
        (
            point
            for point in history
            if point.as_of_ns < as_of_ns and point.available_at_ns <= as_of_ns
        ),
        key=lambda point: point.as_of_ns,
        reverse=True,
    )
    if len({point.as_of_ns for point in valid}) != len(valid):
        raise ValueError("stress history timestamps must be unique")
    prior = tuple(point.stress_score for point in valid[:baseline])
    rank = prior_rank(current, prior, minimum)
    change = current - prior[0] if current is not None and prior else None
    acceleration = (
        current - 2 * prior[0] + prior[1] if current is not None and len(prior) > 1 else None
    )
    persistence = None
    if (
        current is not None
        and rank.median is not None
        and len(prior) >= persistence_observations - 1
    ):
        recent = (current, *prior[: persistence_observations - 1])
        persistence = sum(value > rank.median for value in recent) / persistence_observations
    shock = None
    if current is not None and rank.percentile is not None and rank.median is not None:
        denominator = abs(current) + abs(rank.median)
        shock = max(0.0, 2 * rank.percentile - 1) * (
            max(0.0, current - rank.median) / denominator if denominator > 0 else 0.0
        )
    return StressHistorySummary(
        rank.sample_count, rank.percentile, change, acceleration, persistence, shock
    )
