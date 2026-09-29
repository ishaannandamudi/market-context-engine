"""Causal horizon returns and transparent bounded relative-strength scoring."""

from dataclasses import dataclass
from math import isfinite
from typing import Literal

from mce.config.settings import SectorScoreWeights
from mce.features.breadth.math import weighted_return
from mce.features.returns import simple_return
from mce.features.windows import FeatureWindowReader, WindowSpec, WindowUnit
from mce.models.context import ContextContributor
from mce.models.market_data import BarObservation


def horizon_return(
    reader: FeatureWindowReader,
    symbol: str,
    current: BarObservation | None,
    as_of_ns: int,
    horizon: WindowSpec,
) -> float | None:
    if current is None or current.bar_end_ns is None:
        return None
    sample = reader.select(symbol, as_of_ns, horizon)
    reference = sample.reference_price
    if reference is None and horizon.unit == WindowUnit.ELAPSED and sample.bars:
        first = min(sample.bars, key=lambda bar: bar.bar_start_ns or 0)
        if first.bar_start_ns is not None:
            info = reader.calendar.session_for_timestamp(first.bar_start_ns)
            if (
                first.bar_start_ns == info.regular_open_ns
                and as_of_ns - horizon.length == info.regular_open_ns
            ):
                reference = first.open
    if not sample.bars or sample.bars[-1].bar_end_ns != current.bar_end_ns or reference is None:
        return None
    return simple_return(reference, current.close)


@dataclass(frozen=True)
class HorizonAggregate:
    equal_return: float | None
    weighted_return: float | None
    member_coverage: float
    weight_coverage: float | None


def aggregate_returns(
    values: tuple[tuple[float | None, float | None], ...],
    *,
    minimum_member_coverage: float,
    minimum_weight_coverage: float,
) -> HorizonAggregate:
    expected = len(values)
    valid = tuple((result, weight) for result, weight in values if result is not None)
    coverage = len(valid) / expected if expected else 0.0
    equal = sum(result / len(valid) for result, _ in valid) if valid else None
    if coverage < minimum_member_coverage:
        equal = None
    all_weighted = bool(values) and all(weight is not None for _, weight in values)
    if not all_weighted:
        return HorizonAggregate(equal, None, coverage, None)
    total = sum(weight for _, weight in values if weight is not None)
    pairs = tuple((result, weight) for result, weight in valid if weight is not None)
    weighted, weight_coverage = weighted_return(pairs, total)
    if weight_coverage is None or weight_coverage < minimum_weight_coverage:
        weighted = None
    return HorizonAggregate(equal, weighted, coverage, weight_coverage)


def bounded_return(value: float, scale: float) -> float:
    if scale <= 0 or not isfinite(value):
        raise ValueError("sector return normalization needs finite value and positive scale")
    return value / (abs(value) + scale)


def relative_score(
    sector: str,
    relative_returns: tuple[tuple[str, float | None], ...],
    breadth: float | None,
    weights: SectorScoreWeights,
    scale: float,
) -> tuple[float | None, tuple[ContextContributor, ...]]:
    configured = (weights.short_relative, weights.medium_relative, weights.long_relative)
    terms = [
        (name, bounded_return(value, scale), weight)
        for (name, value), weight in zip(relative_returns, configured, strict=True)
        if value is not None and weight > 0
    ]
    if not terms:
        return None, ()
    if breadth is not None:
        relative_weight = sum(weight for _, _, weight in terms)
        intensity = (
            sum(abs(value) * weight for _, value, weight in terms) / relative_weight
            if relative_weight > 0
            else 0.0
        )
        terms.append(("breadth", breadth * intensity, weights.breadth_confirmation))
    total = sum(weight for _, _, weight in terms)
    if total <= 0:
        return None, ()
    score = sum(value * weight / total for _, value, weight in terms)
    contributors = tuple(
        ContextContributor(
            name=f"sector.{sector}.score.{name}",
            value=value,
            normalized_value=value,
            contribution=value * weight / total,
            source_feature=(
                f"sector.{sector}.breadth"
                if name == "breadth"
                else f"sector.{sector}.relative_return.{name}"
            ),
        )
        for name, value, weight in terms
    )
    return max(-1.0, min(1.0, score)), contributors


def cross_horizon_agreement(
    values: tuple[float | None, ...],
) -> Literal["positive", "negative", "mixed", "unavailable"]:
    available = tuple(value for value in values if value is not None)
    if not available:
        return "unavailable"
    if all(value > 0 for value in available):
        return "positive"
    if all(value < 0 for value in available):
        return "negative"
    return "mixed"
