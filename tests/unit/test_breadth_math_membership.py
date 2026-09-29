from math import isclose

import pytest
from pydantic import ValidationError

from mce.config.settings import BreadthScoreWeights
from mce.features.breadth.history import BreadthHistoryPoint, summarize_history
from mce.features.breadth.math import (
    breadth_score_components,
    contribution_concentration,
    direction_counts,
    weighted_return,
)
from mce.features.breadth.membership import (
    MembershipTimeline,
    UniverseMembership,
    UniverseSnapshot,
)
from mce.models.enums import DataQuality
from mce.serialization import canonical_json_str


def test_point_in_time_membership_add_remove_weights_and_availability() -> None:
    timeline = MembershipTimeline(
        (
            UniverseMembership(symbol=" aaa ", effective_from_ns=0, effective_to_ns=20, weight=1),
            UniverseMembership(symbol="AAA", effective_from_ns=20, weight=3),
            UniverseMembership(symbol="BBB", effective_from_ns=10, effective_to_ns=30, weight=2),
            UniverseMembership(symbol="CCC", effective_from_ns=0, available_at_ns=25),
        ),
        source_quality=DataQuality.DEGRADED,
    )
    early = timeline.members(9)
    middle = timeline.members(15)
    later = timeline.members(25)
    assert [member.symbol for member in early.members] == ["AAA"]
    assert [member.symbol for member in middle.members] == ["AAA", "BBB"]
    assert [member.symbol for member in later.members] == ["AAA", "BBB", "CCC"]
    assert middle.members[0].weight == 1
    assert later.members[0].weight == 3
    assert [member.symbol for member in timeline.members(30).members] == ["AAA", "CCC"]
    assert later.source_quality == DataQuality.DEGRADED
    assert canonical_json_str(timeline.members(25)) == canonical_json_str(timeline.members(25))
    unordered = UniverseSnapshot(
        as_of_ns=25,
        members=tuple(reversed(later.members)),
        source="config",
        source_quality=DataQuality.GOOD,
    )
    assert [item.symbol for item in unordered.members] == ["AAA", "BBB", "CCC"]


def test_membership_validation_rejects_overlaps_duplicates_and_bad_weights() -> None:
    with pytest.raises(ValueError):
        MembershipTimeline((UniverseMembership(symbol="AAA"), UniverseMembership(symbol="aaa")))
    with pytest.raises(ValueError):
        MembershipTimeline(
            (
                UniverseMembership(symbol="AAA", effective_from_ns=0, effective_to_ns=20),
                UniverseMembership(symbol="AAA", effective_from_ns=19),
            )
        )
    with pytest.raises(ValidationError):
        UniverseMembership(symbol="AAA", effective_from_ns=20, effective_to_ns=20)
    with pytest.raises(ValidationError):
        UniverseMembership(symbol="AAA", weight=-1)
    with pytest.raises(ValidationError):
        UniverseMembership(symbol="AAA", weight=float("inf"))
    with pytest.raises(ValueError):
        MembershipTimeline.from_symbols(("AAA",)).members(-1)


def test_direction_counts_tolerance_and_safe_ratio() -> None:
    result = direction_counts((0.1, -0.2, 1e-8, -1e-8), 1e-8)
    assert result == (1, 1, 2, 0.25, 0.25, 0.0, 1.0)
    assert direction_counts((0.1, 0.2), 0) == (2, 0, 0, 1, 0, 1, None)
    assert direction_counts((-0.1, -0.2), 0)[5] == -1
    assert direction_counts((), 0) == (0, 0, 0, None, None, None, None)


def test_weight_coverage_and_concentration_invariants() -> None:
    weighted, coverage = weighted_return(((0.1, 3), (-0.1, 1)), 5)
    assert weighted == pytest.approx(0.05)
    assert coverage == pytest.approx(0.8)
    assert weighted_return((), 1) == (None, 0.0)
    assert weighted_return(((0.1, 0),), 1) == (None, 0.0)
    assert weighted_return(((0.1, 1),), 0) == (None, None)
    equal = contribution_concentration(((0.1, 1), (0.1, 1), (0.1, 1)))
    dominant = contribution_concentration(((0.1, 10), (0.1, 1), (0.1, 1)))
    assert equal is not None and dominant is not None
    assert equal == pytest.approx((1 / 3, 3))
    assert dominant[0] > equal[0] and dominant[1] < equal[1]
    assert contribution_concentration(((1, 1), (1, 2))) == contribution_concentration(
        ((10, 1), (10, 2))
    )
    assert contribution_concentration(((0, 1), (0, 1))) is None
    assert contribution_concentration(((-0.1, -1),)) is None


def test_score_components_and_causal_history_math() -> None:
    score, components = breadth_score_components(1, (1, 1, 1), BreadthScoreWeights())
    assert score == 1
    assert sum(value * weight for _, value, weight in components) == pytest.approx(score)
    score, components = breadth_score_components(-1, (None, None, None), BreadthScoreWeights())
    assert score == -1 and len(components) == 1
    assert breadth_score_components(None, (None, None, None), BreadthScoreWeights())[0] is None
    prior = tuple(BreadthHistoryPoint(-1, 0) for _ in range(10))
    improving = summarize_history(BreadthHistoryPoint(1, 1), prior, 3, 10)
    assert improving.momentum is not None and improving.momentum > 0
    assert improving.acceleration is not None and improving.acceleration > 0
    assert improving.thrust is not None and improving.thrust > 0
    assert improving.persistence is not None and 0 <= improving.persistence <= 1
    assert summarize_history(BreadthHistoryPoint(1, 1), prior[:4], 3, 10).momentum is None
    assert isclose(
        summarize_history(
            BreadthHistoryPoint(0, 0.5),
            tuple(BreadthHistoryPoint(0, 0.5) for _ in range(10)),
            3,
            10,
        ).momentum
        or 0,
        0,
    )
