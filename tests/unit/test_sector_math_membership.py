from math import isclose

import pytest
from pydantic import ValidationError

from mce.config.settings import SectorScoreWeights, SectorSettings
from mce.features.sectors.breadth import reference_participation, sector_direction
from mce.features.sectors.membership import (
    SectorMembership,
    SectorMembershipTimeline,
    SectorUniverseSnapshot,
)
from mce.features.sectors.rotation import (
    leadership_hhi,
    leadership_persistence,
    rank_scores,
    rotation_velocity,
)
from mce.features.sectors.strength import (
    aggregate_returns,
    bounded_return,
    cross_horizon_agreement,
    relative_score,
)
from mce.features.sectors.volatility import volatility_context
from mce.models.enums import DataQuality


def test_sector_membership_add_remove_reclassify_and_availability() -> None:
    timeline = SectorMembershipTimeline(
        (
            SectorMembership(sector=" Technology ", symbol=" abc ", effective_to_ns=20, weight=1),
            SectorMembership(sector="energy", symbol="ABC", effective_from_ns=20, weight=3),
            SectorMembership(
                sector="energy", symbol="DEF", effective_from_ns=10, effective_to_ns=30
            ),
            SectorMembership(sector="utilities", symbol="UVW", available_at_ns=25),
        ),
        source_quality=DataQuality.DEGRADED,
    )
    assert [(item.sector, item.symbol) for item in timeline.members(15).members] == [
        ("energy", "DEF"),
        ("technology", "ABC"),
    ]
    assert [(item.sector, item.symbol) for item in timeline.members(25).members] == [
        ("energy", "ABC"),
        ("energy", "DEF"),
        ("utilities", "UVW"),
    ]
    assert timeline.members(25).members[0].weight == 3
    assert len(timeline.members(30).members) == 2
    assert timeline.members(25).source_quality == DataQuality.DEGRADED


def test_sector_membership_rejects_overlap_duplicates_and_invalid_values() -> None:
    with pytest.raises(ValueError):
        SectorMembershipTimeline(
            (
                SectorMembership(sector="technology", symbol="ABC"),
                SectorMembership(sector="energy", symbol="ABC"),
            )
        )
    with pytest.raises(ValueError):
        SectorMembershipTimeline(
            (
                SectorMembership(sector="technology", symbol="ABC"),
                SectorMembership(sector="technology", symbol="ABC"),
            ),
            exclusive=False,
        )
    nonexclusive = SectorMembershipTimeline(
        (
            SectorMembership(sector="technology", symbol="ABC"),
            SectorMembership(sector="energy", symbol="ABC"),
        ),
        exclusive=False,
    )
    assert len(nonexclusive.members(1).members) == 2
    with pytest.raises(ValueError):
        SectorMembershipTimeline(
            (
                SectorMembership(sector="technology", symbol="ABC", effective_to_ns=100),
                SectorMembership(
                    sector="energy", symbol="ABC", effective_from_ns=10, effective_to_ns=20
                ),
                SectorMembership(
                    sector="technology", symbol="ABC", effective_from_ns=30, effective_to_ns=40
                ),
            ),
            exclusive=False,
        )
    with pytest.raises(ValidationError):
        SectorMembership(sector="technology", symbol="ABC", weight=-1)
    with pytest.raises(ValidationError):
        SectorMembership(sector="bad/name", symbol="ABC")
    with pytest.raises(ValidationError):
        SectorMembership(sector="technology", symbol="ABC", effective_from_ns=2, effective_to_ns=2)
    with pytest.raises(ValueError):
        nonexclusive.members(-1)
    with pytest.raises(ValidationError):
        SectorUniverseSnapshot(
            as_of_ns=1,
            members=(SectorMembership(sector="technology", symbol="ABC", effective_from_ns=2),),
            source="synthetic",
            source_quality=DataQuality.GOOD,
        )


def test_sector_score_math_coverage_and_ordering() -> None:
    aggregate = aggregate_returns(
        ((0.1, 3), (-0.1, 1), (None, 1)),
        minimum_member_coverage=0.5,
        minimum_weight_coverage=0.5,
    )
    assert aggregate.member_coverage == pytest.approx(2 / 3)
    assert aggregate.weight_coverage == pytest.approx(0.8)
    assert aggregate.equal_return == pytest.approx(0)
    assert aggregate.weighted_return == pytest.approx(0.05)
    suppressed = aggregate_returns(
        ((0.1, 3), (None, 1)),
        minimum_member_coverage=0.8,
        minimum_weight_coverage=0.8,
    )
    assert suppressed.equal_return is None and suppressed.weighted_return is None
    assert (
        aggregate_returns(
            ((0.1, None),), minimum_member_coverage=0.8, minimum_weight_coverage=0.8
        ).weighted_return
        is None
    )
    score, contributors = relative_score(
        "technology",
        (("session", 0.02), ("1d", 0.01), ("5d", -0.01)),
        0.5,
        SectorScoreWeights(),
        0.01,
    )
    assert score is not None and -1 <= score <= 1
    assert sum(item.contribution or 0 for item in contributors) == pytest.approx(score)
    unavailable_weighted_score, _ = relative_score(
        "technology",
        (("session", 0.02), ("1d", None), ("5d", None)),
        1,
        SectorScoreWeights(
            short_relative=0,
            medium_relative=0.9,
            long_relative=0,
            breadth_confirmation=0.1,
        ),
        0.01,
    )
    assert unavailable_weighted_score is None
    assert bounded_return(-0.02, 0.01) == pytest.approx(-2 / 3)
    assert cross_horizon_agreement((1, 2, 3)) == "positive"
    assert cross_horizon_agreement((-1, -2)) == "negative"
    assert cross_horizon_agreement((1, -1)) == "mixed"
    assert cross_horizon_agreement((None,)) == "unavailable"
    assert sector_direction((0.01, -0.01, 0), 1e-8) == (1, 1, 1, 0)
    reference = reference_participation(((101, (100.0,) * 5), (99, (100.0,) * 5)), 5)
    assert reference.fraction == 0.5
    assert reference_participation(((101, (None,) * 5),), 5).fraction is None


def test_rank_ties_percentiles_persistence_rotation_and_hhi() -> None:
    ranks = rank_scores({"zeta": 0.2, "alpha": 0.2, "beta": -0.2, "missing": None})
    assert ranks == {"alpha": (1, 1), "zeta": (2, 0.5), "beta": (3, 0)}
    assert rank_scores({"only": 0}) == {"only": (1, 0.5)}
    assert leadership_hhi({"a": 1, "b": 1, "c": -1}) == pytest.approx(0.5)
    assert leadership_hhi({"a": -1, "b": 0}) is None
    previous = rank_scores({"alpha": -1, "zeta": 1, "beta": 0})
    assert rotation_velocity(ranks, (previous,), 1) is not None
    assert rotation_velocity(ranks, (ranks,), 1) == 0
    assert rotation_velocity(ranks, (), 1) is None
    assert leadership_persistence("alpha", ranks, (ranks, previous), 3) == pytest.approx(2 / 3)
    assert leadership_persistence("alpha", ranks, (), 3) is None
    gradual = (
        rank_scores({"a": 0.3, "b": 0.2, "c": 0.1}),
        rank_scores({"a": 0.2, "b": 0.3, "c": 0.1}),
        rank_scores({"a": 0.1, "b": 0.3, "c": 0.2}),
    )
    assert rotation_velocity(gradual[2], (gradual[1], gradual[0]), 2) == pytest.approx(1 / 3)


def test_volatility_context_and_settings_validation() -> None:
    settings = SectorSettings(
        volatility_windows_sessions=(2, 3, 4),
        volatility_baseline_sessions=3,
        minimum_volatility_baseline_samples=2,
    )
    series = (0.01, -0.01, 0.02, -0.02, 0.01, -0.01, 0.02)
    context = volatility_context(series, tuple(value / 2 for value in series), settings, 252)
    assert context.medium is not None and context.market is not None
    assert context.relative is not None and context.relative > 1
    assert context.percentile is not None and 0 <= context.percentile <= 1
    assert volatility_context(series, (None,) * len(series), settings, 252).relative is None
    assert isclose(bounded_return(0, 0.01), 0)
    with pytest.raises(ValidationError):
        SectorSettings(volatility_windows_sessions=(5, 5, 60))
    with pytest.raises(ValidationError):
        SectorSettings(minimum_member_coverage=1.1)
    with pytest.raises(ValidationError):
        SectorSettings(groups={"technology": ("ABC",), "energy": ("ABC",)})
    with pytest.raises(ValidationError):
        SectorSettings(groups={"Technology": ("ABC",), " technology ": ("DEF",)})
    assert (
        SectorSettings(
            groups={"technology": ("ABC",), "energy": ("ABC",)}, exclusive_membership=False
        ).exclusive_membership
        is False
    )
    with pytest.raises(ValidationError):
        SectorScoreWeights(short_relative=0.9)
    with pytest.raises(ValidationError):
        SectorScoreWeights(
            short_relative=0,
            medium_relative=0,
            long_relative=0,
            breadth_confirmation=1,
        )
