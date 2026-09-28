"""Explicit-clock freshness evaluation with no wall-time dependency."""

from mce.models.context import SourceFreshness


def evaluate_freshness(
    source: str,
    latest_effective_timestamp_ns: int | None,
    as_of_ns: int,
    freshness_limit_ns: int,
) -> SourceFreshness | None:
    """Return None for absent input; age exactly at the limit is fresh."""
    if as_of_ns < 0 or freshness_limit_ns <= 0:
        raise ValueError("as_of_ns must be nonnegative and freshness_limit_ns positive")
    if latest_effective_timestamp_ns is None:
        return None
    if latest_effective_timestamp_ns < 0 or latest_effective_timestamp_ns > as_of_ns:
        raise ValueError("latest effective timestamp must lie in [0, as_of_ns]")
    age = as_of_ns - latest_effective_timestamp_ns
    return SourceFreshness(
        source=source,
        latest_effective_timestamp_ns=latest_effective_timestamp_ns,
        age_ns=age,
        freshness_limit_ns=freshness_limit_ns,
        is_stale=age > freshness_limit_ns,
    )
