"""Deterministic coarse quality signals; confidence modeling is deferred."""

from collections.abc import Iterable, Mapping

from mce.models.context import SourceFreshness
from mce.models.enums import DataQuality, QualityFlag
from mce.models.quality import QualityTaggedModel
from mce.models.symbol import normalize_symbol
from mce.providers.base import ProviderStatus


class QualityAssessment(QualityTaggedModel):
    data_quality: DataQuality


def assess_quality(
    *,
    expected_symbols: Iterable[str],
    observed_symbols: Iterable[str],
    freshness: Iterable[SourceFreshness] = (),
    expected_sources: Iterable[str] = (),
    provider_statuses: Mapping[str, ProviderStatus] | None = None,
    rejected_count: int = 0,
    insufficient_history: bool = False,
    timestamp_mismatch: bool = False,
) -> QualityAssessment:
    """Assess availability, staleness, and ingestion faults without inventing scores."""
    if rejected_count < 0:
        raise ValueError("rejected_count must be nonnegative")
    expected = {normalize_symbol(symbol) for symbol in expected_symbols}
    observed = {normalize_symbol(symbol) for symbol in observed_symbols}
    flags: set[QualityFlag] = set()
    if not expected:
        flags.add(QualityFlag.MISSING_DATA)
        quality = DataQuality.UNKNOWN
    elif not observed:
        flags.update({QualityFlag.MISSING_DATA, QualityFlag.LOW_COVERAGE})
        quality = DataQuality.UNUSABLE
    else:
        if not expected.issubset(observed):
            flags.update({QualityFlag.PARTIAL_UNIVERSE, QualityFlag.LOW_COVERAGE})
        quality = DataQuality.GOOD
    if any(item.is_stale for item in freshness):
        flags.update({QualityFlag.STALE_DATA, QualityFlag.SOURCE_STALE})
    required_sources = set(expected_sources)
    if required_sources - set(provider_statuses or {}):
        flags.update({QualityFlag.PROVIDER_DEGRADED, QualityFlag.LOW_COVERAGE})
    if provider_statuses is not None and any(
        status != ProviderStatus.HEALTHY for status in provider_statuses.values()
    ):
        flags.add(QualityFlag.PROVIDER_DEGRADED)
    if rejected_count:
        flags.add(QualityFlag.REJECTED_DATA)
    if insufficient_history:
        flags.add(QualityFlag.INSUFFICIENT_HISTORY)
    if timestamp_mismatch:
        flags.add(QualityFlag.TIMESTAMP_MISMATCH)
    if quality == DataQuality.GOOD and flags:
        quality = DataQuality.DEGRADED
    return QualityAssessment(data_quality=quality, quality_flags=frozenset(flags))
