"""Additive schema-1.0 ensemble projection; upstream headlines remain untouched."""

from mce.features.ensemble.models import EnsembleContextFeatureSet
from mce.models.context import MarketContextSnapshot
from mce.version import __version__


def apply_ensemble_to_snapshot(
    result: EnsembleContextFeatureSet,
    snapshot: MarketContextSnapshot,
) -> MarketContextSnapshot:
    if result.as_of_ns != snapshot.as_of_ns:
        raise ValueError("ensemble and snapshot clocks differ")
    values = {
        name: value
        for name, value in (snapshot.feature_values or {}).items()
        if not name.startswith("ensemble.")
    }
    values["ensemble.headline"] = result.headline.value
    values["ensemble.config_fingerprint"] = result.config_fingerprint
    values["ensemble.conflicts.count"] = len(result.conflicts)
    if result.deterministic_regime is not None:
        values["ensemble.deterministic_regime"] = result.deterministic_regime
    if result.hmm_state_id is not None:
        values["ensemble.hmm_state_id"] = result.hmm_state_id
    for name, value in result.scorecard.model_dump().items():
        if name != "quality_flags" and value is not None:
            values[f"ensemble.{name}"] = value
    for item in result.groups:
        prefix = f"ensemble.group.{item.name}"
        if item.score is not None:
            values[f"{prefix}.score"] = item.score
        values[f"{prefix}.reliability"] = item.reliability
        values[f"{prefix}.configured_weight"] = item.configured_weight
        values[f"{prefix}.adjusted_weight"] = item.adjusted_weight
        values[f"{prefix}.normalized_active_weight"] = item.normalized_active_weight
    for context_item in result.context_reliability:
        values[f"ensemble.reliability.{context_item.family}"] = context_item.reliability
    payload = snapshot.model_dump(mode="python")
    payload.update(
        engine_version=__version__,
        feature_values=values,
        quality_flags=snapshot.quality_flags | result.quality_flags,
    )
    return MarketContextSnapshot.model_validate(payload)
