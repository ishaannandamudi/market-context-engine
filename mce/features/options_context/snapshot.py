"""Additive options evidence in the existing snapshot schema."""

from mce.features.options_context.models import OptionsContextFeatureSet
from mce.models.context import (
    ImpliedVolatilityContext,
    MarketContextSnapshot,
    SourceFreshness,
    TermStructurePoint,
    VolatilitySkewContext,
    VolatilityTermStructureContext,
)
from mce.models.enums import DataQuality, QualityFlag
from mce.version import __version__


def apply_options_context_to_snapshot(
    result: OptionsContextFeatureSet, snapshot: MarketContextSnapshot | None = None
) -> MarketContextSnapshot:
    base = snapshot or MarketContextSnapshot(as_of_ns=result.as_of_ns)
    if base.as_of_ns != result.as_of_ns:
        raise ValueError("options context and snapshot clocks differ")
    values = {
        name: value
        for name, value in (base.feature_values or {}).items()
        if not name.startswith("options.")
    }
    values["options.coverage"] = result.coverage
    values["options.expected_underlyings"] = len(result.expected_underlyings)
    values["options.available_underlyings"] = len(result.available_underlyings)
    if result.atm_iv_dispersion is not None:
        values["options.iv.dispersion"] = result.atm_iv_dispersion
    if result.options_stress_score is not None:
        values["options.stress.score"] = result.options_stress_score
    freshness = [
        item for item in (base.source_freshness or ()) if not item.source.startswith("options.")
    ]
    for context in result.per_underlying:
        source = context.source
        prefix = f"options.underlying.{source.underlying_symbol}"
        values[f"{prefix}.mode"] = source.market_data_mode
        values[f"{prefix}.source_engine"] = source.source_engine
        values[f"{prefix}.source_version"] = source.source_engine_version
        values[f"{prefix}.source_as_of_ns"] = source.as_of_ns
        values[f"{prefix}.available_at_ns"] = source.available_at_ns
        values[f"{prefix}.calculation_mode"] = source.calculation_mode
        values[f"{prefix}.source_quality"] = source.quality.value
        if source.source_fingerprint is not None:
            values[f"{prefix}.source_fingerprint"] = source.source_fingerprint
        values[f"{prefix}.market_data_age_ns"] = context.market_data_age_ns
        values[f"{prefix}.headline_eligible"] = context.headline_eligible
        values[f"{prefix}.surface.tenor_coverage"] = source.surface_quality.tenor_coverage
        values[f"{prefix}.surface.strike_coverage"] = source.surface_quality.strike_coverage
        values[f"{prefix}.surface.point_count"] = len(source.surface_points)
        values[f"{prefix}.surface.independent_neighbor_count"] = (
            source.surface_quality.independent_neighbor_count
        )
        values[f"{prefix}.surface.stale_contract_count"] = (
            source.surface_quality.stale_contract_count
        )
        values[f"{prefix}.surface.crossed_quote_count"] = source.surface_quality.crossed_quote_count
        values[f"{prefix}.surface.own_contract_fallback"] = (
            source.surface_quality.own_contract_fallback
        )
        if context.atm_iv is not None:
            values[f"{prefix}.atm_iv"] = context.atm_iv
        if context.relative_to_broad_iv is not None:
            values[f"{prefix}.relative_iv"] = context.relative_to_broad_iv
        if context.relative_to_broad_skew is not None:
            values[f"{prefix}.relative_skew"] = context.relative_to_broad_skew
        if context.relative_to_broad_term is not None:
            values[f"{prefix}.relative_term"] = context.relative_to_broad_term
        freshness.append(
            SourceFreshness(
                source=f"options.{source.underlying_symbol.lower()}",
                latest_effective_timestamp_ns=source.data_as_of_ns,
                age_ns=context.market_data_age_ns,
                is_stale=QualityFlag.STALE_DATA in context.quality_flags,
            )
        )
    primary = next(
        (
            item
            for item in result.per_underlying
            if item.source.underlying_symbol == result.primary_symbol
        ),
        None,
    )
    implied = None
    term_context = None
    skew_context = None
    if primary is not None:
        if primary.atm_iv is not None:
            values["options.iv.atm"] = primary.atm_iv
            if primary.iv_history is not None and primary.iv_history.percentile is not None:
                values["options.iv.percentile"] = primary.iv_history.percentile
            if primary.iv_history is not None and primary.iv_history.robust_z is not None:
                values["options.iv.robust_z"] = primary.iv_history.robust_z
            implied = ImpliedVolatilityContext(
                reference_symbol=primary.source.underlying_symbol,
                implied_volatility=primary.atm_iv,
                percentile=primary.iv_history.percentile if primary.iv_history else None,
                source=primary.source.source_engine,
                as_of_ns=primary.source.data_as_of_ns,
            )
        for name, history_comparison in (
            ("term", primary.term_history),
            ("skew", primary.skew_history),
            ("iv_rv", primary.dislocation_history),
        ):
            if history_comparison is not None and history_comparison.percentile is not None:
                values[f"options.{name}.percentile"] = history_comparison.percentile
            if history_comparison is not None and history_comparison.robust_z is not None:
                values[f"options.{name}.robust_z"] = history_comparison.robust_z
        if primary.term is not None:
            term = primary.term
            term_context = VolatilityTermStructureContext(
                reference_symbol=primary.source.underlying_symbol,
                points=tuple(
                    TermStructurePoint(
                        tenor_days=max(1, round(item.dte_days)),
                        implied_volatility=item.implied_volatility,
                    )
                    for item in term.tenors
                ),
                source=primary.source.source_engine,
                as_of_ns=primary.source.data_as_of_ns,
            )
            for name, item in (("short", term.short), ("medium", term.medium), ("long", term.long)):
                if item is not None:
                    values[f"options.term.{name}"] = item.implied_volatility
            for name, value in (
                ("short_long_spread", term.short_long_spread),
                ("short_long_ratio", term.short_long_ratio),
                ("slope", term.slope_per_day),
                ("curvature", term.curvature),
            ):
                if value is not None:
                    values[f"options.term.{name}"] = value
        if primary.iv_rv is not None:
            comparison = primary.iv_rv
            if comparison.spread is not None:
                values["options.iv_rv.spread"] = comparison.spread
            if comparison.ratio is not None:
                values["options.iv_rv.ratio"] = comparison.ratio
            if comparison.vrp_proxy is not None:
                values["options.vrp_proxy"] = comparison.vrp_proxy
            if comparison.horizon_mismatch_days is not None:
                values["options.iv_rv.horizon_mismatch_days"] = comparison.horizon_mismatch_days
        if (
            primary.headline_eligible
            and primary.skew is not None
            and primary.skew.put_iv is not None
            and primary.skew.call_iv is not None
        ):
            skew = primary.skew
            skew_context = VolatilitySkewContext(
                reference_symbol=primary.source.underlying_symbol,
                put_call_skew=skew.risk_reversal_call_minus_put,
                source=primary.source.source_engine,
                as_of_ns=primary.source.data_as_of_ns,
            )
            for name, value in (
                ("downside", skew.downside),
                ("upside", skew.upside),
                ("risk_reversal_call_minus_put", skew.risk_reversal_call_minus_put),
                ("butterfly", skew.butterfly),
                ("atm_call_minus_put", skew.atm_call_minus_put),
            ):
                if value is not None:
                    values[f"options.skew.{name}"] = value
        if primary.stress is not None:
            for name in ("iv", "skew", "term", "dislocation"):
                value = getattr(primary.stress, name)
                if value is not None:
                    values[f"options.stress.{name}"] = value
    quality_order = {
        DataQuality.UNKNOWN: 0,
        DataQuality.GOOD: 1,
        DataQuality.DEGRADED: 2,
        DataQuality.POOR: 3,
        DataQuality.UNUSABLE: 4,
    }
    quality = max((base.data_quality, result.data_quality), key=lambda item: quality_order[item])
    payload = base.model_dump(mode="python")
    payload.update(
        engine_version=__version__,
        feature_values=values,
        implied_volatility_context=implied,
        volatility_term_structure=term_context,
        volatility_skew_context=skew_context,
        source_freshness=tuple(freshness),
        quality_flags=base.quality_flags | result.quality_flags,
        data_quality=quality,
    )
    return MarketContextSnapshot.model_validate(payload)
