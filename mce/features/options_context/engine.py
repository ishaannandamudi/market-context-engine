"""Provider-free consumption of precomputed options-market evidence."""

from math import isfinite
from statistics import median, pstdev

from mce.config.settings import MCEConfig
from mce.features.options_context.contracts import ATMIVPoint, OptionsMarketContextInput
from mce.features.options_context.history import compare_history
from mce.features.options_context.models import (
    OptionsContextFeatureSet,
    OptionsContextHistory,
    UnderlyingOptionsContext,
)
from mce.features.options_context.realized_implied import compare_implied_realized
from mce.features.options_context.skew import skew_context
from mce.features.options_context.stress import options_stress
from mce.features.options_context.term_structure import term_structure
from mce.features.volatility.models import RealizedVolatilityFeatureSet
from mce.models.enums import DataQuality, QualityFlag
from mce.reproducibility import config_fingerprint


def _atm(input_context: OptionsMarketContextInput, target_days: int) -> ATMIVPoint | None:
    combined = (point for point in input_context.atm_points if point.combination == "combined")
    return min(
        combined,
        key=lambda point: (abs(point.dte_days - target_days), point.dte_days),
        default=None,
    )


class OptionsContextFeatureEngine:
    def __init__(self, config: MCEConfig) -> None:
        self.config = config
        self.settings = config.features.options_context

    def compute(
        self,
        options_inputs: tuple[OptionsMarketContextInput, ...],
        as_of_ns: int,
        *,
        realized_volatility_context: RealizedVolatilityFeatureSet | None = None,
        historical_options_context: OptionsContextHistory | None = None,
    ) -> OptionsContextFeatureSet:
        if as_of_ns < 0:
            raise ValueError("options as_of_ns must be nonnegative")
        settings = self.settings
        if not settings.enabled:
            return OptionsContextFeatureSet(
                as_of_ns=as_of_ns,
                config_fingerprint=config_fingerprint(self.config),
                expected_underlyings=(),
                available_underlyings=(),
                coverage=0,
                per_underlying=(),
                data_quality=DataQuality.UNKNOWN,
            )
        expected = settings.broad_underlyings + settings.sector_underlyings
        selected: dict[str, OptionsMarketContextInput] = {}
        for item in options_inputs:
            if item.underlying_symbol not in expected or item.available_at_ns > as_of_ns:
                continue
            if item.data_as_of_ns > as_of_ns or item.as_of_ns > as_of_ns:
                continue
            required_scope = (
                "broad" if item.underlying_symbol in settings.broad_underlyings else "sector"
            )
            if item.scope != required_scope:
                raise ValueError("options input scope does not match configured underlying")
            previous_source = selected.get(item.underlying_symbol)
            if previous_source is None or (item.data_as_of_ns, item.available_at_ns) > (
                previous_source.data_as_of_ns,
                previous_source.available_at_ns,
            ):
                selected[item.underlying_symbol] = item
            elif (item.data_as_of_ns, item.available_at_ns) == (
                previous_source.data_as_of_ns,
                previous_source.available_at_ns,
            ) and item != previous_source:
                raise ValueError("ambiguous same-clock options source revision")
        history = historical_options_context or OptionsContextHistory()
        results: list[UnderlyingOptionsContext] = []
        flags: set[QualityFlag] = set()
        for symbol in expected:
            source = selected.get(symbol)
            if source is None:
                flags.add(QualityFlag.MISSING_OPTIONS_CONTEXT)
                continue
            age = as_of_ns - source.data_as_of_ns
            item_flags: set[QualityFlag] = set()
            surface = source.surface_quality
            reliable_surface = tuple(
                point
                for point in source.surface_points
                if point.quality == DataQuality.GOOD
                or (
                    settings.minimum_quality == "degraded" and point.quality == DataQuality.DEGRADED
                )
            )
            sparse = (
                len(reliable_surface) < settings.minimum_surface_points
                or surface.tenor_coverage < 0.5
                or surface.strike_coverage < 0.5
                or surface.independent_neighbor_count == 0
            )
            if sparse:
                item_flags.add(QualityFlag.OPTIONS_SURFACE_SPARSE)
            if source.market_data_mode in ("eod", "delayed"):
                item_flags.add(QualityFlag.OPTIONS_MODE_LIMITED)
            if age > settings.maximum_market_age_ns or surface.stale_contract_count:
                item_flags.add(QualityFlag.STALE_DATA)
            if source.quality not in (DataQuality.GOOD, DataQuality.DEGRADED):
                item_flags.add(QualityFlag.PROVIDER_DEGRADED)
            if source.underlying_price is None:
                item_flags.add(QualityFlag.OPTIONS_UNDERLYING_PRICE_MISSING)
            if surface.crossed_quote_count or surface.extrapolated:
                item_flags.add(QualityFlag.REJECTED_DATA)
            fresh_atm = tuple(
                point
                for point in source.atm_points
                if as_of_ns - point.data_as_of_ns <= settings.maximum_market_age_ns
            )
            fresh_surface = tuple(
                point
                for point in source.surface_points
                if as_of_ns - point.data_as_of_ns <= settings.maximum_market_age_ns
            )
            if len(fresh_atm) != len(source.atm_points) or len(fresh_surface) != len(
                source.surface_points
            ):
                item_flags.add(QualityFlag.STALE_DATA)
            atm = _atm(source, settings.target_tenor_days)
            circular = (
                surface.own_contract_fallback
                or (atm is not None and atm.evidence_origin == "own_contract_fallback")
                or any(
                    point.evidence_origin == "own_contract_fallback"
                    for point in source.surface_points
                )
            )
            if circular:
                item_flags.add(QualityFlag.OPTIONS_CIRCULAR_EVIDENCE)
            headline = (
                atm is not None
                and source.market_data_mode in ("live", "synthetic")
                and age <= settings.maximum_market_age_ns
                and source.quality in (DataQuality.GOOD, DataQuality.DEGRADED)
                and (settings.minimum_quality == "degraded" or source.quality == DataQuality.GOOD)
                and atm.quality in (DataQuality.GOOD, DataQuality.DEGRADED)
                and (settings.minimum_quality == "degraded" or atm.quality == DataQuality.GOOD)
                and not sparse
                and not circular
                and not surface.stale_contract_count
                and not surface.crossed_quote_count
                and not surface.extrapolated
                and len(fresh_surface) >= settings.minimum_surface_points
                and atm in fresh_atm
                and QualityFlag.STALE_DATA not in item_flags
            )
            valid_atm = tuple(
                point
                for point in source.atm_points
                if point.quality not in (DataQuality.POOR, DataQuality.UNUSABLE)
            )
            term = term_structure(valid_atm, settings.target_tenor_days, settings.minimum_tenors)
            skew = (
                skew_context(atm, source.atm_points, source.surface_points, settings)
                if atm
                else None
            )
            iv_rv = (
                compare_implied_realized(
                    atm, symbol, as_of_ns, realized_volatility_context, settings
                )
                if atm
                else None
            )
            if iv_rv is not None and iv_rv.rv_horizon_sessions is not None and not iv_rv.comparable:
                item_flags.add(QualityFlag.OPTIONS_HORIZON_MISMATCH)
            if iv_rv is not None and iv_rv.rv_horizon_sessions is None:
                item_flags.add(QualityFlag.MISSING_VOLATILITY_DATA)
            if iv_rv is not None and not headline:
                iv_rv = iv_rv.model_copy(update={"vrp_proxy": None})
            prior_points = history.prior(symbol, as_of_ns, settings.history_window)
            if len(prior_points) < settings.minimum_history:
                item_flags.add(QualityFlag.INSUFFICIENT_HISTORY)
            iv_history = compare_history(
                atm.implied_volatility if atm else None,
                tuple(point.atm_iv for point in prior_points),
                settings.minimum_history,
            )
            skew_history = compare_history(
                skew.downside if skew else None,
                tuple(point.downside_skew for point in prior_points),
                settings.minimum_history,
            )
            term_history = compare_history(
                term.short_long_spread,
                tuple(point.term_short_long_spread for point in prior_points),
                settings.minimum_history,
            )
            dislocation_history = compare_history(
                iv_rv.spread if iv_rv else None,
                tuple(point.iv_rv_spread for point in prior_points),
                settings.minimum_history,
            )
            stress = (
                options_stress(iv_history, skew_history, term, dislocation_history, settings)
                if headline
                else None
            )
            quality = (
                DataQuality.POOR
                if not headline
                else DataQuality.DEGRADED
                if item_flags
                else DataQuality.GOOD
            )
            results.append(
                UnderlyingOptionsContext(
                    source=source,
                    atm_iv=atm.implied_volatility if atm else None,
                    atm_tenor_days=atm.dte_days if atm else None,
                    iv_history=iv_history,
                    skew_history=skew_history,
                    term_history=term_history,
                    dislocation_history=dislocation_history,
                    term=term,
                    skew=skew,
                    iv_rv=iv_rv,
                    stress=stress,
                    market_data_age_ns=age,
                    headline_eligible=headline,
                    data_quality=quality,
                    quality_flags=frozenset(item_flags),
                )
            )
            flags.update(item_flags)
        broad_results = tuple(
            result
            for result in results
            if result.source.underlying_symbol in settings.broad_underlyings
        )
        primary = next((result for result in broad_results if result.headline_eligible), None)
        if primary is None:
            primary = broad_results[0] if broad_results else None
        if primary is not None:
            updated: list[UnderlyingOptionsContext] = []
            for result in results:
                if result.source.scope != "sector":
                    updated.append(result)
                    continue
                updated.append(
                    result.model_copy(
                        update={
                            "relative_to_broad_iv": result.atm_iv - primary.atm_iv
                            if result.atm_iv is not None and primary.atm_iv is not None
                            else None,
                            "relative_to_broad_skew": result.skew.downside - primary.skew.downside
                            if result.skew is not None
                            and primary.skew is not None
                            and result.skew.downside is not None
                            and primary.skew.downside is not None
                            else None,
                            "relative_to_broad_term": result.term.short_long_spread
                            - primary.term.short_long_spread
                            if result.term is not None
                            and primary.term is not None
                            and result.term.short_long_spread is not None
                            and primary.term.short_long_spread is not None
                            else None,
                        }
                    )
                )
            results = updated
        broad_scores = tuple(
            result.stress.score
            for result in results
            if result.source.underlying_symbol in settings.broad_underlyings
            and result.stress is not None
            and result.stress.score is not None
        )
        broad_coverage = len(broad_scores) / len(settings.broad_underlyings)
        score = (
            float(median(broad_scores))
            if broad_coverage >= settings.minimum_broad_coverage and broad_scores
            else None
        )
        ivs = tuple(
            result.atm_iv
            for result in results
            if result.headline_eligible and result.atm_iv is not None
        )
        dispersion = float(pstdev(ivs)) if len(ivs) >= 2 else None
        if dispersion is not None and not isfinite(dispersion):
            dispersion = None
        coverage = len(results) / len(expected) if expected else 0.0
        if coverage < 1:
            flags.add(QualityFlag.PARTIAL_UNIVERSE)
        quality = (
            DataQuality.POOR if not results else DataQuality.DEGRADED if flags else DataQuality.GOOD
        )
        return OptionsContextFeatureSet(
            as_of_ns=as_of_ns,
            config_fingerprint=config_fingerprint(self.config),
            expected_underlyings=expected,
            available_underlyings=tuple(result.source.underlying_symbol for result in results),
            coverage=coverage,
            per_underlying=tuple(results),
            primary_symbol=primary.source.underlying_symbol if primary is not None else None,
            atm_iv_dispersion=dispersion,
            options_stress_score=score,
            data_quality=quality,
            quality_flags=frozenset(flags),
        )
