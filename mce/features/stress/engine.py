"""Explicit-clock broad liquidity and transparent continuous stress orchestration."""

from datetime import date

from mce.calendar.service import MarketSessionService
from mce.config.settings import MCEConfig
from mce.features.breadth.membership import MembershipTimeline, UniverseProvider
from mce.features.breadth.models import BreadthFeatureSet
from mce.features.correlation.models import CorrelationFeatureSet
from mce.features.dispersion.models import DispersionFeatureSet
from mce.features.sectors.models import SectorFeatureSet
from mce.features.stress.composite import gap_stress, weighted_score
from mce.features.stress.history import collect_liquidity_history, summarize_stress_history
from mce.features.stress.liquidity import build_liquidity
from mce.features.stress.models import (
    LiquidityStressFeatureSet,
    StressContext,
    StressHistoryPoint,
)
from mce.features.volatility.models import RealizedVolatilityFeatureSet
from mce.models.enums import DataQuality, QualityFlag
from mce.replay.store import CausalDataStore


class LiquidityStressFeatureEngine:
    def __init__(
        self,
        config: MCEConfig,
        universe_provider: UniverseProvider | None = None,
        calendar: MarketSessionService | None = None,
    ) -> None:
        self.config = config
        if calendar is not None and calendar.settings.model_dump(
            exclude={"universe"}
        ) != config.market.model_dump(exclude={"universe"}):
            raise ValueError("calendar settings must match result-relevant market configuration")
        self.calendar = calendar or MarketSessionService(config.market)
        self.universe_provider = universe_provider or MembershipTimeline.from_symbols(
            config.features.liquidity.universe
        )

    def compute(
        self,
        store: CausalDataStore,
        as_of_ns: int,
        *,
        volatility_context: RealizedVolatilityFeatureSet | None = None,
        breadth_context: BreadthFeatureSet | None = None,
        sector_context: SectorFeatureSet | None = None,
        correlation_context: CorrelationFeatureSet | None = None,
        dispersion_context: DispersionFeatureSet | None = None,
        stress_history: tuple[StressHistoryPoint, ...] = (),
    ) -> LiquidityStressFeatureSet:
        if as_of_ns < 0:
            raise ValueError("as_of_ns must be nonnegative")
        for context in (
            volatility_context,
            breadth_context,
            sector_context,
            correlation_context,
            dispersion_context,
        ):
            if context is not None and context.as_of_ns != as_of_ns:
                raise ValueError("upstream context and stress as_of_ns must match")
        membership = self.universe_provider.members(as_of_ns)
        if membership.as_of_ns != as_of_ns:
            raise ValueError("universe provider returned a different as_of_ns")
        info = self.calendar.session_for_timestamp(as_of_ns)
        day: date | None = info.trading_date if info.is_trading_day else None
        elapsed_ns = 0
        if day is not None:
            opened, closed = self.calendar.regular_session_bounds(day)
            elapsed_ns = max(0, min(as_of_ns, closed) - opened)
        settings = self.config.features.liquidity
        history = (
            collect_liquidity_history(
                store,
                self.calendar,
                self.universe_provider,
                day,
                elapsed_ns,
                settings.historical_sessions,
                self.config.provider.source,
                self.config.freshness.bar_limit_ns,
                settings.bar_cadence_ns,
            )
            if day is not None and elapsed_ns > 0 and settings.enabled
            else None
        )
        if history is None:
            from mce.features.stress.history import LiquidityHistory

            history = LiquidityHistory((), (), (), {}, {})
        liquidity, gaps = build_liquidity(
            store, self.calendar, membership, day, as_of_ns, history, self.config
        )
        stress_settings = self.config.features.stress
        breadth_stress = (
            (1 - breadth_context.breadth_score) / 2
            if breadth_context is not None and breadth_context.breadth_score is not None
            else None
        )
        sector_stress = None
        if sector_context is not None:
            horizon = self.config.features.sectors.medium_horizon.name
            returns = tuple(
                value
                for item in sector_context.sectors.values()
                if (value := item.absolute_returns.get(horizon)) is not None
            )
            if len(returns) >= self.config.features.sectors.minimum_valid_sectors:
                sector_stress = sum(value < 0 for value in returns) / len(returns)
        dispersion_stress = (
            dispersion_context.dispersion_score * breadth_stress
            if dispersion_context is not None
            and dispersion_context.dispersion_score is not None
            and breadth_stress is not None
            else None
        )
        gap_component = (
            gap_stress(
                gaps.down_fraction, gaps.negative_magnitude, gaps.percentile, stress_settings
            )
            if gaps.coverage >= settings.minimum_member_coverage
            else None
        )
        liquidity_stress = (
            1 - liquidity.liquidity_score if liquidity.liquidity_score is not None else None
        )
        weights = stress_settings.score_weights
        terms = (
            (
                "volatility",
                volatility_context.volatility_score if volatility_context else None,
                weights.volatility,
            ),
            (
                "correlation",
                correlation_context.systemic_co_movement_score if correlation_context else None,
                weights.correlation,
            ),
            ("breadth", breadth_stress, weights.breadth),
            ("liquidity", liquidity_stress, weights.liquidity),
            ("gap", gap_component, weights.gap),
            ("dispersion", dispersion_stress, weights.dispersion),
            ("sector", sector_stress, weights.sector),
        )
        score, active_weights, contributors = (
            weighted_score(terms, stress_settings.minimum_context_components, "stress")
            if stress_settings.enabled
            else (None, (), ())
        )
        summary = summarize_stress_history(
            score,
            stress_history,
            as_of_ns,
            stress_settings.historical_baseline_observations,
            stress_settings.minimum_history_samples,
            stress_settings.persistence_observations,
        )
        stressed_fraction = gaps.stressed_member_fraction
        systemic = (
            score * stressed_fraction
            if score is not None and stressed_fraction is not None
            else None
        )
        localized = (
            score * (1 - stressed_fraction)
            if score is not None and stressed_fraction is not None
            else None
        )
        flags = set(liquidity.quality_flags)
        for context in (
            volatility_context,
            breadth_context,
            sector_context,
            correlation_context,
            dispersion_context,
        ):
            if context is not None:
                flags.update(context.quality_flags)
        if score is None:
            flags.add(QualityFlag.MISSING_DATA)
        if score is not None and summary.percentile is None:
            flags.add(QualityFlag.INSUFFICIENT_HISTORY)
        evidence_quality = (len(active_weights) / 7) * (0.5 + 0.5 * liquidity.evidence_quality)
        quality = (
            DataQuality.POOR
            if score is None
            else DataQuality.DEGRADED
            if flags
            else DataQuality.GOOD
        )
        stress = StressContext(
            gap_up_fraction=gaps.up_fraction,
            gap_down_fraction=gaps.down_fraction,
            gap_magnitude=gaps.magnitude,
            gap_dispersion=gaps.dispersion,
            gap_percentile=gaps.percentile,
            gap_coverage=gaps.coverage,
            volatility_stress=terms[0][1],
            correlation_stress=terms[1][1],
            breadth_stress=breadth_stress,
            liquidity_stress=liquidity_stress,
            gap_stress=gap_component,
            dispersion_context_stress=dispersion_stress,
            sector_stress=sector_stress,
            stress_score=score,
            systemic_stress=systemic,
            localized_stress=localized,
            stressed_member_fraction=stressed_fraction,
            stress_percentile=summary.percentile,
            stress_change=summary.change,
            stress_acceleration=summary.acceleration,
            stress_persistence=summary.persistence,
            stress_shock_score=summary.shock,
            historical_sample_count=summary.sample_count,
            active_weights=active_weights,
            evidence_quality=evidence_quality,
            data_quality=quality,
            contributors=contributors,
            quality_flags=frozenset(flags),
        )
        order = {
            DataQuality.UNKNOWN: 0,
            DataQuality.GOOD: 1,
            DataQuality.DEGRADED: 2,
            DataQuality.POOR: 3,
            DataQuality.UNUSABLE: 4,
        }
        top_quality = max((liquidity.data_quality, stress.data_quality), key=order.__getitem__)
        return LiquidityStressFeatureSet(
            as_of_ns=as_of_ns,
            liquidity=liquidity,
            stress=stress,
            data_quality=top_quality,
            quality_flags=liquidity.quality_flags | stress.quality_flags,
        )
