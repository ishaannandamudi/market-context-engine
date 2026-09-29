"""Explicit-clock point-in-time cross-sectional dispersion orchestration."""

from dataclasses import dataclass
from datetime import date

from mce.calendar.service import MarketSessionService
from mce.config.settings import DispersionSettings, MCEConfig
from mce.errors import CalendarRangeError
from mce.features.breadth.membership import MembershipTimeline, UniverseProvider, UniverseSnapshot
from mce.features.correlation.models import CorrelationFeatureSet
from mce.features.dispersion.concentration import movement_concentration
from mce.features.dispersion.context import sector_summary, volatility_summary
from mce.features.dispersion.cross_section import CrossSection, cross_section, weighted_std
from mce.features.dispersion.decomposition import SectorDecomposition, decompose
from mce.features.dispersion.history import summarize_history
from mce.features.dispersion.models import (
    CorrelationDispersionContext,
    DispersionFeatureSet,
    HorizonDispersion,
    SectorReturnDispersion,
)
from mce.features.dispersion.returns import read_member
from mce.features.sectors.membership import SectorMembershipTimeline, SectorUniverseProvider
from mce.features.sectors.models import SectorFeatureSet
from mce.features.volatility.models import RealizedVolatilityFeatureSet
from mce.features.windows import FeatureWindowReader, WindowSpec
from mce.models.context import ContextContributor
from mce.models.enums import DataQuality, QualityFlag
from mce.replay.store import CausalDataStore


@dataclass(frozen=True)
class _CrossSectionAt:
    values: dict[str, dict[str, float]]
    stale: tuple[str, ...]
    missing: tuple[str, ...]


def _membership_factor(quality: DataQuality) -> float:
    return {
        DataQuality.GOOD: 1.0,
        DataQuality.DEGRADED: 0.5,
        DataQuality.POOR: 0.25,
        DataQuality.UNUSABLE: 0.0,
        DataQuality.UNKNOWN: 0.5,
    }[quality]


def _horizon_stats(
    values: dict[str, dict[str, float]],
    name: str,
    expected: int,
    settings: DispersionSettings,
) -> tuple[HorizonDispersion, CrossSection | None, dict[str, float]]:
    usable = {symbol: item[name] for symbol, item in values.items() if name in item}
    stats = cross_section(
        tuple(usable.values()),
        minimum=settings.minimum_members,
        tail_quantiles=settings.tail_quantiles,
    )
    result = HorizonDispersion(
        name=name,
        valid_member_count=len(usable),
        member_coverage=len(usable) / expected if expected else 0.0,
        return_std=stats.std if stats else None,
        return_mad=stats.mad if stats else None,
    )
    return result, stats, usable


class DispersionFeatureEngine:
    def __init__(
        self,
        config: MCEConfig,
        universe_provider: UniverseProvider | None = None,
        sector_provider: SectorUniverseProvider | None = None,
        calendar: MarketSessionService | None = None,
    ) -> None:
        self.config = config
        if calendar is not None and calendar.settings.model_dump(
            exclude={"universe"}
        ) != config.market.model_dump(exclude={"universe"}):
            raise ValueError("calendar settings must match result-relevant market configuration")
        self.calendar = calendar or MarketSessionService(config.market)
        self.universe_provider = universe_provider or MembershipTimeline.from_symbols(
            config.features.dispersion.universe
        )
        self.sector_provider = sector_provider or (
            SectorMembershipTimeline.from_groups(
                config.features.sectors.groups,
                exclusive=config.features.sectors.exclusive_membership,
            )
            if config.features.sectors.groups
            else None
        )

    def _members(self, as_of_ns: int) -> UniverseSnapshot:
        result = self.universe_provider.members(as_of_ns)
        if result.as_of_ns != as_of_ns:
            raise ValueError("universe provider returned a different as_of_ns")
        return result

    def _cross_section_at(
        self,
        reader: FeatureWindowReader,
        universe: UniverseSnapshot,
        as_of_ns: int,
        horizons: tuple[WindowSpec, ...],
    ) -> _CrossSectionAt:
        values: dict[str, dict[str, float]] = {}
        stale: list[str] = []
        missing: list[str] = []
        for member in universe.members:
            record = read_member(
                reader,
                self.calendar,
                member.symbol,
                as_of_ns,
                horizons,
                self.config.freshness.bar_limit_ns,
            )
            if record.stale:
                stale.append(member.symbol)
            if record.missing:
                missing.append(member.symbol)
            values[member.symbol] = {
                name: value for name, value in record.values if value is not None
            }
        return _CrossSectionAt(values, tuple(sorted(stale)), tuple(sorted(missing)))

    def _history(
        self,
        reader: FeatureWindowReader,
        day: date,
        settings: DispersionSettings,
    ) -> tuple[float | None, ...]:
        slots: list[float | None] = []
        cursor = day
        for _ in range(max(settings.historical_baseline_sessions, 5)):
            try:
                cursor = self.calendar.previous_trading_day(cursor)
            except CalendarRangeError:
                break
            _, endpoint = self.calendar.regular_session_bounds(cursor)
            universe = self._members(endpoint)
            if universe.source_quality == DataQuality.UNUSABLE:
                slots.append(None)
                continue
            panel = self._cross_section_at(reader, universe, endpoint, (settings.medium_horizon,))
            horizon, _, _ = _horizon_stats(
                panel.values, settings.medium_horizon.name, len(universe.members), settings
            )
            slots.append(
                horizon.return_std
                if horizon.member_coverage >= settings.minimum_member_coverage
                else None
            )
        return tuple(slots)

    def _decomposition(
        self, as_of_ns: int, returns: dict[str, float], settings: DispersionSettings
    ) -> SectorDecomposition | None:
        if self.sector_provider is None:
            return None
        membership = self.sector_provider.members(as_of_ns)
        if membership.as_of_ns != as_of_ns:
            raise ValueError("sector provider returned a different as_of_ns")
        if not membership.exclusive or membership.source_quality == DataQuality.UNUSABLE:
            return None
        assignments = {item.symbol: item.sector for item in membership.members}
        return decompose(
            returns,
            assignments,
            minimum_sectors=settings.minimum_sector_count,
            minimum_coverage=settings.minimum_sector_coverage,
        )

    def _previous_snapshot_std(
        self, reader: FeatureWindowReader, as_of_ns: int, settings: DispersionSettings
    ) -> float | None:
        previous = as_of_ns - self.config.snapshot.cadence_ns
        if previous < 0:
            return None
        info = self.calendar.session_for_timestamp(previous)
        if info.trading_date is None:
            return None
        opened, closed = self.calendar.regular_session_bounds(info.trading_date)
        if not opened < previous <= closed:
            return None
        universe = self._members(previous)
        panel = self._cross_section_at(reader, universe, previous, (settings.medium_horizon,))
        horizon, _, _ = _horizon_stats(
            panel.values, settings.medium_horizon.name, len(universe.members), settings
        )
        return (
            horizon.return_std
            if horizon.member_coverage >= settings.minimum_member_coverage
            else None
        )

    def compute(
        self,
        store: CausalDataStore,
        as_of_ns: int,
        *,
        sector_context: SectorFeatureSet | None = None,
        volatility_context: RealizedVolatilityFeatureSet | None = None,
        correlation_context: CorrelationFeatureSet | None = None,
    ) -> DispersionFeatureSet:
        if as_of_ns < 0:
            raise ValueError("as_of_ns must be nonnegative")
        for context in (sector_context, volatility_context, correlation_context):
            if context is not None and context.as_of_ns != as_of_ns:
                raise ValueError("context and dispersion as_of_ns must match")
        settings = self.config.features.dispersion
        universe = self._members(as_of_ns)
        symbols = tuple(member.symbol for member in universe.members)
        flags = set(universe.quality_flags)
        if (
            universe.source_quality != DataQuality.GOOD
            or settings.price_adjustment_status != "adjusted"
        ):
            flags.add(QualityFlag.PROVIDER_DEGRADED)
        info = self.calendar.session_for_timestamp(as_of_ns)
        day = info.trading_date if info.is_trading_day else None
        reader = FeatureWindowReader(
            store, self.calendar, settings.bar_cadence_ns, self.config.provider.source
        )
        horizons = (
            settings.short_horizon,
            settings.medium_horizon,
            settings.long_horizon,
            *settings.additional_horizons,
        )
        panel = (
            self._cross_section_at(reader, universe, as_of_ns, horizons)
            if settings.enabled and day is not None
            else _CrossSectionAt({}, (), ())
        )
        horizon_results = tuple(
            _horizon_stats(panel.values, item.name, len(symbols), settings) for item in horizons
        )
        medium_index = 1
        primary_horizon, primary, primary_returns = horizon_results[medium_index]
        short = horizon_results[0][0].return_std
        medium = primary_horizon.return_std
        long = horizon_results[2][0].return_std
        valid_symbols = tuple(sorted(primary_returns))
        coverage = len(valid_symbols) / len(symbols) if symbols else 0.0
        missing = tuple(sorted(set(symbols) - set(valid_symbols) - set(panel.stale)))
        if coverage < 1:
            flags.update((QualityFlag.PARTIAL_UNIVERSE, QualityFlag.MISSING_DATA))
        if panel.stale:
            flags.update((QualityFlag.STALE_DATA, QualityFlag.SOURCE_STALE))
        if primary is None:
            flags.add(QualityFlag.INSUFFICIENT_HISTORY)
        if coverage < settings.minimum_member_coverage:
            flags.add(QualityFlag.LOW_COVERAGE)
        if day is None:
            flags.add(QualityFlag.OUTSIDE_SESSION)
        all_weights = bool(universe.members) and all(
            member.weight is not None for member in universe.members
        )
        has_any_weight = any(member.weight is not None for member in universe.members)
        weight_coverage = weighted_dispersion = None
        weighted_hhi = weighted_effective = None
        ordered_returns = tuple(primary_returns[symbol] for symbol in valid_symbols)
        concentration = movement_concentration(ordered_returns)
        if all_weights:
            weights = {member.symbol: member.weight for member in universe.members}
            total = sum(value for value in weights.values() if value is not None)
            selected_weights = tuple(weights[symbol] or 0.0 for symbol in valid_symbols)
            weight_coverage = sum(selected_weights) / total if total > 0 else None
            if weight_coverage is None or weight_coverage < settings.minimum_weight_coverage:
                flags.add(QualityFlag.MISSING_DATA)
            if weight_coverage is not None and weight_coverage >= settings.minimum_weight_coverage:
                weighted_dispersion = weighted_std(ordered_returns, selected_weights)
                weighted_concentration = movement_concentration(ordered_returns, selected_weights)
                if weighted_concentration is not None:
                    weighted_hhi, weighted_effective = weighted_concentration
        elif has_any_weight:
            flags.add(QualityFlag.MISSING_DATA)
        decomposition = self._decomposition(as_of_ns, primary_returns, settings)
        sector_std = sector_relative = sector_mad = sector_spread = None
        sector_valid_count = 0
        if sector_context is not None:
            sector = sector_summary(
                sector_context, settings.medium_horizon.name, settings.minimum_sector_count
            )
            sector_std, sector_relative = sector.absolute_std, sector.relative_std
            sector_mad, sector_spread = sector.relative_mad, sector.spread
            sector_valid_count = sector.valid_count
            flags.update(sector_context.quality_flags)
            if sector_std is None:
                flags.add(QualityFlag.MISSING_SECTOR_DATA)
        elif decomposition is not None:
            means = tuple(value for _, value in decomposition.sector_means)
            sector_stats = cross_section(means, minimum=settings.minimum_sector_count)
            sector_std = sector_stats.std if sector_stats else None
            sector_valid_count = len(means)
            # Subtracting the same broad-market return leaves cross-sector std unchanged.
            reference = read_member(
                reader,
                self.calendar,
                self.config.features.sectors.reference_symbol,
                as_of_ns,
                (settings.medium_horizon,),
                self.config.freshness.bar_limit_ns,
            )
            reference_return = reference.values[0][1]
            if sector_stats is not None and reference_return is not None:
                relative = cross_section(
                    tuple(value - reference_return for value in means),
                    minimum=settings.minimum_sector_count,
                )
                if relative is not None:
                    sector_relative = relative.std
                    sector_mad = relative.mad
                    sector_spread = relative.full_range
        if self.sector_provider is not None and decomposition is None and primary is not None:
            flags.add(QualityFlag.MISSING_SECTOR_DATA)
        volatility_std = volatility_mad = relative_volatility = vol_coverage = None
        vol_count = 0
        if volatility_context is not None:
            flags.update(volatility_context.quality_flags)
            volatility = volatility_summary(
                volatility_context,
                symbols,
                settings.minimum_volatility_coverage,
                settings.minimum_members,
            )
            volatility_std, volatility_mad = volatility.std, volatility.mad
            relative_volatility, vol_coverage = volatility.relative_std, volatility.coverage
            vol_count = volatility.valid_count
            if volatility_std is None:
                flags.add(QualityFlag.MISSING_VOLATILITY_DATA)
        history = (
            self._history(reader, day, settings) if day is not None and settings.enabled else ()
        )
        previous_std = (
            self._previous_snapshot_std(reader, as_of_ns, settings)
            if settings.enabled and medium is not None
            else None
        )
        history_result = summarize_history(
            medium if coverage >= settings.minimum_member_coverage else None,
            history,
            baseline_sessions=settings.historical_baseline_sessions,
            minimum_samples=settings.minimum_baseline_samples,
            persistence_observations=settings.persistence_observations,
        )
        if settings.historical_baseline_sessions and history_result.percentile is None:
            flags.add(QualityFlag.INSUFFICIENT_HISTORY)
        score = (
            history_result.percentile
            if primary is not None
            and coverage >= settings.minimum_member_coverage
            and universe.source_quality != DataQuality.UNUSABLE
            else None
        )
        contributors = (
            (
                ContextContributor(
                    name="dispersion.historical_percentile",
                    value=medium,
                    normalized_value=score,
                    contribution=score,
                    source_feature="dispersion.percentile",
                ),
            )
            if score is not None and medium is not None
            else ()
        )
        corr_result = None
        if correlation_context is not None:
            flags.update(correlation_context.quality_flags)
            correlation_score = correlation_context.systemic_co_movement_score
            percentile = history_result.percentile
            differentiation = (
                percentile * (1 - correlation_score)
                if percentile is not None and correlation_score is not None
                else None
            )
            corr_result = CorrelationDispersionContext(
                average_correlation=correlation_context.average_pairwise_correlation,
                systemic_correlation_score=correlation_score,
                correlation_percentile=correlation_context.correlation_percentile,
                return_dispersion=medium,
                dispersion_percentile=percentile,
                differentiation_score=differentiation,
                dispersion_minus_correlation=percentile - correlation_score
                if percentile is not None and correlation_score is not None
                else None,
                quality_flags=correlation_context.quality_flags,
            )
            if correlation_score is None:
                flags.add(QualityFlag.MISSING_DATA)
        adjustment = 1.0 if settings.price_adjustment_status == "adjusted" else 0.5
        context_factor = 1.0
        if volatility_context is not None and volatility_std is None:
            context_factor *= 0.9
        if (
            correlation_context is not None
            and correlation_context.systemic_co_movement_score is None
        ):
            context_factor *= 0.9
        if self.sector_provider is not None and decomposition is None:
            context_factor *= 0.9
        evidence_quality = (
            coverage * _membership_factor(universe.source_quality) * adjustment * context_factor
        )
        quality = (
            DataQuality.UNUSABLE
            if primary is None or universe.source_quality == DataQuality.UNUSABLE
            else DataQuality.POOR
            if score is None
            else DataQuality.DEGRADED
            if flags
            else DataQuality.GOOD
        )
        sector_counts = dict(decomposition.sector_counts) if decomposition else {}
        sector_means = dict(decomposition.sector_means) if decomposition else {}
        per_sector = (
            tuple(
                SectorReturnDispersion(
                    sector=name,
                    member_count=sector_counts[name],
                    mean_return=sector_means[name],
                    return_std=std,
                )
                for name, std in decomposition.sector_std
            )
            if decomposition is not None
            else ()
        )
        return DispersionFeatureSet(
            as_of_ns=as_of_ns,
            expected_symbols=symbols,
            valid_symbols=valid_symbols,
            stale_symbols=panel.stale,
            missing_symbols=missing,
            expected_member_count=len(symbols),
            valid_member_count=len(valid_symbols),
            member_coverage=coverage,
            weight_coverage=weight_coverage,
            return_dispersion=medium,
            robust_return_dispersion=primary.scaled_mad if primary else None,
            return_mad=primary.mad if primary else None,
            return_iqr=primary.iqr if primary else None,
            return_tail_spread=primary.tail_spread if primary else None,
            return_range=primary.full_range if primary else None,
            weighted_return_dispersion=weighted_dispersion,
            upside_dispersion=primary.upside_std if primary else None,
            downside_dispersion=primary.downside_std if primary else None,
            downside_upside_spread=primary.downside_upside_spread if primary else None,
            cross_sectional_skewness=primary.skewness if primary else None,
            cross_sectional_excess_kurtosis=primary.excess_kurtosis if primary else None,
            horizons=tuple(item[0] for item in horizon_results),
            short_dispersion=short,
            medium_dispersion=medium,
            long_dispersion=long,
            short_long_ratio=short / long
            if short is not None and long is not None and long > 1e-12
            else None,
            dispersion_acceleration=(short - medium) / medium
            if short is not None and medium is not None and medium > 1e-12
            else None,
            dispersion_change=medium - previous_std
            if medium is not None and previous_std is not None
            else None,
            session_change=medium - history[0]
            if medium is not None and history and history[0] is not None
            else None,
            five_session_change=medium - history[4]
            if medium is not None and len(history) >= 5 and history[4] is not None
            else None,
            baseline_sample_count=history_result.sample_count,
            dispersion_percentile=history_result.percentile,
            dispersion_robust_z=history_result.robust_z,
            dispersion_shock_score=history_result.shock,
            dispersion_persistence=history_result.persistence,
            sector_dispersion=sector_std,
            sector_relative_dispersion=sector_relative,
            sector_relative_mad=sector_mad,
            sector_spread=sector_spread,
            sector_valid_count=sector_valid_count,
            sector_return_dispersion=per_sector,
            decomposition_member_count=decomposition.member_count if decomposition else 0,
            decomposition_coverage=decomposition.coverage if decomposition else None,
            decomposition_total_variance=decomposition.total_variance if decomposition else None,
            within_sector_variance=decomposition.within_variance if decomposition else None,
            between_sector_variance=decomposition.between_variance if decomposition else None,
            within_sector_share=decomposition.within_share if decomposition else None,
            between_sector_share=decomposition.between_share if decomposition else None,
            volatility_dispersion=volatility_std,
            robust_volatility_dispersion=volatility_mad,
            relative_volatility_dispersion=relative_volatility,
            volatility_member_count=vol_count,
            volatility_coverage=vol_coverage,
            movement_hhi=concentration[0] if concentration else None,
            effective_movers=concentration[1] if concentration else None,
            weighted_movement_hhi=weighted_hhi,
            weighted_effective_movers=weighted_effective,
            correlation_context=corr_result,
            dispersion_score=score,
            evidence_quality=evidence_quality,
            data_quality=quality,
            contributors=contributors,
            quality_flags=frozenset(flags),
        )
