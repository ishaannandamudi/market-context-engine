"""Explicit-clock, point-in-time correlation and systemic co-movement engine."""

from dataclasses import dataclass
from datetime import date
from statistics import median

import numpy as np

from mce.calendar.service import MarketSessionService
from mce.config.settings import CorrelationSettings, MCEConfig
from mce.errors import CalendarRangeError
from mce.features.breadth.membership import MembershipTimeline, UniverseProvider, UniverseSnapshot
from mce.features.correlation.diagnostics import (
    complete_link_groups,
    eigen_diagnostics,
    shrink_identity,
    summary,
    weighted_average,
)
from mce.features.correlation.matrix import MatrixEstimate, estimate
from mce.features.correlation.models import (
    CorrelationFeatureSet,
    CorrelationMatrix,
    HorizonCorrelation,
    SectorPairCorrelation,
)
from mce.features.correlation.returns import CorrelationReturnReader
from mce.features.sectors.membership import SectorMembershipTimeline, SectorUniverseProvider
from mce.models.context import ContextContributor
from mce.models.enums import DataQuality, QualityFlag
from mce.replay.store import CausalDataStore


@dataclass(frozen=True)
class _Window:
    estimate: MatrixEstimate | None
    expected: int
    aligned: int
    coverage: float
    stale: tuple[str, ...]
    missing: tuple[str, ...]
    constant: tuple[str, ...]
    average: float | None


def _prior_distribution(current: float, prior: tuple[float, ...]) -> tuple[float, float | None]:
    percentile = (
        sum(value < current for value in prior) + 0.5 * sum(value == current for value in prior)
    ) / len(prior)
    center = median(prior)
    mad = median(abs(value - center) for value in prior)
    z = (current - center) / (1.4826 * mad) if mad > 0 else (0.0 if current == center else None)
    return percentile, z


class CorrelationFeatureEngine:
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
            config.features.correlation.universe
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
        snapshot = self.universe_provider.members(as_of_ns)
        if snapshot.as_of_ns != as_of_ns:
            raise ValueError("universe provider returned a different as_of_ns")
        return snapshot

    def _window(
        self,
        reader: CorrelationReturnReader,
        universe: UniverseSnapshot,
        day: date,
        as_of_ns: int,
        bars: int,
        settings: CorrelationSettings,
        *,
        daily: bool = False,
    ) -> _Window:
        data: dict[str, dict[int, float]] = {}
        stale: list[str] = []
        missing: list[str] = []
        constant: list[str] = []
        for member in universe.members:
            series = (
                reader.daily(member.symbol, day, as_of_ns, bars)
                if daily
                else reader.intraday(member.symbol, day, as_of_ns, bars)
            )
            if not series.latest_price_available:
                (stale if series.latest_effective_ns is not None else missing).append(member.symbol)
                continue
            if not series.latest_available:
                missing.append(member.symbol)
                continue
            if (
                not daily
                and series.latest_effective_ns is not None
                and as_of_ns - series.latest_effective_ns > self.config.freshness.bar_limit_ns
            ):
                stale.append(member.symbol)
                continue
            if len(series.values) / bars < settings.minimum_time_coverage:
                continue
            values = tuple(series.values.values())
            if len(values) >= 2 and max(values) - min(values) <= 1e-12:
                constant.append(member.symbol)
                continue
            data[member.symbol] = series.values
        # A series can become constant on the common intersection. Remove it before
        # estimating, while keeping all remaining columns on the same timestamps.
        while len(data) >= settings.minimum_symbols:
            common = set.intersection(*(set(item) for item in data.values()))
            if len(common) < settings.minimum_aligned_samples:
                break
            degenerate = tuple(
                symbol
                for symbol, values in sorted(data.items())
                if max(values[t] for t in common) - min(values[t] for t in common) <= 1e-12
            )
            if not degenerate:
                break
            for symbol in degenerate:
                constant.append(symbol)
                del data[symbol]
        common = set.intersection(*(set(item) for item in data.values())) if data else set()
        aligned = len(common)
        coverage = aligned / bars
        result = (
            estimate(
                data, bars, settings.minimum_aligned_samples, settings.robust_estimator_enabled
            )
            if len(data) >= settings.minimum_symbols and coverage >= settings.minimum_time_coverage
            else None
        )
        average = summary(np.asarray(result.matrix))[0] if result else None
        return _Window(
            result,
            bars,
            aligned,
            coverage,
            tuple(sorted(stale)),
            tuple(sorted(missing)),
            tuple(sorted(constant)),
            average,
        )

    def _historical(
        self,
        reader: CorrelationReturnReader,
        day: date,
        settings: CorrelationSettings,
    ) -> tuple[float | None, ...]:
        values: list[float | None] = []
        cursor = day
        for _ in range(max(settings.historical_baseline_sessions, 5)):
            try:
                cursor = self.calendar.previous_trading_day(cursor)
            except CalendarRangeError:
                break
            _, endpoint = self.calendar.regular_session_bounds(cursor)
            universe = self._members(endpoint)
            window = self._window(
                reader, universe, cursor, endpoint, settings.medium_window_bars, settings
            )
            if (
                window.average is None
                or window.estimate is None
                or not universe.members
                or universe.source_quality == DataQuality.UNUSABLE
            ):
                values.append(None)
                continue
            if (
                len(window.estimate.symbols) / len(universe.members)
                >= settings.minimum_symbol_coverage
            ):
                values.append(window.average)
            else:
                values.append(None)
        return tuple(values)

    def _sectors(
        self, as_of_ns: int, estimate_result: MatrixEstimate
    ) -> tuple[tuple[SectorPairCorrelation, ...], float | None, float | None]:
        if self.sector_provider is None:
            return (), None, None
        membership = self.sector_provider.members(as_of_ns)
        if membership.as_of_ns != as_of_ns:
            raise ValueError("sector provider returned a different as_of_ns")
        if not membership.exclusive:
            return (), None, None
        sector = {item.symbol: item.sector for item in membership.members}
        pairs: dict[tuple[str, str], list[float]] = {}
        matrix = estimate_result.matrix
        for i, first in enumerate(estimate_result.symbols):
            for j in range(i + 1, len(estimate_result.symbols)):
                second = estimate_result.symbols[j]
                if first in sector and second in sector:
                    pair = (min(sector[first], sector[second]), max(sector[first], sector[second]))
                    pairs.setdefault(pair, []).append(matrix[i][j])
        summaries = tuple(
            SectorPairCorrelation(
                first=first,
                second=second,
                pair_count=len(values),
                average=sum(values) / len(values),
            )
            for (first, second), values in sorted(pairs.items())
        )
        within = [
            value
            for (first, second), values in pairs.items()
            if first == second
            for value in values
        ]
        cross = [
            value
            for (first, second), values in pairs.items()
            if first != second
            for value in values
        ]
        return (
            summaries,
            (sum(within) / len(within) if within else None),
            (sum(cross) / len(cross) if cross else None),
        )

    def compute(self, store: CausalDataStore, as_of_ns: int) -> CorrelationFeatureSet:
        if as_of_ns < 0:
            raise ValueError("as_of_ns must be nonnegative")
        settings = self.config.features.correlation
        universe = self._members(as_of_ns)
        expected_symbols = tuple(item.symbol for item in universe.members)
        flags = set(universe.quality_flags)
        if universe.source_quality != DataQuality.GOOD:
            flags.add(QualityFlag.PROVIDER_DEGRADED)
        info = self.calendar.session_for_timestamp(as_of_ns)
        day = info.trading_date if info.is_trading_day else None
        reader = CorrelationReturnReader(
            store, self.calendar, self.config.provider.source, settings.bar_cadence_ns
        )
        if settings.enabled and day is not None:
            short = self._window(
                reader, universe, day, as_of_ns, settings.short_window_bars, settings
            )
            medium = self._window(
                reader, universe, day, as_of_ns, settings.medium_window_bars, settings
            )
            long = self._window(
                reader, universe, day, as_of_ns, settings.long_window_bars, settings
            )
            _, closed = self.calendar.regular_session_bounds(day)
            try:
                daily_day = day if as_of_ns >= closed else self.calendar.previous_trading_day(day)
            except CalendarRangeError:
                daily_day = None
            daily_results = tuple(
                (
                    sessions,
                    self._window(
                        reader, universe, daily_day, as_of_ns, sessions, settings, daily=True
                    ),
                )
                for sessions in settings.daily_horizons_sessions
                if daily_day is not None
            )
        else:
            short = _Window(None, settings.short_window_bars, 0, 0.0, (), (), (), None)
            medium = _Window(None, settings.medium_window_bars, 0, 0.0, (), (), (), None)
            long = _Window(None, settings.long_window_bars, 0, 0.0, (), (), (), None)
            daily_results = ()
        primary = medium.estimate
        valid_symbols = primary.symbols if primary is not None else ()
        symbol_coverage = len(valid_symbols) / len(expected_symbols) if expected_symbols else 0.0
        if len(valid_symbols) < len(expected_symbols):
            flags.update((QualityFlag.PARTIAL_UNIVERSE, QualityFlag.MISSING_DATA))
        if medium.stale:
            flags.update((QualityFlag.STALE_DATA, QualityFlag.SOURCE_STALE))
        if medium.missing:
            flags.add(QualityFlag.MISSING_DATA)
        if medium.constant or primary is None:
            flags.add(QualityFlag.INSUFFICIENT_HISTORY)
        if (
            symbol_coverage < settings.minimum_symbol_coverage
            or medium.coverage < settings.minimum_time_coverage
        ):
            flags.add(QualityFlag.LOW_COVERAGE)
        if not expected_symbols:
            flags.add(QualityFlag.MISSING_DATA)
        if day is None:
            flags.add(QualityFlag.OUTSIDE_SESSION)
        average = median_pair = dispersion = spearman_average = None
        weighted = weight_coverage = eigenvalue = share = dimension = mode = score = None
        matrix_model = rank_model = shrunk_model = None
        groups: tuple[tuple[str, ...], ...] = ()
        sector_pairs: tuple[SectorPairCorrelation, ...] = ()
        within = cross = None
        contributors: tuple[ContextContributor, ...] = ()
        if primary is not None:
            matrix = np.asarray(primary.matrix)
            average, median_pair, dispersion = summary(matrix)
            matrix_model = CorrelationMatrix(symbols=primary.symbols, values=primary.matrix)
            if primary.spearman_matrix is not None:
                rank_model = CorrelationMatrix(
                    symbols=primary.symbols, values=primary.spearman_matrix
                )
                spearman_average = summary(np.asarray(primary.spearman_matrix))[0]
            shrunk = shrink_identity(matrix, settings.shrinkage_strength)
            shrunk_model = CorrelationMatrix(
                symbols=primary.symbols,
                values=tuple(tuple(float(value) for value in row) for row in shrunk),
            )
            eigen = eigen_diagnostics(shrunk)
            if eigen is not None:
                eigenvalue, share, dimension = eigen
                mode = max(
                    0.0, min(1.0, (share - 1 / len(valid_symbols)) / (1 - 1 / len(valid_symbols)))
                )
            else:
                flags.add(QualityFlag.REJECTED_DATA)
            weights = {item.symbol: item.weight for item in universe.members}
            if all(value is not None for value in weights.values()) and weights:
                total_weight = sum(value for value in weights.values() if value is not None)
                valid_weight = sum(weights[symbol] or 0.0 for symbol in valid_symbols)
                weight_coverage = valid_weight / total_weight if total_weight > 0 else None
                if (
                    weight_coverage is not None
                    and weight_coverage >= settings.minimum_symbol_coverage
                ):
                    weighted = weighted_average(
                        matrix, tuple(weights[symbol] or 0.0 for symbol in valid_symbols)
                    )
            if settings.clustering_enabled:
                groups = complete_link_groups(
                    matrix, valid_symbols, settings.cluster_correlation_threshold
                )
            sector_pairs, within, cross = self._sectors(as_of_ns, primary)
            if (
                mode is not None
                and share is not None
                and average is not None
                and symbol_coverage >= settings.minimum_symbol_coverage
                and medium.coverage >= settings.minimum_time_coverage
                and universe.source_quality != DataQuality.UNUSABLE
            ):
                positive = max(0.0, average)
                weights_config = settings.score_weights
                average_part = weights_config.average * positive
                mode_part = weights_config.market_mode * positive * mode
                score = min(1.0, max(0.0, average_part + mode_part))
                contributors = (
                    ContextContributor(
                        name="correlation.positive_average",
                        value=average,
                        normalized_value=positive,
                        contribution=average_part,
                        source_feature="correlation.average",
                    ),
                    ContextContributor(
                        name="correlation.market_mode",
                        value=share,
                        normalized_value=mode,
                        contribution=mode_part,
                        source_feature="correlation.first_eigen_share",
                    ),
                )
        history = (
            self._historical(reader, day, settings) if day is not None and settings.enabled else ()
        )
        percentile = robust_z = None
        prior = tuple(
            value for value in history[: settings.historical_baseline_sessions] if value is not None
        )
        if average is not None and len(prior) >= settings.minimum_baseline_samples:
            percentile, robust_z = _prior_distribution(average, prior)
        elif settings.historical_baseline_sessions and settings.enabled:
            flags.add(QualityFlag.INSUFFICIENT_HISTORY)
        previous_interval = None
        if day is not None and settings.enabled:
            previous_asof = max(0, as_of_ns - settings.bar_cadence_ns)
            previous_universe = self._members(previous_asof)
            previous_interval = self._window(
                reader, previous_universe, day, previous_asof, settings.medium_window_bars, settings
            ).average
        evidence_quality = (
            symbol_coverage
            * medium.coverage
            * {
                DataQuality.GOOD: 1.0,
                DataQuality.DEGRADED: 0.5,
                DataQuality.POOR: 0.25,
                DataQuality.UNUSABLE: 0.0,
                DataQuality.UNKNOWN: 0.5,
            }[universe.source_quality]
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
        return CorrelationFeatureSet(
            as_of_ns=as_of_ns,
            expected_symbols=expected_symbols,
            valid_symbols=valid_symbols,
            stale_symbols=medium.stale,
            missing_symbols=medium.missing,
            constant_symbols=medium.constant,
            expected_symbol_count=len(expected_symbols),
            valid_symbol_count=len(valid_symbols),
            symbol_coverage=symbol_coverage,
            expected_observations=medium.expected,
            aligned_sample_count=medium.aligned,
            time_coverage=medium.coverage,
            matrix_dimension=len(valid_symbols),
            correlation_matrix=matrix_model,
            spearman_matrix=rank_model,
            shrunk_matrix=shrunk_model,
            average_pairwise_correlation=average,
            median_pairwise_correlation=median_pair,
            weighted_average_correlation=weighted,
            correlation_dispersion=dispersion,
            spearman_average=spearman_average,
            short_correlation=short.average,
            medium_correlation=medium.average,
            long_correlation=long.average,
            daily_horizons=tuple(
                HorizonCorrelation(
                    name=f"{sessions}d",
                    expected_observations=result.expected,
                    aligned_sample_count=result.aligned,
                    time_coverage=result.coverage,
                    symbol_coverage=(
                        len(result.estimate.symbols) / len(expected_symbols)
                        if result.estimate is not None and expected_symbols
                        else 0.0
                    ),
                    average=result.average,
                    matrix=(
                        CorrelationMatrix(
                            symbols=result.estimate.symbols,
                            values=result.estimate.matrix,
                        )
                        if result.estimate is not None
                        else None
                    ),
                )
                for sessions, result in daily_results
            ),
            correlation_acceleration=short.average - medium.average
            if short.average is not None and medium.average is not None
            else None,
            correlation_change=average - previous_interval
            if average is not None and previous_interval is not None
            else None,
            session_change=average - history[0]
            if average is not None and history and history[0] is not None
            else None,
            five_session_change=average - history[4]
            if average is not None and len(history) >= 5 and history[4] is not None
            else None,
            correlation_percentile=percentile,
            correlation_robust_z=robust_z,
            largest_eigenvalue=eigenvalue,
            first_eigenvalue_share=share,
            effective_dimension=dimension,
            market_mode_strength=mode,
            systemic_co_movement_score=score,
            cluster_groups=groups,
            sector_pairs=sector_pairs,
            within_sector_correlation=within,
            cross_sector_correlation=cross,
            within_cross_spread=within - cross
            if within is not None and cross is not None
            else None,
            weight_coverage=weight_coverage,
            evidence_quality=evidence_quality,
            data_quality=quality,
            contributors=contributors,
            quality_flags=frozenset(flags),
        )
