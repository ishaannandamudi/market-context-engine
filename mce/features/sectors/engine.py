"""Explicit-clock sector participation, relative strength, volatility and rotation."""

from collections import defaultdict
from datetime import date
from zoneinfo import ZoneInfo

from mce.calendar.service import MarketSessionService
from mce.config.settings import MCEConfig, SectorSettings
from mce.errors import CalendarRangeError
from mce.features.breadth.membership import UniverseMembership
from mce.features.breadth.prices import BreadthPriceReader
from mce.features.returns import simple_return
from mce.features.sectors.breadth import reference_participation, sector_direction
from mce.features.sectors.membership import (
    SectorMembershipTimeline,
    SectorUniverseProvider,
    SectorUniverseSnapshot,
)
from mce.features.sectors.models import SectorContext, SectorFeatureSet, SectorMemberFeatures
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
    horizon_return,
    relative_score,
)
from mce.features.sectors.volatility import daily_composites, volatility_context
from mce.features.windows import FeatureWindowReader
from mce.models.enums import DataQuality, QualityFlag
from mce.models.market_data import BarObservation
from mce.models.time import ns_to_datetime
from mce.quality.freshness import evaluate_freshness
from mce.replay.store import CausalDataStore


def _quality(score: float | None, flags: set[QualityFlag]) -> DataQuality:
    if score is None:
        return DataQuality.UNUSABLE
    return DataQuality.DEGRADED if flags else DataQuality.GOOD


def _membership_factor(quality: DataQuality) -> float:
    return {
        DataQuality.GOOD: 1.0,
        DataQuality.DEGRADED: 0.5,
        DataQuality.POOR: 0.25,
        DataQuality.UNUSABLE: 0.0,
        DataQuality.UNKNOWN: 0.5,
    }[quality]


def _difference(first: float | None, second: float | None) -> float | None:
    return first - second if first is not None and second is not None else None


class SectorFeatureEngine:
    def __init__(
        self,
        config: MCEConfig,
        universe_provider: SectorUniverseProvider | None = None,
        calendar: MarketSessionService | None = None,
    ) -> None:
        self.config = config
        if calendar is not None and calendar.settings.model_dump(
            exclude={"universe"}
        ) != config.market.model_dump(exclude={"universe"}):
            raise ValueError("calendar settings must match market configuration")
        self.calendar = calendar or MarketSessionService(config.market)
        self.universe_provider = universe_provider or SectorMembershipTimeline.from_groups(
            config.features.sectors.groups,
            exclusive=config.features.sectors.exclusive_membership,
        )

    def _members_at(self, as_of_ns: int) -> SectorUniverseSnapshot:
        result = self.universe_provider.members(as_of_ns)
        if result.as_of_ns != as_of_ns:
            raise ValueError("sector provider returned a snapshot for a different as_of_ns")
        return result

    def _stale(self, bar: BarObservation, as_of_ns: int) -> bool:
        limit = self.config.freshness.bar_limit_ns
        freshness = evaluate_freshness(bar.source, bar.effective_timestamp_ns, as_of_ns, limit)
        return bool(freshness and freshness.is_stale) or (
            bar.bar_end_ns is not None and as_of_ns - bar.bar_end_ns > limit
        )

    def _current(
        self, reader: BreadthPriceReader, symbol: str, day: date | None, as_of_ns: int
    ) -> BarObservation | None:
        bar = reader.current(symbol, day, as_of_ns) if day is not None else None
        return bar if bar is not None and not self._stale(bar, as_of_ns) else None

    def _completed_day(self, as_of_ns: int) -> date | None:
        local = (
            ns_to_datetime(as_of_ns)
            .astimezone(ZoneInfo(self.config.market.exchange_timezone))
            .date()
        )
        try:
            if self.calendar.is_trading_day(local):
                _, closed = self.calendar.regular_session_bounds(local)
                if closed <= as_of_ns:
                    return local
            return self.calendar.previous_trading_day(local)
        except CalendarRangeError:
            return None

    def _core(
        self,
        reader: BreadthPriceReader,
        windows: FeatureWindowReader,
        as_of_ns: int,
        *,
        references: bool,
    ) -> dict[str, SectorContext]:
        settings = self.config.features.sectors
        universe = self._members_at(as_of_ns)
        info = self.calendar.session_for_timestamp(as_of_ns)
        day = info.trading_date if info.is_trading_day else None
        try:
            previous = self.calendar.previous_trading_day(day) if day is not None else None
        except CalendarRangeError:
            previous = None
        horizons = (
            settings.short_horizon,
            settings.medium_horizon,
            settings.long_horizon,
            *settings.additional_horizons,
        )
        market_bar = self._current(reader, settings.reference_symbol, day, as_of_ns)
        market_returns = {
            horizon.name: horizon_return(
                windows, settings.reference_symbol, market_bar, as_of_ns, horizon
            )
            for horizon in horizons
        }
        grouped: dict[str, list[UniverseMembership]] = defaultdict(list)
        for item in universe.members:
            grouped[item.sector].append(item)
        result: dict[str, SectorContext] = {}
        for sector, assignments in sorted(grouped.items()):
            expected = len(assignments)
            member_models: dict[str, SectorMemberFeatures] = {}
            by_horizon: dict[str, list[tuple[float | None, float | None]]] = {
                horizon.name: [] for horizon in horizons
            }
            session_returns: list[float] = []
            current_and_prior: list[tuple[float, tuple[float | None, ...]]] = []
            stale_count = 0
            for member in assignments:
                symbol = member.symbol
                original = reader.current(symbol, day, as_of_ns) if day is not None else None
                stale = original is not None and self._stale(original, as_of_ns)
                stale_count += int(stale)
                current = original if not stale else None
                previous_close = (
                    reader.session_close(symbol, previous, as_of_ns)
                    if previous is not None
                    else None
                )
                session_return = (
                    simple_return(previous_close, current.close)
                    if previous_close is not None and current is not None
                    else None
                )
                if session_return is not None:
                    session_returns.append(session_return)
                    if references and day is not None and current is not None:
                        current_and_prior.append(
                            (
                                current.close,
                                reader.prior_closes(
                                    symbol, day, max(settings.breadth_reference_sessions), as_of_ns
                                ),
                            )
                        )
                member_flags: set[QualityFlag] = set()
                if stale:
                    member_flags.add(QualityFlag.STALE_DATA)
                elif session_return is None:
                    member_flags.add(QualityFlag.MISSING_DATA)
                member_models[symbol] = SectorMemberFeatures(
                    symbol=symbol,
                    weight=member.weight,
                    current_price=original.close if original is not None else None,
                    previous_close=previous_close,
                    session_return=session_return,
                    stale=stale,
                    quality_flags=frozenset(member_flags),
                )
                for horizon in horizons:
                    by_horizon[horizon.name].append(
                        (
                            horizon_return(windows, symbol, current, as_of_ns, horizon),
                            member.weight,
                        )
                    )
            valid = len(session_returns)
            coverage = valid / expected
            advances, declines, unchanged, net = sector_direction(
                tuple(session_returns), self.config.features.breadth.unchanged_return_tolerance
            )
            ref_results = (
                {
                    f"{length}d": reference_participation(tuple(current_and_prior), length)
                    for length in settings.breadth_reference_sessions
                }
                if references
                else {}
            )
            aggregate = {
                name: aggregate_returns(
                    tuple(values),
                    minimum_member_coverage=settings.minimum_member_coverage,
                    minimum_weight_coverage=settings.minimum_weight_coverage,
                )
                for name, values in by_horizon.items()
            }
            absolute = {name: item.equal_return for name, item in aggregate.items()}
            weighted = {name: item.weighted_return for name, item in aggregate.items()}
            horizon_coverage = {name: item.member_coverage for name, item in aggregate.items()}
            weight_coverage = {name: item.weight_coverage for name, item in aggregate.items()}
            proxy_symbol = settings.proxy_mapping.get(sector)
            proxy_bar = (
                self._current(reader, proxy_symbol, day, as_of_ns)
                if proxy_symbol is not None
                else None
            )
            proxy = {
                horizon.name: horizon_return(windows, proxy_symbol, proxy_bar, as_of_ns, horizon)
                if proxy_symbol is not None
                else None
                for horizon in horizons
            }
            relative = {
                name: _difference(value, market_returns[name]) for name, value in absolute.items()
            }
            divergence = {name: _difference(absolute[name], value) for name, value in proxy.items()}
            proxy_previous = (
                reader.session_close(proxy_symbol, previous, as_of_ns)
                if proxy_symbol is not None and previous is not None
                else None
            )
            proxy_session = (
                simple_return(proxy_previous, proxy_bar.close)
                if proxy_previous is not None and proxy_bar is not None
                else None
            )
            breadth_proxy = (
                net - (1 if proxy_session > 0 else -1 if proxy_session < 0 else 0)
                if net is not None and proxy_session is not None
                else None
            )
            enough_members = expected >= settings.minimum_member_count
            breadth = (
                net if enough_members and coverage >= settings.minimum_member_coverage else None
            )
            names = (
                settings.short_horizon.name,
                settings.medium_horizon.name,
                settings.long_horizon.name,
            )
            relative_triplet = tuple((name, relative[name]) for name in names)
            score, contributors = relative_score(
                sector, relative_triplet, breadth, settings.score_weights, settings.return_scale
            )
            if (
                not enough_members
                or coverage < settings.minimum_member_coverage
                or universe.source_quality == DataQuality.UNUSABLE
            ):
                score, contributors = None, ()
            medium_return = absolute[settings.medium_horizon.name]
            absolute_score = (
                bounded_return(medium_return, settings.return_scale)
                if medium_return is not None
                else None
            )
            short_rel = relative[settings.short_horizon.name]
            medium_rel = relative[settings.medium_horizon.name]
            momentum = (
                short_rel - medium_rel if short_rel is not None and medium_rel is not None else None
            )
            flags = set(universe.quality_flags)
            if (
                universe.source_quality != DataQuality.GOOD
                or settings.price_adjustment_status != "adjusted"
            ):
                flags.add(QualityFlag.PROVIDER_DEGRADED)
            if valid < expected:
                flags.update({QualityFlag.PARTIAL_UNIVERSE, QualityFlag.MISSING_DATA})
            if stale_count:
                flags.update({QualityFlag.STALE_DATA, QualityFlag.SOURCE_STALE})
            if not enough_members or coverage < settings.minimum_member_coverage:
                flags.add(QualityFlag.LOW_COVERAGE)
            if all(relative[name] is None for name in names):
                flags.add(QualityFlag.MISSING_DATA)
            if any(market_returns[name] is None for name in names):
                flags.add(QualityFlag.MISSING_DATA)
            if proxy_symbol is not None and all(value is None for value in proxy.values()):
                flags.add(QualityFlag.MISSING_DATA)
            if references and any(item.comparable_count < valid for item in ref_results.values()):
                flags.add(QualityFlag.INSUFFICIENT_HISTORY)
            if any(item.weight is None for item in assignments) and any(
                item.weight is not None for item in assignments
            ):
                flags.add(QualityFlag.MISSING_DATA)
            relative_available = sum(relative[name] is not None for name in names) / 3
            weight_factor = weight_coverage[settings.medium_horizon.name]
            evidence = (
                coverage
                * relative_available
                * _membership_factor(universe.source_quality)
                * (1.0 if settings.price_adjustment_status == "adjusted" else 0.5)
                * (weight_factor if weight_factor is not None else 1.0)
            )
            result[sector] = SectorContext(
                sector=sector,
                as_of_ns=as_of_ns,
                members=member_models,
                expected_member_count=expected,
                valid_member_count=valid,
                stale_member_count=stale_count,
                member_coverage=coverage,
                weight_coverage=weight_coverage[settings.medium_horizon.name],
                advancers=advances,
                decliners=declines,
                unchanged=unchanged,
                net_advance_fraction=net,
                breadth_references=ref_results,
                breadth_score=breadth,
                absolute_returns=absolute,
                horizon_coverage=horizon_coverage,
                weighted_returns=weighted,
                weight_coverage_by_horizon=weight_coverage,
                proxy_returns=proxy,
                relative_returns=relative,
                proxy_divergence=divergence,
                breadth_proxy_divergence=breadth_proxy,
                cross_horizon_agreement=cross_horizon_agreement(
                    tuple(value for _, value in relative_triplet)
                ),
                absolute_strength_score=absolute_score,
                relative_strength_score=score,
                sector_momentum=momentum,
                evidence_quality=evidence,
                data_quality=_quality(score, flags),
                contributors=contributors,
                quality_flags=frozenset(flags),
            )
        return result

    def compute(self, store: CausalDataStore, as_of_ns: int) -> SectorFeatureSet:
        if as_of_ns < 0:
            raise ValueError("as_of_ns must be nonnegative")
        settings: SectorSettings = self.config.features.sectors
        if not settings.enabled:
            return SectorFeatureSet(
                as_of_ns=as_of_ns,
                sectors={},
                market_coverage=0.0,
                evidence_quality=0.0,
                data_quality=DataQuality.UNKNOWN,
            )
        reader = BreadthPriceReader(store, self.calendar, self.config.provider.source)
        windows = FeatureWindowReader(
            store, self.calendar, settings.bar_cadence_ns, self.config.provider.source
        )
        current = self._core(reader, windows, as_of_ns, references=True)
        current_scores = {name: item.relative_strength_score for name, item in current.items()}
        ranks = rank_scores(current_scores)
        history: list[dict[str, SectorContext]] = []
        history_ranks: list[dict[str, tuple[int, float]]] = []
        cursor = self.calendar.session_for_timestamp(as_of_ns).trading_date
        if cursor is None:
            cursor = self._completed_day(as_of_ns)
        for _ in range(
            max(settings.persistence_window_sessions - 1, settings.rotation_window_sessions)
        ):
            if cursor is None:
                break
            try:
                cursor = self.calendar.previous_trading_day(cursor)
            except CalendarRangeError:
                break
            _, endpoint = self.calendar.regular_session_bounds(cursor)
            past = self._core(reader, windows, endpoint, references=False)
            history.append(past)
            history_ranks.append(
                rank_scores({name: item.relative_strength_score for name, item in past.items()})
            )
        completed = self._completed_day(as_of_ns)
        daily_count = max(
            settings.volatility_windows_sessions[-1],
            settings.volatility_baseline_sessions + settings.volatility_windows_sessions[0],
        )
        composites, market_daily = daily_composites(
            reader, self.calendar, self._members_at, completed, daily_count, settings
        )
        finished: dict[str, SectorContext] = {}
        for name, item in current.items():
            vol = volatility_context(
                composites.get(name, ()),
                market_daily,
                settings,
                self.config.features.volatility.trading_sessions_per_year,
            )
            prior_item = history[0].get(name) if history else None
            prior_rank = history_ranks[0].get(name) if history_ranks else None
            rank = ranks.get(name)
            flags = set(item.quality_flags)
            if vol.medium is None:
                flags.add(QualityFlag.INSUFFICIENT_HISTORY)
            if vol.market is None:
                flags.add(QualityFlag.MISSING_VOLATILITY_DATA)
            if rank is None:
                flags.add(QualityFlag.MISSING_SECTOR_DATA)
            persistence = (
                leadership_persistence(
                    name, ranks, tuple(history_ranks), settings.persistence_window_sessions
                )
                if rank is not None
                else None
            )
            if persistence is None:
                flags.add(QualityFlag.INSUFFICIENT_HISTORY)
            payload = item.model_dump(mode="python")
            payload.update(
                volatility_short=vol.short,
                realized_volatility=vol.medium,
                volatility_long=vol.long,
                volatility_percentile=vol.percentile,
                market_volatility=vol.market,
                relative_volatility=vol.relative,
                rank=rank[0] if rank else None,
                percentile=rank[1] if rank else None,
                leadership_persistence=persistence,
                rank_change=prior_rank[0] - rank[0] if prior_rank and rank else None,
                relative_strength_change=(
                    item.relative_strength_score - prior_item.relative_strength_score
                    if prior_item is not None
                    and item.relative_strength_score is not None
                    and prior_item.relative_strength_score is not None
                    else None
                ),
                sector_acceleration=(
                    item.sector_momentum - prior_item.sector_momentum
                    if prior_item is not None
                    and item.sector_momentum is not None
                    and prior_item.sector_momentum is not None
                    else None
                ),
                evidence_quality=item.evidence_quality * (1.0 if vol.medium is not None else 0.5),
                quality_flags=frozenset(flags),
                data_quality=_quality(item.relative_strength_score, flags),
            )
            finished[name] = SectorContext.model_validate(payload)
        valid_names = tuple(name for name, _ in sorted(ranks.items(), key=lambda pair: pair[1][0]))
        enough = len(valid_names) >= settings.minimum_valid_sectors
        leader = valid_names[0] if enough else None
        laggard = valid_names[-1] if enough else None
        previous_ranks = history_ranks[0] if history_ranks else {}
        previous_order = sorted(previous_ranks.items(), key=lambda pair: pair[1][0])
        previous_leader = (
            previous_order[0][0] if len(previous_order) >= settings.minimum_valid_sectors else None
        )
        spread = (
            _difference(current_scores[leader], current_scores[laggard])
            if leader is not None and laggard is not None
            else None
        )
        market_coverage = len(ranks) / len(current) if current else 0.0
        flags = (
            set().union(*(item.quality_flags for item in finished.values())) if finished else set()
        )
        if not enough:
            flags.add(QualityFlag.LOW_COVERAGE)
        if not current:
            flags.add(QualityFlag.MISSING_SECTOR_DATA)
        velocity = (
            rotation_velocity(ranks, tuple(history_ranks), settings.rotation_window_sessions)
            if enough
            else None
        )
        if velocity is None:
            flags.add(QualityFlag.INSUFFICIENT_HISTORY)
        concentration = leadership_hhi(current_scores)
        evidence = (
            sum(item.evidence_quality for item in finished.values()) / len(finished)
            if finished
            else 0.0
        )
        return SectorFeatureSet(
            as_of_ns=as_of_ns,
            sectors=finished,
            leader_sector=leader,
            laggard_sector=laggard,
            previous_leader_sector=previous_leader,
            leader_changed=leader != previous_leader
            if leader is not None and previous_leader is not None
            else None,
            leadership_spread=spread,
            leadership_concentration=concentration,
            sector_score_dispersion=spread,
            rotation_velocity=velocity,
            market_coverage=market_coverage,
            evidence_quality=evidence,
            data_quality=_quality(spread, flags),
            contributors=tuple(term for item in finished.values() for term in item.contributors),
            quality_flags=frozenset(flags),
        )
