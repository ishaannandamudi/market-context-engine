"""Explicit-clock market breadth over point-in-time membership and completed bars."""

from datetime import date
from math import isclose

from mce.calendar.service import MarketSessionService
from mce.config.settings import BreadthSettings, MCEConfig
from mce.errors import CalendarRangeError
from mce.features.breadth.history import (
    BreadthHistoryPoint,
    summarize_history,
)
from mce.features.breadth.math import (
    breadth_score_components,
    contribution_concentration,
    direction_counts,
    median_return,
    weighted_return,
)
from mce.features.breadth.membership import (
    MembershipTimeline,
    UniverseProvider,
    UniverseSnapshot,
)
from mce.features.breadth.models import BreadthFeatureSet, MemberBreadth, ReferenceParticipation
from mce.features.breadth.prices import BreadthPriceReader
from mce.features.returns import simple_return
from mce.models.context import ContextContributor
from mce.models.enums import DataQuality, QualityFlag
from mce.quality.freshness import evaluate_freshness
from mce.replay.store import CausalDataStore


def _reference(values: tuple[float | None, ...], length: int, current: float) -> bool | None:
    sample = values[:length]
    if len(sample) != length or any(item is None for item in sample):
        return None
    prices = tuple(item for item in sample if item is not None)
    reference = sum(item / length for item in prices)
    return current > reference and not isclose(current, reference, rel_tol=1e-12)


def _new_extreme(
    values: tuple[float | None, ...], length: int, current: float
) -> tuple[bool | None, bool | None]:
    sample = values[:length]
    if len(sample) != length or any(item is None for item in sample):
        return None, None
    prices = tuple(item for item in sample if item is not None)
    return current > max(prices), current < min(prices)


def _fraction(value: int, denominator: int) -> float | None:
    return value / denominator if denominator else None


def _reference_result(
    members: tuple[MemberBreadth, ...], attribute: str, sessions: int
) -> ReferenceParticipation:
    observations = tuple(
        getattr(item, attribute) for item in members if item.simple_return is not None
    )
    valid = tuple(value for value in observations if value is not None)
    above = sum(value is True for value in valid)
    return ReferenceParticipation(
        sessions=sessions,
        above_count=above,
        comparable_count=len(valid),
        fraction=_fraction(above, len(valid)),
    )


class BreadthFeatureEngine:
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
            config.market.universe.breadth
        )

    def _universe_at(self, as_of_ns: int) -> UniverseSnapshot:
        universe = self.universe_provider.members(as_of_ns)
        if universe.as_of_ns != as_of_ns:
            raise ValueError("universe provider returned a snapshot for a different as_of_ns")
        return universe

    def _is_stale(self, source: str, effective_ns: int, end_ns: int, as_of_ns: int) -> bool:
        limit = self.config.freshness.bar_limit_ns
        freshness = evaluate_freshness(source, effective_ns, as_of_ns, limit)
        return bool(freshness and freshness.is_stale) or as_of_ns - end_ns > limit

    def _historical_point(
        self, reader: BreadthPriceReader, day: date, as_of_ns: int, settings: BreadthSettings
    ) -> BreadthHistoryPoint | None:
        universe = self._universe_at(as_of_ns)
        if not universe.members:
            return None
        try:
            previous_day = self.calendar.previous_trading_day(day)
        except CalendarRangeError:
            return None
        returns: list[float] = []
        for member in universe.members:
            current = reader.current(member.symbol, day, as_of_ns)
            previous = reader.session_close(member.symbol, previous_day, as_of_ns)
            if current is None or current.bar_end_ns is None or previous is None:
                continue
            if self._is_stale(
                current.source, current.effective_timestamp_ns, current.bar_end_ns, as_of_ns
            ):
                continue
            result = simple_return(previous, current.close)
            if result is not None:
                returns.append(result)
        if len(returns) / len(universe.members) < settings.minimum_member_coverage or not returns:
            return None
        _, _, _, advance, _, net, _ = direction_counts(
            tuple(returns), settings.unchanged_return_tolerance
        )
        if advance is None or net is None:
            return None
        return BreadthHistoryPoint(net, advance)

    def _prior_history(
        self,
        reader: BreadthPriceReader,
        day: date,
        settings: BreadthSettings,
    ) -> tuple[BreadthHistoryPoint, ...]:
        points: list[BreadthHistoryPoint] = []
        cursor = day
        for _ in range(settings.momentum_long_window):
            try:
                cursor = self.calendar.previous_trading_day(cursor)
            except CalendarRangeError:
                break
            _, close_ns = self.calendar.regular_session_bounds(cursor)
            point = self._historical_point(reader, cursor, close_ns, settings)
            if point is None:
                break
            points.append(point)
        return tuple(points)

    def compute(self, store: CausalDataStore, as_of_ns: int) -> BreadthFeatureSet:
        if as_of_ns < 0:
            raise ValueError("as_of_ns must be nonnegative")
        settings = self.config.features.breadth
        short_n, medium_n, long_n = settings.reference_windows_sessions
        universe = self._universe_at(as_of_ns)
        expected = len(universe.members)
        empty_refs = (
            ReferenceParticipation(sessions=short_n, above_count=0, comparable_count=0),
            ReferenceParticipation(sessions=medium_n, above_count=0, comparable_count=0),
            ReferenceParticipation(sessions=long_n, above_count=0, comparable_count=0),
        )
        if not settings.enabled:
            return BreadthFeatureSet(
                as_of_ns=as_of_ns,
                per_member={
                    member.symbol: MemberBreadth(symbol=member.symbol, weight=member.weight)
                    for member in universe.members
                },
                expected_member_count=expected,
                available_member_count=0,
                stale_member_count=0,
                coverage_ratio=0.0,
                advancers=0,
                decliners=0,
                unchanged=0,
                short_reference=empty_refs[0],
                medium_reference=empty_refs[1],
                long_reference=empty_refs[2],
                new_high_low_window_sessions=settings.new_high_low_window_sessions,
                evidence_quality=0.0,
                data_quality=DataQuality.UNKNOWN,
            )
        flags: set[QualityFlag] = set(universe.quality_flags)
        if (
            universe.source_quality != DataQuality.GOOD
            or settings.price_adjustment_status != "adjusted"
        ):
            flags.add(QualityFlag.PROVIDER_DEGRADED)
        info = self.calendar.session_for_timestamp(as_of_ns)
        day = info.trading_date if info.is_trading_day else None
        try:
            previous_day = self.calendar.previous_trading_day(day) if day is not None else None
        except CalendarRangeError:
            previous_day = None
        reader = BreadthPriceReader(store, self.calendar, self.config.provider.source)
        prior_days: list[date] = []
        if day is not None:
            cursor = day
            for _ in range(max(long_n, settings.new_high_low_window_sessions)):
                try:
                    cursor = self.calendar.previous_trading_day(cursor)
                except CalendarRangeError:
                    break
                prior_days.append(cursor)
        prior_dates = tuple(prior_days)
        members: dict[str, MemberBreadth] = {}
        stale_count = 0
        for member in sorted(universe.members, key=lambda item: item.symbol):
            symbol = member.symbol
            current = reader.current(symbol, day, as_of_ns) if day is not None else None
            previous = (
                reader.session_close(symbol, previous_day, as_of_ns)
                if previous_day is not None
                else None
            )
            stale = (
                current is not None
                and current.bar_end_ns is not None
                and self._is_stale(
                    current.source, current.effective_timestamp_ns, current.bar_end_ns, as_of_ns
                )
            )
            stale_count += int(stale)
            result = (
                simple_return(previous, current.close)
                if current is not None and previous is not None and not stale
                else None
            )
            values = (
                reader.closes_on_days(symbol, prior_dates, as_of_ns)
                if result is not None and day is not None
                else ()
            )
            new_high, new_low = (
                _new_extreme(values, settings.new_high_low_window_sessions, current.close)
                if current is not None and result is not None
                else (None, None)
            )
            member_flags: set[QualityFlag] = set()
            if stale:
                member_flags.add(QualityFlag.STALE_DATA)
            elif result is None:
                member_flags.add(QualityFlag.MISSING_DATA)
            members[symbol] = MemberBreadth(
                symbol=symbol,
                weight=member.weight,
                current_price=current.close if current is not None else None,
                previous_close=previous,
                simple_return=result,
                stale=stale,
                above_short_reference=_reference(values, short_n, current.close)
                if result is not None and current is not None
                else None,
                above_medium_reference=_reference(values, medium_n, current.close)
                if result is not None and current is not None
                else None,
                above_long_reference=_reference(values, long_n, current.close)
                if result is not None and current is not None
                else None,
                new_high=new_high,
                new_low=new_low,
                quality_flags=frozenset(member_flags),
            )
        valid = tuple(item for item in members.values() if item.simple_return is not None)
        returns = tuple(item.simple_return for item in valid if item.simple_return is not None)
        available = len(valid)
        coverage = available / expected if expected else 0.0
        advance, decline, unchanged, advance_fraction, decline_fraction, net, ratio = (
            direction_counts(returns, settings.unchanged_return_tolerance)
        )
        if available < expected:
            flags.update({QualityFlag.PARTIAL_UNIVERSE, QualityFlag.MISSING_DATA})
        if coverage < settings.minimum_member_coverage:
            flags.add(QualityFlag.LOW_COVERAGE)
        if stale_count:
            flags.update({QualityFlag.STALE_DATA, QualityFlag.SOURCE_STALE})
        if not expected:
            flags.add(QualityFlag.MISSING_BREADTH)
        refs = (
            _reference_result(tuple(members.values()), "above_short_reference", short_n),
            _reference_result(tuple(members.values()), "above_medium_reference", medium_n),
            _reference_result(tuple(members.values()), "above_long_reference", long_n),
        )
        if any(item.comparable_count < available for item in refs):
            flags.add(QualityFlag.INSUFFICIENT_HISTORY)
        extrema = tuple(item for item in valid if item.new_high is not None)
        if len(extrema) < available:
            flags.add(QualityFlag.INSUFFICIENT_HISTORY)
        high_fraction = _fraction(sum(item.new_high is True for item in extrema), len(extrema))
        low_fraction = _fraction(sum(item.new_low is True for item in extrema), len(extrema))
        score, score_terms = breadth_score_components(
            net,
            (refs[0].fraction, refs[1].fraction, refs[2].fraction),
            settings.score_weights,
        )
        if (
            coverage < settings.minimum_member_coverage
            or universe.source_quality == DataQuality.UNUSABLE
        ):
            score, score_terms = None, ()
        all_weighted = bool(universe.members) and all(
            item.weight is not None for item in universe.members
        )
        weighted_value: float | None = None
        weight_coverage: float | None = None
        if all_weighted:
            total_weight = sum(item.weight or 0.0 for item in universe.members)
            weighted_pairs = tuple(
                (item.simple_return, item.weight)
                for item in valid
                if item.simple_return is not None and item.weight is not None
            )
            weighted_value, weight_coverage = weighted_return(weighted_pairs, total_weight)
            if weight_coverage is None or weight_coverage < settings.minimum_weight_coverage:
                weighted_value = None
                flags.add(QualityFlag.LOW_COVERAGE)
        elif expected:
            flags.add(QualityFlag.MISSING_DATA)
        equal = sum(value / available for value in returns) if available else None
        index_return: float | None = None
        if settings.index_proxy_symbol is not None and day is not None and previous_day is not None:
            proxy = reader.current(settings.index_proxy_symbol, day, as_of_ns)
            proxy_previous = reader.session_close(
                settings.index_proxy_symbol, previous_day, as_of_ns
            )
            if (
                proxy is not None
                and proxy.bar_end_ns is not None
                and proxy_previous is not None
                and not self._is_stale(
                    proxy.source, proxy.effective_timestamp_ns, proxy.bar_end_ns, as_of_ns
                )
            ):
                index_return = simple_return(proxy_previous, proxy.close)
        weighted_for_concentration = (
            all_weighted and sum(item.weight or 0.0 for item in universe.members) > 0
        )
        concentration = contribution_concentration(
            tuple(
                (item.simple_return, (item.weight or 0.0) if weighted_for_concentration else 1.0)
                for item in valid
                if item.simple_return is not None
            )
        )
        history = (
            self._prior_history(reader, day, settings)
            if day is not None
            and net is not None
            and advance_fraction is not None
            and coverage >= settings.minimum_member_coverage
            else ()
        )
        if (
            net is not None
            and coverage >= settings.minimum_member_coverage
            and len(history) < settings.momentum_long_window
        ):
            flags.add(QualityFlag.INSUFFICIENT_HISTORY)
        historical = summarize_history(
            BreadthHistoryPoint(net, advance_fraction)
            if net is not None
            and advance_fraction is not None
            and coverage >= settings.minimum_member_coverage
            else None,
            history,
            settings.momentum_short_window,
            settings.momentum_long_window,
        )
        reference_coverage = (
            sum(item.comparable_count for item in refs) / (3 * available) if available else 0.0
        )
        membership_factor = {
            DataQuality.GOOD: 1.0,
            DataQuality.DEGRADED: 0.5,
            DataQuality.POOR: 0.25,
            DataQuality.UNUSABLE: 0.0,
            DataQuality.UNKNOWN: 0.5,
        }[universe.source_quality]
        adjustment_factor = 1.0 if settings.price_adjustment_status == "adjusted" else 0.5
        evidence_quality = coverage * reference_coverage * membership_factor * adjustment_factor
        quality = (
            DataQuality.UNUSABLE
            if score is None
            else DataQuality.DEGRADED
            if flags
            else DataQuality.GOOD
        )
        reference_names = dict(
            zip(
                ("short_reference", "medium_reference", "long_reference"),
                settings.reference_windows_sessions,
                strict=True,
            )
        )
        contributors = tuple(
            ContextContributor(
                name=f"breadth.{name}",
                value=value,
                normalized_value=value,
                contribution=value * weight,
                source_feature=(
                    "breadth.net_advance_fraction"
                    if name == "net_advance"
                    else f"breadth.above_reference.{reference_names[name]}d"
                ),
            )
            for name, value, weight in score_terms
        )
        return BreadthFeatureSet(
            as_of_ns=as_of_ns,
            per_member=members,
            expected_member_count=expected,
            available_member_count=available,
            stale_member_count=stale_count,
            coverage_ratio=coverage,
            weight_coverage=weight_coverage,
            advancers=advance,
            decliners=decline,
            unchanged=unchanged,
            advance_fraction=advance_fraction,
            decline_fraction=decline_fraction,
            net_advance_fraction=net,
            advance_decline_ratio=ratio,
            short_reference=refs[0],
            medium_reference=refs[1],
            long_reference=refs[2],
            new_high_low_window_sessions=settings.new_high_low_window_sessions,
            new_high_fraction=high_fraction,
            new_low_fraction=low_fraction,
            net_new_high_fraction=high_fraction - low_fraction
            if high_fraction is not None and low_fraction is not None
            else None,
            equal_weight_return=equal,
            median_member_return=median_return(returns),
            weighted_return=weighted_value,
            equal_weight_divergence=equal - weighted_value
            if equal is not None and weighted_value is not None
            else None,
            index_proxy_return=index_return,
            index_proxy_divergence=equal - index_return
            if equal is not None and index_return is not None
            else None,
            directional_index_divergence=net
            - (1 if index_return > 0 else -1 if index_return < 0 else 0)
            if net is not None and index_return is not None
            else None,
            contribution_hhi=concentration[0] if concentration else None,
            effective_contributors=concentration[1] if concentration else None,
            contribution_basis="weighted" if weighted_for_concentration else "equal",
            participation_score=net,
            breadth_score=score,
            breadth_momentum=historical.momentum,
            breadth_acceleration=historical.acceleration,
            breadth_thrust=historical.thrust,
            participation_persistence=historical.persistence,
            evidence_quality=evidence_quality,
            data_quality=quality,
            contributors=contributors,
            quality_flags=frozenset(flags),
        )
