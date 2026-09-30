"""Completed-session, point-in-time return panels with explicit missingness."""

from datetime import date
from hashlib import sha256
from math import log
from zoneinfo import ZoneInfo

from mce.calendar.service import MarketSessionService
from mce.config.settings import FactorSettings, MCEConfig
from mce.errors import CalendarRangeError
from mce.features.breadth.membership import MembershipTimeline, UniverseProvider
from mce.features.correlation.returns import CorrelationReturnReader
from mce.features.factors.models import ReturnPanel
from mce.models.enums import DataQuality, QualityFlag
from mce.models.time import ns_to_datetime
from mce.replay.store import CausalDataStore
from mce.serialization.canonical import canonical_json_bytes

_QUALITY_RANK = {
    DataQuality.GOOD: 0,
    DataQuality.DEGRADED: 1,
    DataQuality.UNKNOWN: 2,
    DataQuality.POOR: 3,
    DataQuality.UNUSABLE: 4,
}


class FactorPanelBuilder:
    def __init__(
        self,
        config: MCEConfig,
        universe_provider: UniverseProvider | None = None,
        calendar: MarketSessionService | None = None,
    ) -> None:
        self.config = config
        self.settings: FactorSettings = config.features.factor
        self.calendar = calendar or MarketSessionService(config.market)
        if calendar is not None and calendar.settings.model_dump(
            exclude={"universe"}
        ) != config.market.model_dump(exclude={"universe"}):
            raise ValueError("calendar settings must match factor configuration")
        self.universe_provider = universe_provider or MembershipTimeline.from_symbols(
            self.settings.universe
        )

    def _completed_day(self, as_of_ns: int) -> date | None:
        day = (
            ns_to_datetime(as_of_ns)
            .astimezone(ZoneInfo(self.config.market.exchange_timezone))
            .date()
        )
        try:
            if self.calendar.is_trading_day(day):
                _, close = self.calendar.regular_session_bounds(day)
                if as_of_ns >= close:
                    return day
            return self.calendar.previous_trading_day(day)
        except CalendarRangeError:
            return None

    def build(self, store: CausalDataStore, as_of_ns: int) -> ReturnPanel:
        if as_of_ns < 0:
            raise ValueError("as_of_ns must be nonnegative")
        current = self.universe_provider.members(as_of_ns)
        if current.as_of_ns != as_of_ns:
            raise ValueError("factor universe provider returned a different clock")
        expected = tuple(member.symbol for member in current.members)
        membership_hash = sha256(canonical_json_bytes(current))
        quality = current.source_quality
        flags: set[QualityFlag] = set(current.quality_flags)
        end_day = self._completed_day(as_of_ns)
        if end_day is None:
            return ReturnPanel(
                as_of_ns,
                (),
                expected,
                (),
                (),
                self.settings.window_sessions,
                source_quality=current.source_quality,
                quality_flags=current.quality_flags,
                membership_fingerprint=membership_hash.hexdigest(),
            )
        days = [end_day]
        for _ in range(self.settings.window_sessions):
            try:
                days.append(self.calendar.previous_trading_day(days[-1]))
            except CalendarRangeError:
                break
        days.reverse()
        endpoints = tuple((day, self.calendar.regular_session_bounds(day)[1]) for day in days)
        memberships: dict[int, set[str]] = {}
        for _, endpoint in endpoints:
            snapshot = self.universe_provider.members(endpoint)
            if snapshot.as_of_ns != endpoint:
                raise ValueError("factor universe provider returned a different historical clock")
            membership_hash.update(canonical_json_bytes(snapshot))
            if _QUALITY_RANK[snapshot.source_quality] > _QUALITY_RANK[quality]:
                quality = snapshot.source_quality
            flags.update(snapshot.quality_flags)
            memberships[endpoint] = {member.symbol for member in snapshot.members}
        reader = CorrelationReturnReader(
            store, self.calendar, self.config.provider.source, self.settings.bar_cadence_ns
        )
        data: dict[str, dict[int, float]] = {}
        available: dict[str, dict[int, int]] = {}
        stale: list[str] = []
        missing: list[str] = []
        last_endpoint = endpoints[-1][1]
        for symbol in expected:
            closes = {
                endpoint: bar
                for day, endpoint in endpoints
                if (bar := reader.daily_close(symbol, day, as_of_ns)) is not None
            }
            values = {
                finish: log(closes[finish].close / closes[start].close)
                for (_, start), (_, finish) in zip(endpoints, endpoints[1:], strict=False)
                if symbol in memberships[start]
                and symbol in memberships[finish]
                and start in closes
                and finish in closes
            }
            availability = {
                finish: max(
                    closes[finish].effective_timestamp_ns, closes[start].effective_timestamp_ns
                )
                for (_, start), (_, finish) in zip(endpoints, endpoints[1:], strict=False)
                if finish in values
            }
            if last_endpoint not in values:
                (stale if values else missing).append(symbol)
            elif len(values) / self.settings.window_sessions >= self.settings.minimum_time_coverage:
                data[symbol] = values
                available[symbol] = availability
            else:
                missing.append(symbol)
        symbols = tuple(sorted(data))
        common = set.intersection(*(set(data[symbol]) for symbol in symbols)) if symbols else set()
        times = tuple(sorted(common))
        matrix = tuple(tuple(data[symbol][time] for symbol in symbols) for time in times)
        row_availability = tuple(
            max(available[symbol][time] for symbol in symbols) for time in times
        )
        return ReturnPanel(
            as_of_ns,
            times,
            expected,
            symbols,
            matrix,
            self.settings.window_sessions,
            tuple(sorted(missing)),
            tuple(sorted(stale)),
            quality,
            frozenset(flags),
            membership_hash.hexdigest(),
            row_availability,
        )
