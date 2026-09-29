"""Reuse MCE-3 window selection and MCE-6 horizon return semantics."""

from dataclasses import dataclass

from mce.calendar.service import MarketSessionService
from mce.features.sectors.strength import horizon_return
from mce.features.windows import FeatureWindowReader, WindowSpec
from mce.models.market_data import BarObservation


@dataclass(frozen=True)
class MemberReturn:
    symbol: str
    values: tuple[tuple[str, float | None], ...]
    stale: bool
    missing: bool


def read_member(
    reader: FeatureWindowReader,
    calendar: MarketSessionService,
    symbol: str,
    as_of_ns: int,
    horizons: tuple[WindowSpec, ...],
    freshness_limit_ns: int,
) -> MemberReturn:
    current: BarObservation | None = reader.current_bar(symbol, as_of_ns)
    info = calendar.session_for_timestamp(as_of_ns)
    day = info.trading_date
    if day is None:
        return MemberReturn(symbol, tuple((item.name, None) for item in horizons), False, True)
    opened, closed = calendar.regular_session_bounds(day)
    if current is None or current.bar_start_ns is None or current.bar_end_ns is None:
        return MemberReturn(symbol, tuple((item.name, None) for item in horizons), False, True)
    if not opened <= current.bar_start_ns < current.bar_end_ns <= closed:
        return MemberReturn(symbol, tuple((item.name, None) for item in horizons), True, False)
    if (
        as_of_ns - current.effective_timestamp_ns > freshness_limit_ns
        or as_of_ns - current.bar_end_ns > freshness_limit_ns
    ):
        return MemberReturn(symbol, tuple((item.name, None) for item in horizons), True, False)
    return MemberReturn(
        symbol,
        tuple(
            (item.name, horizon_return(reader, symbol, current, as_of_ns, item))
            for item in horizons
        ),
        False,
        False,
    )
