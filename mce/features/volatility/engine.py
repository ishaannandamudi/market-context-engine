"""Provider-free, explicit-clock realized-volatility feature orchestration."""

from datetime import date
from zoneinfo import ZoneInfo

from mce.calendar.service import MarketSessionService
from mce.config.settings import MCEConfig, VolatilitySettings
from mce.errors import CalendarRangeError
from mce.features.volatility.estimators import (
    Estimate,
    SessionOHLC,
    close_to_close,
    garman_klass,
    overnight_context,
    parkinson,
    rogers_satchell,
    yang_zhang,
)
from mce.features.volatility.intraday import completed_intraday_bars, measure_intraday
from mce.features.volatility.models import (
    DailyVolatilityWindow,
    EstimatorValue,
    IntradayVolatilityWindow,
    RealizedVolatilityFeatureSet,
    SymbolVolatilityFeatures,
)
from mce.features.volatility.rolling import (
    CompletedSessionReader,
    contiguous_tail,
    estimator_consensus,
    previous_short_volatility,
    prior_distribution,
    prior_rolling_volatilities,
)
from mce.features.windows import FeatureWindowReader
from mce.models.context import ContextContributor
from mce.models.enums import DataQuality, QualityFlag
from mce.models.time import ns_to_datetime
from mce.quality.freshness import evaluate_freshness
from mce.replay.store import CausalDataStore


def _public_estimate(estimate: Estimate | None) -> EstimatorValue | None:
    if estimate is None:
        return None
    return EstimatorValue(
        annualized_volatility=estimate.annualized_volatility,
        variance_per_session=estimate.variance_per_session,
        sample_count=estimate.sample_count,
        sum_squared_returns=estimate.sum_squared_returns,
    )


def _daily_window(
    slots: tuple[SessionOHLC | None, ...], sessions: int, settings: VolatilitySettings
) -> DailyVolatilityWindow:
    full = contiguous_tail(slots, sessions + 1)
    ranges = contiguous_tail(slots, sessions)
    observed = sum(item is not None for item in slots[-(sessions + 1) :])
    coverage = observed / (sessions + 1)
    annual = settings.trading_sessions_per_year
    cc = close_to_close(full, annual) if full else None
    enabled = settings.enabled_estimators
    pk = parkinson(ranges, annual) if ranges and "parkinson" in enabled else None
    gk = garman_klass(ranges, annual) if ranges and "garman_klass" in enabled else None
    rs = rogers_satchell(ranges, annual) if ranges and "rogers_satchell" in enabled else None
    yz = yang_zhang(full, annual) if full and "yang_zhang" in enabled else None
    context = overnight_context(full) if full else None
    comparable = tuple(item.annualized_volatility for item in (cc, pk, gk, rs, yz) if item)
    consensus = estimator_consensus(comparable)
    flags: set[QualityFlag] = set()
    if cc is None:
        flags.add(QualityFlag.INSUFFICIENT_HISTORY)
    if coverage < settings.minimum_window_coverage:
        flags.add(QualityFlag.INSUFFICIENT_WINDOW_COVERAGE)
    return DailyVolatilityWindow(
        name=f"{sessions}d",
        sessions=sessions,
        close_to_close=_public_estimate(cc),
        parkinson=_public_estimate(pk),
        garman_klass=_public_estimate(gk),
        rogers_satchell=_public_estimate(rs),
        yang_zhang=_public_estimate(yz),
        overnight_variance_share=context[0] if context else None,
        intraday_variance_share=context[1] if context else None,
        latest_gap_magnitude=context[2] if context else None,
        estimator_median=consensus[0] if consensus else None,
        estimator_range=consensus[1] if consensus else None,
        estimator_std=consensus[2] if consensus else None,
        estimator_agreement=consensus[3] if consensus else None,
        observed_sessions=observed,
        expected_sessions=sessions + 1,
        coverage_ratio=coverage,
        quality_flags=frozenset(flags),
    )


class RealizedVolatilityEngine:
    def __init__(self, config: MCEConfig, calendar: MarketSessionService | None = None) -> None:
        self.config = config
        if calendar is not None and calendar.settings.model_dump(
            exclude={"universe"}
        ) != config.market.model_dump(exclude={"universe"}):
            raise ValueError("calendar settings must match result-relevant market configuration")
        self.calendar = calendar or MarketSessionService(config.market)

    def _latest_completed_day(self, as_of_ns: int) -> date | None:
        local_day = (
            ns_to_datetime(as_of_ns)
            .astimezone(ZoneInfo(self.config.market.exchange_timezone))
            .date()
        )
        try:
            if self.calendar.is_trading_day(local_day):
                _, closed = self.calendar.regular_session_bounds(local_day)
                if closed <= as_of_ns:
                    return local_day
            return self.calendar.previous_trading_day(local_day)
        except CalendarRangeError:
            return None

    def _symbol(
        self,
        store: CausalDataStore,
        symbol: str,
        as_of_ns: int,
        completed_day: date | None,
        settings: VolatilitySettings,
    ) -> SymbolVolatilityFeatures:
        flags: set[QualityFlag] = set()
        reader = CompletedSessionReader(
            store, self.calendar, self.config.provider.source, settings.bar_cadence_ns
        )
        history_count = max(
            settings.long_window_sessions + 1,
            settings.baseline_sessions + settings.short_window_sessions + 1,
        )
        slots = (
            reader.history(symbol, completed_day, history_count, as_of_ns)
            if completed_day is not None
            else ()
        )
        daily = {
            f"{sessions}d": _daily_window(slots, sessions, settings)
            for sessions in settings.session_horizons
        }
        flags.update(flag for item in daily.values() for flag in item.quality_flags)
        short = daily[f"{settings.short_window_sessions}d"].close_to_close
        medium = daily[f"{settings.medium_window_sessions}d"].close_to_close
        long = daily[f"{settings.long_window_sessions}d"].close_to_close
        short_value = short.annualized_volatility if short else None
        medium_value = medium.annualized_volatility if medium else None
        long_value = long.annualized_volatility if long else None
        epsilon = 1e-12
        short_long = (
            short_value / long_value
            if short_value is not None and long_value is not None and long_value > epsilon
            else None
        )
        medium_long = (
            medium_value / long_value
            if medium_value is not None and long_value is not None and long_value > epsilon
            else None
        )
        acceleration = (
            (short_value - medium_value) / medium_value
            if short_value is not None and medium_value is not None and medium_value > epsilon
            else None
        )
        prior_short = previous_short_volatility(
            slots, settings.short_window_sessions, settings.trading_sessions_per_year
        )
        change = (
            short_value - prior_short
            if short_value is not None and prior_short is not None
            else None
        )
        baseline = prior_rolling_volatilities(
            slots,
            settings.short_window_sessions,
            settings.baseline_sessions,
            settings.trading_sessions_per_year,
        )
        distribution = (
            prior_distribution(short_value, baseline, settings.minimum_baseline_samples)
            if short_value is not None
            else None
        )
        if distribution is None:
            flags.add(QualityFlag.INSUFFICIENT_HISTORY)

        intraday: dict[str, IntradayVolatilityWindow] = {}
        local_day = (
            ns_to_datetime(as_of_ns)
            .astimezone(ZoneInfo(self.config.market.exchange_timezone))
            .date()
        )
        try:
            trading_today = self.calendar.is_trading_day(local_day)
        except CalendarRangeError:
            trading_today = False
        if trading_today:
            opened, closed = self.calendar.regular_session_bounds(local_day)
            if as_of_ns >= opened:
                bars, duplicate = completed_intraday_bars(
                    store,
                    self.calendar,
                    self.config.provider.source,
                    symbol,
                    local_day,
                    as_of_ns,
                    settings.bar_cadence_ns,
                )
                for name, length in (*settings.intraday_horizons_ns.items(), ("session", None)):
                    measured = measure_intraday(
                        bars,
                        opened,
                        closed,
                        as_of_ns,
                        length,
                        settings.bar_cadence_ns,
                        settings.trading_sessions_per_year,
                        duplicate,
                    )
                    item_flags: set[QualityFlag] = set()
                    if measured.annualized_volatility is None:
                        item_flags.add(QualityFlag.INSUFFICIENT_HISTORY)
                    if (
                        measured.expected_return_count
                        and measured.coverage_ratio < settings.minimum_window_coverage
                    ):
                        item_flags.add(QualityFlag.INSUFFICIENT_WINDOW_COVERAGE)
                    if duplicate:
                        item_flags.add(QualityFlag.TIMESTAMP_MISMATCH)
                    intraday[name] = IntradayVolatilityWindow(
                        name=name,
                        annualized_volatility=measured.annualized_volatility,
                        realized_variance=measured.realized_variance,
                        bipower_variation=measured.bipower_variation,
                        jump_variation=measured.jump_variation,
                        jump_share=measured.jump_share,
                        downside_semivariance=measured.downside_semivariance,
                        upside_semivariance=measured.upside_semivariance,
                        downside_share=measured.downside_share,
                        observed_return_count=measured.observed_return_count,
                        expected_return_count=measured.expected_return_count,
                        coverage_ratio=measured.coverage_ratio,
                        quality_flags=frozenset(item_flags),
                    )
                    flags.update(item_flags)
        latest = FeatureWindowReader(
            store, self.calendar, settings.bar_cadence_ns, self.config.provider.source
        ).current_bar(symbol, as_of_ns)
        if latest is None:
            flags.add(QualityFlag.MISSING_DATA)
        else:
            freshness = evaluate_freshness(
                latest.source,
                latest.effective_timestamp_ns,
                as_of_ns,
                self.config.freshness.bar_limit_ns,
            )
            if freshness is not None and freshness.is_stale:
                flags.update({QualityFlag.STALE_DATA, QualityFlag.SOURCE_STALE})
        medium_window = daily[f"{settings.medium_window_sessions}d"]
        contributors = tuple(
            ContextContributor(
                name=f"volatility.{symbol}.{name}",
                value=value,
                normalized_value=value if name == "percentile" else None,
                source_feature=f"volatility.{name}.{symbol}",
            )
            for name, value in (
                ("percentile", distribution[0] if distribution else None),
                ("short_long_ratio", short_long),
                ("estimator_agreement", medium_window.estimator_agreement),
            )
            if value is not None
        )
        return SymbolVolatilityFeatures(
            symbol=symbol,
            as_of_ns=as_of_ns,
            intraday=intraday,
            daily=daily,
            rv_short=short_value,
            rv_medium=medium_value,
            rv_long=long_value,
            short_long_ratio=short_long,
            medium_long_ratio=medium_long,
            volatility_acceleration=acceleration,
            volatility_change_1d=change,
            volatility_percentile=distribution[0] if distribution else None,
            robust_z=distribution[1] if distribution else None,
            volatility_shock_score=distribution[2] if distribution else None,
            baseline_sample_count=len(baseline),
            sample_count=medium.sample_count if medium else 0,
            expected_count=settings.medium_window_sessions,
            coverage_ratio=medium_window.coverage_ratio,
            quality_flags=frozenset(flags),
            contributors=contributors,
        )

    def compute(self, store: CausalDataStore, as_of_ns: int) -> RealizedVolatilityFeatureSet:
        if as_of_ns < 0:
            raise ValueError("as_of_ns must be nonnegative")
        settings = self.config.features.volatility
        symbols = self.config.market.universe.broad_market
        if not settings.enabled:
            return RealizedVolatilityFeatureSet(
                as_of_ns=as_of_ns,
                per_symbol={},
                index_percentiles={},
                available_symbol_count=0,
                expected_symbol_count=len(symbols),
                coverage_ratio=0.0,
                data_quality=DataQuality.UNKNOWN,
            )
        if not symbols:
            raise ValueError("broad_market universe must be nonempty for volatility computation")
        if settings.index_weights and set(settings.index_weights) != set(symbols):
            raise ValueError("index_weights must name every broad_market symbol exactly")
        completed_day = self._latest_completed_day(as_of_ns)
        per_symbol = {
            symbol: self._symbol(store, symbol, as_of_ns, completed_day, settings)
            for symbol in symbols
        }
        observed = {
            symbol: item.rv_medium
            for symbol, item in per_symbol.items()
            if item.rv_medium is not None
        }
        expected_count = len(symbols)
        coverage = len(observed) / expected_count
        flags = set().union(*(item.quality_flags for item in per_symbol.values()))
        if len(observed) < expected_count:
            flags.update({QualityFlag.PARTIAL_UNIVERSE, QualityFlag.LOW_COVERAGE})
        raw_weights = settings.index_weights or dict.fromkeys(symbols, 1.0)
        maximum_weight = max(raw_weights.values())
        weights = {symbol: value / maximum_weight for symbol, value in raw_weights.items()}
        weight_total = sum(weights[symbol] for symbol in observed)
        market_vol = (
            sum(observed[symbol] * weights[symbol] for symbol in sorted(observed)) / weight_total
            if weight_total and coverage >= settings.minimum_market_coverage
            else None
        )
        percentiles = {
            symbol: item.volatility_percentile
            for symbol, item in per_symbol.items()
            if item.volatility_percentile is not None
        }
        percentile_weight = sum(weights[symbol] for symbol in percentiles)
        score = (
            sum(percentiles[symbol] * weights[symbol] for symbol in sorted(percentiles))
            / percentile_weight
            if percentile_weight
            and len(percentiles) / expected_count >= settings.minimum_market_coverage
            else None
        )
        range_value = (
            max(observed.values()) - min(observed.values()) if len(observed) >= 2 else None
        )
        minimum_vol = min(observed.values()) if len(observed) >= 2 else None
        ratio = (
            max(observed.values()) / minimum_vol
            if minimum_vol is not None and minimum_vol > 1e-12
            else None
        )
        contributors = (
            tuple(
                ContextContributor(
                    name=f"volatility.index.{symbol}.percentile",
                    value=percentiles[symbol],
                    normalized_value=percentiles[symbol],
                    contribution=percentiles[symbol] * weights[symbol] / percentile_weight,
                    source_feature=f"volatility.percentile.{symbol}",
                )
                for symbol in sorted(percentiles)
            )
            if score is not None
            else ()
        )
        quality = (
            DataQuality.UNUSABLE
            if market_vol is None
            else DataQuality.DEGRADED
            if flags
            else DataQuality.GOOD
        )
        return RealizedVolatilityFeatureSet(
            as_of_ns=as_of_ns,
            per_symbol=per_symbol,
            market_realized_volatility=market_vol,
            volatility_score=score,
            index_percentiles=percentiles,
            cross_index_volatility_range=range_value,
            cross_index_volatility_ratio=ratio,
            available_symbol_count=len(observed),
            expected_symbol_count=expected_count,
            coverage_ratio=coverage,
            data_quality=quality,
            contributors=contributors,
            quality_flags=frozenset(flags),
        )
