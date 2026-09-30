"""Causal same-session-point normalization with explicit prior history."""

from mce.calendar.service import MarketSessionService
from mce.config.settings import MCEConfig
from mce.features.breadth.membership import MembershipTimeline, UniverseProvider
from mce.features.correlation.models import CorrelationFeatureSet
from mce.features.seasonality.baseline import SeasonalityHistoryBuilder, finite_median
from mce.features.seasonality.clock import bucket_grid, session_relative_time
from mce.features.seasonality.models import (
    IntradaySeasonalityFeatureSet,
    MarketSeasonalMetric,
    MetricName,
    SeasonalHistoryIndex,
    SeasonallyAdjustedMetric,
    SymbolSeasonality,
)
from mce.features.seasonality.normalization import normalize
from mce.features.seasonality.readers import current_bucket_values, read_session_metrics
from mce.models.enums import DataQuality, QualityFlag
from mce.replay.store import CausalDataStore
from mce.reproducibility import config_fingerprint

_SYMBOL_METRICS: tuple[MetricName, ...] = (
    "volume",
    "cumulative_volume",
    "dollar_volume",
    "volatility",
    "quoted_spread",
    "spread",
    "illiquidity",
)


def _market_metric(
    metric: MetricName,
    per_symbol: tuple[SymbolSeasonality, ...],
    expected: int,
    minimum_members: int,
    minimum_coverage: float,
) -> MarketSeasonalMetric:
    selected = tuple(
        item
        for member in per_symbol
        for item in member.metrics
        if item.metric == metric and item.ratio is not None
    )
    count = len(selected)
    coverage = count / expected if expected else 0.0
    if count < minimum_members or coverage < minimum_coverage:
        return MarketSeasonalMetric(metric, None, None, None, count, coverage)
    ratios = [item.ratio for item in selected if item.ratio is not None]
    percentiles = [item.percentile for item in selected if item.percentile is not None]
    z_scores = [item.robust_z for item in selected if item.robust_z is not None]
    return MarketSeasonalMetric(
        metric,
        finite_median(sorted(ratios)),
        finite_median(sorted(percentiles)) if percentiles else None,
        finite_median(sorted(z_scores)) if z_scores else None,
        count,
        coverage,
    )


class IntradaySeasonalityEngine:
    def __init__(
        self,
        config: MCEConfig,
        universe_provider: UniverseProvider | None = None,
        calendar: MarketSessionService | None = None,
    ) -> None:
        self.config = config
        self.settings = config.features.seasonality
        if calendar is not None and calendar.settings.model_dump(
            mode="python"
        ) != config.market.model_dump(mode="python"):
            raise ValueError("calendar settings must match result-relevant market configuration")
        self.calendar = calendar or MarketSessionService(config.market)
        self.universe_provider = universe_provider or MembershipTimeline.from_symbols(
            self.settings.universe
        )

    def build_history(
        self,
        store: CausalDataStore,
        as_of_ns: int,
        *,
        correlation_history: tuple[CorrelationFeatureSet, ...] = (),
    ) -> SeasonalHistoryIndex:
        return SeasonalityHistoryBuilder(self.config, self.universe_provider, self.calendar).build(
            store, as_of_ns, correlation_history=correlation_history
        )

    def compute(
        self,
        store: CausalDataStore,
        as_of_ns: int,
        history: SeasonalHistoryIndex | None = None,
        *,
        correlation: CorrelationFeatureSet | None = None,
    ) -> IntradaySeasonalityFeatureSet:
        settings = self.settings
        fingerprint = config_fingerprint(self.config)
        clock = session_relative_time(self.calendar, as_of_ns, settings)
        if correlation is not None and correlation.as_of_ns > as_of_ns:
            raise ValueError("future correlation context cannot enter seasonal result")
        if history is not None and (
            clock is None
            or history.target_date != clock.trading_date
            or history.built_as_of_ns > as_of_ns
            or history.config_fingerprint != fingerprint
        ):
            raise ValueError("seasonal history target, clock, or configuration mismatch")
        flags: set[QualityFlag] = set()
        if clock is None or clock.completed_bucket is None or not settings.enabled:
            flags.add(QualityFlag.OUTSIDE_SESSION if clock is None else QualityFlag.MISSING_DATA)
            return IntradaySeasonalityFeatureSet(
                as_of_ns,
                clock,
                history.fingerprint if history is not None else None,
                fingerprint,
                (),
                (),
                0.0,
                (),
                (),
                None,
                frozenset(flags),
                DataQuality.POOR,
            )
        universe = self.universe_provider.members(as_of_ns)
        if universe.as_of_ns != as_of_ns:
            raise ValueError("seasonal universe provider returned a different clock")
        if universe.source_quality != DataQuality.GOOD:
            flags.add(QualityFlag.PROVIDER_DEGRADED)
        flags.update(universe.quality_flags)
        expected = tuple(sorted(member.symbol for member in universe.members))
        if history is None:
            history = SeasonalHistoryIndex(clock.trading_date, (), as_of_ns, fingerprint, ())
            flags.add(QualityFlag.INSUFFICIENT_SEASONAL_HISTORY)
        bucket = clock.completed_bucket
        grid = bucket_grid(self.calendar, clock.trading_date, settings)
        per_symbol: list[SymbolSeasonality] = []
        available: list[str] = []
        for symbol in expected:
            read = read_session_metrics(
                store,
                self.calendar,
                self.config,
                symbol,
                clock.trading_date,
                as_of_ns,
                current=True,
                buckets=grid,
            )
            values = current_bucket_values(read, bucket.index)
            if "volume" in values:
                available.append(symbol)
            if read.missing_bars:
                flags.add(QualityFlag.MISSING_DATA)
            if read.invalid_quotes:
                flags.add(QualityFlag.REJECTED_DATA)
            if "spread" not in values:
                flags.add(QualityFlag.MISSING_DATA)
            if read.stale_quotes or (
                "volume" not in values
                and any(item.metric == "volume" for item in read.observations)
            ):
                flags.add(QualityFlag.STALE_DATA)
            metrics = tuple(
                normalize(
                    history,
                    metric,
                    symbol,
                    bucket.index,
                    values[metric].value if metric in values else None,
                    settings.minimum_history_sessions,
                )
                for metric in _SYMBOL_METRICS
            )
            per_symbol.append(SymbolSeasonality(symbol, metrics))
        member_coverage = len(available) / len(expected) if expected else 0.0
        if member_coverage < 1:
            flags.add(QualityFlag.PARTIAL_UNIVERSE)
        if member_coverage < settings.minimum_member_coverage:
            flags.add(QualityFlag.LOW_COVERAGE)
        market = tuple(
            _market_metric(
                metric,
                tuple(per_symbol),
                len(expected),
                settings.minimum_members,
                settings.minimum_member_coverage,
            )
            for metric in _SYMBOL_METRICS
        )
        if market[0].ratio is None:
            flags.add(QualityFlag.INSUFFICIENT_SEASONAL_HISTORY)
        adjusted_correlation: SeasonallyAdjustedMetric | None = None
        if (
            correlation is not None
            and correlation.as_of_ns == bucket.end_ns
            and correlation.data_quality not in (DataQuality.POOR, DataQuality.UNUSABLE)
        ):
            adjusted_correlation = normalize(
                history,
                "correlation",
                None,
                bucket.index,
                correlation.average_pairwise_correlation,
                settings.minimum_history_sessions,
            )
        quality = (
            DataQuality.UNUSABLE
            if universe.source_quality == DataQuality.UNUSABLE
            else DataQuality.POOR
            if QualityFlag.LOW_COVERAGE in flags or market[0].ratio is None
            else DataQuality.DEGRADED
            if flags
            else DataQuality.GOOD
        )
        return IntradaySeasonalityFeatureSet(
            as_of_ns,
            clock,
            history.fingerprint if history.prior_dates else None,
            fingerprint,
            expected,
            tuple(available),
            member_coverage,
            tuple(per_symbol),
            market,
            adjusted_correlation,
            frozenset(flags),
            quality,
        )
