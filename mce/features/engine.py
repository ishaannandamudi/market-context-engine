"""Causal completed-bar trend features and transparent broad-market aggregation."""

from mce.calendar.service import MarketSessionService
from mce.config.settings import MCEConfig, TrendSettings
from mce.features.models import (
    RegressionFeature,
    ReturnFeature,
    SymbolTrendFeatures,
    TrendFeatureSet,
)
from mce.features.normalization import signed_tanh
from mce.features.returns import simple_return
from mce.features.trend import (
    distance_from_sma,
    log_price_regression,
    path_efficiency,
    persistence,
    sign_agreement,
)
from mce.features.windows import FeatureWindowReader, WindowSpec, WindowUnit
from mce.models.context import ContextContributor
from mce.models.enums import DataQuality, QualityFlag
from mce.models.market_data import BarObservation
from mce.quality.freshness import evaluate_freshness
from mce.replay.store import CausalDataStore


def _regression_feature(
    reader: FeatureWindowReader,
    symbol: str,
    as_of_ns: int,
    length_ns: int,
    min_samples: int,
    minimum_coverage: float,
) -> tuple[RegressionFeature, tuple[BarObservation, ...]]:
    selection = reader.select(
        symbol, as_of_ns, WindowSpec(name="slope", unit=WindowUnit.ELAPSED, length=length_ns)
    )
    raw = log_price_regression(selection.bars) if len(selection.bars) >= min_samples else None
    flags: set[QualityFlag] = set()
    if raw is None:
        flags.add(QualityFlag.INSUFFICIENT_HISTORY)
    if selection.coverage_ratio is not None and selection.coverage_ratio < minimum_coverage:
        flags.add(QualityFlag.INSUFFICIENT_WINDOW_COVERAGE)
    return (
        RegressionFeature(
            slope_per_hour=raw.slope_per_hour if raw else None,
            intercept=raw.intercept if raw else None,
            r_squared=raw.r_squared if raw else None,
            sample_count=selection.sample_count,
            expected_count=selection.expected_count,
            coverage_ratio=selection.coverage_ratio,
            quality_flags=frozenset(flags),
        ),
        selection.bars,
    )


class TrendFeatureEngine:
    """No provider calls, global state, wall clock, or future-fitted parameters."""

    def __init__(self, config: MCEConfig, calendar: MarketSessionService | None = None) -> None:
        self.config = config
        if calendar is not None and calendar.settings.model_dump(
            exclude={"universe"}
        ) != config.market.model_dump(exclude={"universe"}):
            raise ValueError("calendar settings must match result-relevant market configuration")
        self.calendar = calendar or MarketSessionService(config.market)

    def _symbol(
        self, reader: FeatureWindowReader, symbol: str, as_of_ns: int, settings: TrendSettings
    ) -> SymbolTrendFeatures:
        flags: set[QualityFlag] = set()
        returns: dict[str, ReturnFeature] = {}
        latest = reader.current_bar(symbol, as_of_ns)
        freshness = (
            evaluate_freshness(
                latest.source,
                latest.effective_timestamp_ns,
                as_of_ns,
                self.config.freshness.bar_limit_ns,
            )
            if latest is not None
            else None
        )
        if freshness is not None and freshness.is_stale:
            flags.update({QualityFlag.STALE_DATA, QualityFlag.SOURCE_STALE})
        for spec in settings.horizons:
            selection = reader.select(symbol, as_of_ns, spec)
            current = selection.bars[-1].close if selection.bars else None
            value = simple_return(selection.reference_price, current)
            item_flags: set[QualityFlag] = set()
            if value is None:
                item_flags.add(QualityFlag.INSUFFICIENT_HISTORY)
            if (
                selection.coverage_ratio is not None
                and selection.coverage_ratio < settings.minimum_window_coverage
            ):
                item_flags.add(QualityFlag.INSUFFICIENT_WINDOW_COVERAGE)
            returns[spec.name] = ReturnFeature(
                name=f"trend.return.{spec.name}",
                value=value,
                reference_price=selection.reference_price,
                current_price=current,
                sample_count=selection.sample_count,
                expected_count=selection.expected_count,
                coverage_ratio=selection.coverage_ratio,
                quality_flags=frozenset(item_flags),
            )
            flags.update(item_flags)
        short, _ = _regression_feature(
            reader,
            symbol,
            as_of_ns,
            settings.short_slope_window_ns,
            settings.min_regression_samples,
            settings.minimum_window_coverage,
        )
        long, long_bars = _regression_feature(
            reader,
            symbol,
            as_of_ns,
            settings.long_slope_window_ns,
            settings.min_regression_samples,
            settings.minimum_window_coverage,
        )
        flags.update(short.quality_flags)
        flags.update(long.quality_flags)
        persist = (
            persistence(long_bars) if len(long_bars) >= settings.min_regression_samples else None
        )
        efficiency = path_efficiency(long_bars) if persist is not None else None
        distance = distance_from_sma(long_bars) if persist is not None else None
        acceleration = (
            short.slope_per_hour - long.slope_per_hour
            if short.slope_per_hour is not None and long.slope_per_hour is not None
            else None
        )
        values = tuple(item.value for item in returns.values() if item.value is not None)
        agreement, disagreement = sign_agreement(values)
        if disagreement:
            flags.add(QualityFlag.HORIZON_DISAGREEMENT)
        if latest is None:
            flags.add(QualityFlag.MISSING_DATA)

        # Each raw diagnostic remains in the result; only fixed-scale transforms enter the score.
        direction = (
            1.0
            if long.slope_per_hour is not None and long.slope_per_hour > 0
            else -1.0
            if long.slope_per_hour is not None and long.slope_per_hour < 0
            else 0.0
        )
        normalized: dict[str, tuple[float, float, float]] = {}
        weights = settings.score_weights
        if values:
            mean_return = sum(value / len(values) for value in values)
            normalized["returns"] = (
                mean_return,
                signed_tanh(mean_return, settings.return_scale),
                weights.returns,
            )
        if long.slope_per_hour is not None and long.r_squared is not None:
            normalized["slope"] = (
                long.slope_per_hour,
                signed_tanh(long.slope_per_hour, settings.slope_scale_per_hour) * long.r_squared,
                weights.slope,
            )
        if persist is not None:
            normalized["persistence"] = (persist, direction * persist, weights.persistence)
        if acceleration is not None:
            normalized["acceleration"] = (
                acceleration,
                signed_tanh(acceleration, settings.acceleration_scale_per_hour),
                weights.acceleration,
            )
        if distance is not None:
            normalized["distance"] = (
                distance,
                signed_tanh(distance, settings.return_scale),
                weights.distance,
            )
        active_weight = sum(weight for _, _, weight in normalized.values())
        score = (
            sum(mapped * weight for _, mapped, weight in normalized.values()) / active_weight
            if active_weight > 0 and len(long_bars) >= settings.min_regression_samples
            else None
        )
        contributors = tuple(
            ContextContributor(
                name=f"{symbol}.trend.{name}",
                value=raw,
                normalized_value=mapped,
                contribution=mapped * weight / active_weight,
                source_feature=f"trend.{name}",
            )
            for name, (raw, mapped, weight) in sorted(normalized.items())
            if active_weight > 0
        )
        if score is None:
            flags.add(QualityFlag.INSUFFICIENT_HISTORY)
        return SymbolTrendFeatures(
            symbol=symbol,
            as_of_ns=as_of_ns,
            returns=returns,
            slope_short=short,
            slope_long=long,
            persistence=persist,
            path_efficiency=efficiency,
            directional_agreement=agreement,
            acceleration_per_hour=acceleration,
            distance_from_sma=distance,
            trend_score=score,
            sample_count=long.sample_count,
            expected_count=long.expected_count,
            coverage_ratio=long.coverage_ratio,
            contributors=contributors,
            quality_flags=frozenset(flags),
        )

    def compute(self, store: CausalDataStore, as_of_ns: int) -> TrendFeatureSet:
        if as_of_ns < 0:
            raise ValueError("as_of_ns must be nonnegative")
        settings = self.config.features.trend
        symbols = self.config.market.universe.broad_market
        if not settings.enabled:
            return TrendFeatureSet(
                as_of_ns=as_of_ns,
                symbols={},
                index_scores={},
                available_symbol_count=0,
                expected_symbol_count=len(symbols),
                coverage_ratio=0.0,
                data_quality=DataQuality.UNKNOWN,
            )
        if not symbols:
            raise ValueError("broad_market universe must be nonempty for trend computation")
        if settings.index_weights and set(settings.index_weights) != set(symbols):
            raise ValueError("index_weights must name every broad_market symbol exactly")
        reader = FeatureWindowReader(
            store, self.calendar, settings.bar_cadence_ns, self.config.provider.source
        )
        per_symbol = {
            symbol: self._symbol(reader, symbol, as_of_ns, settings) for symbol in symbols
        }
        scores = {
            symbol: item.trend_score
            for symbol, item in per_symbol.items()
            if item.trend_score is not None
        }
        expected = len(symbols)
        available = len(scores)
        coverage = available / expected
        flags = set().union(*(item.quality_flags for item in per_symbol.values()))
        if available < expected:
            flags.update({QualityFlag.PARTIAL_UNIVERSE, QualityFlag.LOW_COVERAGE})
        raw_weights = settings.index_weights or dict.fromkeys(symbols, 1.0)
        largest_weight = max(raw_weights.values())
        scaled_weights = {symbol: weight / largest_weight for symbol, weight in raw_weights.items()}
        weight_total = sum(scaled_weights[symbol] for symbol in scores)
        direction_score = (
            sum(scores[symbol] * scaled_weights[symbol] for symbol in sorted(scores)) / weight_total
            if weight_total > 0 and coverage >= settings.minimum_market_coverage
            else None
        )
        if weight_total:
            signs = [
                (
                    1 if scores[symbol] > 0 else -1 if scores[symbol] < 0 else 0,
                    scaled_weights[symbol],
                )
                for symbol in sorted(scores)
            ]
            nonzero_weight = sum(weight for sign, weight in signs if sign)
            index_agreement = (
                abs(sum(sign * weight for sign, weight in signs)) / nonzero_weight
                if nonzero_weight
                else 0.0
            )
            if any(sign > 0 for sign, _ in signs) and any(sign < 0 for sign, _ in signs):
                flags.add(QualityFlag.INDEX_DISAGREEMENT)
        else:
            index_agreement = None
        trend_score = (
            direction_score * index_agreement
            if direction_score is not None and index_agreement is not None
            else None
        )
        contributors = (
            tuple(
                ContextContributor(
                    name=f"trend.index.{symbol}",
                    value=scores[symbol],
                    normalized_value=scores[symbol],
                    contribution=scores[symbol] * scaled_weights[symbol] / weight_total,
                    source_feature=f"trend.index.{symbol}",
                )
                for symbol in sorted(scores)
            )
            if weight_total
            else ()
        )
        quality = (
            DataQuality.UNUSABLE
            if direction_score is None
            else DataQuality.DEGRADED
            if flags
            else DataQuality.GOOD
        )
        return TrendFeatureSet(
            as_of_ns=as_of_ns,
            symbols=per_symbol,
            index_scores=scores,
            market_direction_score=direction_score,
            trend_score=trend_score,
            index_agreement=index_agreement,
            available_symbol_count=available,
            expected_symbol_count=expected,
            coverage_ratio=coverage,
            data_quality=quality,
            contributors=contributors,
            quality_flags=frozenset(flags),
        )
