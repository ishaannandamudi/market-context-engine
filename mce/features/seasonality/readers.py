"""Completed regular-session bar and quote metrics, with no partial-bucket fill."""

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from datetime import date
from math import isfinite, log

from mce.calendar.service import MarketSessionService
from mce.config.settings import MCEConfig
from mce.features.seasonality.clock import bucket_grid
from mce.features.seasonality.models import MetricName, SeasonalObservation, SessionBucket
from mce.models.market_data import BarObservation, QuoteObservation
from mce.replay.store import CausalDataStore


@dataclass(frozen=True, slots=True)
class BucketRead:
    observations: tuple[SeasonalObservation, ...]
    missing_bars: bool
    invalid_quotes: bool
    stale_quotes: bool


def read_session_metrics(
    store: CausalDataStore,
    calendar: MarketSessionService,
    config: MCEConfig,
    symbol: str,
    day: date,
    as_of_ns: int,
    *,
    current: bool,
    buckets: tuple[SessionBucket, ...] | None = None,
) -> BucketRead:
    settings = config.features.seasonality
    opened, closed = calendar.regular_session_bounds(day)
    buckets = buckets if buckets is not None else bucket_grid(calendar, day, settings)
    cutoff = min(as_of_ns, closed)
    cadence = settings.bar_cadence_ns
    slots = tuple(range(opened + cadence, closed + 1, cadence))
    selected: dict[int, BarObservation] = {}
    for bar in store.bars(symbol, opened, cutoff + 1, as_of_ns):
        start, end = bar.bar_start_ns, bar.bar_end_ns
        if (
            bar.source != config.provider.source
            or start is None
            or end is None
            or end > cutoff
            or start < opened
            or end - start != cadence
            or (end - opened) % cadence != 0
            or bar.open <= 0
            or bar.close <= 0
        ):
            continue
        prior_bar = selected.get(end)
        if prior_bar is None or (bar.effective_timestamp_ns, bar.close) > (
            prior_bar.effective_timestamp_ns,
            prior_bar.close,
        ):
            selected[end] = bar
    quote_ends = tuple(bucket.end_ns for bucket in buckets)
    current_quote_bucket = bisect_right(quote_ends, as_of_ns) - 1 if current else None
    selected_quotes: dict[int, QuoteObservation] = {}
    invalid_quotes = False
    for quote in store.quotes(symbol, opened, cutoff + 1, as_of_ns):
        if quote.source != config.provider.source or quote.event_timestamp_ns >= closed:
            continue
        index = bisect_left(quote_ends, quote.event_timestamp_ns)
        if index >= len(buckets) or quote.event_timestamp_ns <= buckets[index].start_ns:
            continue
        prior_quote = selected_quotes.get(index)
        if prior_quote is None or (
            quote.event_timestamp_ns,
            quote.effective_timestamp_ns,
            quote.bid,
            quote.ask,
        ) > (
            prior_quote.event_timestamp_ns,
            prior_quote.effective_timestamp_ns,
            prior_quote.bid,
            prior_quote.ask,
        ):
            selected_quotes[index] = quote
    observations: list[SeasonalObservation] = []
    missing_bars = False
    stale_quotes = False
    cumulative_complete = True
    cumulative_volume = 0.0
    cumulative_available = opened
    for bucket in buckets:
        if bucket.end_ns > as_of_ns:
            break
        expected = slots[bisect_right(slots, bucket.start_ns) : bisect_right(slots, bucket.end_ns)]
        complete = bool(expected) and all(end in selected for end in expected)
        if not complete:
            missing_bars = True
            cumulative_complete = False
        else:
            bars = tuple(selected[end] for end in expected)
            volume = sum(bar.volume for bar in bars)
            dollar = sum(bar.volume * bar.close for bar in bars)
            variance = sum((log(bar.close) - log(bar.open)) ** 2 for bar in bars)
            if not all(isfinite(value) for value in (volume, dollar, variance)):
                missing_bars = True
                cumulative_complete = False
            else:
                cumulative_volume += volume
                available = max(bucket.end_ns, *(bar.effective_timestamp_ns for bar in bars))
                cumulative_available = max(cumulative_available, available)
                bucket_metrics: tuple[tuple[MetricName, float], ...] = (
                    ("volume", volume),
                    ("dollar_volume", dollar),
                    ("volatility", variance),
                )
                for metric, value in bucket_metrics:
                    observations.append(
                        SeasonalObservation(
                            metric, symbol, day, bucket.index, bucket.end_ns, available, value
                        )
                    )
                if dollar > 0:
                    impact = abs(log(bars[-1].close) - log(bars[0].open)) / dollar
                    observations.append(
                        SeasonalObservation(
                            "illiquidity",
                            symbol,
                            day,
                            bucket.index,
                            bucket.end_ns,
                            available,
                            impact,
                        )
                    )
                if cumulative_complete and isfinite(cumulative_volume):
                    observations.append(
                        SeasonalObservation(
                            "cumulative_volume",
                            symbol,
                            day,
                            bucket.index,
                            bucket.end_ns,
                            cumulative_available,
                            cumulative_volume,
                        )
                    )
                elif not isfinite(cumulative_volume):
                    missing_bars = True
                    cumulative_complete = False
        current_quote = selected_quotes.get(bucket.index)
        if current and bucket.index != current_quote_bucket:
            continue
        if current_quote is not None and (
            current_quote.bid <= 0 or current_quote.ask < current_quote.bid
        ):
            invalid_quotes = True
        elif current_quote is not None and (
            bucket.end_ns - current_quote.event_timestamp_ns > settings.quote_freshness_ns
            or current
            and as_of_ns - current_quote.effective_timestamp_ns > settings.quote_freshness_ns
        ):
            stale_quotes = True
        elif current_quote is not None:
            scale = max(current_quote.ask, current_quote.bid)
            relative_bid = current_quote.bid / scale
            relative_ask = current_quote.ask / scale
            spread = (relative_ask - relative_bid) / ((relative_ask + relative_bid) / 2)
            quoted_spread = current_quote.ask - current_quote.bid
            if isfinite(spread) and isfinite(quoted_spread):
                quote_metrics: tuple[tuple[MetricName, float], ...] = (
                    ("quoted_spread", quoted_spread),
                    ("spread", spread),
                )
                for metric, value in quote_metrics:
                    observations.append(
                        SeasonalObservation(
                            metric,
                            symbol,
                            day,
                            bucket.index,
                            bucket.end_ns,
                            max(bucket.end_ns, current_quote.effective_timestamp_ns),
                            value,
                        )
                    )
            else:
                invalid_quotes = True
    return BucketRead(tuple(observations), missing_bars, invalid_quotes, stale_quotes)


def current_bucket_values(
    read: BucketRead, bucket_index: int
) -> dict[MetricName, SeasonalObservation]:
    return {item.metric: item for item in read.observations if item.bucket_index == bucket_index}
