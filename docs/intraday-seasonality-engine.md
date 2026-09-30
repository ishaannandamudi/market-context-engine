# Intraday seasonality and session-relative normalization (MCE-16)

`IntradaySeasonalityEngine` reads completed XNYS regular-session one-minute bars and causal quotes at an explicit UTC nanosecond clock. It builds an immutable, prior-session-only `SeasonalHistoryIndex`, then compares current measurements with the **same session bucket** for each symbol. This is descriptive context. It never forecasts the rest of the session or changes earlier engines' raw values.

## Clock and buckets

`SessionRelativeTime` exposes actual calendar open and close, elapsed and remaining nanoseconds, elapsed fraction, early-close status, and the latest **completed** bucket. At the open there is no completed bucket. At an exact bucket boundary the ending bucket becomes available. A bucket ending after the clock is never read. The exact exchange close completes the last regular bucket; after that timestamp the seasonality clock is unavailable. Holidays and weekends have no session clock. Exchange schedules, not fixed UTC hours, handle EST, EDT, and shortened sessions.

The default `wall_clock` mode uses consecutive five-minute intervals from regular open, ending at actual close. A shortened session has fewer buckets, so a historical early close contributes no sample to nonexistent late buckets. `session_fraction` mode partitions the actual regular-session length into configurable equal-index bins. Boundaries are aligned downward to complete one-minute bar slots; every bin is nonempty. This compares, for example, the first seventh of a normal day with the first seventh of a shortened day, although those spans have different wall-clock durations. It is useful for stage-of-session context, but raw cumulative volumes across different-length sessions may differ because the elapsed minutes differ. Choose wall-clock mode for same elapsed time, fraction mode for same stage of session. Settings and mode enter the configuration fingerprint.

## Prior-session index and causality

`build_history(store, T)` selects up to `baseline_sessions` XNYS trading dates strictly before T's trading date (default 60). For each current point-in-time member it reads historical data only on dates when that member belonged to the universe at the historical close. The membership provider enforces `effective_from_ns <= T < effective_to_ns` and `available_at_ns <= T`; later additions and corrections cannot enter earlier snapshots. The index stores immutable `(metric, symbol, bucket, historical date)` observations, rejects duplicate/out-of-window/future-available samples, and records the target date, build clock, configuration fingerprint and content fingerprint. Reusing it for another day, future build clock or configuration is rejected. A current day's later bucket never enters the prior-session baseline. Callers can reuse the index across current-day snapshots; rebuild explicitly if a newly active member needs history. No global cache or hidden mutable state is used.

Bars must match the configured source, cover one complete cadence interval within regular session, have positive open and close, end by T, and be effectively available by T. A bucket needs **every** expected one-minute bar. A missing bar removes that bucket's bar metrics; cumulative volume after a gap stays unavailable. Missing history is absent, never forward-filled or zero-filled. Quotes must be causal, within regular hours and their bucket, noncrossed with positive bid, and no older than `quote_freshness_ns` (default 60 seconds) at bucket end. The current completed bucket's quote must also be fresh at T. A stale or crossed quote does not become a normal spread.

## Raw measurements and normalization

For each complete bucket and symbol:

| Metric | Raw calculation | Units |
| --- | --- | --- |
| `volume` | Sum of bar volumes | Shares |
| `cumulative_volume` | Sum from regular open through the completed bucket, with no missing bucket | Shares |
| `dollar_volume` | Sum of `bar.volume × bar.close` | Input currency units |
| `volatility` | Sum of `ln(bar.close/bar.open)^2` | Squared log return (realized-variance proxy) |
| `quoted_spread` | `ask−bid` from the latest valid fresh quote in the bucket | Input price units |
| `spread` | `(ask−bid)/((ask+bid)/2)` from the same quote | Fraction of midpoint |
| `illiquidity` | `abs(ln(last close/first open))/bucket dollar volume` | Absolute log return per currency unit; unavailable at zero dollar volume |
| `correlation` | Optional precomputed MCE-7 average pairwise correlation at the **exact** completed bucket end | Unitless |

The volatility proxy uses bar open-to-close returns, which can omit interbar jumps; it is not the MCE-4 estimator. Illiquidity is an observed price-response proxy, not execution cost. The engine consumes supplied correlation contexts and never reruns the correlation engine. Historical correlation contexts must be prior-session and exactly bucket-ended; a future context is rejected and an off-boundary current context is ignored. Breadth direction and MCE-9 stress are not de-seasonalized here.

For every `(metric, symbol, bucket)`, the prior sample distribution yields median, mean, sample standard deviation (`ddof=1`), median absolute deviation (MAD), and 25th/75th quantiles using NumPy's explicit `method="linear"`. Expected value is the **median**. At least `minimum_history_sessions` prior observations (default 20) are required. `sample_count / number_of_prior_dates` is baseline coverage. Normalization is:

- `ratio = current / median` only when median > `1e−24` and the finite result is representable.
- `percentile = (count(prior < current) + 0.5 × count(prior = current))/N` (midrank ties).
- `robust_z = (current − median)/(1.4826 × MAD)` when MAD > `1e−24`. A current value equal to a zero-MAD median gives zero; other zero-MAD cases give `None`.

If current or baseline data is missing, normalized values are `None` and raw data remains in `SeasonallyAdjustedMetric`. No NaN or infinity is emitted. A high volume, variance, spread, or illiquidity ratio means more of that observed quantity than usual for the bucket; it is not one generic good/bad score. Correlation can be negative, so its ratio is only defined when the historical median is positive. Percentile and robust z remain available with a sufficient nonpositive correlation baseline.

`seasonality.adjusted_rvol` aliases the market median of eligible **per-symbol cumulative-volume ratios**. It compares current cumulative volume with prior cumulative volume through the equivalent completed bucket. MCE-9 `liquidity.relative_volume` remains its existing raw same-elapsed-time output, and MCE-4/MCE-7 raw outputs retain their semantics. A normal high-volume opening can have ratio near one; a later genuine spike can have ratio well above one.

## Market aggregation, quality and snapshot

The engine retains each expected point-in-time member and its raw/normalized metrics. `member_coverage` is members with a complete latest bucket divided by expected members. For each metric, `normalized_symbols` and `coverage` count members with a valid ratio. When both `minimum_members` (default three) and `minimum_member_coverage` (default 0.8) are met, market ratio, percentile, and robust z are separate equal-member medians; no single volume curve is imposed across symbols. Low coverage, missing bars, stale/crossed quotes, insufficient historical samples, and degraded membership are flagged. A headline with insufficient coverage is withheld; raw per-symbol measurements remain inspectable. A ratio can be missing even when percentile exists (zero baseline), so ratio coverage is conservative.

`apply_seasonality_to_snapshot` adds continuous `seasonality.*` feature values to schema 1.0, including session fraction, coverage, volume/cumulative-volume/dollar-volume/volatility/quoted-spread/relative-spread/illiquidity ratios, percentiles and robust z values when available, `seasonality.adjusted_rvol`, and optional correlation normalization. It replaces an older `seasonality.*` namespace on replay, preserves all other feature values and categorical regimes, and propagates quality. It never rewrites existing raw RVOL, volatility, correlation, stress, or regime values. The engine version is 0.16.0; statistical feature schema remains 1.0.

`features.seasonality` is typed and fingerprinted: `enabled`, `universe`, `bucket_mode`, `bucket_minutes` or `session_fraction_bins`, one-minute `bar_cadence_ns`, `baseline_sessions`, `minimum_history_sessions`, `minimum_member_coverage`, `minimum_members`, and `quote_freshness_ns`. Fraction mode requires bins and no minute size; wall-clock mode requires minutes and no bins. Data are synthetic in tests and benchmark; no vendor feed, credentials, broker, option pricing, execution, or trading capability is introduced.

The manual benchmark is `.venv/bin/python benchmarks/seasonality.py`. It defaults to 500 members and 60 prior sessions, each with five one-minute bars and five quotes in the first bucket; three current snapshots reuse one built history index. Its reported baseline build and per-snapshot times exclude synthetic data ingestion. This is an opening-bucket benchmark, not a full-session throughput claim or CI timing gate.
