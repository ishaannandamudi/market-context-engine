# Freshness and coarse data quality

`evaluate_freshness(source, latest_effective_timestamp_ns, as_of_ns, freshness_limit_ns)` computes nonnegative `age_ns` from an explicit as-of clock. A missing latest timestamp returns `None`; it never becomes age zero. `age_ns > freshness_limit_ns` is stale, while exactly at the limit remains fresh. Future effective timestamps are rejected. No path reads machine wall time.

`FreshnessSettings` has separate default limits: bars 120 seconds, quotes 10 seconds, trades 30 seconds. These are configuration defaults for infrastructure, not claims that every vendor feed has identical service levels. A caller chooses the appropriate limit and can override it through nested `MCE_FRESHNESS__...` environment variables.

`assess_quality` returns `QualityAssessment(data_quality, quality_flags)` using explicit expected/observed symbols, expected sources, source freshness, provider statuses, rejected count, insufficient-history signal, and timestamp-mismatch signal. Its initial rules are:

| Condition | Quality / flags |
| --- | --- |
| No expected universe | `UNKNOWN`, `MISSING_DATA` |
| Expected symbols but no observed symbols | `UNUSABLE`, `MISSING_DATA`, `LOW_COVERAGE` |
| Some expected symbols missing | `DEGRADED`, `PARTIAL_UNIVERSE`, `LOW_COVERAGE` |
| Some expected sources absent | `PROVIDER_DEGRADED`, `LOW_COVERAGE` |
| Complete coverage and no known issue | `GOOD`, no flags |
| Any staleness, nonhealthy provider, rejection, insufficient history, or timestamp mismatch | Add corresponding flags; otherwise `GOOD` becomes `DEGRADED` |

`POOR` is reserved for a later calibrated policy. These are deterministic ingestion/availability signals, not a confidence model or market regime. `GOOD` means the supplied required-symbol evidence is complete and no supplied issue is known; callers must supply their actual essential universe and freshness evidence. Flags are deduplicated and serialized lexically.
