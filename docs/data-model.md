# Data model

## Observations

`BarObservation`, `QuoteObservation`, and `TradeObservation` are provider-neutral records with source ID and causality timestamps. Prices and sizes are finite, nonnegative Python floats; bars require `low <= open, close <= high`. Zero volume is a reported numeric value; missing bars remain absent. Future numerical arrays may use other internal representations, but these boundary records stay JSON-safe. Source IDs follow `^[a-z][a-z0-9._-]*$` and are open to new adapters.

MCE-2 adds optional `bar_start_ns`/`bar_end_ns` to bars. Older MCE-1 bars remain constructible, but causal ingestion and replay require both bounds; effective availability must be at or after bar end. Quote and trade records have an optional nonnegative integer or nonempty string `provider_sequence`. No trade deduplication is inferred from timestamps alone. Canonical symbols trim surrounding whitespace and use uppercase while preserving punctuation/internal spacing; `provider_symbol` can retain the raw identifier separately.

## Snapshot

`MarketContextSnapshot` has independent `schema_version` (`1.0`) and `engine_version` (`0.5.0`) fields, `as_of_ns` and exact `as_of_iso`, session classification and optional trading date/identifier, categorical market/trend/volatility/correlation/liquidity/dispersion/breadth regimes, optional normalized scores, optional realized volatility, optional index and sector scores, minimal typed options and event context, optional confidence, data quality and flags, optional source freshness, contributors, and typed scalar feature values.

Directional scores are in `[-1, 1]`: market direction, trend, breadth, risk-on/off, index and sector scores. Unit scores are in `[0, 1]`: dispersion, correlation, stress, liquidity, volatility, confidence, event importance/risk, and implied-volatility percentile. MCE-4 `realized_volatility` is annualized decimal market context; `volatility_score` is the causal weighted percentile of per-symbol short volatility, not a forecast. MCE-5 `breadth_score` is bounded directional constituent participation from net advances and reference-level participation. It is null when member coverage is too low. Categorical breadth, volatility, and market regimes remain `UNKNOWN` unless an independent producer sets them.

`BreadthFeatureSet` is a separate immutable typed result with a read-only per-member map. Each `MemberBreadth` retains the point-in-time weight, causal current/previous prices, return, stale state, and reference comparisons. `ReferenceParticipation` carries horizon and numerator/denominator counts. `UniverseMembership` and `UniverseSnapshot` carry explicit effective/availability times and source quality; `MembershipTimeline` rejects overlapping symbol intervals. See [breadth engine](breadth-engine.md) for formula and survivorship-bias semantics.

All regime enums have `UNKNOWN`. `DataQuality` has `GOOD`, `DEGRADED`, `POOR`, `UNUSABLE`, and `UNKNOWN`. Quality flags are an extensible string enum; they are deduplicated as a frozen set and serialized in lexical order. MCE-2 added `MISSING_DATA`, `REJECTED_DATA`, and `OUTSIDE_SESSION`; MCE-3 adds `INSUFFICIENT_WINDOW_COVERAGE`, `HORIZON_DISAGREEMENT`, and `INDEX_DISAGREEMENT`. Flags describe issues without turning missing data into zero. The snapshot defaults uncomputed analytics to `null`; a zero indicates an actual calculated result. `EventRiskStatus` distinguishes `UNKNOWN`, `UNAVAILABLE`, `NONE_SCHEDULED`, and `SCHEDULED`.

`SourceFreshness` describes a source's latest effective time, age as of the snapshot, stale decision, and optional limit. MCE-2 evaluates it against explicit as-of time and configured category-specific thresholds. Contributors are named finite values with optional normalization, contribution, description, and source feature. `feature_values` accepts only finite scalar string, bool, int, or float values. Arbitrary nested objects are rejected.

Public models are frozen against field reassignment. Optional index, sector, and feature maps are copied into read-only mappings; all their values are scalars. Tuples and frozen sets cover the remaining collections. Use canonical serialization at the boundary rather than relying on Python object representation.

## Versioning

The package release is read from `mce.version.__version__` by packaging and snapshots. The public schema is separately defined by `SCHEMA_VERSION`. This release rejects other schema versions when loading a snapshot; explicit migration logic will be needed for a future schema change. Add optional fields with `null` defaults for compatible minor schema updates. Breaking changes require a major schema version and AME coordination.
