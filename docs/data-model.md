# Data model

## Observations

`BarObservation`, `QuoteObservation`, and `TradeObservation` are provider-neutral records with source ID and causality timestamps. Prices and sizes are finite, nonnegative Python floats; bars require `low <= open, close <= high`. This catches obviously corrupt records while leaving unusual but nonnegative provider quotes or zero-volume bars representable and flaggable. Future numerical arrays may use other internal representations, but these boundary records stay JSON-safe. Source IDs follow `^[a-z][a-z0-9._-]*$` and are open to new adapters.

## Snapshot

`MarketContextSnapshot` has independent `schema_version` (`1.0`) and `engine_version` (`0.1.0`) fields, `as_of_ns` and exact `as_of_iso`, session classification and optional trading date/identifier, categorical market/trend/volatility/correlation/liquidity/dispersion/breadth regimes, optional normalized scores, optional realized volatility, optional index and sector scores, minimal typed options and event context, optional confidence, data quality and flags, optional source freshness, contributors, and typed scalar feature values.

Directional scores are in `[-1, 1]`: market direction, trend, breadth, risk-on/off, index and sector scores. Unit scores are in `[0, 1]`: dispersion, correlation, stress, liquidity, volatility, confidence, event importance/risk, and implied-volatility percentile. `realized_volatility` and implied volatility are nonnegative finite raw values, not normalized scores. Scores are conventions for future calculations, not calculations in this phase.

All regime enums have `UNKNOWN`. `DataQuality` has `GOOD`, `DEGRADED`, `POOR`, `UNUSABLE`, and `UNKNOWN`. Quality flags are an extensible string enum; they are deduplicated as a frozen set and serialized in lexical order. Flags describe issues without turning missing data into zero. The snapshot defaults uncomputed analytics to `null`; a zero indicates an actual calculated result. `EventRiskStatus` distinguishes `UNKNOWN`, `UNAVAILABLE`, `NONE_SCHEDULED`, and `SCHEDULED`.

`SourceFreshness` describes a source's latest effective time, age as of the snapshot, stale decision, and optional limit. MCE-1 validates age consistency but defines no threshold policy. Contributors are named finite values with optional normalization, contribution, description, and source feature. `feature_values` accepts only finite scalar string, bool, int, or float values. Arbitrary nested objects are rejected.

Public models are frozen against field reassignment. Optional index, sector, and feature maps are copied into read-only mappings; all their values are scalars. Tuples and frozen sets cover the remaining collections. Use canonical serialization at the boundary rather than relying on Python object representation.

## Versioning

The package release is read from `mce.version.__version__` by packaging and snapshots. The public schema is separately defined by `SCHEMA_VERSION`. This release rejects other schema versions when loading a snapshot; explicit migration logic will be needed for a future schema change. Add optional fields with `null` defaults for compatible minor schema updates. Breaking changes require a major schema version and AME coordination.
