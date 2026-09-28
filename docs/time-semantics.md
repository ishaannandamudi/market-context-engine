# Time semantics and causality

All internal instants are integer nanoseconds since the Unix epoch in UTC. `as_of_ns` is authoritative. `as_of_iso` is an exact ISO-8601 UTC rendering with nine fractional digits and `Z`, derived and checked against the integer. Python `datetime` views lose sub-microsecond precision; retain the original nanosecond integer in records and snapshots. Naive datetimes and pre-epoch timestamps are rejected. No domain model reads the local clock.

Canonical observations use these distinct times:

- `event_timestamp_ns`: when the market event occurred.
- `provider_timestamp_ns` (optional): when a provider says it published or timestamped the record. Its precise meaning depends on the adapter and must be documented there.
- `ingestion_timestamp_ns` (optional): when this MCE deployment received or stored the record. Historical backfills can have ingestion after the historical effective time.
- `effective_timestamp_ns`: earliest time this observation may be used in the chosen replay/live computation, after any relevant publication or pipeline delay. It cannot precede the event or a supplied provider timestamp. Adapters must set it conservatively; MCE-1 cannot infer it from absent provider metadata.

**Causal invariant:** a `MarketContextSnapshot` at time `T` may depend only on information whose effective availability timestamp is at or before `T`. MCE-1 validates directly embedded source freshness and options-context times against `as_of_ns`. Future feature and snapshot producers must enforce the invariant for every contributing observation; a snapshot model alone cannot prove the provenance of an arbitrary score.

`MarketSession` offers `PRE_MARKET`, `REGULAR`, `AFTER_HOURS`, `CLOSED`, and `UNKNOWN`, plus optional trading date and identifier in the snapshot. It is a contract, not a calendar calculation. MCE-2 will add U.S. holiday, early-close, regular-session, and extended-hours interpretation in `America/New_York`; stored instants remain UTC.
