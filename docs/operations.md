# Operation and diagnostics

The pipeline accepts frozen configuration per instance and explicit UTC nanosecond clocks. Validate startup configuration and the DAG with `python -m mce validate-config --config config.json`. Required broad-market bars must be completed and causally available. `PipelineError` preserves stable failure codes in Python; the service returns safe generic errors for unexpected failures.

Stage diagnostics record status (`ok`, `degraded`, `withheld`), failure policy, output quality, reason code, and monotonic runtime duration. Durations are operational only. The snapshot quality aggregates upstream quality conservatively and retains source quality flags; missing optional families appear in diagnostics instead of fabricated zero scores. `GET /ready` reports critical market-source status separately from options/macro health. `GET /health` checks process liveness only.

The service logs snapshot/replay completion at INFO with clock and duration, plus snapshot quality or replay count; rejected requests at WARNING; unexpected failures at ERROR. It never logs request bodies, credentials, environment values, or exception text. In-memory counters track completed, failed, degraded, and replayed snapshots. Service instances hold immutable observations and a stateless pipeline; per-request stores and membership caches make independent concurrent calls safe. The provided `wsgiref` server is synchronous and intended for local research use.

Troubleshooting:

- `INSUFFICIENT_MARKET_DATA`: check configured source, broad-market symbols, bar bounds, exchange regular session, and effective timestamp.
- Withheld network: inspect correlation matrix coverage; the network stage does not invent a graph.
- Withheld seasonality: check for completed prior-session bars. The pipeline can build a causal `SeasonalHistoryIndex` from the store, or accept a supplied matching history.
- Withheld options/macro: inspect source availability, quality, and source mode/vintage metadata.
- Withheld HMM: provide only an available out-of-sample filtered state. `FilteredHMMRegistry` selects records by explicit availability.
- Unexpected replay difference: compare config, canonical observation, membership, options/macro, and artifact fingerprints in `SnapshotLineage`.

Manual profile: `.venv/bin/python -m benchmarks.full_pipeline --symbols 500 --bars 30 --snapshots 2 --prior-sessions 22`. Each prior session supplies two sparse constituent bars and a completed daily SPY bar; 30 current-session minute bars are supplied for every symbol. Prior sessions exercise factor and seasonality history. Options, macro, and filtered HMM contexts are precomputed synthetic inputs. Add `--memory` for `tracemalloc` peak; this changes timing. The benchmark includes replay schedule, observation ingestion, all configured feature stages, snapshot projection, lineage, and fingerprinting. Synthetic input creation and optional context preparation are excluded. `replay_per_snapshot_ms` includes replay ingestion amortized over snapshots; `pipeline_mean_ms` measures snapshot orchestration alone. Do not use local elapsed time as a CI assertion.
