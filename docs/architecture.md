# Architecture

MCE is a standalone U.S. market-context research engine. Its public boundary is a versioned `MarketContextSnapshot`; AME and other consumers should read snapshots rather than import future feature internals. An external options analytics engine may supply options-derived context through a future adapter. MCE does not calculate option prices or execute trades.

## Layers

1. **Provider boundary:** `mce.providers.base` defines capabilities and historical/live protocols. `mce.providers.memory` is a deterministic, credential-free historical implementation. Future vendor adapters must normalize raw records before they reach downstream code. Source identifiers are validated lower-case strings rather than a closed vendor enum.
2. **Canonical observations:** Immutable bar, quote, and trade records carry event, effective, optional provider, and optional ingestion times. Bars may carry start/end bounds; MCE-2 ingestion requires them. These models do not calculate features.
3. **Calendar and ingestion:** `mce.calendar` wraps the XNYS schedule and classifies regular, extended, closed, and unknown time. `mce.ingestion` validates canonical batches, records explicit rejections and duplicates, and preserves suspicious off-session observations with flags.
4. **Causal replay and quality:** `mce.replay` sorts historical observations once, advances an explicit clock, and serves as-of queries from a bounded store. `mce.quality` evaluates freshness and coarse availability/ingestion quality without calculating confidence.
5. **Feature layer:** `mce.features` selects causal completed-bar windows and calculates trend, realized volatility, market breadth, sector context, and correlation. The sector package separates point-in-time assignment, return/score math, breadth, composite daily volatility, ranking/rotation, immutable results, orchestration, and snapshot mapping. The correlation package separates exact-grid return alignment, covariance/matrices, diagnostics, typed results, orchestration, and snapshot mapping. Feature engines read only canonical store data, explicit configuration/time, and typed in-memory membership. Regime and confidence modeling remain future work.
6. **Snapshot output:** The MCE-1 frozen, versioned `MarketContextSnapshot` remains the external contract. Any producer must enforce the causal invariant in [time semantics](time-semantics.md).

The protocols remain narrow. MCE-2 added separate historical quote/trade protocols while retaining the MCE-1 base and bar protocol. Historical replay and feature engines are synchronous; live adapters can remain async behind their protocol later. There are no network providers, database, or service endpoints.
