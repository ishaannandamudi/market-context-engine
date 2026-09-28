# Architecture

MCE is a standalone U.S. market-context research engine. Its public boundary is a versioned `MarketContextSnapshot`; AME and other consumers should read snapshots rather than import future feature internals. An external options analytics engine may supply options-derived context through a future adapter. MCE does not calculate option prices or execute trades.

## Layers

1. **Provider adapters (future):** Vendor, file, or replay code implements the small protocols in `mce.providers.base` and converts source-specific records into canonical observations. Source identifiers are validated lower-case strings rather than a closed vendor enum, so adding a provider does not change the public schema. Capability metadata lets orchestration handle absent data gracefully.
2. **Canonical observations (MCE-1):** Immutable bar, quote, and trade records carry event, effective, optional provider, and optional ingestion times. They do not calculate features.
3. **Calendar and features (future):** Session identification and broad-market feature computation will operate on canonical inputs. No weekday or session-time shortcuts exist in this phase.
4. **Regimes and quality (future):** Regime assignment, confidence assessment, and explanation will use available features and explicit quality flags. Unknown and unavailable remain distinct from zero.
5. **Snapshot output (MCE-1 contract):** Frozen, versioned snapshot model with canonical JSON serialization. Any producer must enforce the causal invariant described in [time semantics](time-semantics.md).

The protocols are deliberately narrow. Historical access defines bars; live access defines a bar stream. Later phases may add focused quote/trade protocols without bloating the base interface. No provider implementation, network request, persistence layer, or service is part of MCE-1.
