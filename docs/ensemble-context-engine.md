# Explainable ensemble market context (MCE-19)

`EnsembleContextFeatureEngine` combines **precomputed descriptive context** at UTC nanosecond clock `T`. It does not run upstream engines, train a meta-model, estimate future returns, or recommend trades. `EnsembleContextInput` requires a schema-1.0 `MarketContextSnapshot` at exactly `T` and optionally accepts typed trend, breadth, sector, volatility, liquidity/stress, correlation, dispersion, factor, network, deterministic regime, transition, filtered HMM, seasonality, options, and macro results. A typed source result supersedes the snapshot fallback for that family. Optional families remain missing, not zero. Future-dated upstream results and source availability are rejected. A source result from before `T` loses reliability with age.

The MCE-10 deterministic classification, MCE-11 stability, and statistical/HMM state retain separate identities. The new headline is `ensemble.headline` in the `feature_values` namespace; `market_regime` and prior feature fields are never overwritten. Snapshot, statistical, options, and macro schemas remain 1.0. The ensemble was introduced in engine version 0.19.0 and remains available in the 1.0.0 system release.

## Evidence groups and orientation

| Group | Family votes and configured within-group weights | Orientation |
| --- | --- | --- |
| Direction | trend 1.00 | Signed `[-1,1]`; positive is upward market direction. |
| Participation | breadth .70, sector breadth .30 | Signed `[-1,1]`; positive is broad participation. |
| Volatility/stress | realized volatility .40, MCE-9 market stress .60 | `[0,1]`; higher is more observed market stress. Correlated volatility and stress are one group vote. |
| Liquidity health | MCE-9 liquidity 1.00 | `[0,1]`; higher is healthier liquidity. |
| Systemic structure | MCE-7 correlation .35, MCE-14 factor dominance .25, MCE-15 network connectivity .25, MCE-8 dispersion differentiation complement .15 | `[0,1]`; higher is more synchronized/systemic. Correlation, factors, and network never receive separate full core weights. The snapshot dispersion fallback uses `1−dispersion_score`; the typed result prefers `1−differentiation_score`. |
| External risk | MCE-17 options stress .35, MCE-18 observed macro stress .35, MCE-18 scheduled-event proximity .30 | `[0,1]`; higher means more external or scheduled information context. This group has **zero weight in the core market score** and is reported separately. Event proximity is distinct from observed macro stress. |

The sector snapshot fallback reads `sector.*.breadth` feature values. `sector_scores` stores relative strength, so it is not used as a participation vote.

For each available family `i`, reliability is `r_i = Q_i × coverage_i × max(0,1−age_i/max_source_age) × 0.85^min(flag_count,4) × modifier_i`, clipped to `[0,1]`. Quality factors are good `1`, degraded `.7`, poor `.2`, unusable/unknown `0`. `max_source_age` defaults to 24 hours. Snapshot fallback core values receive a `.5` freshness modifier when a non-external snapshot source is flagged stale. Options source modes use live/synthetic `1`, delayed `.8`, EOD `.6`; a primary options context that is not headline-eligible receives `0`. Stale macro observations/releases reduce usable-series coverage. An acceptable seasonality baseline showing a normal volatility ratio (`0.8–1.2`) reduces the **reliability** of elevated raw-volatility evidence by `.75`; it never rewrites raw volatility. A missing or poor family is not treated as a calm vote.

An evidence vote is active only when `r_i >= minimum_reliability` (default `.35`). The result exposes its configured within-group weight `c_i`, adjusted weight `c_i r_i`, and normalized active weight `c_i r_i / Σ_j c_j r_j`. Group score is `Σ_i(value_i c_i r_i)/Σ_i(c_i r_i)` across eligible families only. Group reliability is `Σ_i(c_i r_i)/Σ_i c_i` across those same families. Empty or insufficient-quality groups are missing. The result exposes all family and group weights, including zero adjusted weights, so renormalization is visible.

Core configured group weights are direction `.30`, participation `.20`, volatility/stress `.25`, liquidity health `.15`, and systemic structure `.10`. Active group weight is `w_g × group_reliability_g`, renormalized over active core groups. Each group contributes exactly `normalized_weight × oriented_score`, where oriented scores are direction, participation, `1−2×volatility_stress`, `2×liquidity_health−1`, and `1−2×systemic_structure`. Their sum is the signed `ensemble.core_market_score` in `[-1,1]`. Typed contributors reconcile to that sum. The core score is withheld below the configured minimum active groups; a headline additionally requires three of the four critical groups (direction, participation, volatility/stress, liquidity) and four active families by default.

## Disagreement, HMM, and uncertainty

Stable conflict IDs distinguish `direction_vs_breadth`, `realized_vs_options`, `correlation_vs_network`, `factor_vs_network`, `structure_vs_dispersion`, `liquidity_vs_stress`, `macro_vs_market`, `deterministic_vs_hmm`, and `deterministic_vs_core`. Each record retains both values, source families, severity `[0,1]`, and a description. Opposing trend/breadth severity is the smaller absolute magnitude; high options versus calm realized volatility, structure divergences, and macro-versus-market stress use the bounded difference; healthy-liquidity/high-stress severity is the smaller score. `ensemble.disagreement = min(1, Σ conflict_severity / 2)` when at least two eligible families exist, otherwise missing. `ensemble.consensus = 1−disagreement` exactly when disagreement exists. Contradictory evidence is retained rather than averaged away.

Only out-of-sample **filtered**, nonretrospective HMM inference with matching statistical feature schema enters the input. Smoothed and full-path Viterbi outputs fail validation. State IDs have no economic meaning by themselves. A matching latent-state profile and explicitly supplied direction/stress interpretation are required before an HMM can disagree with deterministic classification. HMM reliability is bounded normalized posterior-entropy confidence times convergence, degeneracy, occupancy, quality-flag, and freshness factors. A weak HMM cannot create a regime conflict. This reliability is **not** a model probability; the HMM posterior is **not** a crash probability.

MCE-11 `stability_score` passes through as `ensemble.temporal_stability`; no transition history is recalculated. With active groups, `ensemble.evidence_strength = mean_active_group_reliability × core_group_coverage × (1−disagreement) × (0.5+0.5×|core_market_score|)`, clipped to `[0,1]`; it is not a calibrated probability. `ensemble.uncertainty` is the clipped sum of `.45×disagreement + .30×(1−mean_reliability) + .15×missing_core_group_fraction + .05×HMM_reliability_penalty + .05×temporal_instability`. Missing HMM or stability contributes zero to its specific penalty. Evidence strength and temporal stability remain separate.

## Ordered headline rules

Rules run in configurable precedence order. Each rule trace records its stable ID, condition, matched status, input values, thresholds, and precedence. The first match wins. If critical evidence is insufficient, the headline is `UNKNOWN` and no rule matches. Thresholds below are defaults and can be changed in typed configuration.

| Precedence | Headline | Required supporting evidence and thresholds | Contradictory/limiting evidence |
| --- | --- | --- | --- |
| 1 | `SYSTEMIC_STRESS` | volatility/stress ≥ `.80`, systemic structure ≥ `.65`, plus liquidity ≤ `.25` or independent MCE-10 systemic classification | No rule without both observed stress and structural synchronization. Overrides direction. |
| 2 | `LIQUIDITY_STRESS` | liquidity health ≤ `.25` | Poor liquidity can be visible even while realized volatility is calm. |
| 3 | `VOLATILITY_STRESS` | volatility/stress ≥ `.80` | Liquidity and structure are reported separately. |
| 4 | `BROAD_RISK_OFF` | direction ≤ `−.35`, participation ≤ `−.20` | Severe stress rules take precedence. |
| 5 | `BROAD_RISK_ON` | direction ≥ `.35`, participation ≥ `.20`, stress ≤ `.50`, liquidity ≥ `.55`, disagreement < `.65` | Conflicting or high-stress evidence prevents this label. Nearby events alone do not override it. |
| 6 | `ROTATION` | `|direction| < .35`, structure < `.65`, sector rotation or dispersion evidence ≥ `.60` | Broad synchronized structure blocks rotation. |
| 7 | `CALM_RANGE` | `|direction| ≤ .20`, stress ≤ `.35`, liquidity ≥ `.55` | Elevated stress or weak liquidity blocks calm range. |
| 8 | `EXTERNAL_EVENT_RISK` | opt-in; external risk and eligible scheduled-event proximity ≥ `.85`, `|core_market_score| ≤ .20` | Disabled by default; observed market-state labels above take precedence. Event proximity is **not** expected market movement. |
| 9 | `TRANSITIONAL` | temporal stability ≤ `.35` or disagreement ≥ `.65` | Higher-precedence observed states still win. |
| 10 | `MIXED` | sufficient critical evidence without an earlier match | Holds conflicting or ambiguous context without inventing a directional label. |

MCE-10 `RISK_ON` can coexist with ensemble `BROAD_RISK_ON` when participation, liquidity, and stress corroborate it; a disagreement is explicit when it does not. MCE-10 `SYSTEMIC_STRESS` supports the ensemble systemic rule but never replaces the required observed stress and structure thresholds. `UNKNOWN` means required evidence is absent. No label is a forecast or trading recommendation.

## Causality, projection, and limits

The input contract requires the snapshot at `T`, rejects future source calculation clocks and known future availability, rejects future factor fits, and rejects retrospective HMM inference. Upstream engines remain responsible for their own bar completion, point-in-time membership, vintage resolution, and source freshness. Ensemble replay with the same typed inputs and configuration produces byte-equivalent serialization. The projection replaces only `ensemble.*` scalars; MCE-10 `market_regime`, market stress, options, macro, and statistical source outputs remain intact. Options IV/RV spread is **not** guaranteed mispricing; external risk is **not** a crash probability or a cause of market moves.

Run `.venv/bin/python -m benchmarks.ensemble_context --iterations 10000` for separate reliability, grouping, conflict, rule, projection, and full-engine timings over precomputed contexts. No upstream calculation occurs inside the timed loop. Service/API integration, provider wiring, replay orchestration, and final hardening are deferred to MCE-20.
