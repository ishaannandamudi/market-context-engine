# Deterministic market regimes (MCE-10)

`MarketRegimeEngine.compute(RegimeEvidence, history=...)` classifies **observed** market context. It is a pure decision function: the same current evidence, configuration, and explicit prior states produce the same result. `evidence_from_features` adapts already-computed MCE-3 through MCE-9 results only when every supplied result has the exact clock; `evidence_from_snapshot` adapts populated schema-1.0 scores. Regime code reads no raw data, calls no upstream engine, and uses no wall clock or hidden process state. Its `MarketRegimeState` retains raw evidence, nine dimension labels, the headline, conflicts, rule trace, evidence strength, hysteresis details, transitions, quality, and the configuration fingerprint.

The classification describes the present evidence. It is **not** a forecast, calibrated probability, crash risk estimate, or trading recommendation. Default thresholds are interpretable choices; they were not fitted to future returns or PnL.

## Taxonomy and exact dimension boundaries

The detailed headline taxonomy is `RISK_ON`, `RISK_OFF`, `RANGE_BOUND`, `ROTATION`, `SYSTEMIC_STRESS`, `LIQUIDITY_STRESS`, `TRANSITIONAL`, `MIXED`, and `UNKNOWN`. These labels are mutually exclusive **only at the headline**. The dimensions remain separate. All comparisons use the configured values below; `>=`/`<=` are inclusive, while `<`/`>` are exclusive. Missing input produces `UNKNOWN` for its dimension, never a fabricated neutral score.

| Dimension | Default decision boundaries |
| --- | --- |
| Direction | `POSITIVE` if direction `>= .35`, trend `>= .30`, and breadth is missing or `> −.45`; `NEGATIVE` if direction `<= −.35`, trend `<= −.30`, and breadth is missing or `< .45`. `NEUTRAL` if `abs(direction) < .20` and `abs(trend) < .30`; otherwise `MIXED`. Both direction and trend are required. |
| Trend | `STRONG_UP >= .70`, `UP >= .30`, `STRONG_DOWN <= −.70`, `DOWN <= −.30`, otherwise `SIDEWAYS`. |
| Breadth | Requires breadth score and, when supplied, breadth coverage `>= minimum_evidence_quality` (default .5). `BROAD_POSITIVE` if score `>= .45` and participation `>= .15`; `NARROW_POSITIVE` if score `>= .15` without broad confirmation. Negative mirrors use `<= −.45` and `<= −.15`; otherwise `MIXED`. |
| Volatility | `LOW < .25`, `HIGH >= .70`, `EXTREME >= .90`, otherwise `NORMAL`. |
| Liquidity | Higher is better: `STRESSED < .20`, `THIN < .45`, `HEALTHY >= .75`, otherwise `NORMAL`. |
| Correlation | `LOW < .25`, `HIGH >= .70`, `EXTREME >= .90`, otherwise `NORMAL`. The public `EXTREME` label represents the strongest co-movement range. |
| Dispersion | `LOW < .25`, `HIGH >= .70`, `EXTREME >= .90`, otherwise `NORMAL`. This does not imply high volatility. |
| Stress | `CALM < .30`, `ELEVATED >= .30`, `HIGH >= .65`. `SYSTEMIC` requires stress `>= .85` **and** stressed-member fraction `>= .60`; a high score without that fraction remains `HIGH`. |
| Sector rotation | `ACTIVE_ROTATION >= .70`; at that velocity, coverage `< .5` instead yields `FRAGMENTED`. `STABLE_LEADERSHIP <= .40`; otherwise `MIXED`. The raw sector leadership spread and breadth remain in the scorecard and conflict checks. |

These are dimension-entry thresholds. Hysteresis applies only when an explicit prior state exists, as described below.

## Headline rule table

The table is evaluated in ascending precedence. The first matching rule becomes the **candidate**. The default order is configurable as a validated permutation of the eight stable rule IDs. An eligible final headline may then be held by hysteresis. `rule_trace` contains every rule, its Boolean result, source feature names, thresholds, score, reason, and configured precedence. `candidate_rule_id` is always the first matching rule; `selected_rule_id` identifies the final rule after any hold.

| Priority / stable ID | Headline | Required evidence and threshold | Supporting evidence | Contradictory evidence / resolution |
| --- | --- | --- | --- | --- |
| 0 `regime.systemic_stress` | `SYSTEMIC_STRESS` | stress `>= .85`, breadth `<= −.15`, and either correlation `>= .70` or liquidity `< .20` | stressed-member fraction, volatility, negative sector breadth | Overrides ordinary direction by default; conflicts remain recorded. |
| 1 `regime.liquidity_stress` | `LIQUIDITY_STRESS` | liquidity `< .20` and stress `>= .65` | wide quoted spreads and Amihud in upstream MCE-9 evidence | Overrides ordinary trend; missing liquidity cannot satisfy it. |
| 2 `regime.transitional` | `TRANSITIONAL` | maximum conflict severity `>= .80` | any contradictory raw families | Overrides directional/range/rotation rules, below acute stress rules. |
| 3 `regime.risk_off` | `RISK_OFF` | `NEGATIVE` direction, breadth `<= −.15`, stress `< .85` | negative participation and elevated stress | Strong positive breadth blocks direction classification; systemic rule wins when matched. |
| 4 `regime.risk_on` | `RISK_ON` | `POSITIVE` direction, breadth `>= .15`, stress `< .65`, and liquidity missing or `>= .20` | positive participation, healthy liquidity | Severe liquidity strain, high stress, or conflicting breadth blocks it. |
| 5 `regime.rotation` | `ROTATION` | neutral/mixed direction, active sector rotation `>= .70` with sector coverage `>= .5` when supplied, dispersion `>= .70`, correlation missing or `< .70`, stress `< .65` | sector leadership changes | High co-movement, high stress, fragmented sectors, or firm direction blocks it. |
| 6 `regime.range_bound` | `RANGE_BOUND` | neutral direction, `abs(trend) < .30`, stress `< .30`, volatility `< .70` | stable sectors and low dispersion | Firm direction or elevated stress blocks it. |
| 7 `regime.mixed` | `MIXED` | unconditional fallback after quality gate | partial/nonconforming combinations | Never overrides an earlier matching rule. |

The headline `SYSTEMIC_STRESS` rule uses broad weak breadth plus co-movement **or** liquidity strain, whereas the separate stress dimension uses MCE-9's stressed-member fraction. This is intentional: they summarize related but different evidence. The configurable precedence can change which matched rule wins, and is part of the fingerprint.

## Conflicts and evidence strength

The typed conflict list retains stable IDs, exact source features, a description, and deterministic severity in `[0,1]`:

| Conflict ID suffix | Condition | Severity |
| --- | --- | ---: |
| `trend_up_breadth_down` | trend `>= .45`, breadth `<= −.45` | 1.00 |
| `trend_down_breadth_up` | trend `<= −.45`, breadth `>= .45` | 1.00 |
| `correlation_dispersion` | correlation `>= .70`, dispersion `>= .70` | .60 |
| `stress_healthy_liquidity` | stress `>= .65`, liquidity `>= .75` | .45 |
| `trend_sector_weakness` | trend `>= .45`, sector breadth `<= −.45` | .75 |
| `trend_rotation` | trend `>= .45`, sector rotation `>= .70` | .35 |

The first two conditions use the configurable `breadth_conflict` threshold; other numeric boundaries refer to their named settings. Conflicts are **not** averaged away. Severe conflicts trigger `TRANSITIONAL` unless an earlier acute rule wins, and all conflicts reduce evidence strength.

The eight families are direction/trend, breadth, stress, volatility, liquidity, correlation, dispersion, and sectors. Direction/trend, usable breadth, and stress are **critical**. The default headline gate requires all three, at least four families total, breadth coverage `>= .5` when supplied, evidence quality `>= .5`, and `features.regime.enabled`. Optional families may be absent. `quality_factor = source_coverage × Q(data_quality)` with `Q(GOOD)=1`, `Q(DEGRADED)=.75`, `Q(UNKNOWN)=.5`, `Q(POOR)=.35`, and `Q(UNUSABLE)=0`. `family_factor = .75 + .25 × active_families/8`. `agreement_factor = 1 − .5 × max_conflict_severity`.

`threshold_margin` is a bounded distance from the selected headline's entry boundary: direction for risk-on/off, stress for systemic, inverted liquidity for liquidity stress, rotation velocity for rotation, centrality for range-bound, conflict severity for transitional, and zero for mixed/unknown. Exact calculations are in `classifier.threshold_margin`. The result's `evidence_strength = quality_factor × family_factor × (.5 + .5 × threshold_margin) × agreement_factor`, times `.85` when headline hysteresis holds, clipped to `[0,1]`. An ineligible result has strength zero and headline `UNKNOWN`; strength below the configured `.2` minimum also withholds the headline. This number measures **internal support for a descriptive classification**, not a probability. The typed `RegimeScorecard` exposes every factor and raw input; no hidden evidence weights are renormalized.

## Replayable hysteresis and transitions

`compute` accepts prior `MarketRegimeState` records in `history`. It selects only the latest record whose observation is **strictly before** the current clock and whose availability is no later than that clock. Future records are ignored; duplicate accepted timestamps are rejected. A selected prior state with a different configuration fingerprint is rejected. Omitting history resets all hysteresis. There is no singleton state, process cache, or wall-clock dwell timer.

Default dimension band is `.08`: for example, volatility enters `HIGH` at `.70`, remains `HIGH` through `.62`, and exits below `.62`; `EXTREME` enters at `.90` and remains through `.82`. Liquidity orientation is reversed: `STRESSED` enters below `.20` and remains while score is `<= .28`; `HEALTHY` enters at `.75` and remains down to `.67`. Equivalent `.08` exit bands apply to correlation, dispersion, stress, and active sector rotation. The detailed state lists held dimensions. Headline risk-on/off enters at direction magnitude `.35` and can remain down to `.20` if breadth and stress still support it. Systemic stress and liquidity stress have `.08` exit bands; new systemic stress, severe liquidity stress, and severe conflict candidates can enter immediately under the default priority. Range-bound and rotation have bounded hold conditions. A hold is never applied when the current evidence fails the quality gate. No minimum dwell period is imposed in MCE-10; the documented bands control one-snapshot flicker.

Each classification records the **candidate** and final headline, selected rule, `hysteresis_held`, and held dimensions. On a change from an explicit prior state, `RegimeTransition` records from/to, current timestamp, triggering rule and features, before/after evidence strength, and conflicts. Every changed dimension has a `DimensionTransition`. With no prior state there is no transition. A known initial headline starts at the current clock; observation age remains `None` until prior history exists. Subsequent same-headline observations carry the start and increment observation age; a transition resets it to 1. This is an observation count, not a trading-session duration or a transition probability. Later backfills require replaying the affected explicit history from that point forward.

## Snapshot, quality, and scope

`apply_regime_to_snapshot` fills existing schema-1.0 market, trend, volatility, liquidity, correlation, dispersion, and breadth categorical fields. The compact public `MarketRegime` maps detailed systemic/liquidity stress to `STRESSED`, range-bound to `NEUTRAL`, rotation/transitional/mixed to `MIXED`, and preserves `RISK_ON`, `RISK_OFF`, and `UNKNOWN`. Detailed labels and `regime.*` scalars/strings remain in the typed result and `feature_values`. The snapshot's existing `confidence` field stays untouched; evidence strength has its own stable name. The stress and sector-rotation dimensions have no dedicated schema-1.0 enum fields, so their detailed labels remain in the typed state and feature values.

Poor/unusable quality or low source coverage makes dimensions and headline unknown; missing critical families or too few total families withholds the headline. Input quality flags propagate, and missing/low-coverage gates add flags. The classifier never converts missing scores into zero. Future upstream results fail the exact-clock feature adapter; future history cannot change a past classification. Existing upstream engines remain independent of regime classification. The dependency graph is causal observations → MCE-3–MCE-9 features → typed regime evidence → deterministic dimensions/conflicts/rules → optional explicit-prior hysteresis → snapshot projection.

This phase does not estimate transition probabilities, duration distributions, statistical instability, HMM states, change points, forecasts, or trading actions. Those remain outside MCE-10; MCE-11 may add descriptive transition analytics from historical observations without an HMM.
