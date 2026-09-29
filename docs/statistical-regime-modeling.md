# Statistical regime modeling foundation (MCE-12)

MCE has two separate concepts. The MCE-10 deterministic regime is the rule-based
description in `MarketContextSnapshot`. A future statistical latent state is an
unobserved model label whose integer ID has no fixed market meaning. MCE-12 does not
fit or infer a statistical model and does not project one into the public snapshot.
The snapshot schema remains 1.0.

## Feature contract and causal rows

`STATISTICAL_FEATURE_SCHEMA` version 1.0 contains **18 ordered inputs**. Its SHA-256
fingerprint covers names, order, sources, expected ranges, missing policy, identity
transforms, and normalization defaults. The input order is:

1. `market_direction_score` (required, [-1, 1]);
2. `volatility_score`, `breadth_score`, `dispersion_score`, `correlation_score`,
   `liquidity_score`, `stress_score`;
3. `breadth.net_advance_fraction`, `breadth.participation`, `breadth.momentum`;
4. `sector.leadership.spread`, `sector.rotation.velocity`;
5. `correlation.first_eigen_share`, `dispersion.between_sector_share`;
6. `stress.systemic`, `stress.localized`;
7. `regime_transition.stability`, `regime_transition.intensity`.

This uses one broad score per family and selected participation, structure, and
transition details. It avoids dynamic symbol/sector keys and fixed historical
constituent lists. It deliberately omits regime labels from the input vector.
Bounded [0, 1] scores pass through by default; unbounded/signed detail fields use
robust scaling by default. The transformation rule is identity before scaling.
Feature values outside their declared range, nonfinite values, and nonnumeric values
are rejected.

`StatisticalDatasetBuilder` consumes immutable, previously computed snapshots.
`SnapshotRecord` can attach an availability time and source/replay fingerprint.
Rows are sorted by `as_of_ns`; duplicate clocks are rejected. A row is included only
if its snapshot was available by its own clock. An overall `as_of_ns` cuts off later
snapshots. The exchange calendar supplies trading date and session. The observation
contains the original feature vector, a `True` for observed and `False` for missing
mask, upstream quality and flags, an optional same-clock deterministic regime for
evaluation, and optional source fingerprint. There are no future labels. A matching
regime state is joined only when its availability is at or before the row clock.
The builder cannot repair causality violations in an upstream snapshot; its public
snapshot contract already requires all underlying bars, membership, weights and
other inputs to be effectively available by its `as_of_ns`. Rebuild upstream snapshots
with point-in-time replay when their raw data changes.

Missing required direction rejects a row. Missing optional values remain `None`
and masked until a fitted normalizer imputes the **training median**. No forward fill,
later interpolation, or implicit zero is used. The default quality policy skips
`UNUSABLE` rows; settings can also skip `POOR` or include all rows for explicit
research. An optional column with no observed training value prevents fitting.
Quality is retained on every included row, not converted into sample weights;
optional missingness adds `MISSING_DATA` to its quality flags.

## Training-only preparation and fingerprints

Call `fit_normalizer(dataset, split.train_indices)`. It computes each feature's
median from observed training entries, then computes center/scale from observed
training entries only. `standard` uses mean and population standard deviation;
`robust` uses median and 1.4826 times median absolute deviation; `none` retains
the original units. The default `schema` method uses each feature's declared
method. Zero variance or zero MAD uses scale 1 and a constant-feature flag.
`transform` imputes with the saved medians, scales, and returns the original mask.
It rejects a changed schema version, fingerprint, or feature order. The fitted
transform also rejects a changed result-relevant configuration fingerprint. The fitted
artifact stores methods, medians, centers, scales, constant flags, training time
range, row count, and training-only fingerprint. Persist the artifact with any
future model; do not refit it on validation or test rows.

The dataset fingerprint hashes schema and configuration fingerprints,
normalization setting, row clocks, vectors, masks, quality, regime annotations,
and optional replay/source identities. Including values ensures changed source
content changes the fingerprint even when a provider did not supply a content
fingerprint. Split fingerprints additionally hash every boundary, indices,
embargo, split settings and dataset fingerprint. Model metadata can record the
model type/version, training interval/count, seed, normalizer, split, dataset,
feature/config fingerprints and parameter hash. No wall-clock creation timestamp
enters these hashes. Future stochastic models must take an explicit seed.

## Session splits and walk-forward evaluation

`walk_forward_splits` counts actual XNYS trading sessions, including sessions with
no observation, rather than calendar days or row counts. Each fold has train,
validation and test windows, with configurable embargo sessions between train and
validation and between validation and test. `expanding` keeps the first training
session; `rolling` advances it. The step size is in trading sessions. Empty
segments and insufficient histories fail explicitly. Fit the normalizer and model
on **each fold's training indices**, then evaluate validation/test rows using that
frozen fit. There is no random shuffle. A long rolling input feature at the test
boundary may share earlier market observations with a training feature; embargo
separates row dates but does not claim statistical independence or magically purge
every feature's input window. Choose a gap suited to the longest upstream window.

## State labels and descriptive diagnostics

Latent IDs can permute across fits. Given training-state profiles,
`canonical_state_mapping` orders states by increasing stress centroid, then
volatility centroid, then decreasing direction centroid, with raw ID only as a
tie breaker. Missing centroids sort last. This is a stable *within-fit* convention,
not proof that states in different fits have the same economic meaning. For
cross-fit matching, compare profiles and document the matching rule. Apply any
mapping consistently to probabilities, transition matrices, and artifact metadata
in a future model.

`evaluate_states` reports count occupancy, session-weighted occupancy (each
observed trading session receives total weight 1), centroid, median and dispersion
of observed raw features, dwell lengths in observations, transition counts, self
transition persistence, empirical exit entropy in natural-log units, and a
count-weighted aggregate entropy across states with exits. A state with no exits
has `None` entropy. Deterministic regime contingency and purity use labels only
for evaluation; they are never training targets. Session-weighted occupancy is
descriptive and does not measure wall-clock duration within a session. None of
these diagnostics alone selects the best market model.

## HMM mathematical preparation and causal meaning

For an eventual K-state Gaussian HMM, hidden state `z_t` follows initial
probabilities `π_i=P(z_1=i)` and row-stochastic transitions
`A_ij=P(z_t=j|z_(t-1)=i)`. Emissions are
`b_i(x_t)=N(x_t; μ_i, Σ_i)`, initially with diagonal positive covariance and a
variance floor. Full covariance would require positive-definiteness checks and
regularization. Define `ℓ_t(i)=log b_i(x_t)` and implement all recursions with
log-sum-exp (or proven scaling) to prevent underflow:

```
log α_1(i) = log π_i + ℓ_1(i)
log α_t(j) = ℓ_t(j) + logsumexp_i(log α_(t-1)(i) + log A_ij)
log L = logsumexp_i(log α_T(i))
log β_T(i) = 0
log β_t(i) = logsumexp_j(log A_ij + ℓ_(t+1)(j) + log β_(t+1)(j))
γ_t(i) = exp(log α_t(i) + log β_t(i) - log L)
ξ_t(i,j) = exp(log α_t(i) + log A_ij + ℓ_(t+1)(j)
                + log β_(t+1)(j) - log L)
```

Viterbi replaces `logsumexp` in the forward path with `max` and stores
backpointers. Baum-Welch/EM computes `γ` and `ξ`, then updates `π`, each row of
`A`, emission means and covariance estimates from posterior weights until a
configured tolerance/max-iteration rule is met. A correct implementation must
track likelihood, guard empty states and singular covariances, use a seed for
stochastic initialization, and test reference likelihood/posterior examples.

Filtering `P(z_t|x_1:t, θ)` uses observations only through `t`, but is **causal
historical inference only if θ and the normalizer were fitted before t**.
Smoothing `P(z_t|x_1:T, θ)` with `T>t` and full-sequence Viterbi use later
observations, so they are retrospective research outputs. In-sample filtered
results are also retrospective if fitted on the evaluated period. The inference
contract records mode, training end, and retrospective status and rejects a false
causal claim. MCE-12 does not run forward/backward, EM or Viterbi, and no
statistical output enters historical replay or the deterministic snapshot.

MCE-13 may implement a thoroughly tested diagonal Gaussian HMM, model-size
comparison, change points and clustering. There is no forecasting, trading,
broker integration or production retraining in this foundation.
