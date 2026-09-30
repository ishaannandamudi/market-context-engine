# Statistical regime comparisons (MCE-13)

MCE-13 provides three descriptive research views over the same unchanged
statistical feature schema. MCE-10's deterministic labels remain separate and
may be compared to latent assignments, never used as an unsupervised training
target. No metric below predicts returns or selects a trading strategy.

## K-means baseline

`KMeansRegimeModel` fits normalized training vectors using seeded first-center
selection, deterministic farthest-point remaining centers, bounded Lloyd updates,
and multiple restarts. It selects the lowest training inertia, records every
restart, handles empty clusters without nonfinite centroids, and canonicalizes
cluster IDs by raw-unit stress, volatility, direction and raw-ID tie break.
Assignments are nearest-centroid classifications; k-means ignores sequence order
and temporal persistence. Compare it with the HMM to see what a transition model
adds. `compare_kmeans_cluster_counts` reports training inertia, convergence,
iterations, occupancy balance and empty clusters for each requested K. Inertia
tends to fall with K and is not a stand-alone choice of K.

## Offline mean shifts

`MeanShiftDetector.offline` uses fitted normalized vectors. At boundary `t` it
computes the Euclidean distance between the mean of `X[t−L:t]` and the mean of
`X[t:t+R]`, divided by `sqrt(D)` to keep scale comparable as the feature count
changes. A configurable threshold selects candidates. Candidates are ranked by
descending score with earlier index breaking ties; greedy minimum separation
suppresses neighboring detections, then output is returned in time order.
`right_window_end_ns` and `causal=False` make future dependence explicit.
These are offline historical structural differences, not forecasts.

An optional `causal_scores` path compares two trailing windows ending at T, using
only rows through T and requiring a normalizer fitted before the first scored
row. It emits a raw causal change score, not a confirmed break. Threshold and
spacing count **observations**, not sessions, so data cadence matters.

## Common evaluation

`compare_assignments` exposes MCE-12 occupancy, session-weighted occupancy,
centroids, dwell lengths, transition counts, persistence, exit entropy, and
contingency with available deterministic MCE-10 labels. It additionally reports
adjusted Rand index (ARI) and normalized mutual information (NMI) on rows with
deterministic labels. NMI divides mutual information by the geometric mean of
label entropies; identical single-cluster partitions return 1. Both metrics are
invariant to label permutation. Missing deterministic labels are omitted from
these two comparisons and never filled from the future.

The HMM size report adds AIC/BIC, occupancy balance, mean dwell, state separation
and degeneracy flags. Session occupancy gives equal total weight to each observed
trading day; dwell is measured in observations. Adjacent assignments separated
by unobserved market time are still adjacent in this descriptive sequence, so
interpret switching statistics with the data cadence and gaps in mind. Posterior
probabilities are model-state probabilities, never crash or future-regime odds.
