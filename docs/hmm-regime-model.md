# Diagonal Gaussian hidden-state model (MCE-13)

MCE-13 adds an **unsupervised, descriptive** Gaussian hidden Markov model (HMM).
It does not replace the MCE-10 deterministic regime and does not produce forecasts,
signals, or calibrated probabilities of future market events. A posterior such as
`P(z_t=i | x_1:t, θ)` describes uncertainty about the model's current latent state
under fitted parameters `θ`. The fitted transition matrix is a model assumption
estimated from latent states; it is distinct from MCE-11 observed deterministic
regime exit frequencies.

## Data and assumptions

The model consumes the unchanged MCE-12 version-1.0, 18-feature ordered vector.
Median imputation and per-feature scaling are fitted on training rows only. Missing
masks and upstream quality stay with the dataset. The model uses a K-state Markov
chain (`2 <= K <= 6`) with initial distribution `π`, row-stochastic transition
matrix `A`, state means `μ_i`, and **diagonal** covariance `diag(v_i)`. It does not
model cross-feature covariance within a state. Every variance is floored at the
configured positive `covariance_floor` in **normalized feature units**.

For D dimensions, the log emission density is

```
log b_i(x) = -1/2 [ D log(2π) + Σ_d log v_(i,d)
                     + Σ_d (x_d - μ_(i,d))² / v_(i,d) ]
```

The forward recursion is `log α_1(i)=log π_i+log b_i(x_1)` and
`log α_t(j)=log b_j(x_t)+logsumexp_i(log α_(t-1)(i)+log A_ij)`.
The sequence log likelihood is `logsumexp_i(log α_T(i))`.
The backward recursion is `log β_T(i)=0` and
`log β_t(i)=logsumexp_j(log A_ij+log b_j(x_(t+1))+log β_(t+1)(j))`.
Smoothed `γ_t(i)` and pair posterior `ξ_t(i,j)` normalize products of forward,
transition, emission and backward terms by the sequence likelihood. Viterbi uses
the same log terms with `max` and backpointers. `logsumexp` handles long sequences
without multiplying small probabilities directly; tests compare a hand-derived
two-state example with the implementation.

## Initialization, EM and local optima

Each restart uses a derived seed (`base_seed + restart_index`) to choose an initial
k-means center. Subsequent centers use deterministic farthest-point selection;
Lloyd updates have deterministic tie handling and an empty-cluster fallback. The
initial HMM means and variances come from those clusters. State transition counts
and the first-state probability receive a fixed `0.001` additive pseudocount,
avoiding exact zeros. Baum-Welch then alternates forward/backward posterior
calculation and updates of `π`, `A`, `μ`, and diagonal variances. The covariance
floor prevents singular emissions. A materially decreasing observed likelihood
fails that restart; every restart records its likelihood history, iterations,
convergence and any error. The selected fit has the highest training likelihood
among converged restarts, or the highest finite non-converged fit if none converge.
Non-convergence is visible in quality output. EM finds local optima; restarts do
not prove a global optimum.

Raw state numbers have no meaning. After fitting, MCE-12 canonicalization orders
state centroids by increasing raw-unit stress score, then volatility, then
decreasing market direction, with raw ID breaking exact ties. Initial,
transition, mean, variance, and training-end posterior arrays are all reordered
together. `state_mapping` records raw-to-canonical IDs. A canonical number is a
within-fit convention; comparisons across fits still require profile review.

## Inference and time semantics

`GaussianHMMRegimeModel.infer(mode="filtered")` uses forward observations only.
It returns each row's current-state posterior, maximum-posterior state, posterior
entropy `-Σ_i p_i log p_i` in natural-log units, model fingerprint and upstream
quality flags. It marks a filtered row causal **only if its timestamp is after the
model's training end**. A supplied `previous_filtered` belief is first advanced
through one transition before observing the new row. In the walk-forward runner,
the training-end belief carries into validation only when observations are
contiguous; an embargo or missing-row gap resets to fitted `π`. The same rule
applies between validation and test. This conservative reset avoids pretending
that one transition spans an unknown gap.

`mode="smoothed"` uses backward messages from later rows in the supplied sequence.
`mode="viterbi"` chooses a full-sequence path. Both are labeled retrospective and
are unsuitable as historical replay features. Fitting on a full historical sample
also makes in-sample filtered results retrospective. No HMM result is projected
into `MarketContextSnapshot`; its schema remains 1.0 and the deterministic regime
remains authoritative.

## Model size, artifacts and quality

For K states and D diagonal-Gaussian features, the free-parameter count is
`(K−1) + K(K−1) + 2KD`. `AIC=2k−2logL` and `BIC=k log(n)−2logL`, where `n` is
training rows. The typed size report includes K, likelihood, AIC/BIC,
convergence, iterations, occupancy balance, mean dwell, exit entropy, and
between-centroid separation relative to within-state variance. It does not
declare an automatic winner. Likelihood and information criteria describe fit
under assumptions, not trading performance.

The immutable artifact stores all numerical parameters, covariance floor,
training-end filtered belief, canonical mapping, restart diagnostics, model and
feature versions, configuration, dataset/split and normalizer fingerprints, seed,
and a parameter hash. Canonical JSON serialization is deterministic. Loading
validates dimensions, probability sums, finite values, variance floors,
parameter hash, feature schema and normalizer identity. Quality output reports
convergence, likelihood, AIC/BIC, minimum posterior occupancy, mean entropy,
excessive switching and states with low occupancy, collapsed variance or nearly
duplicate centroids. These are research diagnostics, not validity guarantees.

`run_walk_forward_hmm` fits each fold's normalizer and model only on its training
rows, then freezes them while filtering validation/test rows. The frozen artifact
identity excludes future validation/test content; appending future rows cannot
rewrite earlier fold outputs. Later folds may retrain on their own prescribed
training windows. Upstream rolling features can overlap across a boundary, so
embargo length must reflect the feature construction window; row separation alone
does not establish statistical independence.
