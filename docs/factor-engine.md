# Dynamic factor and latent market structure (MCE-14)

The factor engine is descriptive research analytics. A principal component is a
statistical direction in an observed return panel, **not** an identified economic
cause, return forecast, or trading signal. MCE-10 deterministic regimes and MCE-13
HMM states remain separate. The snapshot schema and 18-feature statistical schema
remain 1.0.

## Point-in-time return panel

`FactorPanelBuilder` uses the configured universe or an explicit
`UniverseProvider`. The latter can use `MembershipTimeline` records with
`effective_from_ns` inclusive, `effective_to_ns` exclusive, and
`available_at_ns` visibility. Current expected symbols are queried at the
calculation clock. A historical close-to-close return enters a symbol's series
only when the symbol was a known member at **both** session close endpoints.
Future membership records cannot backfill past factor panels.
The panel fingerprint includes the point-in-time membership snapshots used at
its endpoints, along with aligned returns, each return's effective availability
time, and coverage diagnostics.

The panel contains the last `window_sessions` completed XNYS regular-session
log returns, `r_(d,s)=log(P_(d,s)/P_(d-1,s))`. Before today's regular close, the
newest possible endpoint is the prior completed trading session. At and after
regular close, today's close is eligible only when its bar is effectively
available. A close may be a full-session bar or the closing bar of the configured
cadence; the existing correlation reader enforces completed regular-session bar
bounds and source identity. All queries use `as_of_ns` availability filtering.
There is no calendar-day subtraction or use of extended-hours prices.

Symbols are sorted lexically. Missing closes, stale latest returns, and short
per-symbol history exclude the affected symbol. Remaining symbols use the
intersection of their return timestamps. Missing values are never zero-filled.
The result exposes expected/valid/missing/stale symbols, symbol coverage
`valid/expected`, aligned sample count, and time coverage
`aligned/window_sessions`. A fit is withheld below configured minimum symbols,
samples, symbol coverage, or time coverage. This complete-case choice may change
the universe and should be considered when comparing windows.

## PCA fit and factor count

Let `R` be the aligned `T × N` matrix of log returns, in log-return units. Every
fit uses only its supplied historical panel. Compute each column's sample mean
`μ_j`. With `matrix_type="covariance"`, set `s_j=1`; with the default
`matrix_type="correlation"`, set `s_j` to the sample standard deviation of that
asset's returns. A constant column prevents fitting and is flagged. The fitted
matrix is `Z_(t,j)=(R_(t,j)−μ_j)/s_j`. Correlation PCA gives equal volatility
scale to assets; covariance PCA preserves raw return variance. Neither applies
future scaling. There is no mixed-scale asset/sector input matrix.

For `N <= T`, use the symmetric sample matrix `C=ZᵀZ/(T−1)` and
`numpy.linalg.eigh`; otherwise use a thin SVD of `Z/sqrt(T−1)` to avoid forming
a large `N × N` matrix. Eigenvalues `λ_i` descend, are clipped only for tiny
numerical negative error, and must describe a positive-semidefinite matrix.
Scores are `F=ZV_K`; loadings are the orthonormal columns `V_K`. The in-window
score series is expressed under this **one fitted window**. It should not be
mistaken for a sequence of historical point-in-time rolling fits.

The default fixed factor count is `min(factor_count, effective rank)`. If
`variance_threshold` is configured, choose the smallest number reaching that
cumulative explained share, capped by `factor_count` and rank. A cap may prevent
reaching the threshold; inspect the reported spectrum. No count is declared
economically optimal. Adjacent eigenvalues with a relative gap at or below
`eigen_gap_tolerance` flag `FACTOR_IDENTIFICATION_WEAK`; individual vectors in a
near-tied subspace need cautious interpretation.

Eigenvector signs are arbitrary. For each factor, the first lexically ordered
symbol with the largest absolute loading is made positive. The PCA factor order
remains descending by eigenvalue. Equal eigenvalues need not have uniquely
identified individual directions even after this sign rule.

## Exact diagnostics

All spectrum metrics use the fitted `T`-row window at the result clock. Set
`p_i=λ_i/Σλ` across the full spectrum, including zero eigenvalues:

| Feature | Formula | Units/range |
| --- | --- | --- |
| `factor.first.explained_variance` | `p_1` | fraction `[0,1]` |
| `factor.top3.explained_variance` | `Σ_(i<=3) p_i` | fraction `[0,1]` |
| `factor.effective_dimension` | `(Σλ)^2/Σλ²` | effective factors `[1,N]` |
| `factor.concentration` | `Σp_i² = 1/D_eff` | HHI `[1/N,1]` |
| `factor.dominance` | `0.6 p_1 + 0.4 Σp_i²` | descriptive score `[0,1]` |

The `dominance` blend combines leading-factor share and whole-spectrum
concentration; it is separate from MCE-7's correlation score. A high value
indicates concentrated common variation under the chosen PCA matrix, not a
probability of a market event.

In raw log-return units, reconstruct the K-factor component
`R_hat=μ + (F V_Kᵀ) diag(s)` and residual `E=R−R_hat`. For each symbol, residual
variance share is `Var(E_j)/Var(R_j)` and common share is its complement. A
zero-variance denominator yields `None`. `factor.residual_variance` projects the
sum of sample residual variances, in squared log-return units;
`factor.residual_variance_share` projects the median residual share, and
`factor.common_variance_share` projects the mean common share. Across fitted
rows, `factor.residual_dispersion` is the average
cross-sectional population standard deviation of `E_t`, in log-return units.
The corresponding raw dispersion is retained in the result. On the fitted
window, summed sample variances satisfy
`total ≈ explained + residual` within numerical tolerance. A frozen
out-of-sample transform need not retain this orthogonality identity.

`factor.volatility` is the sample standard deviation of first-factor scores in
normalized score units over the fitted window. `factor.shock` is the latest
first-factor score minus the mean of **prior** scores in the same fitted window,
divided by their sample standard deviation. It is `None` when there are too few
prior observations or the baseline variance is zero. The score is descriptive;
all fitting and baseline data are available by the result clock.

## Rolling alignment and sector exposures

`factor_history` explicitly passes the previous context to each new rolling
fit; there is no hidden mutable model state. Factors are matched to the prior
window by maximum total absolute cosine similarity of loading vectors on shared
symbols. A deterministic bitmask assignment finds the exact best match for the
small configured K, breaking equal scores lexically. Each match also records
the sign needed to compare current with prior loadings. Per-symbol loading drift
is signed current loading minus prior loading on the shared symbol list.
`factor.loading_stability` is the mean absolute matched cosine, in `[0,1]`.

The two top-K subspaces are QR-orthonormalized on shared symbols. With singular
values `σ_i` of `Q_currentᵀQ_prior`, `factor.subspace_stability` is
`mean(σ_i²)`, in `[0,1]`. This can stay high when individual factors rotate
inside a near-tied subspace. Mapping IDs and signs are metadata for tracking;
the artifact itself keeps variance-ranked PCA order. No alignment is reported
without enough shared symbols.

An optional point-in-time `SectorUniverseProvider` supplies exclusive sector
membership at the calculation clock. Each sector's factor exposure is the
arithmetic mean of member loadings; its dispersion is the population standard
deviation of those loadings. Stable projection names are
`factor.sector.<canonical_sector>.loading.<one-based_factor_id>`. This describes
constituents in the fitted universe; it does not predict sector returns.

## Frozen evaluation, comparisons, and projection

`walk_forward_factors` uses actual XNYS sessions and MCE's temporal split
settings. Each fold fits its PCA artifact on training rows only, then projects
validation and test rows with frozen training means, scales, and loadings. It
reports out-of-sample `1−SSE/SST`, where SST uses training means; negative
values are possible. Later folds refit only at their configured boundary.
Appending future rows leaves earlier fold artifacts and scores unchanged.
It rejects a historical panel whose recorded return availability exceeds that
row's clock, such as a late bar correction imported from a future replay time.
For a full revision history, rebuild each as-of panel from the causal store.
Directly supplied synthetic panels have no bar-availability metadata; their
caller is responsible for ensuring each historical row was available on time.

`compare_factor_contexts` separately groups completed factor contexts by
supplied same-clock deterministic labels. `compare_with_latent_states` accepts
only causal, filtered MCE-13 inference at matching clocks; it rejects smoothed,
Viterbi or in-sample retrospective inference. These functions summarize
first share, dominance, residual dispersion, and factor volatility without
using labels during PCA fit. HMM implementation and inference remain unchanged.
Callers must supply point-in-time labels; these comparisons do not turn factors
or HMM posterior probabilities into forecasts.

`apply_factors_to_snapshot` projects continuous metrics into the existing
`feature_values` map only for causal, sufficiently covered results. It also
projects coverage counts on withheld results. It does not set a categorical
regime or overwrite MCE-10 outputs. The snapshot schema stays 1.0. All numeric
results and artifacts have deterministic fingerprints, finite-value checks,
schema-like model versioning, and validation on load. No broker, network,
training service, or new runtime dependency is involved.

This phase does not estimate residual correlation networks or assign economic
names to factors. Missing symbols and changes in universe composition can affect
comparisons even when the underlying market is similar. Market network and graph
analytics remain deferred to MCE-15.
