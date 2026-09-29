# Advanced cross-sectional dispersion (MCE-8)

`DispersionFeatureEngine.compute(store, as_of_ns, *, sector_context=None,
volatility_context=None, correlation_context=None)` measures differences between
constituent **returns** at an explicit clock. It uses the same completed,
effectively available regular XNYS bars and horizon references as MCE-3 and
MCE-6. The primary panel is the configured medium horizon. A constituent with
no current regular bar, a stale bar, or no horizon reference is excluded from
that horizon; no return is forward filled or set to zero. The engine reads
`MembershipTimeline` and optional `SectorMembershipTimeline` at each requested
clock, including each historical baseline clock. `effective_from_ns` is
inclusive, `effective_to_ns` exclusive, and `available_at_ns` prevents late
knowledge from revising earlier membership. The same rules apply to weights
and sector reclassifications. No constituent list is hard coded.

## Returns and cross-sectional statistics

For valid horizon returns `r_i`, the headline is the **population** standard
deviation `sqrt(sum((r_i - mean(r))²) / N)`, with `N >= minimum_members`.
Units are decimal returns: `0.018` means 1.8 percentage points of cross-sectional
spread. It is equal-weighted and remains separate from optional weighted
dispersion. Weighted dispersion uses nonnegative point-in-time weights on valid
names, normalized to sum to one:
`sqrt(sum(p_i * (r_i - sum(p_j*r_j))²))`. It requires weights for **every**
expected member, positive total weight, and configured valid-weight coverage;
there is no unbiased sample correction.

The engine also exposes raw `MAD = median(|r_i - median(r)|)`, robust sigma
`1.4826 * MAD`, `IQR = Q75 - Q25`, configurable symmetric tail spread
`Q_high - Q_low` (default Q90–Q10), and diagnostic `max(r)-min(r)`. Quantiles
use NumPy `method="linear"` interpolation explicitly. No clipping or
winsorization occurs. Positive and negative subsets each need at least two
returns for their separate population standard deviations; absent subsets
produce `None`. Downside-minus-upside is emitted only when both exist.
Population skew is `mean(((r-mean)/std)³)` for at least three names; population
**excess** kurtosis is `mean(((r-mean)/std)⁴)-3` for at least four. Zero or
near-zero standard deviation withholds both moments. All public numbers are
finite Python scalars.

Short, medium, long, and additional named horizons each retain valid count,
coverage, standard deviation, and MAD. Defaults are 5m, 30m, 1d, session,
5d, and 20d. Short/long ratio requires a nontrivial long denominator;
acceleration is `(short-medium)/medium` when medium is nontrivial. Snapshot
change is current medium dispersion minus the prior configured snapshot-cadence
point within the regular session, using that point's own membership. Session
and five-session changes subtract dispersion at the last and fifth prior XNYS
session close. Missing references leave these diagnostics `None`.

## Prior-only history and score

Historical medium-horizon cross sections are computed at previous XNYS closes,
each with its own causal membership and completed bars. The current observation
is **excluded** from its baseline. A baseline point needs minimum member count
and coverage. Once `minimum_baseline_samples` are present, percentile is the
midrank `(number below current + 0.5 * number equal) / sample count`.
Robust z is `(current - prior median)/(1.4826 * prior MAD)`; when MAD is zero,
it is zero for an exact tie and otherwise unavailable. Shock is
`max(0, 2*percentile-1) * max(0,current-median)/(|current|+|median|)`, with a
zero denominator contributing zero. It lies in `[0,1]` and needs both high
rank and positive jump. Persistence is the fraction of current plus the
configured most recent prior slots strictly above the prior baseline median;
any missing recent slot withholds it. `dispersion_score` is exactly the
prior-only percentile when the primary cross section and coverage are valid.
Its sole score contributor is the percentile term, so contributions sum to
the score. It describes unusual dispersion, not direction or probability.

## Sector, volatility, concentration, and correlation context

With a same-clock `SectorFeatureSet`, sector dispersion is the population std
of its valid absolute medium-horizon sector returns; relative std, MAD, and
top-minus-bottom spread use its valid relative returns. Without that context,
the point-in-time sector decomposition supplies sector means. When a causal
broad-market reference return exists, subtracting it supplies relative sector
diagnostics; a common subtraction leaves cross-sector std unchanged. A sector
context with missing horizons does not silently use a different horizon.

For exclusive point-in-time sectors with sufficient assignment coverage and
at least two sectors, decomposition uses equal-weight constituent returns.
Let `n_s` be valid names in sector `s`, `N=sum(n_s)`, `mu_s` its mean, and
`mu` the grand mean. `between = sum((n_s/N)*(mu_s-mu)²)` and
`within = sum((n_s/N)*mean_i_in_s((r_i-mu_s)²))`. Thus the population total
variance equals `between+within` within numerical tolerance. Shares are
reported only when total variance is nontrivial; per-sector means, counts,
and within-sector standard deviations retain detail. Single-name sectors have
a mean but withhold their own std. Sector return std is distinct from the
count-weighted between component.

An optional same-clock MCE-4 result supplies medium realized volatilities.
Valid constituent volatilities yield cross-sectional std and MAD after the
configured volatility coverage gate. Relative volatility dispersion is that
std divided by positive broad-market realized volatility. Volatility values
are annualized decimals, so the raw volatility std has the same units. The
engine does not calculate an extra volatility history.

Movement concentration uses shares proportional to `|r_i|`; weighted
concentration separately uses `|w_i*r_i|`. HHI is the sum of squared shares,
and effective movers is `1/HHI`. All-zero absolute movement makes both
unavailable, never an invented neutral count. Weighted HHI has the same
weight completeness and coverage gate as weighted dispersion.

The optional same-clock MCE-7 context retains raw average correlation,
systemic co-movement score, correlation percentile, return dispersion, and
dispersion percentile. Continuous differentiation is
`dispersion_percentile * (1 - systemic_co_movement_score)` when both scores
exist; the percentile difference is also exposed. High correlation and high
dispersion are both possible. No categorical quadrant, dispersion regime, or
market regime is assigned.

## Quality, configuration, and output

`member_coverage` is valid medium-horizon names divided by expected
point-in-time names. `weight_coverage` is valid-name weight divided by total
expected weight, present only when every expected weight is valid. Missing and
stale names are listed separately and excluded. Partial universe, low
coverage, stale data, missing sector/volatility inputs, unavailable baseline,
and degraded membership or price-adjustment status set existing quality flags.
Raw diagnostics may remain visible when coverage withholds the headline
score. `evidence_quality` multiplies member coverage, source quality factor,
price-adjustment factor, and applicable context-completeness factors; it is
an input-quality diagnostic, not statistical confidence. Set
`price_adjustment_status="adjusted"` only when supplied prices are consistently
adjusted; unknown/raw status degrades quality. The engine cannot infer or
repair corporate actions from bars.

`features.dispersion` has typed universe, cadence, horizons, count/coverage,
baseline, persistence, quantile, and adjustment settings. The existing
configuration fingerprint includes the entire result-relevant `features`
section. Output is immutable `DispersionFeatureSet`; snapshot projection
adds finite `dispersion.*` scalars, `dispersion_score`, contributors, quality,
and flags to schema 1.0 while preserving existing evidence. Stable names
include `dispersion.return.std`, `.mad`, `.iqr`, `.tail_spread`, `.skew`,
`.excess_kurtosis`, `dispersion.short`, `.medium`, `.long`, `.percentile`,
`.robust_z`, `.shock`, `.persistence`, `.sector`, `.volatility`,
`.within_sector`, `.between_sector`, `.movement_hhi`, and
`dispersion.correlation.differentiation`; the snapshot projector is the full
name catalog. The dependency direction is canonical store/window reader and
membership providers into dispersion; optional typed sector, volatility, and
correlation results enter as inputs. Those engines do not import dispersion.

No network, vendor SDK, credential, or mutable global baseline is involved.
The deterministic benchmark is `benchmarks/dispersion.py`; data creation is
excluded from timed engine computation.
