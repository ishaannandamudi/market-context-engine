# Market graph and network structure (MCE-15)

The network engine describes the **organization** of observed dependence. An edge
is statistical co-movement, not an economic causal link, prediction, or trading
signal. MCE-7 remains the owner of pair correlation and eigenstructure; MCE-14
remains the owner of latent return factors. The network engine consumes their
causal outputs and does not change their fits or MCE-10 categorical regimes.

## Input and point-in-time contract

`NetworkFeatureEngine.compute` accepts one same-clock `CorrelationFeatureSet` and
selects its raw or shrunk `CorrelationMatrix` according to
`features.network.source_matrix`. `DependenceInput` is the explicit interface for
other validated matrices. It requires ordered unique symbols, a symmetric
finite matrix with unit diagonal, correlations in `[-1,1]`, a positive
semidefinite spectrum within numerical tolerance, expected symbols, quality,
and a calculation clock. Matrix construction and membership availability remain
with the upstream MCE-7 engine: it uses only bars and members available by that
clock. A future matrix or future membership record is never consulted by a past
network calculation. Directly supplied research matrices must meet the same
point-in-time contract. Input and graph fingerprints support deterministic
replay.

Only valid matrix symbols become nodes. Expected-but-missing or stale symbols
are reported in `coverage = valid_nodes / expected_nodes`; they are not invented
as zero-degree nodes. Results are withheld below configured minimum node count
or coverage, or for poor/unusable underlying dependence quality. Partial
coverage and disconnected topology carry explicit flags. A point-in-time,
exclusive `SectorUniverseProvider` may attach sector and weight metadata; its
future reclassifications cannot change earlier nodes or sector comparisons.

## Graph construction and distance

For each pair, numerically clip `rho` to `[-1,1]` and set correlation distance
`d = sqrt(2(1-rho))`, in `[0,2]`. A graph edge records signed correlation and
distance. Its nonnegative similarity weight is `w = max(0,rho)`. Negative
correlations can be selected by a negative threshold or top-k rule and remain
topological edges, but contribute zero to strength, eigenvector centrality,
and weighted modularity. Distance is never confused with similarity weight.

Choose one graph mode; default is the sparse threshold graph:

| Mode | Exact inclusion and tie rule |
| --- | --- |
| `threshold` | Include `(i,j)` iff `rho_ij >= correlation_threshold`, including equality. Default threshold `0.6`. |
| `top_k` | Each node picks its `k` largest correlations, ties by lexically ordered neighbor. Include an undirected edge if **either** endpoint picks the other (union policy). |
| `mst` | Kruskal's minimum spanning tree on the complete valid correlation-distance matrix. Sort candidates by `(distance, first_symbol, second_symbol)`; output exactly `N-1` edges. |

The optional MST overlay is also computed for threshold/top-k graphs. It uses
the full valid dependence matrix, so it can connect nodes even when the sparse
main graph is fragmented. No graph library is required.

## Topology and centrality

All metrics use the selected main graph at the result clock with `N` valid
nodes and `E` undirected edges. Components are sorted by decreasing size, then
lexical member list; isolated nodes are singleton components.

| Metric | Exact definition | Range/units |
| --- | --- | --- |
| `network.density` | `2E/[N(N-1)]` | `[0,1]`, zero for `N<2` |
| `network.average_degree` | `2E/N` | edges per node `[0,N-1]` |
| `network.largest_component_fraction` | largest component size `/ N` | `[0,1]` |
| `network.isolated_nodes` | number of degree-zero nodes | count |
| Node degree centrality | `degree_i/(N-1)` | `[0,1]` |
| Node strength | `sum_j max(0,rho_ij)` on selected edges | positive similarity units `[0,N-1]` |
| `network.centralization` | `sum_i(max_degree-degree_i)/[(N-1)(N-2)]` | Freeman degree centralization `[0,1]`, zero for `N<3` |
| `network.hub_hhi` | `sum_i(strength_i / sum_strength)^2` | `[1/N,1]` when positive strength exists |
| `network.effective_hubs` | `1/hub_hhi` | effective node count `[1,N]` |
| `network.connectivity_score` | `(largest_component_fraction + density)/2` | descriptive topology score `[0,1]` |

If total positive strength is zero, hub HHI and effective hubs are unavailable.
The connectivity score is withheld for MST-only graph mode: every complete
tree has one component and exactly `N-1` edges, making that blend uninformative.
Optional nonnegative eigenvector centrality uses the weighted adjacency matrix.
Power iteration on `A+I` avoids bipartite sign cycling, starts from a uniform
positive vector, and L2-normalizes the result. Nodes have nonnegative values;
isolated or entirely zero-weight graphs get zero. Nonconvergence withholds the
eigenvector values and flags topology degradation. No betweenness or all-pairs
shortest paths run by default.

## Communities and sectors

The baseline community algorithm forms connected components from **all matrix
pairs** with `rho >= community_threshold` (default `0.7`), independently of
the selected main graph's edges. This is deterministic threshold-connected
grouping, not a claim of optimized community detection. Communities are ordered
by decreasing size and then lexical members; IDs start at zero. Community
count and largest-community fraction are therefore stable under raw label
permutation. A single homogeneous community is valid.

Weighted modularity evaluates that partition on the **main graph's positive
weights**. With `m = sum_edges w`, node strength `s_i`, and internal edge weight
`m_g` in group `g`, `Q = sum_g [m_g/m - (sum_(i in g)s_i/(2m))^2]`.
It is unavailable when `m=0`; otherwise it lies in `[-1,1]` and may be
negative for a poor partition. It is
not an optimization score or probability.

For available sector assignments, community purity is the sum over communities
of the largest sector count divided by all classified nodes. For each sector,
`dominant_community_share = max_community_count / sector_member_count` and
`fragmentation = 1 - dominant_community_share`; the number of communities
containing that sector is also reported. `cross_sector_edge_fraction` counts
selected edges whose two classified endpoints have different sectors divided
by all selected edges with two classified endpoints; within-sector fraction is
its complement. Purity, dominant share, fragmentation, and edge fractions lie
in `[0,1]`. If no such edges or labels exist, the corresponding fractions are
absent.
These are point-in-time descriptions, not sector forecasts.

## MST and rolling topology

For the full-matrix MST, `total_length = sum_tree_edges d`,
`mean_distance = total_length/(N-1)`, and weighted `diameter` is the longest
shortest-path distance through the tree. The reported central node has maximum
tree degree, breaking ties lexically. Lower mean distance indicates tighter
pair dependence under this distance definition. It does not measure direction
or causation. Total length and diameter lie in `[0,2(N-1)]`; mean distance
lies in `[0,2]`.

`network_history` or `compute(previous=...)` passes an explicit prior context;
there is no mutable hidden history. Comparison requires increasing clocks and
the same configuration, matrix source, and graph mode. On symbols shared by
both graphs:

| Metric | Exact definition | Range |
| --- | --- | --- |
| `network.edge_persistence` | `|E_t intersect E_prev| / |E_t union E_prev|`; empty union gives 1 | `[0,1]` |
| `network.edge_turnover` | `1 - edge_persistence` | `[0,1]` |
| `network.centrality_drift` | mean shared-node absolute change in normalized degree centrality | `[0,1]` |
| `network.community_stability` | NMI of shared-node community labels, invariant to ID permutation; both constant partitions give 1, one constant gives 0 | `[0,1]` |
| MST turnover | `1 - Jaccard` of shared-node MST edge sets | `[0,1]` |
| `network.rewiring` | `(edge_turnover + centrality_drift + 1-community_stability)/3` | descriptive `[0,1]` |

NMI is mutual information divided by the geometric mean of label entropies,
clipped to `[0,1]`. Component-count change and shared-node fraction are also
reported, along with each shared node's signed current-minus-prior degree
centrality. Changes in universe composition are visible through coverage and
shared-node fraction; edge comparison does not fabricate missing nodes. Future
graphs do not rewrite earlier graph artifacts or rewiring results.

## Optional factor-residual comparison and snapshot

`residual_dependence(panel, factor_context)` requires the MCE-14 artifact and
panel fingerprints to match, the same calculation clock, and a causal fitted
factor context. It reconstructs fitted returns, subtracts them, and computes
the residual Pearson matrix using the existing covariance-to-correlation
validator. An unavailable or constant residual matrix is explicit. Supplying
this matrix to the network engine yields a separate residual graph with the
same construction settings. For threshold or top-k mode it reports residual
density and `raw_minus_residual_density`. MST mode does not publish a density
difference because every full tree has the same edge count. Residual graphs are
descriptive; no future factor artifact can enter a past comparison.

`apply_network_to_snapshot` adds continuous `network.*` values and coverage
counts to schema 1.0. It clears earlier `network.*` values on reprojection so
withheld results cannot retain stale headlines. It never assigns a categorical
regime or changes MCE-10 classifications. The statistical feature schema also
remains 1.0. Configured thresholds, mode, source, quality gates, and optional
calculations enter the normal config fingerprint.

Topology depends on threshold and universe composition; compare scores only
under compatible settings and coverage. Threshold communities can chain
through weak bridges, and correlation is an estimate with sampling error.
Network connectivity is not systemic-risk probability. MCE-16 supplies separate
session-relative baselines; it does not rewrite this network topology.
