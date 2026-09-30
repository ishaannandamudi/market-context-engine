"""Immutable dependence, graph, and descriptive network result contracts."""

from dataclasses import asdict, dataclass
from hashlib import sha256
from math import isfinite, sqrt
from typing import Literal

import numpy as np

from mce.features.correlation.models import CorrelationMatrix
from mce.models.enums import DataQuality, QualityFlag
from mce.serialization.canonical import canonical_json_bytes

MatrixSource = Literal["raw", "shrunk", "factor_residual"]
GraphMode = Literal["threshold", "top_k", "mst"]


@dataclass(frozen=True, slots=True)
class DependenceInput:
    as_of_ns: int
    expected_symbols: tuple[str, ...]
    matrix: CorrelationMatrix | None
    source: MatrixSource
    data_quality: DataQuality = DataQuality.GOOD
    quality_flags: frozenset[QualityFlag] = frozenset()
    aligned_samples: int = 0

    def __post_init__(self) -> None:
        if self.as_of_ns < 0 or self.aligned_samples < 0:
            raise ValueError("invalid dependence clock or sample count")
        if self.source not in ("raw", "shrunk", "factor_residual"):
            raise ValueError("unsupported dependence source")
        if self.expected_symbols != tuple(sorted(set(self.expected_symbols))):
            raise ValueError("expected network symbols must be sorted and unique")
        if self.matrix is not None:
            if not set(self.matrix.symbols) <= set(self.expected_symbols):
                raise ValueError("dependence matrix contains nonmember symbols")
            values = np.asarray(self.matrix.values, dtype=np.float64)
            if np.linalg.eigvalsh(values)[0] < -1e-8:
                raise ValueError("dependence matrix must be positive semidefinite")

    @property
    def valid_symbols(self) -> tuple[str, ...]:
        return self.matrix.symbols if self.matrix is not None else ()

    @property
    def coverage(self) -> float:
        return (
            len(self.valid_symbols) / len(self.expected_symbols) if self.expected_symbols else 0.0
        )

    @property
    def fingerprint(self) -> str:
        return sha256(
            canonical_json_bytes(
                {
                    "as_of_ns": self.as_of_ns,
                    "expected_symbols": self.expected_symbols,
                    "matrix": self.matrix.model_dump(mode="python") if self.matrix else None,
                    "source": self.source,
                    "data_quality": self.data_quality,
                    "quality_flags": self.quality_flags,
                    "aligned_samples": self.aligned_samples,
                }
            )
        ).hexdigest()


@dataclass(frozen=True, slots=True)
class NetworkNode:
    symbol: str
    sector: str | None = None
    membership_weight: float | None = None

    def __post_init__(self) -> None:
        if (
            not self.symbol
            or self.membership_weight is not None
            and (not isfinite(self.membership_weight) or self.membership_weight < 0)
        ):
            raise ValueError("invalid network node or membership weight")


@dataclass(frozen=True, slots=True)
class NetworkEdge:
    first: str
    second: str
    correlation: float
    distance: float
    weight: float

    def __post_init__(self) -> None:
        if self.first >= self.second or not all(
            isfinite(value) for value in (self.correlation, self.distance, self.weight)
        ):
            raise ValueError("network edge must be finite and canonically ordered")
        if not -1 <= self.correlation <= 1 or not 0 <= self.distance <= 2:
            raise ValueError("invalid correlation distance")
        if abs(self.weight - max(0.0, self.correlation)) > 1e-9:
            raise ValueError("network edge weight must be positive correlation")
        if abs(self.distance - sqrt(2 * (1 - self.correlation))) > 1e-9:
            raise ValueError("network edge distance disagrees with correlation")


@dataclass(frozen=True, slots=True)
class MarketNetwork:
    as_of_ns: int
    nodes: tuple[NetworkNode, ...]
    edges: tuple[NetworkEdge, ...]
    matrix_source: MatrixSource
    graph_mode: GraphMode
    input_fingerprint: str

    def __post_init__(self) -> None:
        symbols = tuple(node.symbol for node in self.nodes)
        pairs = tuple((edge.first, edge.second) for edge in self.edges)
        if self.as_of_ns < 0 or symbols != tuple(sorted(set(symbols))):
            raise ValueError("network nodes must be sorted and unique")
        if self.graph_mode not in ("threshold", "top_k", "mst"):
            raise ValueError("unsupported network graph mode")
        if pairs != tuple(sorted(set(pairs))) or any(
            first not in symbols or second not in symbols for first, second in pairs
        ):
            raise ValueError("network edges must be unique, sorted member pairs")
        if len(self.input_fingerprint) != 64:
            raise ValueError("invalid network input fingerprint")

    @property
    def fingerprint(self) -> str:
        return sha256(canonical_json_bytes(asdict(self))).hexdigest()


@dataclass(frozen=True, slots=True)
class NetworkTopology:
    components: tuple[tuple[str, ...], ...]
    density: float
    average_degree: float
    median_degree: float
    max_degree: int
    largest_component_fraction: float
    isolated_nodes: int


@dataclass(frozen=True, slots=True)
class NodeCentrality:
    symbol: str
    degree: int
    degree_centrality: float
    strength: float
    eigenvector: float | None


@dataclass(frozen=True, slots=True)
class NetworkCentrality:
    nodes: tuple[NodeCentrality, ...]
    degree_centralization: float
    hub_hhi: float | None
    effective_hubs: float | None
    eigenvector_converged: bool | None


@dataclass(frozen=True, slots=True)
class NetworkCommunities:
    assignments: tuple[tuple[str, int], ...]
    groups: tuple[tuple[str, ...], ...]
    modularity: float | None
    largest_fraction: float


@dataclass(frozen=True, slots=True)
class SectorFragmentation:
    sector: str
    symbols: tuple[str, ...]
    community_count: int
    dominant_community_share: float
    fragmentation: float


@dataclass(frozen=True, slots=True)
class SectorAlignment:
    sectors: tuple[SectorFragmentation, ...]
    community_purity: float | None
    cross_sector_edge_fraction: float | None
    within_sector_edge_fraction: float | None


@dataclass(frozen=True, slots=True)
class MSTTopology:
    edges: tuple[NetworkEdge, ...]
    total_length: float
    mean_distance: float
    diameter: float
    central_node: str


@dataclass(frozen=True, slots=True)
class NetworkDynamics:
    previous_as_of_ns: int
    shared_node_fraction: float
    edge_persistence: float
    edge_turnover: float
    centrality_drift: float
    max_centrality_drift: float
    node_centrality_changes: tuple[tuple[str, float], ...]
    community_stability: float
    component_count_change: int
    mst_edge_turnover: float | None
    rewiring: float


@dataclass(frozen=True, slots=True)
class NetworkContext:
    as_of_ns: int
    input_fingerprint: str
    config_fingerprint: str
    expected_nodes: int
    valid_nodes: int
    coverage: float
    graph: MarketNetwork | None
    topology: NetworkTopology | None
    centrality: NetworkCentrality | None
    communities: NetworkCommunities | None
    sector_alignment: SectorAlignment | None
    mst: MSTTopology | None
    dynamics: NetworkDynamics | None
    connectivity_score: float | None
    residual_density: float | None
    raw_minus_residual_density: float | None
    quality_flags: frozenset[QualityFlag]
    data_quality: DataQuality

    @property
    def fingerprint(self) -> str:
        return sha256(canonical_json_bytes(asdict(self))).hexdigest()
