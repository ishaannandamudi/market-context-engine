"""Explicit prior-snapshot topology comparison without mutable history."""

from collections import Counter
from math import log, sqrt

from mce.features.network.models import NetworkContext, NetworkDynamics, NetworkEdge


def _nmi(first: tuple[int, ...], second: tuple[int, ...]) -> float:
    count = len(first)
    a, b = Counter(first), Counter(second)
    joint = Counter(zip(first, second, strict=True))
    entropy_a = -sum((value / count) * log(value / count) for value in a.values())
    entropy_b = -sum((value / count) * log(value / count) for value in b.values())
    if entropy_a == 0 or entropy_b == 0:
        return 1.0 if entropy_a == entropy_b == 0 else 0.0
    information = sum(
        (value / count) * log(value * count / (a[left] * b[right]))
        for (left, right), value in joint.items()
    )
    return max(0.0, min(1.0, information / sqrt(entropy_a * entropy_b)))


def _pairs(edges: tuple[NetworkEdge, ...], shared: set[str]) -> set[tuple[str, str]]:
    return {
        (edge.first, edge.second)
        for edge in edges
        if edge.first in shared and edge.second in shared
    }


def _jaccard(first: set[tuple[str, str]], second: set[tuple[str, str]]) -> float:
    union = first | second
    return len(first & second) / len(union) if union else 1.0


def compare_networks(current: NetworkContext, previous: NetworkContext) -> NetworkDynamics | None:
    if previous.as_of_ns >= current.as_of_ns:
        raise ValueError("previous network must precede current clock")
    if (
        current.graph is None
        or previous.graph is None
        or current.topology is None
        or previous.topology is None
        or current.centrality is None
        or previous.centrality is None
        or current.communities is None
        or previous.communities is None
        or current.graph.graph_mode != previous.graph.graph_mode
        or current.graph.matrix_source != previous.graph.matrix_source
        or current.config_fingerprint != previous.config_fingerprint
    ):
        return None
    current_nodes = {node.symbol for node in current.graph.nodes}
    previous_nodes = {node.symbol for node in previous.graph.nodes}
    shared = current_nodes & previous_nodes
    if len(shared) < 2:
        return None
    first = _pairs(current.graph.edges, shared)
    second = _pairs(previous.graph.edges, shared)
    persistence = _jaccard(first, second)
    current_degrees = {node.symbol: node.degree_centrality for node in current.centrality.nodes}
    previous_degrees = {node.symbol: node.degree_centrality for node in previous.centrality.nodes}
    changes = tuple(
        (symbol, current_degrees[symbol] - previous_degrees[symbol]) for symbol in sorted(shared)
    )
    drift = [abs(value) for _, value in changes]
    current_labels = dict(current.communities.assignments)
    previous_labels = dict(previous.communities.assignments)
    ordered = sorted(shared)
    stability = _nmi(
        tuple(current_labels[symbol] for symbol in ordered),
        tuple(previous_labels[symbol] for symbol in ordered),
    )
    mst_turnover = None
    if current.mst is not None and previous.mst is not None:
        mst_turnover = 1 - _jaccard(
            _pairs(current.mst.edges, shared), _pairs(previous.mst.edges, shared)
        )
    mean_drift = sum(drift) / len(drift)
    turnover = 1 - persistence
    return NetworkDynamics(
        previous.as_of_ns,
        len(shared) / len(current_nodes | previous_nodes),
        persistence,
        turnover,
        mean_drift,
        max(drift),
        changes,
        stability,
        len(current.topology.components) - len(previous.topology.components),
        mst_turnover,
        (turnover + mean_drift + (1 - stability)) / 3,
    )
