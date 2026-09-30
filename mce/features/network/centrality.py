"""Degree, positive-strength and optional nonnegative eigenvector centrality."""

import numpy as np

from mce.features.network.models import MarketNetwork, NetworkCentrality, NodeCentrality


def centrality(network: MarketNetwork, *, eigenvector_enabled: bool = True) -> NetworkCentrality:
    symbols = tuple(node.symbol for node in network.nodes)
    size = len(symbols)
    positions = {symbol: index for index, symbol in enumerate(symbols)}
    degrees = np.zeros(size, dtype=np.float64)
    adjacency = np.zeros((size, size), dtype=np.float64)
    for edge in network.edges:
        i, j = positions[edge.first], positions[edge.second]
        degrees[i] += 1
        degrees[j] += 1
        adjacency[i, j] = adjacency[j, i] = edge.weight
    strengths = adjacency.sum(axis=1)
    total_strength = float(strengths.sum())
    hhi = (
        float(np.dot(strengths / total_strength, strengths / total_strength))
        if total_strength
        else None
    )
    eigenvector: np.ndarray | None = None
    converged: bool | None = None
    if eigenvector_enabled:
        converged = True
        if total_strength:
            estimate = np.full(size, 1 / np.sqrt(size))
            # A + I has the same eigenvectors and avoids bipartite sign cycling.
            for _ in range(300):
                updated = adjacency @ estimate + estimate
                updated /= np.linalg.norm(updated)
                if np.linalg.norm(updated - estimate) < 1e-10:
                    estimate = updated
                    break
                estimate = updated
            else:
                converged = False
            if converged:
                estimate[strengths == 0] = 0
                estimate /= np.linalg.norm(estimate)
                eigenvector = estimate
        else:
            eigenvector = np.zeros(size)
    maximum = float(max(degrees, default=0.0))
    centralization = (
        float(np.sum(maximum - degrees)) / ((size - 1) * (size - 2)) if size >= 3 else 0.0
    )
    return NetworkCentrality(
        tuple(
            NodeCentrality(
                symbol,
                int(degrees[index]),
                float(degrees[index] / (size - 1)) if size >= 2 else 0.0,
                float(strengths[index]),
                float(eigenvector[index]) if eigenvector is not None else None,
            )
            for index, symbol in enumerate(symbols)
        ),
        centralization,
        hhi,
        1 / hhi if hhi else None,
        converged,
    )
