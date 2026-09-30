"""Deterministic Kruskal tree and linear-time weighted tree diameter."""

from collections import defaultdict

import numpy as np
from numpy.typing import NDArray

from mce.features.network.distances import correlation_distance
from mce.features.network.models import MSTTopology, NetworkEdge


def minimum_spanning_tree(symbols: tuple[str, ...], matrix: NDArray[np.float64]) -> MSTTopology:
    size = len(symbols)
    if (
        size < 2
        or symbols != tuple(sorted(set(symbols)))
        or matrix.shape != (size, size)
        or not np.isfinite(matrix).all()
        or not np.allclose(matrix, matrix.T, atol=1e-9)
        or not np.allclose(np.diag(matrix), 1, atol=1e-9)
        or np.any(np.abs(matrix) > 1 + 1e-9)
    ):
        raise ValueError("MST needs a finite complete dependence matrix")
    candidates = sorted(
        (
            (correlation_distance(float(matrix[i, j])), i, j)
            for i in range(size)
            for j in range(i + 1, size)
        ),
        key=lambda item: (item[0], symbols[item[1]], symbols[item[2]]),
    )
    parent = list(range(size))
    rank = [0] * size

    def root(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    selected: list[NetworkEdge] = []
    for distance, i, j in candidates:
        left, right = root(i), root(j)
        if left == right:
            continue
        if rank[left] < rank[right]:
            left, right = right, left
        parent[right] = left
        if rank[left] == rank[right]:
            rank[left] += 1
        rho = min(1.0, max(-1.0, float(matrix[i, j])))
        selected.append(NetworkEdge(symbols[i], symbols[j], rho, distance, max(0.0, rho)))
        if len(selected) == size - 1:
            break
    if len(selected) != size - 1:
        raise ValueError("MST input graph is disconnected")
    edges = tuple(sorted(selected, key=lambda edge: (edge.first, edge.second)))
    adjacency: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for edge in edges:
        adjacency[edge.first].append((edge.second, edge.distance))
        adjacency[edge.second].append((edge.first, edge.distance))

    def farthest(start: str) -> tuple[str, float]:
        stack = [(start, "", 0.0)]
        best = (start, 0.0)
        while stack:
            node, prior, length = stack.pop()
            if length > best[1] + 1e-12 or (abs(length - best[1]) <= 1e-12 and node < best[0]):
                best = (node, length)
            stack.extend(
                (neighbor, node, length + distance)
                for neighbor, distance in adjacency[node]
                if neighbor != prior
            )
        return best

    endpoint, _ = farthest(symbols[0])
    diameter = farthest(endpoint)[1]
    degrees = {symbol: len(adjacency[symbol]) for symbol in symbols}
    center = min(symbols, key=lambda symbol: (-degrees[symbol], symbol))
    total = sum(edge.distance for edge in edges)
    return MSTTopology(edges, total, total / len(edges), diameter, center)
