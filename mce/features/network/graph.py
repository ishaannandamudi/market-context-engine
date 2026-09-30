"""Immutable graph construction from a typed, ordered dependence matrix."""

import numpy as np

from mce.config.settings import NetworkSettings
from mce.features.network.distances import correlation_distance
from mce.features.network.models import (
    DependenceInput,
    MarketNetwork,
    NetworkEdge,
    NetworkNode,
)
from mce.features.network.mst import minimum_spanning_tree
from mce.features.network.thresholding import threshold_pairs, top_k_pairs


def build_graph(
    dependence: DependenceInput,
    settings: NetworkSettings,
    *,
    sectors: dict[str, str] | None = None,
    weights: dict[str, float] | None = None,
) -> MarketNetwork:
    if dependence.matrix is None:
        raise ValueError("network graph requires a dependence matrix")
    symbols = dependence.matrix.symbols
    matrix = np.asarray(dependence.matrix.values, dtype=np.float64)
    if settings.graph_mode == "threshold":
        pairs = threshold_pairs(matrix, settings.correlation_threshold)
    elif settings.graph_mode == "top_k":
        pairs = top_k_pairs(matrix, settings.top_k)
    else:
        selected_tree = minimum_spanning_tree(symbols, matrix)
        positions = {symbol: index for index, symbol in enumerate(symbols)}
        pairs = tuple(
            (positions[edge.first], positions[edge.second]) for edge in selected_tree.edges
        )
    edges = tuple(
        NetworkEdge(
            symbols[i],
            symbols[j],
            min(1.0, max(-1.0, float(matrix[i, j]))),
            correlation_distance(float(matrix[i, j])),
            max(0.0, min(1.0, float(matrix[i, j]))),
        )
        for i, j in sorted(pairs)
    )
    return MarketNetwork(
        dependence.as_of_ns,
        tuple(
            NetworkNode(
                symbol,
                sectors.get(symbol) if sectors else None,
                weights.get(symbol) if weights else None,
            )
            for symbol in symbols
        ),
        edges,
        dependence.source,
        settings.graph_mode,
        dependence.fingerprint,
    )
