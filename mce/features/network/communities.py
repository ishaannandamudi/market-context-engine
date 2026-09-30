"""Threshold-connected communities, weighted modularity and sector alignment."""

from collections import Counter, defaultdict

import numpy as np

from mce.features.network.models import (
    MarketNetwork,
    NetworkCommunities,
    SectorAlignment,
    SectorFragmentation,
)
from mce.features.network.thresholding import threshold_pairs


def detect_communities(
    network: MarketNetwork, matrix: np.ndarray, threshold: float
) -> NetworkCommunities:
    symbols = tuple(node.symbol for node in network.nodes)
    parent = list(range(len(symbols)))

    def root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for i, j in threshold_pairs(matrix, threshold):
        parent[root(j)] = root(i)
    groups_by_root: dict[int, list[str]] = defaultdict(list)
    for index, symbol in enumerate(symbols):
        groups_by_root[root(index)].append(symbol)
    groups = tuple(
        sorted(
            (tuple(sorted(group)) for group in groups_by_root.values()),
            key=lambda group: (-len(group), group),
        )
    )
    labels = {symbol: group_id for group_id, group in enumerate(groups) for symbol in group}
    strengths: dict[str, float] = defaultdict(float)
    internal: dict[int, float] = defaultdict(float)
    total = 0.0
    for edge in network.edges:
        total += edge.weight
        strengths[edge.first] += edge.weight
        strengths[edge.second] += edge.weight
        if labels[edge.first] == labels[edge.second]:
            internal[labels[edge.first]] += edge.weight
    modularity = None
    if total > 0:
        modularity = sum(
            internal[group_id] / total
            - (sum(strengths[symbol] for symbol in group) / (2 * total)) ** 2
            for group_id, group in enumerate(groups)
        )
    return NetworkCommunities(
        tuple((symbol, labels[symbol]) for symbol in symbols),
        groups,
        float(modularity) if modularity is not None else None,
        len(groups[0]) / len(symbols) if symbols else 0.0,
    )


def sector_alignment(network: MarketNetwork, communities: NetworkCommunities) -> SectorAlignment:
    sector = {node.symbol: node.sector for node in network.nodes if node.sector is not None}
    labels = dict(communities.assignments)
    groups: dict[str, list[str]] = defaultdict(list)
    for symbol, name in sector.items():
        groups[name].append(symbol)
    rows: list[SectorFragmentation] = []
    for name, members in sorted(groups.items()):
        counts = Counter(labels[symbol] for symbol in members)
        largest = max(counts.values()) / len(members)
        rows.append(
            SectorFragmentation(name, tuple(sorted(members)), len(counts), largest, 1 - largest)
        )
    matched = [edge for edge in network.edges if edge.first in sector and edge.second in sector]
    cross = (
        sum(sector[edge.first] != sector[edge.second] for edge in matched) / len(matched)
        if matched
        else None
    )
    classified = 0
    pure = 0
    for group in communities.groups:
        names = [sector[symbol] for symbol in group if symbol in sector]
        if names:
            classified += len(names)
            pure += max(Counter(names).values())
    return SectorAlignment(
        tuple(rows),
        pure / classified if classified else None,
        cross,
        1 - cross if cross is not None else None,
    )
