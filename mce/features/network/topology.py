"""Component and degree structure of a simple undirected market graph."""

from statistics import median

from mce.features.network.models import MarketNetwork, NetworkTopology


def connected_components(network: MarketNetwork) -> tuple[tuple[str, ...], ...]:
    neighbors: dict[str, set[str]] = {node.symbol: set() for node in network.nodes}
    for edge in network.edges:
        neighbors[edge.first].add(edge.second)
        neighbors[edge.second].add(edge.first)
    unseen = set(neighbors)
    groups: list[tuple[str, ...]] = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        stack = [start]
        group: list[str] = []
        while stack:
            node = stack.pop()
            group.append(node)
            for neighbor in sorted(neighbors[node] & unseen, reverse=True):
                unseen.remove(neighbor)
                stack.append(neighbor)
        groups.append(tuple(sorted(group)))
    return tuple(sorted(groups, key=lambda group: (-len(group), group)))


def topology(network: MarketNetwork) -> NetworkTopology:
    size = len(network.nodes)
    degrees = {node.symbol: 0 for node in network.nodes}
    for edge in network.edges:
        degrees[edge.first] += 1
        degrees[edge.second] += 1
    values = tuple(degrees.values())
    components = connected_components(network)
    return NetworkTopology(
        components,
        2 * len(network.edges) / (size * (size - 1)) if size >= 2 else 0.0,
        2 * len(network.edges) / size if size else 0.0,
        float(median(values)) if values else 0.0,
        max(values, default=0),
        len(components[0]) / size if size else 0.0,
        sum(degree == 0 for degree in values),
    )
