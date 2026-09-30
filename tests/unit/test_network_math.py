"""Graph, MST, centrality, modularity, and invariant reference tests."""

from dataclasses import replace

import numpy as np
import pytest
from pydantic import ValidationError

from mce.config.settings import NetworkSettings
from mce.features.network.centrality import centrality
from mce.features.network.communities import detect_communities, sector_alignment
from mce.features.network.distances import correlation_distance
from mce.features.network.graph import build_graph
from mce.features.network.models import NetworkEdge, NetworkNode
from mce.features.network.mst import minimum_spanning_tree
from mce.features.network.topology import connected_components, topology
from tests.fixtures.network_scenarios import blocks, dependence, equicorrelation, star


def test_correlation_distance_and_input_validation() -> None:
    assert correlation_distance(1) == 0
    assert correlation_distance(0) == pytest.approx(2**0.5)
    assert correlation_distance(-1) == 2
    with pytest.raises(ValueError, match="correlation"):
        correlation_distance(float("nan"))
    with pytest.raises(ValueError, match="correlation"):
        correlation_distance(1.01)
    with pytest.raises(ValidationError, match="symmetric"):
        dependence(np.array([[1, 0.2], [0.3, 1]]))
    with pytest.raises(ValueError, match="positive semidefinite"):
        dependence(np.array([[1, 0.9, 0.9], [0.9, 1, -0.9], [0.9, -0.9, 1]]))
    with pytest.raises(ValueError, match="sorted"):
        replace(dependence(equicorrelation()), expected_symbols=("BBB", "AAA"))
    with pytest.raises(ValueError, match="nonmember"):
        replace(dependence(equicorrelation()), expected_symbols=("AAA", "BBB"))
    with pytest.raises(ValueError, match="finite"):
        NetworkEdge("AAA", "AAA", 0.8, 0.5, 0.8)
    with pytest.raises(ValueError, match="distance disagrees"):
        NetworkEdge("AAA", "BBB", 0.8, 0.5, 0.8)
    with pytest.raises(ValueError, match="membership weight"):
        NetworkNode("AAA", membership_weight=-1)
    with pytest.raises(ValueError, match="source"):
        replace(dependence(equicorrelation()), source="future")  # type: ignore[arg-type]


def test_threshold_top_k_ties_negative_edges_and_graph_invariants() -> None:
    complete = dependence(equicorrelation())
    threshold = build_graph(complete, NetworkSettings(correlation_threshold=0.8))
    assert len(threshold.edges) == 6
    assert topology(threshold).density == 1
    assert connected_components(threshold) == (("AAA", "BBB", "CCC", "DDD"),)
    assert all(edge.first < edge.second for edge in threshold.edges)
    boundary = build_graph(complete, NetworkSettings(correlation_threshold=0.8001))
    assert not boundary.edges and topology(boundary).isolated_nodes == 4
    top = build_graph(complete, NetworkSettings(graph_mode="top_k", top_k=1))
    assert tuple((edge.first, edge.second) for edge in top.edges) == (
        ("AAA", "BBB"),
        ("AAA", "CCC"),
        ("AAA", "DDD"),
    )
    # A valid negative-equicorrelation matrix can still contribute topology;
    # signed dependence never becomes a negative centrality weight.
    negative = dependence(equicorrelation(3, -0.4))
    graph = build_graph(negative, NetworkSettings(correlation_threshold=-0.4))
    assert len(graph.edges) == 3 and all(edge.weight == 0 for edge in graph.edges)
    assert centrality(graph).hub_hhi is None
    assert centrality(graph).eigenvector_converged is True
    with pytest.raises(ValueError, match="unique, sorted"):
        replace(threshold, edges=threshold.edges[::-1])


def test_known_mst_ties_tree_diameter_and_invalid_inputs() -> None:
    matrix = np.array([[1, 0.9, 0.2], [0.9, 1, 0.3], [0.2, 0.3, 1]])
    tree = minimum_spanning_tree(("AAA", "BBB", "CCC"), matrix)
    assert tuple((edge.first, edge.second) for edge in tree.edges) == (
        ("AAA", "BBB"),
        ("BBB", "CCC"),
    )
    assert tree.total_length == pytest.approx(correlation_distance(0.9) + correlation_distance(0.3))
    assert tree.diameter == pytest.approx(tree.total_length)
    assert tree.central_node == "BBB"
    tied = minimum_spanning_tree(("AAA", "BBB", "CCC", "DDD"), equicorrelation())
    assert len(tied.edges) == 3
    assert all(edge.first == "AAA" for edge in tied.edges)
    graph = build_graph(dependence(equicorrelation()), NetworkSettings(graph_mode="mst"))
    assert len(graph.edges) == len(graph.nodes) - 1
    assert len(topology(graph).components) == 1
    with pytest.raises(ValueError, match="finite complete"):
        minimum_spanning_tree(("AAA", "BBB"), np.array([[1, np.nan], [np.nan, 1]]))
    with pytest.raises(ValueError, match="finite complete"):
        minimum_spanning_tree(("AAA", "BBB"), np.ones((3, 3)))


def test_complete_star_path_and_disconnected_centrality() -> None:
    full = build_graph(dependence(equicorrelation()), NetworkSettings())
    centered = centrality(full)
    assert centered.degree_centralization == pytest.approx(0)
    assert centered.nodes[0].degree_centrality == pytest.approx(1)
    assert centered.nodes[0].eigenvector == pytest.approx(0.5)
    hub = build_graph(dependence(star()), NetworkSettings(correlation_threshold=0.6))
    result = centrality(hub)
    assert result.degree_centralization == pytest.approx(1)
    assert result.nodes[0].degree == 3 and all(node.degree == 1 for node in result.nodes[1:])
    assert result.nodes[0].eigenvector is not None
    assert result.nodes[1].eigenvector is not None
    assert result.nodes[0].eigenvector > result.nodes[1].eigenvector
    assert result.hub_hhi == pytest.approx(1 / 3)
    assert result.effective_hubs == pytest.approx(3)
    chain = np.fromfunction(lambda i, j: 0.8 ** np.abs(i - j), (4, 4))
    path = build_graph(dependence(chain), NetworkSettings(correlation_threshold=0.7))
    assert len(path.edges) == 3
    assert centrality(path).degree_centralization == pytest.approx(1 / 3)
    two = build_graph(dependence(blocks()), NetworkSettings(correlation_threshold=0.6))
    assert topology(two).components == (("AAA", "BBB"), ("CCC", "DDD"))
    assert topology(two).density == pytest.approx(1 / 3)
    assert centrality(two, eigenvector_enabled=False).eigenvector_converged is None


def test_communities_modularity_and_sector_alignment() -> None:
    matrix = blocks()
    graph = build_graph(
        dependence(matrix),
        NetworkSettings(correlation_threshold=0.6),
        sectors={"AAA": "tech", "BBB": "tech", "CCC": "banks", "DDD": "banks"},
    )
    groups = detect_communities(graph, matrix, 0.7)
    assert groups.groups == (("AAA", "BBB"), ("CCC", "DDD"))
    assert groups.assignments == (("AAA", 0), ("BBB", 0), ("CCC", 1), ("DDD", 1))
    assert groups.modularity == pytest.approx(0.5)
    sectors = sector_alignment(graph, groups)
    assert sectors.community_purity == 1
    assert sectors.cross_sector_edge_fraction == 0
    assert all(row.fragmentation == 0 for row in sectors.sectors)
    mixed = build_graph(
        dependence(matrix),
        NetworkSettings(correlation_threshold=0.6),
        sectors={"AAA": "tech", "CCC": "tech", "BBB": "banks", "DDD": "banks"},
    )
    mixed_result = sector_alignment(mixed, groups)
    assert mixed_result.community_purity == 0.5
    assert mixed_result.cross_sector_edge_fraction == 1
    assert all(row.fragmentation == 0.5 for row in mixed_result.sectors)
    homogeneous = detect_communities(
        build_graph(dependence(equicorrelation()), NetworkSettings()),
        equicorrelation(),
        0.7,
    )
    assert homogeneous.groups == (("AAA", "BBB", "CCC", "DDD"),)
    assert homogeneous.modularity == pytest.approx(0)
