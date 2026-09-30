"""Manual 500-node graph, tree, centrality, community and rolling benchmark."""

import argparse
import platform
from time import perf_counter

import numpy as np

from mce.config.settings import MCEConfig, NetworkSettings
from mce.features.correlation.models import CorrelationMatrix
from mce.features.network.centrality import centrality
from mce.features.network.communities import detect_communities
from mce.features.network.dynamics import compare_networks
from mce.features.network.engine import NetworkFeatureEngine
from mce.features.network.graph import build_graph
from mce.features.network.models import DependenceInput
from mce.features.network.mst import minimum_spanning_tree


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", type=int, default=500)
    parser.add_argument("--snapshots", type=int, default=3)
    args = parser.parse_args()
    if args.nodes < 3 or args.snapshots < 2:
        parser.error("need at least three nodes and two snapshots")
    rng = np.random.default_rng(15)
    symbols = tuple(f"S{index:04d}" for index in range(args.nodes))
    settings = NetworkSettings(correlation_threshold=0.6, community_threshold=0.7)
    base = MCEConfig(_env_file=None)
    config = base.model_copy(
        update={"features": base.features.model_copy(update={"network": settings})}
    )
    engine = NetworkFeatureEngine(config)
    panel_seconds = graph_seconds = tree_seconds = central_seconds = community_seconds = 0.0
    dynamics_seconds = total_seconds = 0.0
    previous = None
    edge_count = 0
    for index in range(args.snapshots):
        loadings = np.zeros((args.nodes, 6))
        loadings[:, 0] = 0.8 + rng.normal(scale=0.04, size=args.nodes)
        loadings[np.arange(args.nodes), 1 + np.arange(args.nodes) % 5] = 0.4
        loadings += rng.normal(scale=0.02, size=loadings.shape)
        covariance = loadings @ loadings.T + np.eye(args.nodes) * 0.3
        scales = np.sqrt(np.diag(covariance))
        matrix = covariance / np.outer(scales, scales)
        np.fill_diagonal(matrix, 1)
        start = perf_counter()
        typed = CorrelationMatrix(
            symbols=symbols,
            values=tuple(tuple(float(value) for value in row) for row in matrix),
        )
        dependence = DependenceInput(index + 1, symbols, typed, "raw", aligned_samples=2500)
        panel_seconds += perf_counter() - start
        start = perf_counter()
        graph = build_graph(dependence, settings)
        graph_seconds += perf_counter() - start
        edge_count = len(graph.edges)
        start = perf_counter()
        minimum_spanning_tree(symbols, matrix)
        tree_seconds += perf_counter() - start
        start = perf_counter()
        centrality(graph)
        central_seconds += perf_counter() - start
        start = perf_counter()
        detect_communities(graph, matrix, settings.community_threshold)
        community_seconds += perf_counter() - start
        start = perf_counter()
        current = engine.from_input(dependence, previous=previous)
        total_seconds += perf_counter() - start
        if previous is not None:
            start = perf_counter()
            compare_networks(current, previous)
            dynamics_seconds += perf_counter() - start
        previous = current
    count = args.snapshots
    print(f"Python {platform.python_version()} on {platform.platform()}")
    print(
        f"nodes={args.nodes} edges={edge_count} snapshots={count} mode={settings.graph_mode} "
        f"threshold={settings.correlation_threshold}"
    )
    print(
        f"input_seconds={panel_seconds / count:.3f} graph_seconds={graph_seconds / count:.3f} "
        f"mst_seconds={tree_seconds / count:.3f} centrality_seconds={central_seconds / count:.3f} "
        f"communities_seconds={community_seconds / count:.3f} "
        f"dynamics_seconds={dynamics_seconds / (count - 1):.3f} "
        f"engine_seconds={total_seconds / count:.3f}"
    )
    print(f"rolling_comparison_included={count - 1} total_engine_seconds={total_seconds:.3f}")


if __name__ == "__main__":
    main()
