"""Causal, descriptive network context from existing dependence evidence."""

from dataclasses import replace
from typing import Literal

import numpy as np

from mce.config.settings import MCEConfig
from mce.features.correlation.matrix import covariance_to_correlation, sample_covariance
from mce.features.correlation.models import CorrelationFeatureSet, CorrelationMatrix
from mce.features.factors.models import FactorContext, ReturnPanel
from mce.features.factors.residuals import reconstruct
from mce.features.network.centrality import centrality
from mce.features.network.communities import detect_communities, sector_alignment
from mce.features.network.dynamics import compare_networks
from mce.features.network.graph import build_graph
from mce.features.network.models import DependenceInput, NetworkContext
from mce.features.network.mst import minimum_spanning_tree
from mce.features.network.topology import topology
from mce.features.sectors.membership import SectorMembershipTimeline, SectorUniverseProvider
from mce.models.enums import DataQuality, QualityFlag
from mce.reproducibility import config_fingerprint


def dependence_from_correlation(
    result: CorrelationFeatureSet, *, source: Literal["raw", "shrunk"] = "raw"
) -> DependenceInput:
    if source not in ("raw", "shrunk"):
        raise ValueError("network correlation source must be raw or shrunk")
    matrix = result.correlation_matrix if source == "raw" else result.shrunk_matrix
    return DependenceInput(
        result.as_of_ns,
        result.expected_symbols,
        matrix,
        source,
        result.data_quality,
        result.quality_flags,
        result.aligned_sample_count,
    )


def residual_dependence(panel: ReturnPanel, context: FactorContext) -> DependenceInput:
    if (
        context.as_of_ns != panel.as_of_ns
        or context.panel_fingerprint != panel.fingerprint
        or context.artifact is None
        or context.artifact.training_fingerprint != panel.fingerprint
        or not context.causal
        or context.artifact.symbols != panel.symbols
        or len(context.score_history) != len(panel.returns)
    ):
        raise ValueError("residual network needs a same-clock causal fitted factor panel")
    residual = np.asarray(panel.returns) - reconstruct(
        context.artifact, np.asarray(context.score_history)
    )
    covariance = sample_covariance(np.asarray(residual, dtype=np.float64))
    matrix = covariance_to_correlation(covariance) if covariance is not None else None
    typed = (
        CorrelationMatrix(
            symbols=panel.symbols,
            values=tuple(tuple(float(value) for value in row) for row in matrix),
        )
        if matrix is not None
        else None
    )
    return DependenceInput(
        panel.as_of_ns,
        panel.expected_symbols,
        typed,
        "factor_residual",
        context.data_quality if typed is not None else DataQuality.POOR,
        context.quality_flags,
        len(panel.returns),
    )


class NetworkFeatureEngine:
    def __init__(
        self, config: MCEConfig, sector_provider: SectorUniverseProvider | None = None
    ) -> None:
        self.config = config
        self.settings = config.features.network
        self.sector_provider = sector_provider or (
            SectorMembershipTimeline.from_groups(
                config.features.sectors.groups,
                exclusive=config.features.sectors.exclusive_membership,
            )
            if config.features.sectors.groups
            else None
        )

    def compute(
        self,
        correlation: CorrelationFeatureSet,
        *,
        previous: NetworkContext | None = None,
        residual: DependenceInput | None = None,
    ) -> NetworkContext:
        dependence = dependence_from_correlation(correlation, source=self.settings.source_matrix)
        return self.from_input(dependence, previous=previous, residual=residual)

    def from_input(
        self,
        dependence: DependenceInput,
        *,
        previous: NetworkContext | None = None,
        residual: DependenceInput | None = None,
    ) -> NetworkContext:
        if previous is not None and previous.as_of_ns >= dependence.as_of_ns:
            raise ValueError("previous network must precede current clock")
        if residual is not None and (
            residual.as_of_ns != dependence.as_of_ns or residual.source != "factor_residual"
        ):
            raise ValueError("residual dependence must match the current clock and source")
        flags = set(dependence.quality_flags)
        if dependence.data_quality != DataQuality.GOOD:
            flags.add(QualityFlag.PROVIDER_DEGRADED)
        if dependence.coverage < 1:
            flags.add(QualityFlag.PARTIAL_UNIVERSE)
        if (
            dependence.matrix is None
            or len(dependence.valid_symbols) < self.settings.minimum_nodes
            or dependence.coverage < self.settings.minimum_coverage
        ):
            flags.add(QualityFlag.INSUFFICIENT_NETWORK_COVERAGE)
        empty = NetworkContext(
            dependence.as_of_ns,
            dependence.fingerprint,
            config_fingerprint(self.config),
            len(dependence.expected_symbols),
            len(dependence.valid_symbols),
            dependence.coverage,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            frozenset(flags),
            DataQuality.POOR,
        )
        if (
            not self.settings.enabled
            or QualityFlag.INSUFFICIENT_NETWORK_COVERAGE in flags
            or dependence.data_quality in (DataQuality.POOR, DataQuality.UNUSABLE)
        ):
            return replace(
                empty,
                data_quality=DataQuality.UNUSABLE
                if dependence.data_quality == DataQuality.UNUSABLE
                else DataQuality.POOR,
            )
        assert dependence.matrix is not None
        sectors: dict[str, str] = {}
        weights: dict[str, float] = {}
        if self.sector_provider is not None:
            membership = self.sector_provider.members(dependence.as_of_ns)
            if membership.as_of_ns != dependence.as_of_ns:
                raise ValueError("sector provider returned a different network clock")
            if membership.exclusive and membership.source_quality not in (
                DataQuality.POOR,
                DataQuality.UNUSABLE,
            ):
                if membership.source_quality != DataQuality.GOOD:
                    flags.add(QualityFlag.PROVIDER_DEGRADED)
                sectors = {item.symbol: item.sector for item in membership.members}
                weights = {
                    item.symbol: item.weight
                    for item in membership.members
                    if item.weight is not None
                }
            else:
                flags.add(QualityFlag.NETWORK_TOPOLOGY_DEGRADED)
        matrix = np.asarray(dependence.matrix.values, dtype=np.float64)
        graph = build_graph(dependence, self.settings, sectors=sectors, weights=weights)
        structure = topology(graph)
        node_centrality = centrality(graph, eigenvector_enabled=self.settings.eigenvector_enabled)
        if node_centrality.eigenvector_converged is False:
            flags.add(QualityFlag.NETWORK_TOPOLOGY_DEGRADED)
        if len(structure.components) > 1:
            flags.add(QualityFlag.NETWORK_DISCONNECTED)
        groups = detect_communities(graph, matrix, self.settings.community_threshold)
        if graph.edges and len(groups.groups) == len(graph.nodes):
            flags.add(QualityFlag.COMMUNITY_STRUCTURE_WEAK)
        alignment = sector_alignment(graph, groups) if sectors else None
        tree = (
            minimum_spanning_tree(dependence.matrix.symbols, matrix)
            if self.settings.mst_enabled
            else None
        )
        score = (
            (structure.largest_component_fraction + structure.density) / 2
            if self.settings.graph_mode != "mst"
            else None
        )
        residual_density = difference = None
        if residual is not None and self.settings.graph_mode != "mst":
            if (
                residual.matrix is None
                or residual.valid_symbols != dependence.valid_symbols
                or residual.data_quality in (DataQuality.POOR, DataQuality.UNUSABLE)
            ):
                flags.add(QualityFlag.NETWORK_TOPOLOGY_DEGRADED)
            else:
                if residual.data_quality != DataQuality.GOOD:
                    flags.add(QualityFlag.NETWORK_TOPOLOGY_DEGRADED)
                adjusted = build_graph(residual, self.settings)
                residual_density = topology(adjusted).density
                difference = structure.density - residual_density
        result = replace(
            empty,
            graph=graph,
            topology=structure,
            centrality=node_centrality,
            communities=groups,
            sector_alignment=alignment,
            mst=tree,
            connectivity_score=score,
            residual_density=residual_density,
            raw_minus_residual_density=difference,
        )
        dynamics = compare_networks(result, previous) if previous is not None else None
        return replace(
            result,
            dynamics=dynamics,
            quality_flags=frozenset(flags),
            data_quality=DataQuality.DEGRADED if flags else DataQuality.GOOD,
        )


def network_history(
    engine: NetworkFeatureEngine, correlations: tuple[CorrelationFeatureSet, ...]
) -> tuple[NetworkContext, ...]:
    if any(a.as_of_ns >= b.as_of_ns for a, b in zip(correlations, correlations[1:], strict=False)):
        raise ValueError("network history clocks must increase")
    contexts: list[NetworkContext] = []
    previous: NetworkContext | None = None
    for correlation in correlations:
        current = engine.compute(correlation, previous=previous)
        contexts.append(current)
        previous = current
    return tuple(contexts)
