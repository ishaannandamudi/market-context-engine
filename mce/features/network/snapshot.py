"""Add continuous network topology into the existing schema-1.0 snapshot."""

from mce.features.network.models import NetworkContext
from mce.models.context import MarketContextSnapshot
from mce.models.enums import DataQuality
from mce.version import __version__


def apply_network_to_snapshot(
    result: NetworkContext, snapshot: MarketContextSnapshot | None = None
) -> MarketContextSnapshot:
    base = snapshot or MarketContextSnapshot(as_of_ns=result.as_of_ns)
    if base.as_of_ns != result.as_of_ns:
        raise ValueError("network result and snapshot clocks differ")
    values = {
        name: value
        for name, value in (base.feature_values or {}).items()
        if not name.startswith("network.")
    }
    values.update(
        {
            "network.expected_nodes": result.expected_nodes,
            "network.valid_nodes": result.valid_nodes,
            "network.coverage": result.coverage,
        }
    )
    if result.graph is not None and result.data_quality not in (
        DataQuality.POOR,
        DataQuality.UNUSABLE,
    ):
        assert result.topology is not None
        assert result.centrality is not None
        assert result.communities is not None
        metrics = {
            "network.density": result.topology.density,
            "network.average_degree": result.topology.average_degree,
            "network.median_degree": result.topology.median_degree,
            "network.max_degree": result.topology.max_degree,
            "network.component_count": len(result.topology.components),
            "network.largest_component_fraction": result.topology.largest_component_fraction,
            "network.isolated_nodes": result.topology.isolated_nodes,
            "network.centralization": result.centrality.degree_centralization,
            "network.hub_hhi": result.centrality.hub_hhi,
            "network.effective_hubs": result.centrality.effective_hubs,
            "network.community_count": len(result.communities.groups),
            "network.largest_community_fraction": result.communities.largest_fraction,
            "network.modularity": result.communities.modularity,
            "network.community_purity": result.sector_alignment.community_purity
            if result.sector_alignment
            else None,
            "network.cross_sector_edge_fraction": result.sector_alignment.cross_sector_edge_fraction
            if result.sector_alignment
            else None,
            "network.mst.total_length": result.mst.total_length if result.mst else None,
            "network.mst.mean_distance": result.mst.mean_distance if result.mst else None,
            "network.mst.diameter": result.mst.diameter if result.mst else None,
            "network.edge_persistence": result.dynamics.edge_persistence
            if result.dynamics
            else None,
            "network.edge_turnover": result.dynamics.edge_turnover if result.dynamics else None,
            "network.centrality_drift": result.dynamics.centrality_drift
            if result.dynamics
            else None,
            "network.community_stability": result.dynamics.community_stability
            if result.dynamics
            else None,
            "network.rewiring": result.dynamics.rewiring if result.dynamics else None,
            "network.connectivity_score": result.connectivity_score,
            "network.residual_density": result.residual_density,
            "network.raw_minus_residual_density": result.raw_minus_residual_density,
        }
        values.update({name: value for name, value in metrics.items() if value is not None})
    quality_order = {
        DataQuality.UNKNOWN: 0,
        DataQuality.GOOD: 1,
        DataQuality.DEGRADED: 2,
        DataQuality.POOR: 3,
        DataQuality.UNUSABLE: 4,
    }
    quality = (
        base.data_quality
        if quality_order[base.data_quality] > quality_order[result.data_quality]
        else result.data_quality
    )
    payload = base.model_dump(mode="python")
    payload.update(
        engine_version=__version__,
        feature_values=values,
        quality_flags=base.quality_flags | result.quality_flags,
        data_quality=quality,
    )
    return MarketContextSnapshot.model_validate(payload)
