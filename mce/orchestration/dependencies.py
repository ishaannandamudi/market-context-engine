"""Small, validated dependency graph and explicit provider wiring."""

from dataclasses import dataclass
from typing import Protocol

from mce.features.breadth.membership import UniverseProvider, UniverseSnapshot
from mce.features.sectors.membership import SectorUniverseProvider, SectorUniverseSnapshot
from mce.providers.base import ProviderHealth


class HealthProvider(Protocol):
    @property
    def provider_name(self) -> str: ...

    def health(self) -> ProviderHealth: ...


@dataclass(frozen=True, slots=True)
class PipelineProviders:
    """Provider-specific adapters end here; engines receive canonical data."""

    breadth_membership: UniverseProvider | None = None
    sector_membership: SectorUniverseProvider | None = None
    correlation_membership: UniverseProvider | None = None
    dispersion_membership: UniverseProvider | None = None
    factor_membership: UniverseProvider | None = None
    liquidity_membership: UniverseProvider | None = None
    seasonality_membership: UniverseProvider | None = None
    market_health: HealthProvider | None = None
    options_health: HealthProvider | None = None
    macro_health: HealthProvider | None = None

    def for_snapshot(self) -> "PipelineProviders":
        """Memoize immutable point-in-time membership only within one snapshot."""

        universe_cache: dict[tuple[int, int], UniverseSnapshot] = {}
        sector_cache: dict[tuple[int, int], SectorUniverseSnapshot] = {}

        def universe(source: UniverseProvider | None) -> UniverseProvider | None:
            return _CachedUniverse(source, universe_cache) if source is not None else None

        def sector(source: SectorUniverseProvider | None) -> SectorUniverseProvider | None:
            return _CachedSectors(source, sector_cache) if source is not None else None

        return PipelineProviders(
            breadth_membership=universe(self.breadth_membership),
            sector_membership=sector(self.sector_membership),
            correlation_membership=universe(self.correlation_membership),
            dispersion_membership=universe(self.dispersion_membership),
            factor_membership=universe(self.factor_membership),
            liquidity_membership=universe(self.liquidity_membership),
            seasonality_membership=universe(self.seasonality_membership),
            market_health=self.market_health,
            options_health=self.options_health,
            macro_health=self.macro_health,
        )


class _CachedUniverse:
    def __init__(
        self, source: UniverseProvider, cache: dict[tuple[int, int], UniverseSnapshot]
    ) -> None:
        self.source = source
        self.cache = cache

    def members(self, as_of_ns: int) -> UniverseSnapshot:
        key = (id(self.source), as_of_ns)
        if key not in self.cache:
            self.cache[key] = self.source.members(as_of_ns)
        return self.cache[key]


class _CachedSectors:
    def __init__(
        self, source: SectorUniverseProvider, cache: dict[tuple[int, int], SectorUniverseSnapshot]
    ) -> None:
        self.source = source
        self.cache = cache

    def members(self, as_of_ns: int) -> SectorUniverseSnapshot:
        key = (id(self.source), as_of_ns)
        if key not in self.cache:
            self.cache[key] = self.source.members(as_of_ns)
        return self.cache[key]


@dataclass(frozen=True, slots=True)
class StageNode:
    stage_id: str
    depends_on: tuple[str, ...] = ()


STAGES: tuple[StageNode, ...] = (
    StageNode("trend"),
    StageNode("volatility"),
    StageNode("breadth"),
    StageNode("sectors"),
    StageNode("correlation"),
    StageNode("dispersion", ("sectors", "volatility", "correlation")),
    StageNode("factors"),
    StageNode("network", ("correlation",)),
    StageNode("seasonality", ("correlation",)),
    StageNode(
        "liquidity_stress", ("volatility", "breadth", "sectors", "correlation", "dispersion")
    ),
    StageNode(
        "regime",
        (
            "trend",
            "volatility",
            "breadth",
            "sectors",
            "correlation",
            "dispersion",
            "liquidity_stress",
        ),
    ),
    StageNode("regime_transition", ("regime",)),
    StageNode("options", ("volatility",)),
    StageNode("macro"),
    StageNode("hmm"),
    StageNode(
        "ensemble",
        (
            "regime",
            "regime_transition",
            "options",
            "macro",
            "hmm",
            "factors",
            "network",
            "seasonality",
        ),
    ),
)


def topological_order(nodes: tuple[StageNode, ...]) -> tuple[str, ...]:
    """Registration order has no effect on execution order."""

    names = [node.stage_id for node in nodes]
    if len(names) != len(set(names)):
        raise ValueError("duplicate stage ID")
    known = set(names)
    dependencies = {node.stage_id: set(node.depends_on) for node in nodes}
    if any(not required <= known for required in dependencies.values()):
        raise ValueError("unknown stage dependency")
    ordered: list[str] = []
    remaining = set(names)
    while remaining:
        ready = sorted(name for name in remaining if dependencies[name] <= set(ordered))
        if not ready:
            raise ValueError("cycle in stage dependencies")
        ordered.extend(ready)
        remaining.difference_update(ready)
    return tuple(ordered)
