"""Causal rolling factor context from completed regular-session return panels."""

from dataclasses import replace

from mce.calendar.service import MarketSessionService
from mce.config.settings import MCEConfig
from mce.features.breadth.membership import UniverseProvider
from mce.features.factors.dynamics import (
    align_factors,
    factor_shock_and_volatility,
    spectrum_diagnostics,
)
from mce.features.factors.exposures import sector_exposures
from mce.features.factors.models import FactorContext, ReturnPanel
from mce.features.factors.panel import FactorPanelBuilder
from mce.features.factors.pca import fit_pca
from mce.features.factors.residuals import decompose
from mce.features.sectors.membership import SectorMembershipTimeline, SectorUniverseProvider
from mce.models.enums import DataQuality, QualityFlag
from mce.replay.store import CausalDataStore
from mce.reproducibility import config_fingerprint


class FactorFeatureEngine:
    def __init__(
        self,
        config: MCEConfig,
        universe_provider: UniverseProvider | None = None,
        sector_provider: SectorUniverseProvider | None = None,
        calendar: MarketSessionService | None = None,
    ) -> None:
        self.config = config
        self.panel_builder = FactorPanelBuilder(config, universe_provider, calendar)
        self.sector_provider = sector_provider or (
            SectorMembershipTimeline.from_groups(
                config.features.sectors.groups,
                exclusive=config.features.sectors.exclusive_membership,
            )
            if config.features.sectors.groups
            else None
        )

    def compute(
        self, store: CausalDataStore, as_of_ns: int, *, previous: FactorContext | None = None
    ) -> FactorContext:
        if previous is not None and previous.as_of_ns >= as_of_ns:
            raise ValueError("prior factor context must precede current clock")
        panel = self.panel_builder.build(store, as_of_ns)
        return self.from_panel(panel, previous=previous)

    def from_panel(
        self, panel: ReturnPanel, *, previous: FactorContext | None = None
    ) -> FactorContext:
        if previous is not None and previous.as_of_ns >= panel.as_of_ns:
            raise ValueError("prior factor context must precede current clock")
        settings = self.config.features.factor
        flags: set[QualityFlag] = set(panel.quality_flags)
        if panel.source_quality != DataQuality.GOOD:
            flags.add(QualityFlag.PROVIDER_DEGRADED)
        if panel.stale_symbols:
            flags.update((QualityFlag.STALE_DATA, QualityFlag.SOURCE_STALE))
        if panel.missing_symbols or panel.symbol_coverage < 1:
            flags.add(QualityFlag.PARTIAL_UNIVERSE)
        if panel.symbol_coverage < settings.minimum_symbol_coverage:
            flags.add(QualityFlag.LOW_COVERAGE)
        if panel.time_coverage < settings.minimum_time_coverage:
            flags.add(QualityFlag.INSUFFICIENT_WINDOW_COVERAGE)
        if len(panel.returns) < settings.minimum_samples:
            flags.add(QualityFlag.INSUFFICIENT_HISTORY)
        if len(panel.symbols) < settings.minimum_symbols:
            flags.add(QualityFlag.MISSING_DATA)
        empty = FactorContext(
            as_of_ns=panel.as_of_ns,
            panel_fingerprint=panel.fingerprint,
            artifact=None,
            expected_symbols=panel.expected_symbols,
            valid_symbols=panel.symbols,
            missing_symbols=panel.missing_symbols,
            stale_symbols=panel.stale_symbols,
            symbol_coverage=panel.symbol_coverage,
            aligned_samples=len(panel.returns),
            expected_samples=panel.expected_samples,
            time_coverage=panel.time_coverage,
            first_share=None,
            top3_share=None,
            effective_dimension=None,
            concentration=None,
            dominance=None,
            current_scores=(),
            score_history=(),
            decomposition=None,
            alignment=None,
            drift_symbols=(),
            loading_drift=(),
            sector_exposures=(),
            shock_z=None,
            factor_volatility=None,
            quality_flags=frozenset(flags),
            data_quality=DataQuality.POOR,
        )
        if panel.source_quality in (DataQuality.POOR, DataQuality.UNUSABLE):
            return replace(empty, data_quality=panel.source_quality)
        if not settings.enabled or any(
            flag in flags
            for flag in (
                QualityFlag.LOW_COVERAGE,
                QualityFlag.INSUFFICIENT_WINDOW_COVERAGE,
                QualityFlag.INSUFFICIENT_HISTORY,
                QualityFlag.MISSING_DATA,
            )
        ):
            return empty
        try:
            artifact, scores = fit_pca(
                panel, settings, config_fingerprint=config_fingerprint(self.config)
            )
        except ValueError:
            flags.add(QualityFlag.FACTOR_IDENTIFICATION_WEAK)
            return replace(empty, quality_flags=frozenset(flags))
        if artifact.near_degenerate or artifact.selected_count < settings.factor_count:
            flags.add(QualityFlag.FACTOR_IDENTIFICATION_WEAK)
        first, top3, effective, dominance = spectrum_diagnostics(artifact.eigenvalues)
        decomposition = decompose(panel, artifact, scores)
        alignment, drift_symbols, drift = (
            align_factors(artifact, previous.artifact)
            if previous is not None and previous.artifact is not None
            else (None, (), ())
        )
        shock, volatility = factor_shock_and_volatility(scores, settings.shock_min_samples)
        exposures = sector_exposures(artifact, self.sector_provider, panel.as_of_ns)
        return replace(
            empty,
            artifact=artifact,
            first_share=first,
            top3_share=top3,
            effective_dimension=effective,
            concentration=1 / effective,
            dominance=dominance,
            current_scores=tuple(float(value) for value in scores[-1]),
            score_history=tuple(tuple(float(value) for value in row) for row in scores),
            decomposition=decomposition,
            alignment=alignment,
            drift_symbols=drift_symbols,
            loading_drift=drift,
            sector_exposures=exposures,
            shock_z=shock,
            factor_volatility=volatility,
            quality_flags=frozenset(flags),
            data_quality=DataQuality.DEGRADED if flags else DataQuality.GOOD,
        )
