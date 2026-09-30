"""Explicit, one-pass orchestration of existing causal feature engines."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, is_dataclass
from hashlib import sha256
from time import perf_counter_ns
from typing import Protocol, TypeVar

from mce.calendar.service import MarketSessionService
from mce.config.settings import MCEConfig
from mce.features.breadth.engine import BreadthFeatureEngine
from mce.features.breadth.membership import UniverseProvider
from mce.features.breadth.snapshot import apply_breadth_to_snapshot
from mce.features.correlation.engine import CorrelationFeatureEngine
from mce.features.correlation.snapshot import apply_correlation_to_snapshot
from mce.features.dispersion.engine import DispersionFeatureEngine
from mce.features.dispersion.snapshot import apply_dispersion_to_snapshot
from mce.features.engine import TrendFeatureEngine
from mce.features.ensemble import EnsembleContextFeatureEngine, EnsembleContextInput
from mce.features.ensemble.snapshot import apply_ensemble_to_snapshot
from mce.features.factors.engine import FactorFeatureEngine
from mce.features.factors.snapshot import apply_factors_to_snapshot
from mce.features.macro.engine import MacroContextFeatureEngine
from mce.features.macro.snapshot import apply_macro_context_to_snapshot
from mce.features.network.engine import NetworkFeatureEngine
from mce.features.network.snapshot import apply_network_to_snapshot
from mce.features.options_context.engine import OptionsContextFeatureEngine
from mce.features.options_context.snapshot import apply_options_context_to_snapshot
from mce.features.regime import MarketRegimeEngine, apply_regime_to_snapshot, evidence_from_features
from mce.features.regime_transition.engine import RegimeTransitionAnalyticsEngine
from mce.features.regime_transition.snapshot import apply_regime_transition_to_snapshot
from mce.features.seasonality.engine import IntradaySeasonalityEngine
from mce.features.seasonality.models import IntradaySeasonalityFeatureSet
from mce.features.seasonality.snapshot import apply_seasonality_to_snapshot
from mce.features.sectors.engine import SectorFeatureEngine
from mce.features.sectors.snapshot import apply_sectors_to_snapshot
from mce.features.snapshot import apply_trend_to_snapshot
from mce.features.stress.engine import LiquidityStressFeatureEngine
from mce.features.stress.snapshot import apply_liquidity_stress_to_snapshot
from mce.features.volatility.engine import RealizedVolatilityEngine
from mce.features.volatility.snapshot import apply_volatility_to_snapshot
from mce.features.windows import FeatureWindowReader
from mce.models.context import MarketContextSnapshot, SourceFreshness
from mce.models.enums import DataQuality, QualityFlag
from mce.orchestration.context import SuppliedContexts
from mce.orchestration.dependencies import STAGES, PipelineProviders, StageNode, topological_order
from mce.orchestration.models import (
    FailurePolicy,
    PipelineDiagnostics,
    PipelineError,
    PipelineErrorCode,
    PipelineResult,
    PipelineStageDiagnostic,
    SnapshotLineage,
    StageStatus,
    snapshot_fingerprint,
)
from mce.replay.store import CausalDataStore
from mce.reproducibility import config_fingerprint
from mce.serialization.canonical import canonical_json_bytes
from mce.version import SCHEMA_VERSION, __version__


class _QualityResult(Protocol):
    @property
    def data_quality(self) -> DataQuality: ...


ResultT = TypeVar("ResultT", bound=_QualityResult)
_QUALITY_ORDER = {
    DataQuality.GOOD: 0,
    DataQuality.UNKNOWN: 1,
    DataQuality.DEGRADED: 2,
    DataQuality.POOR: 3,
    DataQuality.UNUSABLE: 4,
}


class MarketContextPipeline:
    """Stateless per call. Optional inputs and membership adapters are explicit."""

    def __init__(
        self,
        config: MCEConfig,
        providers: PipelineProviders | None = None,
        *,
        stages: tuple[StageNode, ...] = STAGES,
    ) -> None:
        self.config = config
        self.providers = providers or PipelineProviders()
        validated = topological_order(stages)
        self.stage_order = tuple(item.stage_id for item in STAGES)
        if set(validated) != set(self.stage_order):
            raise ValueError("pipeline requires the complete stage graph")
        positions = {name: position for position, name in enumerate(self.stage_order)}
        if any(
            positions[dependency] >= positions[node.stage_id]
            for node in stages
            for dependency in node.depends_on
        ):
            raise ValueError("pipeline execution order violates stage dependencies")
        if not config.market.universe.broad_market:
            raise ValueError("pipeline requires a broad_market universe")
        if not config.features.trend.enabled or not config.features.volatility.enabled:
            raise ValueError("pipeline requires trend and volatility for the base snapshot")
        if config.features.network.enabled and not config.features.correlation.enabled:
            raise ValueError("network requires correlation")
        self.calendar = MarketSessionService(config.market)
        self.fingerprint = config_fingerprint(config)

    @staticmethod
    def _record(
        diagnostics: list[PipelineStageDiagnostic],
        stage: str,
        policy: FailurePolicy,
        action: Callable[[], ResultT] | None,
        *,
        missing_reason: str = "NOT_CONFIGURED",
    ) -> ResultT | None:
        if action is None:
            diagnostics.append(
                PipelineStageDiagnostic(
                    stage_id=stage,
                    status=StageStatus.WITHHELD,
                    policy=policy,
                    duration_ns=0,
                    reason_code=missing_reason,
                )
            )
            return None
        started = perf_counter_ns()
        result = action()
        duration = perf_counter_ns() - started
        quality = result.data_quality
        status = (
            StageStatus.WITHHELD
            if quality in (DataQuality.UNKNOWN, DataQuality.UNUSABLE)
            else StageStatus.DEGRADED
            if quality in (DataQuality.POOR, DataQuality.DEGRADED)
            else StageStatus.OK
        )
        diagnostics.append(
            PipelineStageDiagnostic(
                stage_id=stage,
                status=status,
                policy=policy,
                duration_ns=duration,
                output_quality=quality,
                reason_code="INSUFFICIENT_EVIDENCE" if status == StageStatus.WITHHELD else None,
            )
        )
        return (
            result
            if status != StageStatus.WITHHELD or policy == FailurePolicy.FAIL_SNAPSHOT
            else None
        )

    def compute_snapshot(
        self,
        as_of_ns: int,
        store: CausalDataStore,
        supplied: SuppliedContexts | None = None,
    ) -> PipelineResult:
        if as_of_ns < 0:
            raise PipelineError(PipelineErrorCode.INVALID_AS_OF, "as_of_ns must be nonnegative")
        supplied = supplied or SuppliedContexts()
        providers = self.providers.for_snapshot()
        started = perf_counter_ns()
        diagnostics: list[PipelineStageDiagnostic] = []
        reader = FeatureWindowReader(
            store,
            self.calendar,
            self.config.features.trend.bar_cadence_ns,
            self.config.provider.source,
        )
        latest = tuple(
            bar
            for symbol in self.config.market.universe.broad_market
            if (bar := reader.current_bar(symbol, as_of_ns)) is not None
        )
        if not latest:
            raise PipelineError(
                PipelineErrorCode.INSUFFICIENT_MARKET_DATA,
                "no completed regular-session broad-market bar is available",
            )
        freshest = max(latest, key=lambda item: item.effective_timestamp_ns)
        freshness = SourceFreshness(
            source=freshest.source,
            latest_effective_timestamp_ns=freshest.effective_timestamp_ns,
            age_ns=as_of_ns - freshest.effective_timestamp_ns,
            is_stale=as_of_ns - freshest.effective_timestamp_ns
            > self.config.freshness.bar_limit_ns,
            freshness_limit_ns=self.config.freshness.bar_limit_ns,
        )
        session = self.calendar.session_for_timestamp(as_of_ns)

        def prior_bars_exist(
            configured: tuple[str, ...], provider: UniverseProvider | None
        ) -> bool:
            if session.trading_date is None:
                return False
            opened, _ = self.calendar.regular_session_bounds(session.trading_date)
            symbols = (
                tuple(item.symbol for item in provider.members(as_of_ns).members)
                if provider is not None
                else configured
            )
            return any(store.bars(symbol, 0, opened, as_of_ns) for symbol in symbols)

        flags: set[QualityFlag] = set()
        if freshness.is_stale:
            flags.add(QualityFlag.SOURCE_STALE)
        if len(latest) < len(self.config.market.universe.broad_market):
            flags.add(QualityFlag.PARTIAL_UNIVERSE)
        base = MarketContextSnapshot(
            as_of_ns=as_of_ns,
            market_session=session.session,
            session_trading_date=session.trading_date,
            session_identifier=session.trading_date.isoformat() if session.trading_date else None,
            source_freshness=(freshness,),
            data_quality=DataQuality.DEGRADED if flags else DataQuality.GOOD,
            quality_flags=frozenset(flags),
        )
        required = FailurePolicy.FAIL_SNAPSHOT
        optional = FailurePolicy.WITHHOLD_FEATURE
        trend = self._record(
            diagnostics,
            "trend",
            required,
            lambda: TrendFeatureEngine(self.config, self.calendar).compute(store, as_of_ns),
        )
        volatility = self._record(
            diagnostics,
            "volatility",
            required,
            lambda: RealizedVolatilityEngine(self.config, self.calendar).compute(store, as_of_ns),
        )
        breadth = self._record(
            diagnostics,
            "breadth",
            optional,
            (
                lambda: BreadthFeatureEngine(
                    self.config, providers.breadth_membership, self.calendar
                ).compute(store, as_of_ns)
            )
            if self.config.features.breadth.enabled
            and (providers.breadth_membership is not None or self.config.market.universe.breadth)
            else None,
        )
        sectors = self._record(
            diagnostics,
            "sectors",
            optional,
            (
                lambda: SectorFeatureEngine(
                    self.config, providers.sector_membership, self.calendar
                ).compute(store, as_of_ns)
            )
            if self.config.features.sectors.enabled
            and (providers.sector_membership is not None or self.config.features.sectors.groups)
            else None,
        )
        correlation = self._record(
            diagnostics,
            "correlation",
            optional,
            (
                lambda: CorrelationFeatureEngine(
                    self.config,
                    providers.correlation_membership,
                    providers.sector_membership,
                    self.calendar,
                ).compute(store, as_of_ns)
            )
            if self.config.features.correlation.enabled
            and (
                providers.correlation_membership is not None
                or self.config.features.correlation.universe
            )
            else None,
        )
        dispersion = self._record(
            diagnostics,
            "dispersion",
            optional,
            (
                lambda: DispersionFeatureEngine(
                    self.config,
                    providers.dispersion_membership,
                    providers.sector_membership,
                    self.calendar,
                ).compute(
                    store,
                    as_of_ns,
                    sector_context=sectors,
                    volatility_context=volatility,
                    correlation_context=correlation,
                )
            )
            if self.config.features.dispersion.enabled
            and (
                providers.dispersion_membership is not None
                or self.config.features.dispersion.universe
            )
            else None,
        )
        factor_configured = self.config.features.factor.enabled and (
            providers.factor_membership is not None or self.config.features.factor.universe
        )
        factors = self._record(
            diagnostics,
            "factors",
            optional,
            (
                lambda: FactorFeatureEngine(
                    self.config,
                    providers.factor_membership,
                    providers.sector_membership,
                    self.calendar,
                ).compute(store, as_of_ns, previous=supplied.previous_factor)
            )
            if factor_configured
            and prior_bars_exist(self.config.features.factor.universe, providers.factor_membership)
            else None,
            missing_reason="INSUFFICIENT_PRIOR_HISTORY" if factor_configured else "NOT_CONFIGURED",
        )
        network = self._record(
            diagnostics,
            "network",
            optional,
            (
                lambda: NetworkFeatureEngine(self.config, providers.sector_membership).compute(
                    correlation, previous=supplied.previous_network
                )
            )
            if self.config.features.network.enabled and correlation is not None
            else None,
            missing_reason=(
                "CORRELATION_UNAVAILABLE"
                if self.config.features.network.enabled
                else "NOT_CONFIGURED"
            ),
        )

        def compute_seasonality() -> IntradaySeasonalityFeatureSet:
            engine = IntradaySeasonalityEngine(
                self.config, providers.seasonality_membership, self.calendar
            )
            history = supplied.seasonality_history or engine.build_history(store, as_of_ns)
            return engine.compute(store, as_of_ns, history, correlation=correlation)

        seasonal_configured = self.config.features.seasonality.enabled and (
            providers.seasonality_membership is not None
            or self.config.features.seasonality.universe
        )
        seasonality = self._record(
            diagnostics,
            "seasonality",
            optional,
            compute_seasonality
            if seasonal_configured
            and session.is_trading_day
            and (
                supplied.seasonality_history is not None
                or prior_bars_exist(
                    self.config.features.seasonality.universe,
                    providers.seasonality_membership,
                )
            )
            else None,
            missing_reason="INSUFFICIENT_PRIOR_HISTORY"
            if seasonal_configured
            else "NOT_CONFIGURED",
        )
        liquidity_stress = self._record(
            diagnostics,
            "liquidity_stress",
            optional,
            (
                lambda: LiquidityStressFeatureEngine(
                    self.config, providers.liquidity_membership, self.calendar
                ).compute(
                    store,
                    as_of_ns,
                    volatility_context=volatility,
                    breadth_context=breadth,
                    sector_context=sectors,
                    correlation_context=correlation,
                    dispersion_context=dispersion,
                )
            )
            if (self.config.features.liquidity.enabled or self.config.features.stress.enabled)
            and (
                providers.liquidity_membership is not None
                or self.config.features.liquidity.universe
            )
            else None,
        )
        regime = self._record(
            diagnostics,
            "regime",
            optional,
            (
                lambda: MarketRegimeEngine(self.config).compute(
                    evidence_from_features(
                        as_of_ns,
                        trend=trend,
                        breadth=breadth,
                        volatility=volatility,
                        correlation=correlation,
                        dispersion=dispersion,
                        sectors=sectors,
                        liquidity_stress=liquidity_stress,
                    ),
                    history=supplied.regime_history,
                )
            )
            if self.config.features.regime.enabled
            else None,
        )
        transition = self._record(
            diagnostics,
            "regime_transition",
            optional,
            (
                lambda: RegimeTransitionAnalyticsEngine(self.config, self.calendar).compute(
                    (*supplied.regime_history, regime), as_of_ns
                )
            )
            if self.config.features.regime_transition.enabled and regime is not None
            else None,
            missing_reason=(
                "REGIME_UNAVAILABLE"
                if self.config.features.regime_transition.enabled
                else "NOT_CONFIGURED"
            ),
        )
        options = self._record(
            diagnostics,
            "options",
            optional,
            (
                lambda: (
                    supplied.options_context
                    if supplied.options_context is not None
                    else OptionsContextFeatureEngine(self.config).compute(
                        supplied.options_inputs, as_of_ns, realized_volatility_context=volatility
                    )
                )
            )
            if self.config.features.options_context.enabled
            and (supplied.options_context is not None or supplied.options_inputs)
            else None,
            missing_reason=(
                "OPTIONS_SOURCE_UNAVAILABLE"
                if self.config.features.options_context.enabled
                else "NOT_CONFIGURED"
            ),
        )
        macro_inputs = supplied.macro_inputs
        macro = self._record(
            diagnostics,
            "macro",
            optional,
            (
                lambda: (
                    supplied.macro_context
                    if supplied.macro_context is not None
                    else MacroContextFeatureEngine(self.config).compute(
                        macro_inputs if macro_inputs is not None else (), as_of_ns
                    )
                )
            )
            if self.config.features.macro.enabled
            and (supplied.macro_context is not None or supplied.macro_inputs is not None)
            else None,
            missing_reason=(
                "MACRO_SOURCE_UNAVAILABLE"
                if self.config.features.macro.enabled
                else "NOT_CONFIGURED"
            ),
        )
        hmm = supplied.hmm if self.config.statistical.enabled else None
        self._record(
            diagnostics,
            "hmm",
            optional,
            (lambda: _HMMQuality(hmm)) if hmm is not None else None,
            missing_reason=(
                "MODEL_ARTIFACT_UNAVAILABLE"
                if self.config.statistical.enabled
                else "NOT_CONFIGURED"
            ),
        )
        snapshot = base
        if trend is not None:
            snapshot = apply_trend_to_snapshot(trend, snapshot)
        if volatility is not None:
            snapshot = apply_volatility_to_snapshot(volatility, snapshot)
        if breadth is not None:
            snapshot = apply_breadth_to_snapshot(breadth, snapshot)
        if sectors is not None:
            snapshot = apply_sectors_to_snapshot(sectors, snapshot)
        if correlation is not None:
            snapshot = apply_correlation_to_snapshot(correlation, snapshot)
        if dispersion is not None:
            snapshot = apply_dispersion_to_snapshot(dispersion, snapshot)
        if factors is not None:
            snapshot = apply_factors_to_snapshot(factors, snapshot)
        if network is not None:
            snapshot = apply_network_to_snapshot(network, snapshot)
        if seasonality is not None:
            snapshot = apply_seasonality_to_snapshot(seasonality, snapshot)
        if liquidity_stress is not None:
            snapshot = apply_liquidity_stress_to_snapshot(liquidity_stress, snapshot)
        if regime is not None:
            snapshot = apply_regime_to_snapshot(regime, snapshot)
        if transition is not None:
            snapshot = apply_regime_transition_to_snapshot(transition, snapshot)
        if options is not None:
            snapshot = apply_options_context_to_snapshot(options, snapshot)
        if macro is not None:
            snapshot = apply_macro_context_to_snapshot(macro, snapshot)
        ensemble_input = EnsembleContextInput(
            as_of_ns=as_of_ns,
            snapshot=snapshot,
            trend=trend,
            breadth=breadth,
            sectors=sectors,
            volatility=volatility,
            liquidity_stress=liquidity_stress,
            correlation=correlation,
            dispersion=dispersion,
            deterministic_regime=regime,
            regime_transition=transition,
            factors=factors,
            network=network,
            seasonality=seasonality,
            options=options,
            macro=macro,
            hmm=hmm,
        )
        ensemble = self._record(
            diagnostics,
            "ensemble",
            optional,
            (lambda: EnsembleContextFeatureEngine(self.config).compute(ensemble_input))
            if self.config.features.ensemble.enabled
            else None,
        )
        if ensemble is not None:
            snapshot = apply_ensemble_to_snapshot(ensemble, snapshot)
        stage_quality = [
            item.output_quality
            for item in diagnostics
            if item.output_quality is not None and item.stage_id != "hmm"
        ]
        quality = max((snapshot.data_quality, *stage_quality), key=_QUALITY_ORDER.__getitem__)
        if (
            any(
                item.status == StageStatus.WITHHELD and item.reason_code not in ("NOT_CONFIGURED",)
                for item in diagnostics
            )
            and quality == DataQuality.GOOD
        ):
            quality = DataQuality.DEGRADED
        final_flags = set(snapshot.quality_flags)
        if quality != DataQuality.GOOD:
            final_flags.add(QualityFlag.PROVIDER_DEGRADED)
        snapshot = snapshot.model_copy(
            update={"data_quality": quality, "quality_flags": frozenset(final_flags)}
        )
        source_fingerprints = tuple(
            sorted(
                (
                    name,
                    sha256(
                        canonical_json_bytes(asdict(value) if is_dataclass(value) else value)
                    ).hexdigest(),
                )
                for name, value in (
                    ("options", options),
                    ("macro", macro),
                    ("hmm", hmm),
                    ("factors", factors),
                    ("network", network),
                )
                if value is not None
            )
        )
        memberships = tuple(
            sorted(
                (name, sha256(canonical_json_bytes(provider.members(as_of_ns))).hexdigest())
                for name, provider in (
                    ("breadth", providers.breadth_membership),
                    ("sectors", providers.sector_membership),
                    ("correlation", providers.correlation_membership),
                    ("dispersion", providers.dispersion_membership),
                    ("factors", providers.factor_membership),
                    ("liquidity", providers.liquidity_membership),
                    ("seasonality", providers.seasonality_membership),
                )
                if provider is not None
            )
        )
        lineage = SnapshotLineage(
            config_fingerprint=self.fingerprint,
            market_data_fingerprint=store.causal_fingerprint(as_of_ns),
            membership_fingerprints=memberships,
            source_fingerprints=source_fingerprints,
            engine_version=__version__,
            schema_version=SCHEMA_VERSION,
        )
        operational = PipelineDiagnostics(
            as_of_ns=as_of_ns,
            duration_ns=perf_counter_ns() - started,
            stage_results=tuple(diagnostics),
            quality=quality,
            config_fingerprint=self.fingerprint,
        )
        return PipelineResult(
            snapshot=snapshot,
            lineage=lineage,
            snapshot_fingerprint=snapshot_fingerprint(snapshot, lineage),
            diagnostics=operational,
            regime_state=regime,
            factor_context=factors,
            network_context=network,
        )


class _HMMQuality:
    def __init__(self, context: object) -> None:
        self.data_quality = DataQuality.GOOD if context is not None else DataQuality.UNKNOWN
