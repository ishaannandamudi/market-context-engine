"""Typed, same-clock inputs to descriptive ensemble synthesis."""

from pydantic import InstanceOf, model_validator

from mce.features.breadth.models import BreadthFeatureSet
from mce.features.correlation.models import CorrelationFeatureSet
from mce.features.dispersion.models import DispersionFeatureSet
from mce.features.factors.models import FactorContext
from mce.features.macro.models import MacroContextFeatureSet
from mce.features.models import TrendFeatureSet
from mce.features.network.models import NetworkContext
from mce.features.options_context.models import OptionsContextFeatureSet
from mce.features.regime.models import MarketRegimeState
from mce.features.regime_transition.models import RegimeTransitionFeatureSet
from mce.features.seasonality.models import IntradaySeasonalityFeatureSet
from mce.features.sectors.models import SectorFeatureSet
from mce.features.stress.models import LiquidityStressFeatureSet
from mce.features.volatility.models import RealizedVolatilityFeatureSet
from mce.models._base import FrozenModel
from mce.models.context import MarketContextSnapshot
from mce.models.quality import DirectionalScore, UnitScore
from mce.models.time import TimestampNs
from mce.statistical.features import STATISTICAL_FEATURE_SCHEMA
from mce.statistical.models import LatentStateInference, LatentStateProfile
from mce.statistical.regime.hmm import HMMQuality


class FilteredHMMContext(FrozenModel):
    """Explicit profile interpretation; a numeric state ID has no market meaning."""

    inference: InstanceOf[LatentStateInference]
    profile: InstanceOf[LatentStateProfile] | None = None
    quality: InstanceOf[HMMQuality] | None = None
    direction: DirectionalScore | None = None
    stress: UnitScore | None = None

    @model_validator(mode="after")
    def validate_filtered(self) -> "FilteredHMMContext":
        inference = self.inference
        if not inference.causal or inference.training_end_ns >= inference.as_of_ns:
            raise ValueError("ensemble requires out-of-sample filtered HMM inference")
        if inference.feature_schema_version != STATISTICAL_FEATURE_SCHEMA.version:
            raise ValueError("incompatible HMM feature schema")
        if self.profile is not None and self.profile.state_id != inference.state_id:
            raise ValueError("HMM profile must describe the inferred state")
        if self.profile is not None and not 0 <= self.profile.occupancy_fraction <= 1:
            raise ValueError("HMM profile occupancy must be in [0,1]")
        if (self.direction is not None or self.stress is not None) and self.profile is None:
            raise ValueError("HMM state interpretation requires a matching profile")
        return self


class EnsembleContextInput(FrozenModel):
    as_of_ns: TimestampNs
    snapshot: MarketContextSnapshot
    trend: TrendFeatureSet | None = None
    breadth: BreadthFeatureSet | None = None
    sectors: SectorFeatureSet | None = None
    volatility: RealizedVolatilityFeatureSet | None = None
    liquidity_stress: LiquidityStressFeatureSet | None = None
    correlation: CorrelationFeatureSet | None = None
    dispersion: DispersionFeatureSet | None = None
    deterministic_regime: MarketRegimeState | None = None
    regime_transition: RegimeTransitionFeatureSet | None = None
    hmm: FilteredHMMContext | None = None
    factors: InstanceOf[FactorContext] | None = None
    network: InstanceOf[NetworkContext] | None = None
    seasonality: InstanceOf[IntradaySeasonalityFeatureSet] | None = None
    options: OptionsContextFeatureSet | None = None
    macro: MacroContextFeatureSet | None = None

    @model_validator(mode="after")
    def validate_clocks(self) -> "EnsembleContextInput":
        if self.snapshot.as_of_ns != self.as_of_ns:
            raise ValueError("ensemble snapshot and calculation clocks must match")
        for name in (
            "trend",
            "breadth",
            "sectors",
            "volatility",
            "liquidity_stress",
            "correlation",
            "dispersion",
            "deterministic_regime",
            "regime_transition",
            "factors",
            "network",
            "seasonality",
            "options",
            "macro",
        ):
            item = getattr(self, name)
            if item is not None and item.as_of_ns > self.as_of_ns:
                raise ValueError(f"future {name} context exceeds ensemble clock")
        if (
            self.deterministic_regime is not None
            and self.deterministic_regime.available_at_ns > self.as_of_ns
        ):
            raise ValueError("future deterministic regime availability")
        if self.hmm is not None and self.hmm.inference.as_of_ns > self.as_of_ns:
            raise ValueError("future HMM filtered state")
        if (
            self.factors is not None
            and self.factors.artifact is not None
            and not self.factors.causal
        ):
            raise ValueError("future factor fit")
        if (
            self.network is not None
            and self.network.graph is not None
            and self.network.graph.as_of_ns > self.as_of_ns
        ):
            raise ValueError("future network graph")
        if self.options is not None and any(
            item.source.available_at_ns > self.as_of_ns for item in self.options.per_underlying
        ):
            raise ValueError("future options source availability")
        if self.macro is not None and any(
            item.available_at_ns > self.as_of_ns for item in self.macro.sources
        ):
            raise ValueError("future macro source availability")
        if self.macro is not None:
            for series in self.macro.per_series:
                if (
                    series.observation is not None
                    and series.observation.available_at_ns > self.as_of_ns
                ) or (
                    series.release is not None and series.release.available_at_ns > self.as_of_ns
                ):
                    raise ValueError("future macro observation or release")
            event = self.macro.event
            if event is not None:
                if any(
                    item is not None and item.calendar_available_at_ns > self.as_of_ns
                    for item in (event.next_event, event.next_high_importance)
                ):
                    raise ValueError("future macro calendar announcement")
                if (
                    event.recent_release is not None
                    and event.recent_release.available_at_ns > self.as_of_ns
                ):
                    raise ValueError("future macro release")
        return self
