"""Explicit optional inputs; no hidden provider or model-artifact lookup."""

from dataclasses import dataclass

from mce.features.ensemble.contracts import FilteredHMMContext
from mce.features.factors.models import FactorContext
from mce.features.macro.contracts import MacroContextInput
from mce.features.macro.history import MacroHistoryIndex
from mce.features.macro.models import MacroContextFeatureSet
from mce.features.network.models import NetworkContext
from mce.features.options_context.contracts import OptionsMarketContextInput
from mce.features.options_context.models import OptionsContextFeatureSet
from mce.features.regime.models import MarketRegimeState
from mce.features.seasonality.models import SeasonalHistoryIndex


@dataclass(frozen=True, slots=True)
class SuppliedContexts:
    options_inputs: tuple[OptionsMarketContextInput, ...] = ()
    macro_inputs: tuple[MacroContextInput, ...] | MacroHistoryIndex | None = None
    options_context: OptionsContextFeatureSet | None = None
    macro_context: MacroContextFeatureSet | None = None
    hmm: FilteredHMMContext | None = None
    regime_history: tuple[MarketRegimeState, ...] = ()
    seasonality_history: SeasonalHistoryIndex | None = None
    previous_factor: FactorContext | None = None
    previous_network: NetworkContext | None = None
