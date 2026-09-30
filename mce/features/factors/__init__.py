"""Causal rolling PCA and descriptive latent market structure."""

from mce.features.factors.dynamics import align_factors
from mce.features.factors.engine import FactorFeatureEngine
from mce.features.factors.exposures import (
    compare_factor_contexts,
    compare_with_latent_states,
    sector_exposures,
)
from mce.features.factors.models import FactorArtifact, FactorContext, ReturnPanel
from mce.features.factors.panel import FactorPanelBuilder
from mce.features.factors.pca import fit_pca, transform
from mce.features.factors.rolling import factor_history, walk_forward_factors
from mce.features.factors.snapshot import apply_factors_to_snapshot

__all__ = [
    "FactorArtifact",
    "FactorContext",
    "FactorFeatureEngine",
    "FactorPanelBuilder",
    "ReturnPanel",
    "align_factors",
    "apply_factors_to_snapshot",
    "compare_factor_contexts",
    "compare_with_latent_states",
    "factor_history",
    "fit_pca",
    "sector_exposures",
    "transform",
    "walk_forward_factors",
]
