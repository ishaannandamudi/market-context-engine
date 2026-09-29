"""Point-in-time cross-sectional return and volatility differentiation."""

from mce.features.dispersion.engine import DispersionFeatureEngine
from mce.features.dispersion.models import DispersionFeatureSet
from mce.features.dispersion.snapshot import apply_dispersion_to_snapshot

__all__ = ("DispersionFeatureEngine", "DispersionFeatureSet", "apply_dispersion_to_snapshot")
