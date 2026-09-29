"""Causal research datasets and contracts for future statistical regimes."""

from mce.statistical.dataset import (
    StatisticalDataset,
    StatisticalDatasetBuilder,
    StatisticalObservation,
)
from mce.statistical.features import STATISTICAL_FEATURE_SCHEMA, StatisticalFeatureSchema
from mce.statistical.normalization import FittedNormalizer, fit_normalizer
from mce.statistical.splits import TemporalSplit, walk_forward_splits

__all__ = [
    "FittedNormalizer",
    "STATISTICAL_FEATURE_SCHEMA",
    "StatisticalDataset",
    "StatisticalDatasetBuilder",
    "StatisticalFeatureSchema",
    "StatisticalObservation",
    "TemporalSplit",
    "fit_normalizer",
    "walk_forward_splits",
]
