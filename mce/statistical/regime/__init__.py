"""Statistical latent regimes and descriptive model evaluation."""

from mce.statistical.regime.alignment import canonical_state_mapping, relabel_states
from mce.statistical.regime.changepoint import CausalChangeScore, ChangePoint, MeanShiftDetector
from mce.statistical.regime.clustering import KMeansArtifact, KMeansRegimeModel
from mce.statistical.regime.evaluation import RegimeEvaluation, evaluate_states
from mce.statistical.regime.hmm import GaussianHMMRegimeModel, HMMArtifact, HMMQuality
from mce.statistical.regime.selection import (
    HMMSelectionEntry,
    KMeansSelectionEntry,
    RegimeComparison,
    compare_assignments,
    compare_hmm_state_counts,
    compare_kmeans_cluster_counts,
)
from mce.statistical.regime.walkforward import WalkForwardHMMFold, run_walk_forward_hmm

__all__ = [
    "CausalChangeScore",
    "ChangePoint",
    "GaussianHMMRegimeModel",
    "HMMArtifact",
    "HMMQuality",
    "HMMSelectionEntry",
    "KMeansArtifact",
    "KMeansRegimeModel",
    "KMeansSelectionEntry",
    "MeanShiftDetector",
    "RegimeComparison",
    "RegimeEvaluation",
    "WalkForwardHMMFold",
    "canonical_state_mapping",
    "compare_assignments",
    "compare_hmm_state_counts",
    "compare_kmeans_cluster_counts",
    "evaluate_states",
    "relabel_states",
    "run_walk_forward_hmm",
]
