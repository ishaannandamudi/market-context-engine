"""Freeze each training fit before sequential, causal validation/test filtering."""

from dataclasses import dataclass
from hashlib import sha256

from mce.calendar.service import MarketSessionService
from mce.config.settings import HMMSettings, StatisticalSplitSettings
from mce.serialization.canonical import canonical_json_bytes
from mce.statistical.dataset import StatisticalDataset
from mce.statistical.models import LatentStateInference
from mce.statistical.normalization import fit_normalizer
from mce.statistical.regime.hmm import GaussianHMMRegimeModel, HMMQuality
from mce.statistical.splits import walk_forward_splits


@dataclass(frozen=True, slots=True)
class WalkForwardHMMFold:
    fold_index: int
    train_end_ns: int
    training_fingerprint: str
    normalizer_fingerprint: str
    artifact_fingerprint: str
    quality: HMMQuality
    validation: tuple[LatentStateInference, ...]
    test: tuple[LatentStateInference, ...]


def run_walk_forward_hmm(
    dataset: StatisticalDataset,
    calendar: MarketSessionService,
    split_settings: StatisticalSplitSettings,
    hmm_settings: HMMSettings,
    *,
    random_seed: int,
) -> tuple[WalkForwardHMMFold, ...]:
    if random_seed < 0:
        raise ValueError("random seed must be nonnegative")
    splits = walk_forward_splits(dataset, calendar, split_settings)
    result: list[WalkForwardHMMFold] = []
    for fold_index, split in enumerate(splits):
        training = dataset.take(split.train_indices)
        normalizer = fit_normalizer(dataset, split.train_indices)
        # Exclude future validation/test content from the frozen artifact identity.
        training_split_fingerprint = sha256(
            canonical_json_bytes(
                {
                    "training": training.fingerprint,
                    "dates": split.train_dates,
                    "spec": split.spec_fingerprint,
                }
            )
        ).hexdigest()
        model = GaussianHMMRegimeModel(hmm_settings)
        artifact = model.fit(
            training,
            normalizer,
            random_seed=random_seed + fold_index * 1000,
            split_fingerprint=training_split_fingerprint,
        )
        validation = dataset.take(split.validation_indices)
        test = dataset.take(split.test_indices)
        # Crossing an embargo or unobserved gap resets to the fitted initial belief.
        prior = (
            artifact.training_end_filtered
            if (split.train_indices[-1] + 1 == split.validation_indices[0])
            else None
        )
        validation_outputs = model.infer(validation, mode="filtered", previous_filtered=prior)
        test_prior = (
            validation_outputs[-1].state_probabilities
            if (split.validation_indices[-1] + 1 == split.test_indices[0])
            else None
        )
        test_outputs = model.infer(test, mode="filtered", previous_filtered=test_prior)
        if any(not item.causal for item in validation_outputs + test_outputs):
            raise ValueError("walk-forward HMM produced noncausal inference")
        result.append(
            WalkForwardHMMFold(
                fold_index,
                training.rows[-1].as_of_ns,
                training.fingerprint,
                normalizer.fingerprint,
                artifact.fingerprint,
                model.quality(training),
                validation_outputs,
                test_outputs,
            )
        )
    return tuple(result)
