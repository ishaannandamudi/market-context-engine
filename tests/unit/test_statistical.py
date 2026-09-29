"""MCE-12 causal dataset and statistical preparation contracts."""

from dataclasses import asdict, replace
from datetime import date

import numpy as np
import pytest

from mce.calendar.service import MarketSessionService
from mce.config.settings import MCEConfig, StatisticalSettings, StatisticalSplitSettings
from mce.features.regime.models import MarketRegimeState
from mce.features.regime.taxonomy import HeadlineRegime
from mce.models.context import MarketContextSnapshot
from mce.models.enums import DataQuality, MarketRegime, QualityFlag
from mce.reproducibility import config_fingerprint
from mce.statistical.artifacts import ModelArtifactMetadata
from mce.statistical.dataset import SnapshotRecord, StatisticalDatasetBuilder
from mce.statistical.features import STATISTICAL_FEATURE_SCHEMA, StatisticalFeatureSchema
from mce.statistical.models import LatentStateInference
from mce.statistical.normalization import FittedNormalizer, fit_normalizer
from mce.statistical.regime import canonical_state_mapping, evaluate_states, relabel_states
from mce.statistical.splits import walk_forward_splits
from tests.fixtures.statistical_scenarios import synthetic_dataset


def _snapshots(
    calendar: MarketSessionService, count: int, *, start: date = date(2024, 1, 2)
) -> tuple[MarketContextSnapshot, ...]:
    day = start
    result = []
    for index in range(count):
        _, close = calendar.regular_session_bounds(day)
        result.append(
            MarketContextSnapshot(
                as_of_ns=close,
                market_direction_score=(index % 5 - 2) / 4,
                volatility_score=(index % 4) / 4,
                breadth_score=(index % 3 - 1) / 2,
                dispersion_score=None if index == 0 else (index % 3) / 4,
                correlation_score=(index % 5) / 5,
                liquidity_score=(index % 4) / 4,
                stress_score=(index % 4) / 4,
                feature_values={
                    "breadth.net_advance_fraction": (index % 5 - 2) / 4,
                    "breadth.participation": 0.5,
                    "breadth.momentum": float(index % 3),
                    "sector.leadership.spread": float(index % 4),
                    "sector.rotation.velocity": float(index % 3),
                    "correlation.first_eigen_share": 0.6,
                    "dispersion.between_sector_share": 0.3,
                    "stress.systemic": 0.4,
                    "stress.localized": 0.2,
                    "regime_transition.stability": 0.8,
                    "regime_transition.intensity": float(index % 5),
                },
                market_regime=MarketRegime.RISK_ON if index % 2 else MarketRegime.RISK_OFF,
                data_quality=DataQuality.GOOD,
            )
        )
        if index + 1 < count:
            day = calendar.next_trading_day(day)
    return tuple(result)


@pytest.fixture
def prepared() -> tuple[MCEConfig, MarketSessionService, tuple[MarketContextSnapshot, ...]]:
    config = MCEConfig(_env_file=None)
    calendar = MarketSessionService(config.market)
    return config, calendar, _snapshots(calendar, 18)


def test_feature_schema_order_ranges_and_missing(prepared: tuple) -> None:
    config, calendar, snapshots = prepared
    schema = STATISTICAL_FEATURE_SCHEMA
    assert schema.version == "1.0"
    assert len(schema.names) == 18
    assert schema.names[:3] == ("market_direction_score", "volatility_score", "breadth_score")
    assert len(schema.fingerprint) == 64
    assert schema.fingerprint == StatisticalFeatureSchema(schema.version, schema.specs).fingerprint
    vector, mask = schema.extract(snapshots[0])
    assert vector[0] == -0.5 and mask[0]
    assert vector[3] is None and not mask[3]
    assert (
        QualityFlag.MISSING_DATA
        in StatisticalDatasetBuilder(config, calendar).build((snapshots[0],)).rows[0].quality_flags
    )
    with pytest.raises(ValueError, match="required"):
        schema.extract(MarketContextSnapshot(as_of_ns=snapshots[0].as_of_ns))
    with pytest.raises(ValueError, match="finite numeric"):
        schema.extract(
            snapshots[0].model_copy(update={"feature_values": {"breadth.momentum": float("nan")}})
        )
    with pytest.raises(ValueError, match="outside expected range"):
        schema.extract(snapshots[0].model_copy(update={"volatility_score": 2.0}))
    altered = StatisticalFeatureSchema("1.0", tuple(reversed(schema.specs)))
    assert altered.fingerprint != schema.fingerprint
    with pytest.raises(ValueError, match="mismatch"):
        StatisticalDatasetBuilder(
            config.model_copy(update={"statistical": StatisticalSettings(feature_set="1.0")}),
            calendar,
            StatisticalFeatureSchema("2.0", schema.specs),
        )


def test_causal_builder_future_snapshots_late_availability_and_replay(prepared: tuple) -> None:
    config, calendar, snapshots = prepared
    builder = StatisticalDatasetBuilder(config, calendar)
    past = builder.build(snapshots[:3])
    future = snapshots[3].model_copy(update={"market_direction_score": -1.0})
    with_future = builder.build(snapshots[:3] + (future,), as_of_ns=snapshots[2].as_of_ns)
    assert past == with_future
    assert past.fingerprint == with_future.fingerprint
    assert past.rows[0].deterministic_regime == MarketRegime.RISK_OFF.value
    late = builder.build((SnapshotRecord(snapshots[0], snapshots[1].as_of_ns), snapshots[1]))
    assert len(late.rows) == 1
    assert late.rows[0].as_of_ns == snapshots[1].as_of_ns
    with pytest.raises(ValueError, match="duplicate snapshot"):
        builder.build((snapshots[0], snapshots[0]))
    with pytest.raises(ValueError, match="snapshot schema"):
        builder.build((snapshots[0].model_copy(update={"schema_version": "2.0"}),))
    with pytest.raises(ValueError, match="availability"):
        SnapshotRecord(snapshots[0], snapshots[0].as_of_ns - 1)


def test_future_regime_and_upstream_frozen_snapshot_causality(prepared: tuple) -> None:
    config, calendar, snapshots = prepared
    builder = StatisticalDatasetBuilder(config, calendar)
    future_state = MarketRegimeState.model_construct(
        as_of_ns=snapshots[1].as_of_ns,
        available_at_ns=snapshots[1].as_of_ns,
        headline_regime=HeadlineRegime.SYSTEMIC_STRESS,
    )
    base = builder.build(snapshots[:1])
    candidate = builder.build(snapshots[:1], regime_states=(future_state,))
    assert base == candidate
    assert base.rows[0].feature_vector == candidate.rows[0].feature_vector
    # Frozen, previously constructed snapshots are unaffected by later source revisions.
    revised_future = snapshots[1].model_copy(update={"feature_values": {"breadth.momentum": 99.0}})
    assert builder.build(snapshots[:1] + (revised_future,), as_of_ns=snapshots[0].as_of_ns) == base


def test_quality_and_dataset_fingerprint(prepared: tuple) -> None:
    config, calendar, snapshots = prepared
    bad = snapshots[0].model_copy(update={"data_quality": DataQuality.UNUSABLE})
    assert len(StatisticalDatasetBuilder(config, calendar).build((bad, snapshots[1])).rows) == 1
    include = config.model_copy(
        update={"statistical": StatisticalSettings(quality_policy="include")}
    )
    assert len(StatisticalDatasetBuilder(include, calendar).build((bad, snapshots[1])).rows) == 2
    one = StatisticalDatasetBuilder(config, calendar).build((snapshots[0],))
    changed = snapshots[0].model_copy(update={"market_direction_score": 0.25})
    assert (
        one.fingerprint != StatisticalDatasetBuilder(config, calendar).build((changed,)).fingerprint
    )
    assert config_fingerprint(config) != config_fingerprint(include)


def test_train_only_normalization_and_round_trip(prepared: tuple) -> None:
    config, calendar, snapshots = prepared
    dataset = StatisticalDatasetBuilder(config, calendar).build(snapshots)
    indices = tuple(range(12))
    fitted = fit_normalizer(dataset, indices, "standard")
    assert fitted.training_end_ns == snapshots[11].as_of_ns
    assert fitted.impute_medians[0] == pytest.approx(
        np.median([s.market_direction_score for s in snapshots[:12]])
    )
    changed = snapshots[-1].model_copy(update={"market_direction_score": 1.0})
    changed_data = StatisticalDatasetBuilder(config, calendar).build(snapshots[:-1] + (changed,))
    assert fit_normalizer(changed_data, indices, "standard") == fitted
    missing_future = snapshots[-1].model_copy(update={"dispersion_score": None})
    future_data = StatisticalDatasetBuilder(config, calendar).build(
        snapshots[:-1] + (missing_future,)
    )
    future_values, future_mask = fitted.transform(future_data.take((17,)))
    assert not future_mask[0, 3]
    assert future_values[0, 3] == pytest.approx(
        (fitted.impute_medians[3] - fitted.centers[3]) / fitted.scales[3]
    )
    values, masks = fitted.transform(dataset.take((12, 13)))
    assert values.shape == (2, 18) and masks.shape == (2, 18)
    _, initial_mask = fitted.transform(dataset.take((0,)))
    assert not initial_mask[0, 3]
    assert FittedNormalizer(**asdict(fitted)) == fitted
    assert fitted.fingerprint == FittedNormalizer(**asdict(fitted)).fingerprint
    robust = fit_normalizer(dataset, indices, "robust")
    assert robust.centers[0] == fitted.impute_medians[0]
    assert robust.scales[0] > 0
    assert robust.constant_features[8] and robust.scales[8] == 1.0
    none = fit_normalizer(dataset, indices, "none")
    assert none.centers[0] == 0.0 and none.scales[0] == 1.0
    with pytest.raises(ValueError, match="schema mismatch"):
        fitted.transform(
            replace(dataset, schema=StatisticalFeatureSchema("2.0", dataset.schema.specs))
        )
    with pytest.raises(ValueError, match="configuration mismatch"):
        fitted.transform(replace(dataset, config_fingerprint="f" * 64))
    with pytest.raises(ValueError, match="observed"):
        fit_normalizer(dataset, (0,), "standard")
    with pytest.raises(ValueError, match="indices"):
        fit_normalizer(dataset, (1, 0))


def test_session_walk_forward_embargo_and_modes(prepared: tuple) -> None:
    config, calendar, snapshots = prepared
    dataset = StatisticalDatasetBuilder(config, calendar).build(snapshots)
    settings = StatisticalSplitSettings(
        train_sessions=5,
        validation_sessions=2,
        test_sessions=2,
        embargo_sessions=1,
        step_sessions=2,
    )
    expanding = walk_forward_splits(dataset, calendar, settings)
    assert len(expanding) >= 2
    first, second = expanding[:2]
    assert len(first.train_indices) == 5 and len(second.train_indices) == 7
    assert first.train_indices[-1] < first.validation_indices[0] < first.test_indices[0]
    assert first.validation_indices[0] - first.train_indices[-1] == 2
    assert first.test_indices[0] - first.validation_indices[-1] == 2
    assert first.fingerprint == walk_forward_splits(dataset, calendar, settings)[0].fingerprint
    assert first.train_dates[0] == date(2024, 1, 2)
    rolling = walk_forward_splits(
        dataset, calendar, settings.model_copy(update={"mode": "rolling"})
    )
    assert len(rolling[0].train_indices) == len(rolling[1].train_indices) == 5
    assert rolling[1].train_indices[0] == 2
    with pytest.raises(ValueError, match="insufficient"):
        walk_forward_splits(dataset.take(tuple(range(5))), calendar, settings)


def test_split_counts_exchange_sessions_over_weekend_and_holiday() -> None:
    config = MCEConfig(_env_file=None)
    calendar = MarketSessionService(config.market)
    snapshots = _snapshots(calendar, 5, start=date(2024, 1, 12))
    assert [calendar.trading_date_for_timestamp(s.as_of_ns) for s in snapshots] == [
        date(2024, 1, 12),
        date(2024, 1, 16),
        date(2024, 1, 17),
        date(2024, 1, 18),
        date(2024, 1, 19),
    ]
    dataset = StatisticalDatasetBuilder(config, calendar).build(snapshots)
    split = walk_forward_splits(
        dataset,
        calendar,
        StatisticalSplitSettings(
            train_sessions=1,
            validation_sessions=1,
            test_sessions=1,
            embargo_sessions=1,
        ),
    )[0]
    assert split.train_dates == (date(2024, 1, 12), date(2024, 1, 12))
    assert split.validation_dates == (date(2024, 1, 17), date(2024, 1, 17))
    assert split.test_dates == (date(2024, 1, 19), date(2024, 1, 19))


def test_state_alignment_profiles_and_evaluation(prepared: tuple) -> None:
    config, calendar, snapshots = prepared
    dataset = StatisticalDatasetBuilder(config, calendar).build(snapshots[:6])
    states = (2, 2, 2, 7, 7, 2)
    evaluation = evaluate_states(dataset, states)
    assert sum(p.observations for p in evaluation.profiles) == 6
    assert evaluation.persistence_fraction == pytest.approx(0.6)
    assert evaluation.transition_counts == ((2, 2, 2), (2, 7, 1), (7, 2, 1), (7, 7, 1))
    assert evaluation.dwell_lengths == ((2, (3, 1)), (7, (2,)))
    assert evaluation.weighted_exit_entropy_nats == pytest.approx(0.0)
    assert evaluation.deterministic_purity is not None
    assert sum(value for _, value in evaluation.session_weighted_occupancy) == pytest.approx(1.0)
    mapping = canonical_state_mapping(evaluation.profiles, dataset.schema.names)
    assert set(mapping.values()) == {0, 1}
    assert relabel_states(states, mapping) == tuple(mapping[state] for state in states)
    permuted = tuple(7 if state == 2 else 2 for state in states)
    other = evaluate_states(dataset, permuted)
    canon_other = canonical_state_mapping(other.profiles, dataset.schema.names)
    assert relabel_states(states, mapping) == relabel_states(permuted, canon_other)
    with pytest.raises(ValueError, match="unmapped"):
        relabel_states((10,), mapping)
    with pytest.raises(ValueError, match="one-to-one"):
        evaluate_states(dataset, (1,))


def test_artifact_reproducibility_and_inference_guard(prepared: tuple) -> None:
    config, calendar, snapshots = prepared
    dataset = StatisticalDatasetBuilder(config, calendar).build(snapshots)
    split = walk_forward_splits(
        dataset,
        calendar,
        StatisticalSplitSettings(
            train_sessions=5,
            validation_sessions=2,
            test_sessions=2,
            embargo_sessions=1,
        ),
    )[0]
    normalizer = fit_normalizer(dataset, split.train_indices)
    kwargs = {
        "model_type": "example",
        "model_version": "0",
        "config_fingerprint": dataset.config_fingerprint,
        "dataset_fingerprint": dataset.fingerprint,
        "split_fingerprint": split.fingerprint,
        "normalizer": normalizer,
        "random_seed": 7,
        "parameters": {"means": [[1.0, 2.0]]},
    }
    artifact = ModelArtifactMetadata.from_fit(**kwargs)
    assert artifact.fingerprint == ModelArtifactMetadata.from_fit(**kwargs).fingerprint
    assert (
        artifact.parameter_hash
        != ModelArtifactMetadata.from_fit(
            **(kwargs | {"parameters": {"means": [[2.0, 1.0]]}})
        ).parameter_hash
    )
    with pytest.raises(ValueError, match="configuration mismatch"):
        ModelArtifactMetadata.from_fit(**(kwargs | {"config_fingerprint": "f" * 64}))
    with pytest.raises(ValueError, match="retrospective"):
        LatentStateInference(
            snapshots[0].as_of_ns, 0, (1.0,), "filtered", False, snapshots[5].as_of_ns, "0", "1.0"
        )
    with pytest.raises(ValueError, match="retrospective"):
        LatentStateInference(
            snapshots[10].as_of_ns, 0, (1.0,), "smoothed", False, snapshots[5].as_of_ns, "0", "1.0"
        )
    valid = LatentStateInference(
        snapshots[10].as_of_ns, 0, (1.0,), "filtered", False, snapshots[5].as_of_ns, "0", "1.0"
    )
    assert not valid.retrospective


@pytest.mark.parametrize(
    "scenario", ["two_state", "three_state", "overlapping", "rapid", "long_dwell"]
)
def test_synthetic_scenarios_quality_missing_and_switching(scenario: str) -> None:
    dataset, states = synthetic_dataset(scenario)
    assert len(dataset.rows) == len(states) == 30
    assert dataset.rows[0].source_quality == DataQuality.DEGRADED
    assert not dataset.rows[0].feature_mask[9]
    assert dataset.fingerprint == synthetic_dataset(scenario)[0].fingerprint
    evaluation = evaluate_states(dataset, states)
    assert sum(profile.occupancy_fraction for profile in evaluation.profiles) == pytest.approx(1)
    if scenario == "rapid":
        assert evaluation.persistence_fraction == 0
    if scenario == "long_dwell":
        assert evaluation.persistence_fraction is not None
        assert evaluation.persistence_fraction > 0.9


def test_normalizer_and_inference_reject_invalid_artifacts(prepared: tuple) -> None:
    config, calendar, snapshots = prepared
    dataset = StatisticalDatasetBuilder(config, calendar).build(snapshots)
    fitted = fit_normalizer(dataset, tuple(range(8)))
    with pytest.raises(ValueError, match="scales"):
        FittedNormalizer(**(asdict(fitted) | {"scales": (0.0,) + fitted.scales[1:]}))
    with pytest.raises(ValueError, match="vectors"):
        FittedNormalizer(**(asdict(fitted) | {"centers": (0.0,)}))
    with pytest.raises(ValueError, match="probabilities"):
        LatentStateInference(
            snapshots[10].as_of_ns,
            0,
            (float("nan"),),
            "filtered",
            False,
            snapshots[7].as_of_ns,
            "0",
            "1.0",
        )
    with pytest.raises(ValueError, match="sum to one"):
        LatentStateInference(
            snapshots[10].as_of_ns,
            0,
            (0.2, 0.2),
            "filtered",
            False,
            snapshots[7].as_of_ns,
            "0",
            "1.0",
        )


def test_alignment_ties_and_exit_entropy(prepared: tuple) -> None:
    config, calendar, snapshots = prepared
    dataset = StatisticalDatasetBuilder(config, calendar).build(snapshots[:8])
    states = (1, 1, 2, 1, 3, 1, 2, 1)
    evaluation = evaluate_states(dataset, states)
    entropies = dict(evaluation.exit_entropy_nats)
    assert entropies[1] == pytest.approx(-(2 / 3) * np.log(2 / 3) - (1 / 3) * np.log(1 / 3))
    assert entropies[2] == 0
    assert entropies[3] == 0
    assert evaluation.weighted_exit_entropy_nats is not None
    assert evaluation.weighted_exit_entropy_nats > 0
    tied = tuple(replace(profile, centroid=(0.0,) * 18) for profile in evaluation.profiles)
    assert canonical_state_mapping(tied, dataset.schema.names) == {1: 0, 2: 1, 3: 2}
    with pytest.raises(ValueError, match="unique"):
        canonical_state_mapping((tied[0], tied[0]), dataset.schema.names)
    with pytest.raises(ValueError, match="schema"):
        canonical_state_mapping((replace(tied[0], centroid=(1.0,)),), dataset.schema.names)
    with pytest.raises(ValueError, match="features"):
        canonical_state_mapping((tied[0],), ("x",) * 18)


def test_empty_transform_empty_split_and_invariant_guards(prepared: tuple) -> None:
    config, calendar, snapshots = prepared
    dataset = StatisticalDatasetBuilder(config, calendar).build(snapshots)
    fitted = fit_normalizer(dataset, tuple(range(8)))
    values, masks = fitted.transform(dataset.take(()))
    assert values.shape == masks.shape == (0, 18)
    with pytest.raises(ValueError, match="empty"):
        walk_forward_splits(dataset.take(()), calendar, StatisticalSplitSettings())
    with pytest.raises(IndexError, match="range"):
        dataset.take((len(dataset.rows),))
    with pytest.raises(ValueError, match="feature vector"):
        replace(dataset.rows[0], feature_vector=(0.0,))
    with pytest.raises(ValueError, match="mask"):
        replace(dataset.rows[0], feature_mask=(False,) * 18)
    with pytest.raises(ValueError, match="finite"):
        replace(
            dataset.rows[0], feature_vector=(float("nan"),) + dataset.rows[0].feature_vector[1:]
        )
    with pytest.raises(ValueError, match="required"):
        replace(
            dataset,
            rows=(
                replace(
                    dataset.rows[0],
                    feature_vector=(None,) + dataset.rows[0].feature_vector[1:],
                    feature_mask=(False,) + dataset.rows[0].feature_mask[1:],
                ),
            ),
        )
