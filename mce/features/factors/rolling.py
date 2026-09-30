"""Explicit-state rolling history and session-based frozen factor evaluation."""

from dataclasses import dataclass
from datetime import date
from zoneinfo import ZoneInfo

import numpy as np

from mce.calendar.service import MarketSessionService
from mce.config.settings import FactorSettings, MCEConfig, StatisticalSplitSettings
from mce.features.factors.engine import FactorFeatureEngine
from mce.features.factors.models import FactorArtifact, FactorContext, ReturnPanel
from mce.features.factors.pca import fit_pca, transform
from mce.models.time import ns_to_datetime
from mce.replay.store import CausalDataStore
from mce.reproducibility import config_fingerprint


def factor_history(
    engine: FactorFeatureEngine, store: CausalDataStore, as_of_ns: tuple[int, ...]
) -> tuple[FactorContext, ...]:
    if any(a >= b for a, b in zip(as_of_ns, as_of_ns[1:], strict=False)):
        raise ValueError("factor history clocks must increase")
    result: list[FactorContext] = []
    previous: FactorContext | None = None
    for clock in as_of_ns:
        current = engine.compute(store, clock, previous=previous)
        result.append(current)
        previous = current
    return tuple(result)


@dataclass(frozen=True, slots=True)
class WalkForwardFactorFold:
    fold_index: int
    train_end_ns: int
    artifact: FactorArtifact
    validation_timestamps_ns: tuple[int, ...]
    validation_scores: tuple[tuple[float, ...], ...]
    test_timestamps_ns: tuple[int, ...]
    test_scores: tuple[tuple[float, ...], ...]
    validation_explained_share: float | None
    test_explained_share: float | None


def _frozen_explained(
    panel: ReturnPanel, artifact: FactorArtifact, scores: np.ndarray
) -> float | None:
    x = np.asarray(panel.returns)
    baseline = np.asarray(artifact.means)
    fitted = scores @ np.asarray(artifact.loadings).T * np.asarray(artifact.scales) + baseline
    total = float(np.sum((x - baseline) ** 2))
    return 1 - float(np.sum((x - fitted) ** 2)) / total if total > 1e-24 else None


def walk_forward_factors(
    panel: ReturnPanel,
    config: MCEConfig,
    calendar: MarketSessionService,
    split: StatisticalSplitSettings,
    *,
    factor_settings: FactorSettings | None = None,
) -> tuple[WalkForwardFactorFold, ...]:
    """Fit each fold on its training rows; project validation/test without refitting."""
    if not panel.timestamps_ns:
        raise ValueError("cannot split an empty factor panel")
    if panel.row_available_at_ns and any(
        available > timestamp
        for timestamp, available in zip(panel.timestamps_ns, panel.row_available_at_ns, strict=True)
    ):
        raise ValueError("walk-forward factor panel contains late historical revisions")
    zone = ZoneInfo(config.market.exchange_timezone)
    dates = tuple(
        ns_to_datetime(timestamp).astimezone(zone).date() for timestamp in panel.timestamps_ns
    )
    sessions = [dates[0]]
    while sessions[-1] < dates[-1]:
        sessions.append(calendar.next_trading_day(sessions[-1]))
    by_day: dict[date, list[int]] = {}
    for index, day in enumerate(dates):
        by_day.setdefault(day, []).append(index)

    def indices(start: int, finish: int) -> tuple[int, ...]:
        return tuple(index for day in sessions[start:finish] for index in by_day.get(day, ()))

    settings = factor_settings or config.features.factor
    result: list[WalkForwardFactorFold] = []
    fold = 0
    while True:
        train_start = 0 if split.mode == "expanding" else fold * split.step_sessions
        train_end = split.train_sessions + fold * split.step_sessions
        val_start = train_end + split.embargo_sessions
        val_end = val_start + split.validation_sessions
        test_start = val_end + split.embargo_sessions
        test_end = test_start + split.test_sessions
        if test_end > len(sessions):
            break
        train_idx, val_idx, test_idx = (
            indices(train_start, train_end),
            indices(val_start, val_end),
            indices(test_start, test_end),
        )
        if not train_idx or not val_idx or not test_idx:
            raise ValueError("factor fold has an empty segment")
        train = panel.take(train_idx)
        validation = panel.take(val_idx)
        test = panel.take(test_idx)
        artifact, _ = fit_pca(train, settings, config_fingerprint=config_fingerprint(config))
        val_scores = transform(artifact, validation)
        test_scores = transform(artifact, test)
        result.append(
            WalkForwardFactorFold(
                fold,
                train.timestamps_ns[-1],
                artifact,
                validation.timestamps_ns,
                tuple(tuple(float(value) for value in row) for row in val_scores),
                test.timestamps_ns,
                tuple(tuple(float(value) for value in row) for row in test_scores),
                _frozen_explained(validation, artifact, val_scores),
                _frozen_explained(test, artifact, test_scores),
            )
        )
        fold += 1
    if not result:
        raise ValueError("insufficient trading sessions for factor folds")
    return tuple(result)
