"""Manual 50k-row, 18-feature statistical preparation benchmark."""

import argparse
import platform
from datetime import date
from time import perf_counter

from mce.calendar.service import MarketSessionService
from mce.config.settings import MCEConfig, StatisticalSplitSettings
from mce.models.context import MarketContextSnapshot
from mce.models.enums import DataQuality, MarketRegime
from mce.statistical.dataset import StatisticalDatasetBuilder
from mce.statistical.normalization import fit_normalizer
from mce.statistical.regime import evaluate_states
from mce.statistical.splits import walk_forward_splits

MINUTE = 60_000_000_000


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--observations", type=int, default=50_000)
    args = parser.parse_args()
    if not 1_000 <= args.observations <= 100_000:
        parser.error("observations must be 1,000–100,000")
    config = MCEConfig(_env_file=None)
    calendar = MarketSessionService(config.market)
    snapshots: list[MarketContextSnapshot] = []
    day = date(2020, 1, 2)
    while len(snapshots) < args.observations:
        opened, _ = calendar.regular_session_bounds(day)
        for slot in range(100):
            if len(snapshots) >= args.observations:
                break
            index = len(snapshots)
            phase = index % 1000
            snapshots.append(
                MarketContextSnapshot(
                    as_of_ns=opened + (slot + 1) * 3 * MINUTE,
                    market_direction_score=(phase % 200 - 100) / 100,
                    volatility_score=(phase % 100) / 100,
                    breadth_score=(phase % 150 - 75) / 75,
                    dispersion_score=(phase % 80) / 80,
                    correlation_score=(phase % 90) / 90,
                    liquidity_score=(phase % 70) / 70,
                    stress_score=(phase % 60) / 60,
                    feature_values={
                        "breadth.net_advance_fraction": (phase % 200 - 100) / 100,
                        "breadth.participation": (phase % 90) / 90,
                        "breadth.momentum": (phase % 30) / 10,
                        "sector.leadership.spread": (phase % 40) / 10,
                        "sector.rotation.velocity": (phase % 50) / 10,
                        "correlation.first_eigen_share": (phase % 90) / 90,
                        "dispersion.between_sector_share": (phase % 80) / 80,
                        "stress.systemic": (phase % 70) / 70,
                        "stress.localized": (phase % 60) / 60,
                        "regime_transition.stability": (phase % 90) / 90,
                        "regime_transition.intensity": (phase % 40) / 10,
                    },
                    market_regime=MarketRegime.RISK_ON
                    if (index // 500) % 2
                    else MarketRegime.RISK_OFF,
                    data_quality=DataQuality.GOOD,
                )
            )
        if len(snapshots) < args.observations:
            day = calendar.next_trading_day(day)
    start = perf_counter()
    dataset = StatisticalDatasetBuilder(config, calendar).build(tuple(snapshots))
    vector_seconds = perf_counter() - start
    settings = StatisticalSplitSettings(
        train_sessions=100,
        validation_sessions=20,
        test_sessions=20,
        embargo_sessions=2,
        step_sessions=40,
    )
    start = perf_counter()
    splits = walk_forward_splits(dataset, calendar, settings)
    split_seconds = perf_counter() - start
    start = perf_counter()
    fitted = fit_normalizer(dataset, splits[0].train_indices)
    values, mask = fitted.transform(dataset)
    normalize_seconds = perf_counter() - start
    states = tuple((index // 500) % 3 for index in range(len(dataset.rows)))
    start = perf_counter()
    evaluation = evaluate_states(dataset, states)
    evaluation_seconds = perf_counter() - start
    print(f"Python {platform.python_version()} on {platform.platform()}")
    print(
        f"rows={len(dataset.rows)} features={values.shape[1]} folds={len(splits)} "
        f"missing={mask.size - int(mask.sum())} states={len(evaluation.profiles)}"
    )
    print(f"vector_seconds={vector_seconds:.3f}")
    print(f"split_seconds={split_seconds:.3f}")
    print(f"normalization_seconds={normalize_seconds:.3f}")
    print(f"evaluation_seconds={evaluation_seconds:.3f}")


if __name__ == "__main__":
    main()
