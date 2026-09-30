"""Manual 10k-row diagonal HMM, k-means and mean-shift benchmark."""

import argparse
import platform
from datetime import date
from time import perf_counter

from mce.calendar.service import MarketSessionService
from mce.config.settings import (
    ChangePointSettings,
    HMMSettings,
    KMeansSettings,
    MCEConfig,
)
from mce.models.context import MarketContextSnapshot
from mce.models.enums import DataQuality
from mce.statistical.dataset import StatisticalDatasetBuilder
from mce.statistical.normalization import fit_normalizer
from mce.statistical.regime.changepoint import MeanShiftDetector
from mce.statistical.regime.clustering import KMeansRegimeModel
from mce.statistical.regime.hmm import GaussianHMMRegimeModel

MINUTE = 60_000_000_000


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--observations", type=int, default=10_000)
    args = parser.parse_args()
    if not 1_000 <= args.observations <= 20_000:
        parser.error("observations must be 1,000–20,000")
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
            state = (index // 400) % 3
            noise = (index % 17 - 8) / 200
            direction = (0.6, 0.0, -0.6)[state] + noise
            stress = (0.1, 0.5, 0.9)[state]
            snapshots.append(
                MarketContextSnapshot(
                    as_of_ns=opened + (slot + 1) * 3 * MINUTE,
                    market_direction_score=direction,
                    volatility_score=(0.15, 0.5, 0.85)[state],
                    breadth_score=direction,
                    dispersion_score=0.2 + state * 0.2,
                    correlation_score=0.2 + state * 0.2,
                    liquidity_score=0.8 - state * 0.2,
                    stress_score=stress,
                    feature_values={
                        "breadth.net_advance_fraction": direction,
                        "breadth.participation": 0.4 + state * 0.1,
                        "breadth.momentum": noise,
                        "sector.leadership.spread": 0.2 + state * 0.1,
                        "sector.rotation.velocity": 0.1 + state * 0.1,
                        "correlation.first_eigen_share": 0.2 + state * 0.2,
                        "dispersion.between_sector_share": 0.2 + state * 0.2,
                        "stress.systemic": stress,
                        "stress.localized": 0.1 + state * 0.1,
                        "regime_transition.stability": 0.9 - state * 0.2,
                        "regime_transition.intensity": 0.1 + state * 0.2,
                    },
                    data_quality=DataQuality.GOOD,
                )
            )
        if len(snapshots) < args.observations:
            day = calendar.next_trading_day(day)
    dataset = StatisticalDatasetBuilder(config, calendar).build(tuple(snapshots))
    train_count = int(len(dataset.rows) * 0.8)
    training = dataset.take(tuple(range(train_count)))
    test = dataset.take(tuple(range(train_count, len(dataset.rows))))
    normalizer = fit_normalizer(dataset, tuple(range(train_count)))

    hmm = GaussianHMMRegimeModel(HMMSettings(state_count=3, max_iterations=15, restarts=2))
    start = perf_counter()
    artifact = hmm.fit(training, normalizer, random_seed=7)
    hmm_fit = perf_counter() - start
    start = perf_counter()
    filtered = hmm.infer(test, mode="filtered")
    hmm_infer = perf_counter() - start

    kmeans = KMeansRegimeModel(KMeansSettings(cluster_count=3, max_iterations=50, restarts=2))
    start = perf_counter()
    kmeans_artifact = kmeans.fit(training, normalizer, random_seed=7)
    kmeans_fit = perf_counter() - start
    start = perf_counter()
    cluster_labels = kmeans.infer(test)
    kmeans_infer = perf_counter() - start

    detector = MeanShiftDetector(ChangePointSettings(left_window=20, right_window=20))
    start = perf_counter()
    change_points = detector.offline(dataset, normalizer)
    change_seconds = perf_counter() - start
    print(f"Python {platform.python_version()} on {platform.platform()}")
    print(
        f"observations={len(dataset.rows)} dimensions={len(dataset.schema.specs)} "
        f"train={len(training.rows)} test={len(test.rows)} K=3"
    )
    print(
        f"hmm_iterations={artifact.restarts[artifact.selected_restart].iterations} "
        f"hmm_fit_seconds={hmm_fit:.3f} hmm_filtered_seconds={hmm_infer:.3f} "
        f"filtered_rows={len(filtered)}"
    )
    kmeans_iterations = kmeans_artifact.restarts[kmeans_artifact.selected_restart].iterations
    print(
        f"kmeans_iterations={kmeans_iterations} "
        f"kmeans_fit_seconds={kmeans_fit:.3f} kmeans_infer_seconds={kmeans_infer:.3f} "
        f"clustered_rows={len(cluster_labels)}"
    )
    print(f"offline_change_seconds={change_seconds:.3f} points={len(change_points)}")


if __name__ == "__main__":
    main()
