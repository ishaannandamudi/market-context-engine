"""Manual 500-symbol return-panel, PCA, frozen-transform and alignment benchmark."""

import argparse
import platform
from time import perf_counter

import numpy as np

from mce.config.settings import FactorSettings
from mce.features.factors.dynamics import align_factors
from mce.features.factors.models import ReturnPanel
from mce.features.factors.pca import fit_pca, transform
from mce.features.factors.residuals import decompose


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--observations", type=int, default=2500)
    parser.add_argument("--symbols", type=int, default=500)
    parser.add_argument("--factors", type=int, default=5)
    args = parser.parse_args()
    if args.observations < 100 or args.symbols < 3 or not 1 <= args.factors <= 10:
        parser.error("need at least 100 observations, 3 symbols, and 1-10 factors")
    rng = np.random.default_rng(14)
    market = rng.normal(size=(args.observations, 1))
    sector = rng.normal(size=(args.observations, 5))
    exposures = rng.normal(size=(5, args.symbols))
    matrix = (
        0.004 * market
        + 0.0015 * sector @ exposures
        + rng.normal(scale=0.002, size=(args.observations, args.symbols))
    )
    symbols = tuple(f"S{index:04d}" for index in range(args.symbols))
    start = perf_counter()
    panel = ReturnPanel(
        args.observations,
        tuple(range(1, args.observations + 1)),
        symbols,
        symbols,
        tuple(tuple(float(value) for value in row) for row in matrix),
        args.observations,
    )
    panel_seconds = perf_counter() - start
    settings = FactorSettings(
        window_sessions=args.observations,
        minimum_samples=100,
        factor_count=args.factors,
        minimum_symbols=3,
    )
    train_count = int(args.observations * 0.8)
    training = panel.take(tuple(range(train_count)))
    testing = panel.take(tuple(range(train_count, args.observations)))
    start = perf_counter()
    artifact, training_scores = fit_pca(training, settings, config_fingerprint="a" * 64)
    fit_seconds = perf_counter() - start
    start = perf_counter()
    held_out = transform(artifact, testing)
    transform_seconds = perf_counter() - start
    start = perf_counter()
    residual = decompose(training, artifact, training_scores)
    residual_seconds = perf_counter() - start
    start = perf_counter()
    next_artifact, _ = fit_pca(
        panel.take(tuple(range(train_count + 20))), settings, config_fingerprint="a" * 64
    )
    alignment, _, _ = align_factors(next_artifact, artifact)
    refit_align_seconds = perf_counter() - start
    print(f"Python {platform.python_version()} on {platform.platform()}")
    print(
        f"observations={args.observations} symbols={args.symbols} K={artifact.selected_count} "
        f"method={artifact.method}"
    )
    print(
        f"panel_seconds={panel_seconds:.3f} fit_seconds={fit_seconds:.3f} "
        f"transform_seconds={transform_seconds:.3f} transformed_rows={len(held_out)}"
    )
    print(
        f"residual_seconds={residual_seconds:.3f} "
        f"refit_and_alignment_seconds={refit_align_seconds:.3f} "
        f"subspace_similarity={alignment.subspace_similarity if alignment else None}"
    )
    print(f"common_share={residual.mean_common_share}")


if __name__ == "__main__":
    main()
