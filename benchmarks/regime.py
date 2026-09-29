"""Manual 1,000-snapshot deterministic regime and transition benchmark."""

import argparse
import platform
from time import perf_counter

from mce.config.settings import MCEConfig
from mce.features.regime import MarketRegimeEngine, RegimeEvidence

MINUTE = 60_000_000_000


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshots", type=int, default=1000)
    args = parser.parse_args()
    if not 10 <= args.snapshots <= 100_000:
        parser.error("snapshots must be 10–100000")
    engine = MarketRegimeEngine(MCEConfig(_env_file=None))
    clock = 1_719_580_000_000_000_000
    contexts = []
    for index in range(args.snapshots):
        mode = (index // 25) % 4
        direction = (0.8, -0.8, -0.8, 0.0)[mode]
        contexts.append(
            RegimeEvidence(
                as_of_ns=clock + index * MINUTE,
                market_direction_score=direction,
                trend_score=direction,
                breadth_score=(0.7, -0.7, -0.7, 0.0)[mode],
                breadth_participation_score=(0.6, -0.6, -0.6, 0.0)[mode],
                volatility_score=(0.3, 0.5, 0.9, 0.2)[mode],
                liquidity_score=(0.8, 0.6, 0.15, 0.8)[mode],
                correlation_score=(0.4, 0.5, 0.85, 0.2)[mode],
                dispersion_score=(0.4, 0.5, 0.8, 0.2)[mode],
                stress_score=(0.2, 0.5, 0.9, 0.1)[mode],
                stressed_member_fraction=(0.2, 0.4, 0.8, 0.1)[mode],
                sector_rotation_velocity=0.2,
            )
        )
    began = perf_counter()
    results = []
    for context in contexts:
        previous = (results[-1],) if results else ()
        results.append(engine.compute(context, history=previous))
    elapsed = perf_counter() - began
    headline_transitions = sum(item.transition is not None for item in results)
    dimension_transitions = sum(len(item.dimension_transitions) for item in results)
    print(f"Python {platform.python_version()} on {platform.platform()}")
    print(
        f"snapshots={len(results)} transitions={headline_transitions} "
        f"dimension_transitions={dimension_transitions}"
    )
    print(f"total_seconds={elapsed:.3f} seconds_per_snapshot={elapsed / len(results):.6f}")


if __name__ == "__main__":
    main()
