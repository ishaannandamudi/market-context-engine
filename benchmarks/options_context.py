"""Manual provider-neutral MCE-17 benchmark using deterministic synthetic context."""

import argparse
import platform
from time import perf_counter

from mce.features.options_context import (
    ATMIVPoint,
    OptionsContextAdapter,
    OptionsContextFeatureEngine,
    OptionsContextHistory,
    SurfaceIVPoint,
    apply_options_context_to_snapshot,
)
from mce.features.options_context.contracts import DAY_NS, SurfaceQuality
from mce.features.options_context.history import compare_history
from mce.features.options_context.realized_implied import compare_implied_realized
from mce.features.options_context.skew import skew_context
from mce.features.options_context.stress import options_stress
from mce.features.options_context.term_structure import term_structure
from tests.fixtures.options_scenarios import (
    BASE,
    options_config,
    options_history,
    options_input,
    rv_context,
)

SYMBOLS = ("SPY", "QQQ", "IWM", "XLK", "XLF")
TENORS = (7, 14, 21, 30, 45, 60, 90)


def _payload(symbol: str, sector: bool) -> dict[str, object]:
    source = options_input(symbol=symbol, scope="sector" if sector else "broad")
    atm = tuple(
        ATMIVPoint(
            dte_days=days,
            expiration_ns=BASE + days * DAY_NS,
            implied_volatility=0.18 + days * 0.0003,
            combination="combined",
            source_method="synthetic_surface",
            data_as_of_ns=BASE,
            available_at_ns=BASE,
        )
        for days in TENORS
    )
    wings = tuple(
        SurfaceIVPoint(
            dte_days=days,
            expiration_ns=BASE + days * DAY_NS,
            side=side,  # type: ignore[arg-type]
            coordinate_system="delta",
            coordinate=(-1 if side == "put" else 1) * coordinate,
            implied_volatility=0.18
            + days * 0.0003
            + (0.04 if side == "put" else -0.01) * coordinate / 0.25,
            source_method="synthetic_surface",
            data_as_of_ns=BASE,
            available_at_ns=BASE,
        )
        for days in TENORS
        for coordinate in (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55)
        for side in ("put", "call")
    )
    quality = SurfaceQuality(
        tenor_coverage=1, strike_coverage=1, independent_neighbor_count=len(wings)
    )
    return source.model_copy(
        update={"atm_points": atm, "surface_points": wings, "surface_quality": quality}
    ).model_dump(mode="json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshots", type=int, default=20)
    args = parser.parse_args()
    if args.snapshots < 1:
        parser.error("snapshots must be positive")
    config = options_config(broad=SYMBOLS[:3], sectors=SYMBOLS[3:], minimum_history=10)
    history = OptionsContextHistory(
        points=tuple(
            point for symbol in SYMBOLS for point in options_history(symbol=symbol, count=60).points
        )
    )
    payloads = tuple(_payload(symbol, symbol in SYMBOLS[3:]) for symbol in SYMBOLS)
    adapter = OptionsContextAdapter()
    started = perf_counter()
    sources = tuple(adapter.translate(payload) for payload in payloads)
    parse_seconds = perf_counter() - started
    engine = OptionsContextFeatureEngine(config)
    realized = rv_context(0.2)
    current = sources[0]
    settings = config.features.options_context
    started = perf_counter()
    term = term_structure(current.atm_points, settings.target_tenor_days, settings.minimum_tenors)
    term_seconds = perf_counter() - started
    started = perf_counter()
    skew = skew_context(current.atm_points[3], current.atm_points, current.surface_points, settings)
    skew_seconds = perf_counter() - started
    started = perf_counter()
    iv_rv = compare_implied_realized(current.atm_points[3], "SPY", BASE, realized, settings)
    iv_rv_seconds = perf_counter() - started
    started = perf_counter()
    comparison = compare_history(
        0.2, tuple(point.atm_iv for point in history.prior("SPY", BASE, 60)), 10
    )
    history_seconds = perf_counter() - started
    started = perf_counter()
    stress = options_stress(comparison, comparison, term, comparison, settings)
    stress_seconds = perf_counter() - started
    started = perf_counter()
    results = tuple(
        engine.compute(
            sources,
            BASE + index * 1_000_000_000,
            realized_volatility_context=realized if index == 0 else None,
            historical_options_context=history,
        )
        for index in range(args.snapshots)
    )
    compute_seconds = perf_counter() - started
    started = perf_counter()
    snapshots = tuple(apply_options_context_to_snapshot(result) for result in results)
    project_seconds = perf_counter() - started
    print(f"Python {platform.python_version()} on {platform.platform()}")
    print(
        f"underlyings={len(SYMBOLS)} tenors={len(TENORS)} "
        f"surface_points={sum(len(item.surface_points) for item in sources)} "
        f"historical_points={len(history.points)} snapshots={len(snapshots)}"
    )
    print(
        f"validation_adaptation_seconds={parse_seconds:.6f} term_seconds={term_seconds:.6f} "
        f"skew_seconds={skew_seconds:.6f} iv_rv_seconds={iv_rv_seconds:.6f} "
        f"history_seconds={history_seconds:.6f} stress_seconds={stress_seconds:.6f}"
    )
    print(
        f"engine_seconds_per_snapshot={compute_seconds / args.snapshots:.6f} "
        f"snapshot_projection_seconds_per_snapshot={project_seconds / args.snapshots:.6f} "
        f"last_quality={results[-1].data_quality.value} term_slope={term.slope_per_day} "
        f"skew_downside={skew.downside} iv_rv_spread={iv_rv.spread} stress_score={stress.score}"
    )


if __name__ == "__main__":
    main()
