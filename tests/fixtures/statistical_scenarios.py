"""Fixed synthetic latent patterns; labels are fixture IDs, not market truth."""

from datetime import date

from mce.calendar.service import MarketSessionService
from mce.config.settings import MCEConfig
from mce.models.context import MarketContextSnapshot
from mce.models.enums import DataQuality
from mce.statistical.dataset import StatisticalDataset, StatisticalDatasetBuilder


def synthetic_states(kind: str, count: int = 30) -> tuple[int, ...]:
    if kind == "two_state":
        return tuple((index // 5) % 2 for index in range(count))
    if kind == "three_state":
        return tuple((index // 5) % 3 for index in range(count))
    if kind == "overlapping":
        return tuple((index // 3) % 3 for index in range(count))
    if kind == "rapid":
        return tuple(index % 2 for index in range(count))
    if kind == "long_dwell":
        return tuple(0 if index < count - 2 else 1 for index in range(count))
    raise ValueError("unknown synthetic scenario")


def synthetic_dataset(kind: str, count: int = 30) -> tuple[StatisticalDataset, tuple[int, ...]]:
    config = MCEConfig(_env_file=None)
    calendar = MarketSessionService(config.market)
    states = synthetic_states(kind, count)
    day = date(2024, 1, 2)
    snapshots = []
    for index, state in enumerate(states):
        _, close = calendar.regular_session_bounds(day)
        noise = (index % 3 - 1) * (0.12 if kind == "overlapping" else 0.01)
        values = {
            "breadth.net_advance_fraction": (0.5, 0.0, -0.5)[state],
            "breadth.participation": 0.5,
            "breadth.momentum": float(state),
            "sector.leadership.spread": 0.2,
            "sector.rotation.velocity": 0.3,
            "correlation.first_eigen_share": 0.4,
            "dispersion.between_sector_share": 0.5,
            "stress.systemic": (0.1, 0.5, 0.9)[state],
            "stress.localized": 0.2,
            "regime_transition.stability": 0.8,
            "regime_transition.intensity": 0.2,
        }
        if index % 7 == 0:
            del values["breadth.momentum"]
        snapshots.append(
            MarketContextSnapshot(
                as_of_ns=close,
                market_direction_score=max(-1.0, min(1.0, (0.7, 0.0, -0.7)[state] + noise)),
                volatility_score=max(0.0, min(1.0, (0.15, 0.6, 0.9)[state] + noise)),
                breadth_score=(0.5, 0.0, -0.5)[state],
                dispersion_score=0.3,
                correlation_score=0.4,
                liquidity_score=0.5,
                stress_score=(0.1, 0.5, 0.9)[state],
                feature_values=values,
                data_quality=DataQuality.DEGRADED if index % 11 == 0 else DataQuality.GOOD,
            )
        )
        if index + 1 < count:
            day = calendar.next_trading_day(day)
    return StatisticalDatasetBuilder(config, calendar).build(tuple(snapshots)), states
