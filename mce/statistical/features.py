"""Versioned, ordered, deliberately small statistical input schema."""

from dataclasses import dataclass
from hashlib import sha256
from math import isfinite
from typing import Literal

from mce.models.context import MarketContextSnapshot
from mce.serialization.canonical import canonical_json_bytes

NormalizeMethod = Literal["none", "standard", "robust"]


@dataclass(frozen=True, slots=True)
class StatisticalFeatureSpec:
    name: str
    source_family: str
    source_key: str
    required: bool = False
    expected_range: tuple[float | None, float | None] = (None, None)
    transform: Literal["identity"] = "identity"
    missing_policy: Literal["reject", "train_median"] = "train_median"
    normalization: NormalizeMethod = "robust"

    def __post_init__(self) -> None:
        if not self.name or not self.source_family or not self.source_key:
            raise ValueError("feature identifiers must be nonempty")
        if self.required != (self.missing_policy == "reject"):
            raise ValueError("required features must reject missing values")
        lower, upper = self.expected_range
        if lower is not None and upper is not None and lower > upper:
            raise ValueError("invalid expected range")


@dataclass(frozen=True, slots=True)
class StatisticalFeatureSchema:
    version: str
    specs: tuple[StatisticalFeatureSpec, ...]

    def __post_init__(self) -> None:
        if (
            not self.version
            or not self.specs
            or len({s.name for s in self.specs}) != len(self.specs)
        ):
            raise ValueError("schema needs a version and unique ordered features")

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self.specs)

    @property
    def fingerprint(self) -> str:
        from dataclasses import asdict

        return sha256(
            canonical_json_bytes(
                {"version": self.version, "specs": [asdict(s) for s in self.specs]}
            )
        ).hexdigest()

    def extract(
        self, snapshot: MarketContextSnapshot
    ) -> tuple[tuple[float | None, ...], tuple[bool, ...]]:
        values: list[float | None] = []
        mask: list[bool] = []
        for spec in self.specs:
            raw = (
                getattr(snapshot, spec.source_key)
                if spec.source_family == "snapshot"
                else (snapshot.feature_values or {}).get(spec.source_key)
            )
            if raw is None:
                if spec.required:
                    raise ValueError(f"required statistical feature missing: {spec.name}")
                values.append(None)
                mask.append(False)
                continue
            if isinstance(raw, bool) or not isinstance(raw, (int, float)) or not isfinite(raw):
                raise ValueError(f"statistical feature must be finite numeric: {spec.name}")
            value = float(raw)
            lower, upper = spec.expected_range
            if (lower is not None and value < lower) or (upper is not None and value > upper):
                raise ValueError(f"statistical feature outside expected range: {spec.name}")
            values.append(value)
            mask.append(True)
        return tuple(values), tuple(mask)


def _score(name: str, *, required: bool = False, signed: bool = False) -> StatisticalFeatureSpec:
    return StatisticalFeatureSpec(
        name=name,
        source_family="snapshot",
        source_key=name,
        required=required,
        expected_range=(-1.0, 1.0) if signed else (0.0, 1.0),
        missing_policy="reject" if required else "train_median",
        normalization="none",
    )


def _detail(
    name: str, family: str, bounds: tuple[float | None, float | None] = (None, None)
) -> StatisticalFeatureSpec:
    return StatisticalFeatureSpec(
        name=name,
        source_family=family,
        source_key=name,
        expected_range=bounds,
        normalization="none" if bounds == (0.0, 1.0) else "robust",
    )


STATISTICAL_FEATURE_SCHEMA = StatisticalFeatureSchema(
    version="1.0",
    specs=(
        _score("market_direction_score", required=True, signed=True),
        _score("volatility_score"),
        _score("breadth_score", signed=True),
        _score("dispersion_score"),
        _score("correlation_score"),
        _score("liquidity_score"),
        _score("stress_score"),
        _detail("breadth.net_advance_fraction", "breadth", (-1.0, 1.0)),
        _detail("breadth.participation", "breadth", (0.0, 1.0)),
        _detail("breadth.momentum", "breadth"),
        _detail("sector.leadership.spread", "sector"),
        _detail("sector.rotation.velocity", "sector"),
        _detail("correlation.first_eigen_share", "correlation", (0.0, 1.0)),
        _detail("dispersion.between_sector_share", "dispersion", (0.0, 1.0)),
        _detail("stress.systemic", "stress", (0.0, 1.0)),
        _detail("stress.localized", "stress", (0.0, 1.0)),
        _detail("regime_transition.stability", "regime_transition", (0.0, 1.0)),
        _detail("regime_transition.intensity", "regime_transition"),
    ),
)
