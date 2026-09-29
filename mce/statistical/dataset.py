"""Point-in-time rows from already-causal public snapshots."""

from dataclasses import dataclass
from datetime import date
from hashlib import sha256
from math import isfinite

from mce.calendar.service import MarketSessionService
from mce.config.settings import MCEConfig
from mce.features.regime.models import MarketRegimeState
from mce.models.context import MarketContextSnapshot
from mce.models.enums import DataQuality, MarketSession, QualityFlag
from mce.reproducibility import config_fingerprint
from mce.serialization.canonical import canonical_json_bytes
from mce.statistical.features import STATISTICAL_FEATURE_SCHEMA, StatisticalFeatureSchema
from mce.version import SCHEMA_VERSION


@dataclass(frozen=True, slots=True)
class SnapshotRecord:
    snapshot: MarketContextSnapshot
    available_at_ns: int | None = None
    source_fingerprint: str | None = None

    def __post_init__(self) -> None:
        if self.available_at_ns is not None and self.available_at_ns < self.snapshot.as_of_ns:
            raise ValueError("availability cannot precede snapshot as-of")

    @property
    def effective_availability_ns(self) -> int:
        return self.available_at_ns if self.available_at_ns is not None else self.snapshot.as_of_ns


@dataclass(frozen=True, slots=True)
class StatisticalObservation:
    as_of_ns: int
    trading_date: date
    session: MarketSession
    feature_vector: tuple[float | None, ...]
    feature_mask: tuple[bool, ...]
    source_quality: DataQuality
    quality_flags: frozenset[QualityFlag]
    deterministic_regime: str | None
    source_fingerprint: str | None

    def __post_init__(self) -> None:
        if self.as_of_ns < 0:
            raise ValueError("observation clock must be nonnegative")
        if len(self.feature_vector) != len(self.feature_mask):
            raise ValueError("feature vector/mask length mismatch")
        if any(
            (value is not None) != present
            for value, present in zip(self.feature_vector, self.feature_mask, strict=True)
        ):
            raise ValueError("feature mask must describe original missingness")
        if any(value is not None and not isfinite(value) for value in self.feature_vector):
            raise ValueError("observation features must be finite")


@dataclass(frozen=True, slots=True)
class StatisticalDataset:
    rows: tuple[StatisticalObservation, ...]
    schema: StatisticalFeatureSchema
    config_fingerprint: str
    normalization_method: str

    def __post_init__(self) -> None:
        if any(len(row.feature_vector) != len(self.schema.specs) for row in self.rows):
            raise ValueError("row does not match feature schema")
        if any(
            row.feature_mask[index] is False
            for row in self.rows
            for index, spec in enumerate(self.schema.specs)
            if spec.required
        ):
            raise ValueError("required statistical feature missing from dataset")
        if any(a.as_of_ns >= b.as_of_ns for a, b in zip(self.rows, self.rows[1:], strict=False)):
            raise ValueError("dataset rows must have unique increasing timestamps")

    @property
    def fingerprint(self) -> str:
        digest = sha256(
            canonical_json_bytes(
                {
                    "schema": self.schema.fingerprint,
                    "config": self.config_fingerprint,
                    "normalization": self.normalization_method,
                }
            )
        )
        for row in self.rows:
            digest.update(
                canonical_json_bytes(
                    {
                        "as_of_ns": row.as_of_ns,
                        "trading_date": row.trading_date,
                        "session": row.session,
                        "vector": row.feature_vector,
                        "mask": row.feature_mask,
                        "quality": row.source_quality,
                        "flags": row.quality_flags,
                        "regime": row.deterministic_regime,
                        "source": row.source_fingerprint,
                    }
                )
            )
        return digest.hexdigest()

    def take(self, indices: tuple[int, ...]) -> "StatisticalDataset":
        if any(index < 0 or index >= len(self.rows) for index in indices):
            raise IndexError("dataset row index out of range")
        return StatisticalDataset(
            rows=tuple(self.rows[index] for index in indices),
            schema=self.schema,
            config_fingerprint=self.config_fingerprint,
            normalization_method=self.normalization_method,
        )


class StatisticalDatasetBuilder:
    def __init__(
        self,
        config: MCEConfig,
        calendar: MarketSessionService | None = None,
        schema: StatisticalFeatureSchema = STATISTICAL_FEATURE_SCHEMA,
    ) -> None:
        if config.statistical.feature_set != schema.version:
            raise ValueError("configured statistical feature schema mismatch")
        self.config = config
        self.calendar = calendar or MarketSessionService(config.market)
        self.schema = schema

    def build(
        self,
        snapshots: tuple[SnapshotRecord | MarketContextSnapshot, ...],
        *,
        as_of_ns: int | None = None,
        regime_states: tuple[MarketRegimeState, ...] = (),
    ) -> StatisticalDataset:
        if as_of_ns is not None and as_of_ns < 0:
            raise ValueError("as_of_ns must be nonnegative")
        states: dict[int, MarketRegimeState] = {}
        for state in regime_states:
            if state.as_of_ns in states:
                raise ValueError("duplicate deterministic regime timestamp")
            states[state.as_of_ns] = state
        records = [
            item if isinstance(item, SnapshotRecord) else SnapshotRecord(item) for item in snapshots
        ]
        records.sort(key=lambda item: item.snapshot.as_of_ns)
        rows: list[StatisticalObservation] = []
        seen: set[int] = set()
        for record in records:
            snapshot = record.snapshot
            clock = snapshot.as_of_ns
            if as_of_ns is not None and clock > as_of_ns:
                break
            if clock in seen:
                raise ValueError("duplicate snapshot timestamp")
            seen.add(clock)
            if snapshot.schema_version != SCHEMA_VERSION:
                raise ValueError("snapshot schema version mismatch")
            if record.effective_availability_ns > clock:
                continue
            policy = self.config.statistical.quality_policy
            if policy == "reject_unusable" and snapshot.data_quality == DataQuality.UNUSABLE:
                continue
            if policy == "reject_poor" and snapshot.data_quality in (
                DataQuality.POOR,
                DataQuality.UNUSABLE,
            ):
                continue
            info = self.calendar.session_for_timestamp(clock)
            if info.trading_date is None:
                raise ValueError("statistical row must belong to a trading date")
            vector, mask = self.schema.extract(snapshot)
            flags = snapshot.quality_flags
            if not all(mask):
                flags = flags | frozenset({QualityFlag.MISSING_DATA})
            matched_state = states.get(clock)
            if matched_state is not None and matched_state.available_at_ns <= clock:
                regime = matched_state.headline_regime.value
            elif snapshot.market_regime.value != "unknown":
                regime = snapshot.market_regime.value
            else:
                regime = None
            rows.append(
                StatisticalObservation(
                    as_of_ns=clock,
                    trading_date=info.trading_date,
                    session=info.session,
                    feature_vector=vector,
                    feature_mask=mask,
                    source_quality=snapshot.data_quality,
                    quality_flags=flags,
                    deterministic_regime=regime,
                    source_fingerprint=record.source_fingerprint,
                )
            )
        return StatisticalDataset(
            rows=tuple(rows),
            schema=self.schema,
            config_fingerprint=config_fingerprint(self.config),
            normalization_method=self.config.statistical.normalization,
        )
