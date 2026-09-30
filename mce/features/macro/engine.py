"""Provider-free, vintage-aware descriptive macro context at an explicit clock."""

from math import isfinite
from statistics import median

from mce.config.settings import MCEConfig
from mce.features.macro.contracts import MacroCategory, MacroContextInput, SeriesCadence
from mce.features.macro.events import event_context
from mce.features.macro.history import MacroHistoryIndex
from mce.features.macro.models import (
    MacroContextFeatureSet,
    MacroDimension,
    MacroSeriesContext,
    MacroSourceMetadata,
    MacroStressContext,
)
from mce.features.macro.rates import policy_context, yield_curve_context
from mce.features.macro.surprises import compare_values, surprise_context
from mce.models.enums import DataQuality, QualityFlag
from mce.reproducibility import config_fingerprint


def _cadence_limit(cadence: SeriesCadence, config: MCEConfig) -> int:
    return int(getattr(config.features.macro.cadence_age, f"{cadence.value}_ns"))


def _dimension(name: str, series: tuple[MacroSeriesContext, ...]) -> MacroDimension:
    groups = {
        "inflation": (MacroCategory.INFLATION,),
        "growth": (
            MacroCategory.GROWTH,
            MacroCategory.ACTIVITY,
            MacroCategory.CONSUMER,
            MacroCategory.HOUSING,
        ),
        "labor": (MacroCategory.EMPLOYMENT, MacroCategory.UNEMPLOYMENT, MacroCategory.WAGES),
    }
    members = tuple(
        item
        for item in series
        if item.category in groups[name]
        and not item.stale
        and item.data_quality in (DataQuality.GOOD, DataQuality.DEGRADED)
    )
    primary = max(
        members,
        key=lambda item: (
            max(
                item.observation.available_at_ns if item.observation else -1,
                item.release.available_at_ns if item.release else -1,
            ),
            item.source,
            item.series_id,
        ),
        default=None,
    )
    normalized = tuple(
        item.surprise.oriented_z
        for item in members
        if item.surprise is not None
        and item.surprise.oriented_z is not None
        and not item.release_stale
        and item.data_quality in (DataQuality.GOOD, DataQuality.DEGRADED)
    )
    return MacroDimension(
        name=name,  # type: ignore[arg-type]
        primary_series_id=primary.series_id if primary else None,
        latest_value=primary.observation.value if primary and primary.observation else None,
        unit=primary.unit if primary else None,
        latest_surprise=primary.surprise.raw
        if primary and primary.surprise and not primary.release_stale
        else None,
        latest_surprise_z=primary.surprise.normalized_z
        if primary and primary.surprise and not primary.release_stale
        else None,
        normalized_signal=float(median(normalized)) if normalized else None,
        contributing_series=len(normalized),
    )


def _stress(
    series: tuple[MacroSeriesContext, ...],
    curve_change: float | None,
    curve_inversion: float | None,
    config: MCEConfig,
) -> MacroStressContext:
    settings = config.features.macro
    rates = (
        min(1.0, abs(curve_change) / settings.rate_move_scale_pp)
        if curve_change is not None
        else None
    )
    curve = (
        min(1.0, curve_inversion / settings.curve_inversion_scale_pp)
        if curve_inversion is not None
        else None
    )
    surprise_values = tuple(
        abs(item.surprise.normalized_z)
        for item in series
        if item.surprise is not None
        and item.surprise.normalized_z is not None
        and not item.stale
        and not item.release_stale
        and item.data_quality in (DataQuality.GOOD, DataQuality.DEGRADED)
    )
    surprise = (
        min(1.0, max(surprise_values) / settings.surprise_z_scale) if surprise_values else None
    )
    candidates = (
        ("rates", rates, settings.stress_weights.rates),
        ("curve", curve, settings.stress_weights.curve),
        ("surprise", surprise, settings.stress_weights.surprise),
    )
    active = tuple(
        (name, value, weight)
        for name, value, weight in candidates
        if value is not None and weight > 0
    )
    total = sum(weight for _, _, weight in active)
    score = (
        sum(value * weight for _, value, weight in active) / total
        if len(active) >= settings.minimum_stress_components and total > 0
        else None
    )
    return MacroStressContext(
        rates=rates,
        curve=curve,
        surprise=surprise,
        score=score,
        active_weights=tuple((name, weight / total) for name, _, weight in active) if total else (),
    )


class MacroContextFeatureEngine:
    def __init__(self, config: MCEConfig) -> None:
        self.config = config
        self.settings = config.features.macro

    def compute(
        self,
        inputs: tuple[MacroContextInput, ...] | MacroHistoryIndex,
        as_of_ns: int,
    ) -> MacroContextFeatureSet:
        if as_of_ns < 0:
            raise ValueError("macro as_of_ns must be nonnegative")
        fingerprint = config_fingerprint(self.config)
        if not self.settings.enabled:
            return MacroContextFeatureSet(
                as_of_ns=as_of_ns, config_fingerprint=fingerprint, data_quality=DataQuality.UNKNOWN
            )
        index = inputs if isinstance(inputs, MacroHistoryIndex) else MacroHistoryIndex(inputs)
        visible_sources = index.visible_sources(as_of_ns)
        source_metadata = tuple(MacroSourceMetadata.from_input(item) for item in visible_sources)
        source_quality = {item.source: item.quality for item in visible_sources}
        flags: set[QualityFlag] = set()
        if not visible_sources or any(
            item.quality in (DataQuality.POOR, DataQuality.UNUSABLE, DataQuality.UNKNOWN)
            for item in visible_sources
        ):
            flags.add(QualityFlag.MACRO_SOURCE_DEGRADED)
        per_series: list[MacroSeriesContext] = []
        latest_releases = []
        for key in index.series_keys:
            source, series_id = key
            observation = index.observation(key, as_of_ns)
            release = index.release(key, as_of_ns)
            if observation is None and release is None:
                continue
            reference = observation or release
            assert reference is not None
            cadence = reference.cadence
            stale = (
                observation is not None
                and as_of_ns - observation.period_end_ns > _cadence_limit(cadence, self.config)
            )
            release_stale = (
                release is not None
                and as_of_ns - release.period_end_ns > _cadence_limit(cadence, self.config)
            )
            if observation is None:
                stale = release_stale
            if stale or release_stale:
                flags.add(QualityFlag.MACRO_SERIES_STALE)
            previous = None
            comparison = None
            if observation is not None:
                prior = index.prior_observations(
                    key, observation.period_end_ns, as_of_ns, self.settings.history_limit
                )
                previous = prior[-1].value if prior else None
                comparison = compare_values(
                    observation.value,
                    tuple(item.value for item in prior),
                    self.settings.minimum_history,
                )
                if comparison.sample_count < self.settings.minimum_history:
                    flags.add(QualityFlag.INSUFFICIENT_MACRO_HISTORY)
            surprise = None
            if release is not None:
                prior_releases = index.prior_releases(
                    key, release.period_end_ns, as_of_ns, self.settings.history_limit
                )
                surprise = surprise_context(release, prior_releases, self.settings.minimum_history)
                latest_releases.append(release)
                if (
                    surprise.raw is not None
                    and surprise.historical_sample_count < self.settings.minimum_history
                ):
                    flags.add(QualityFlag.INSUFFICIENT_MACRO_HISTORY)
            upstream_quality = source_quality.get(source, DataQuality.UNKNOWN)
            record_qualities = tuple(
                quality
                for quality in (
                    observation.quality if observation is not None else None,
                    release.quality if release is not None else None,
                )
                if quality is not None
            )
            record_quality = (
                DataQuality.POOR
                if any(
                    item in (DataQuality.POOR, DataQuality.UNUSABLE, DataQuality.UNKNOWN)
                    for item in record_qualities
                )
                else DataQuality.DEGRADED
                if DataQuality.DEGRADED in record_qualities
                else DataQuality.GOOD
            )
            acceptable = upstream_quality in (
                DataQuality.GOOD,
                DataQuality.DEGRADED,
            ) and record_quality in (DataQuality.GOOD, DataQuality.DEGRADED)
            if self.settings.minimum_quality == "good":
                acceptable = (
                    upstream_quality == DataQuality.GOOD and record_quality == DataQuality.GOOD
                )
            quality = (
                DataQuality.POOR
                if not acceptable or stale
                else DataQuality.DEGRADED
                if upstream_quality == DataQuality.DEGRADED
                or record_quality == DataQuality.DEGRADED
                else DataQuality.GOOD
            )
            if not acceptable:
                flags.add(QualityFlag.MACRO_SOURCE_DEGRADED)
            per_series.append(
                MacroSeriesContext(
                    source=source,
                    series_id=series_id,
                    category=reference.category,
                    unit=reference.unit,
                    cadence=cadence,
                    observation=observation,
                    previous_value=previous,
                    change=observation.value - previous
                    if observation is not None
                    and previous is not None
                    and isfinite(observation.value - previous)
                    else None,
                    history=comparison,
                    release=release,
                    surprise=surprise,
                    stale=stale,
                    release_stale=release_stale,
                    data_quality=quality,
                )
            )
        series_tuple = tuple(per_series)
        calendar = index.calendar(as_of_ns)
        if calendar is None:
            flags.add(QualityFlag.MACRO_EVENT_CONTEXT_UNAVAILABLE)
        eligible_sources = frozenset(
            item.source
            for item in visible_sources
            if item.quality == DataQuality.GOOD
            or (
                self.settings.minimum_quality == "degraded" and item.quality == DataQuality.DEGRADED
            )
        )
        event = event_context(
            calendar, tuple(latest_releases), as_of_ns, self.settings, eligible_sources
        )
        if not eligible_sources:
            event = event.model_copy(
                update={
                    "proximity_score": None,
                    "upcoming_count_24h": None,
                    "high_importance_count_24h": None,
                    "weighted_count_24h": None,
                    "event_count_window": None,
                    "weighted_count_window": None,
                }
            )
        elif calendar is not None and event.proximity_score is None:
            flags.add(QualityFlag.MACRO_SOURCE_DEGRADED)
        eligible = tuple(
            item
            for item in series_tuple
            if item.data_quality in (DataQuality.GOOD, DataQuality.DEGRADED)
        )
        policy = policy_context(eligible, index, as_of_ns, self.settings)
        curve = yield_curve_context(eligible, index, as_of_ns, self.settings)
        if (
            any(item.category == MacroCategory.TREASURY_YIELD for item in series_tuple)
            and curve.spread_2s10s_pp is None
        ):
            flags.add(QualityFlag.MACRO_PARTIAL_CURVE)
        dimensions = tuple(
            _dimension(name, series_tuple) for name in ("inflation", "growth", "labor")
        )
        stress = _stress(
            eligible, curve.change_10y_1d_pp, curve.inversion_magnitude_pp, self.config
        )
        quality = (
            DataQuality.POOR
            if not visible_sources
            else DataQuality.DEGRADED
            if flags
            else DataQuality.GOOD
        )
        return MacroContextFeatureSet(
            as_of_ns=as_of_ns,
            config_fingerprint=fingerprint,
            sources=source_metadata,
            event=event,
            per_series=series_tuple,
            dimensions=dimensions,
            policy=policy,
            curve=curve,
            stress=stress,
            data_quality=quality,
            quality_flags=frozenset(flags),
        )
