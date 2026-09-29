"""Explicit prior-state bands; no hidden mutable classifier memory."""

from dataclasses import replace

from mce.config.settings import RegimeSettings
from mce.features.regime.classifier import DimensionLabels
from mce.features.regime.models import MarketRegimeState, RegimeEvidence
from mce.features.regime.taxonomy import HeadlineRegime, SectorRotationState, StressState
from mce.models.enums import CorrelationRegime, DispersionRegime, LiquidityRegime, VolatilityRegime


def stabilize_dimensions(
    raw: DimensionLabels,
    evidence: RegimeEvidence,
    prior: MarketRegimeState | None,
    config: RegimeSettings,
) -> tuple[DimensionLabels, tuple[str, ...]]:
    if prior is None:
        return raw, ()
    labels = raw
    held: list[str] = []
    band = config.hysteresis_band
    v = evidence.volatility_score
    if v is not None:
        if (
            prior.volatility_regime == VolatilityRegime.EXTREME
            and raw.volatility == VolatilityRegime.HIGH
            and v >= config.volatility_extreme - band
        ):
            labels = replace(labels, volatility=VolatilityRegime.EXTREME)
            held.append("volatility")
        elif (
            prior.volatility_regime == VolatilityRegime.HIGH
            and raw.volatility == VolatilityRegime.NORMAL
            and v >= config.volatility_high - band
        ):
            labels = replace(labels, volatility=VolatilityRegime.HIGH)
            held.append("volatility")
        elif (
            prior.volatility_regime == VolatilityRegime.LOW
            and raw.volatility == VolatilityRegime.NORMAL
            and v <= config.volatility_low + band
        ):
            labels = replace(labels, volatility=VolatilityRegime.LOW)
            held.append("volatility")
    liq = evidence.liquidity_score
    if liq is not None:
        if (
            prior.liquidity_regime == LiquidityRegime.STRESSED
            and raw.liquidity == LiquidityRegime.THIN
            and liq <= config.liquidity_stressed + band
        ):
            labels = replace(labels, liquidity=LiquidityRegime.STRESSED)
            held.append("liquidity")
        elif (
            prior.liquidity_regime == LiquidityRegime.THIN
            and raw.liquidity == LiquidityRegime.NORMAL
            and liq <= config.liquidity_degraded + band
        ):
            labels = replace(labels, liquidity=LiquidityRegime.THIN)
            held.append("liquidity")
        elif (
            prior.liquidity_regime == LiquidityRegime.HEALTHY
            and raw.liquidity == LiquidityRegime.NORMAL
            and liq >= config.liquidity_healthy - band
        ):
            labels = replace(labels, liquidity=LiquidityRegime.HEALTHY)
            held.append("liquidity")
    r = evidence.correlation_score
    if r is not None:
        if (
            prior.correlation_regime == CorrelationRegime.EXTREME
            and raw.correlation == CorrelationRegime.HIGH
            and r >= config.correlation_systemic - band
        ):
            labels = replace(labels, correlation=CorrelationRegime.EXTREME)
            held.append("correlation")
        elif (
            prior.correlation_regime == CorrelationRegime.HIGH
            and raw.correlation == CorrelationRegime.NORMAL
            and r >= config.correlation_high - band
        ):
            labels = replace(labels, correlation=CorrelationRegime.HIGH)
            held.append("correlation")
        elif (
            prior.correlation_regime == CorrelationRegime.LOW
            and raw.correlation == CorrelationRegime.NORMAL
            and r <= config.correlation_low + band
        ):
            labels = replace(labels, correlation=CorrelationRegime.LOW)
            held.append("correlation")
    x = evidence.dispersion_score
    if x is not None:
        if (
            prior.dispersion_regime == DispersionRegime.EXTREME
            and raw.dispersion == DispersionRegime.HIGH
            and x >= config.dispersion_extreme - band
        ):
            labels = replace(labels, dispersion=DispersionRegime.EXTREME)
            held.append("dispersion")
        elif (
            prior.dispersion_regime == DispersionRegime.HIGH
            and raw.dispersion == DispersionRegime.NORMAL
            and x >= config.dispersion_high - band
        ):
            labels = replace(labels, dispersion=DispersionRegime.HIGH)
            held.append("dispersion")
        elif (
            prior.dispersion_regime == DispersionRegime.LOW
            and raw.dispersion == DispersionRegime.NORMAL
            and x <= config.dispersion_low + band
        ):
            labels = replace(labels, dispersion=DispersionRegime.LOW)
            held.append("dispersion")
    s = evidence.stress_score
    if s is not None:
        if (
            prior.stress_regime == StressState.SYSTEMIC
            and raw.stress == StressState.HIGH
            and s >= config.stress_systemic - band
            and evidence.stressed_member_fraction is not None
            and evidence.stressed_member_fraction >= config.systemic_member_fraction
        ):
            labels = replace(labels, stress=StressState.SYSTEMIC)
            held.append("stress")
        elif (
            prior.stress_regime == StressState.HIGH
            and raw.stress == StressState.ELEVATED
            and s >= config.stress_high - band
        ):
            labels = replace(labels, stress=StressState.HIGH)
            held.append("stress")
        elif (
            prior.stress_regime == StressState.ELEVATED
            and raw.stress == StressState.CALM
            and s >= config.stress_elevated - band
        ):
            labels = replace(labels, stress=StressState.ELEVATED)
            held.append("stress")
    rotation = evidence.sector_rotation_velocity
    if (
        rotation is not None
        and prior.sector_rotation_state == SectorRotationState.ACTIVE_ROTATION
        and raw.sector == SectorRotationState.MIXED
        and rotation >= config.rotation_active - band
    ):
        labels = replace(labels, sector=SectorRotationState.ACTIVE_ROTATION)
        held.append("sector")
    return labels, tuple(held)


def stabilize_headline(
    candidate: HeadlineRegime,
    evidence: RegimeEvidence,
    prior: MarketRegimeState | None,
    config: RegimeSettings,
    maximum_conflict: float,
) -> tuple[HeadlineRegime, bool]:
    if prior is None or prior.headline_regime in (HeadlineRegime.UNKNOWN, candidate):
        return candidate, False
    previous = prior.headline_regime
    s, liq, d, b, r, x, rotation = (
        evidence.stress_score,
        evidence.liquidity_score,
        evidence.market_direction_score,
        evidence.breadth_score,
        evidence.correlation_score,
        evidence.dispersion_score,
        evidence.sector_rotation_velocity,
    )
    band = config.hysteresis_band
    if candidate == HeadlineRegime.SYSTEMIC_STRESS:
        return candidate, False
    if previous == HeadlineRegime.SYSTEMIC_STRESS:
        hold = (
            s is not None
            and s >= config.stress_systemic - band
            and b is not None
            and b <= -max(0.0, config.breadth_confirmation - band)
            and (
                (r is not None and r >= config.correlation_high - band)
                or (liq is not None and liq <= config.liquidity_stressed + band)
            )
        )
    elif candidate in (HeadlineRegime.LIQUIDITY_STRESS, HeadlineRegime.TRANSITIONAL):
        return candidate, False
    elif previous == HeadlineRegime.LIQUIDITY_STRESS:
        hold = (
            liq is not None
            and liq <= config.liquidity_stressed + band
            and s is not None
            and s >= config.stress_high - band
        )
    elif previous == HeadlineRegime.RISK_ON:
        hold = (
            d is not None
            and d >= config.direction_exit
            and b is not None
            and b >= -config.breadth_confirmation
            and s is not None
            and s < config.stress_high
            and maximum_conflict < config.conflict_override_severity
        )
    elif previous == HeadlineRegime.RISK_OFF:
        hold = (
            d is not None
            and d <= -config.direction_exit
            and b is not None
            and b <= config.breadth_confirmation
            and s is not None
            and s < config.stress_systemic
            and maximum_conflict < config.conflict_override_severity
        )
    elif previous == HeadlineRegime.ROTATION:
        hold = (
            rotation is not None
            and rotation >= config.rotation_active - band
            and x is not None
            and x >= config.dispersion_high - band
            and s is not None
            and s < config.stress_high
        )
    elif previous == HeadlineRegime.RANGE_BOUND:
        hold = (
            d is not None
            and abs(d) < config.direction_entry
            and s is not None
            and s < config.stress_elevated + band
        )
    elif previous == HeadlineRegime.TRANSITIONAL:
        hold = maximum_conflict >= config.conflict_override_severity - band
    else:
        hold = False
    return (previous, True) if hold else (candidate, False)
