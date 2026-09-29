"""Detailed descriptive taxonomy; public snapshot enums remain schema-1.0 stable."""

from enum import StrEnum


class HeadlineRegime(StrEnum):
    RISK_ON = "risk_on"
    RISK_OFF = "risk_off"
    RANGE_BOUND = "range_bound"
    ROTATION = "rotation"
    SYSTEMIC_STRESS = "systemic_stress"
    LIQUIDITY_STRESS = "liquidity_stress"
    TRANSITIONAL = "transitional"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class DirectionState(StrEnum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class BreadthState(StrEnum):
    BROAD_POSITIVE = "broad_positive"
    NARROW_POSITIVE = "narrow_positive"
    BROAD_NEGATIVE = "broad_negative"
    NARROW_NEGATIVE = "narrow_negative"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class StressState(StrEnum):
    CALM = "calm"
    ELEVATED = "elevated"
    HIGH = "high"
    SYSTEMIC = "systemic"
    UNKNOWN = "unknown"


class SectorRotationState(StrEnum):
    STABLE_LEADERSHIP = "stable_leadership"
    ACTIVE_ROTATION = "active_rotation"
    FRAGMENTED = "fragmented"
    MIXED = "mixed"
    UNKNOWN = "unknown"
