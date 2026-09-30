"""Stable categorical values in the public JSON contract."""

from enum import StrEnum


class MarketRegime(StrEnum):
    RISK_ON = "risk_on"
    RISK_OFF = "risk_off"
    MIXED = "mixed"
    NEUTRAL = "neutral"
    STRESSED = "stressed"
    UNKNOWN = "unknown"


class TrendRegime(StrEnum):
    STRONG_UP = "strong_up"
    UP = "up"
    SIDEWAYS = "sideways"
    DOWN = "down"
    STRONG_DOWN = "strong_down"
    UNKNOWN = "unknown"


class VolatilityRegime(StrEnum):
    VERY_LOW = "very_low"
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    EXTREME = "extreme"
    UNKNOWN = "unknown"


class CorrelationRegime(StrEnum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    EXTREME = "extreme"
    UNKNOWN = "unknown"


class LiquidityRegime(StrEnum):
    HEALTHY = "healthy"
    NORMAL = "normal"
    THIN = "thin"
    STRESSED = "stressed"
    UNKNOWN = "unknown"


class DispersionRegime(StrEnum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    EXTREME = "extreme"
    UNKNOWN = "unknown"


class BreadthRegime(StrEnum):
    STRONG_POSITIVE = "strong_positive"
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    STRONG_NEGATIVE = "strong_negative"
    UNKNOWN = "unknown"


class MarketSession(StrEnum):
    PRE_MARKET = "pre_market"
    REGULAR = "regular"
    AFTER_HOURS = "after_hours"
    CLOSED = "closed"
    UNKNOWN = "unknown"


class BarInterval(StrEnum):
    ONE_MINUTE = "1m"
    FIVE_MINUTES = "5m"
    DAILY = "1d"


class DataQuality(StrEnum):
    GOOD = "good"
    DEGRADED = "degraded"
    POOR = "poor"
    UNUSABLE = "unusable"
    UNKNOWN = "unknown"


class QualityFlag(StrEnum):
    STALE_DATA = "stale_data"
    PARTIAL_UNIVERSE = "partial_universe"
    MISSING_BREADTH = "missing_breadth"
    MISSING_SECTOR_DATA = "missing_sector_data"
    MISSING_VOLATILITY_DATA = "missing_volatility_data"
    MISSING_OPTIONS_CONTEXT = "missing_options_context"
    PROVIDER_DEGRADED = "provider_degraded"
    INSUFFICIENT_HISTORY = "insufficient_history"
    SESSION_UNKNOWN = "session_unknown"
    TIMESTAMP_MISMATCH = "timestamp_mismatch"
    SOURCE_STALE = "source_stale"
    LOW_COVERAGE = "low_coverage"
    MISSING_DATA = "missing_data"
    REJECTED_DATA = "rejected_data"
    OUTSIDE_SESSION = "outside_session"
    INSUFFICIENT_WINDOW_COVERAGE = "insufficient_window_coverage"
    HORIZON_DISAGREEMENT = "horizon_disagreement"
    INDEX_DISAGREEMENT = "index_disagreement"
    FACTOR_IDENTIFICATION_WEAK = "factor_identification_weak"
    INSUFFICIENT_NETWORK_COVERAGE = "insufficient_network_coverage"
    NETWORK_DISCONNECTED = "network_disconnected"
    COMMUNITY_STRUCTURE_WEAK = "community_structure_weak"
    NETWORK_TOPOLOGY_DEGRADED = "network_topology_degraded"
    INSUFFICIENT_SEASONAL_HISTORY = "insufficient_seasonal_history"
    OPTIONS_MODE_LIMITED = "options_mode_limited"
    OPTIONS_SURFACE_SPARSE = "options_surface_sparse"
    OPTIONS_CIRCULAR_EVIDENCE = "options_circular_evidence"
    OPTIONS_HORIZON_MISMATCH = "options_horizon_mismatch"
    OPTIONS_UNDERLYING_PRICE_MISSING = "options_underlying_price_missing"
    INSUFFICIENT_MACRO_HISTORY = "insufficient_macro_history"
    MACRO_SOURCE_DEGRADED = "macro_source_degraded"
    MACRO_EVENT_CONTEXT_UNAVAILABLE = "macro_event_context_unavailable"
    MACRO_SERIES_STALE = "macro_series_stale"
    MACRO_PARTIAL_CURVE = "macro_partial_curve"


class EventRiskStatus(StrEnum):
    UNKNOWN = "unknown"
    UNAVAILABLE = "unavailable"
    NONE_SCHEDULED = "none_scheduled"
    SCHEDULED = "scheduled"
