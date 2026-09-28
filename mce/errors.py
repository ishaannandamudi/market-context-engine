"""Stable exception categories for future adapters and orchestration."""


class MCEError(Exception):
    """Base error raised by MCE operations."""


class ConfigurationError(MCEError):
    """A configuration cannot be used."""


class ProviderError(MCEError):
    """A provider operation failed."""


class ProviderUnavailableError(ProviderError):
    """A provider cannot currently serve a request."""


class DataQualityError(MCEError):
    """Input quality is insufficient for the requested operation."""


class CausalityError(MCEError):
    """An operation would use information unavailable at its as-of time."""
