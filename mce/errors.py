"""Stable exception categories for future adapters and orchestration."""


class MCEError(Exception):
    """Base error raised by MCE operations."""


class ConfigurationError(MCEError):
    """A configuration cannot be used."""


class ProviderError(MCEError):
    """A provider operation failed."""


class ProviderUnavailableError(ProviderError):
    """A provider cannot currently serve a request."""


class ProviderAuthenticationError(ProviderError):
    """Provider credentials or configuration are invalid."""


class ProviderEntitlementError(ProviderError):
    """The requested capability is not entitled or supported."""


class ProviderTimeoutError(ProviderError):
    """The provider did not respond in time."""


class ProviderRateLimitError(ProviderError):
    """The provider rejected a request due to a rate limit."""


class ProviderMalformedDataError(ProviderError):
    """The provider emitted a record that cannot be normalized."""


class DataQualityError(MCEError):
    """Input quality is insufficient for the requested operation."""


class CausalityError(MCEError):
    """An operation would use information unavailable at its as-of time."""


class CalendarRangeError(MCEError):
    """A date lies outside the configured exchange-calendar range."""
