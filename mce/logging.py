"""Opt-in logging helpers; importing MCE never configures root logging."""

import logging
import time


class UTCFormatter(logging.Formatter):
    """Format logging timestamps in UTC when an application configures logging."""

    def converter(self, timestamp: float | None) -> time.struct_time:
        return time.gmtime(timestamp)


def get_logger(component: str | None = None) -> logging.Logger:
    """Return a predictably named library logger without adding handlers."""
    return logging.getLogger("mce" if component is None else f"mce.{component}")
