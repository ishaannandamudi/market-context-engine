"""Isolated JSON-compatible boundary; source-specific mapping belongs here."""

from collections.abc import Mapping
from json import dumps
from typing import Any

from mce.features.options_context.contracts import OptionsMarketContextInput


class OptionsContextAdapter:
    """Parse version-1.0 payloads without importing the options engine package."""

    def translate(self, payload: Mapping[str, Any]) -> OptionsMarketContextInput:
        return OptionsMarketContextInput.model_validate_json(dumps(dict(payload), allow_nan=False))
