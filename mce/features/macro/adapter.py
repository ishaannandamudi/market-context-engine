"""JSON-compatible transport boundary independent of any provider client."""

from collections.abc import Mapping
from json import dumps
from typing import Any

from mce.features.macro.contracts import MacroContextInput


class MacroContextAdapter:
    def translate(self, payload: Mapping[str, Any]) -> MacroContextInput:
        return MacroContextInput.model_validate_json(dumps(dict(payload), allow_nan=False))
