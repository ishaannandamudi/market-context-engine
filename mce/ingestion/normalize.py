"""Strict adapter-boundary normalization without repairing market values."""

from collections.abc import Mapping
from typing import TypeVar

from pydantic import ValidationError

from mce.errors import ProviderMalformedDataError
from mce.models.market_data import MarketObservation

ObservationT = TypeVar("ObservationT", bound=MarketObservation)


def normalize_observation(model: type[ObservationT], payload: Mapping[str, object]) -> ObservationT:
    """Validate a provider record and retain a raw provider symbol when normalized."""
    try:
        fields = dict(payload)
        raw_symbol = fields.get("symbol")
        if isinstance(raw_symbol, str) and fields.get("provider_symbol") is None:
            fields["provider_symbol"] = raw_symbol
        return model.model_validate(fields)
    except (ValidationError, ValueError, TypeError) as exc:
        raise ProviderMalformedDataError(f"invalid {model.__name__} record") from exc
