"""Provider-neutral symbol normalization without assuming alphabetic tickers."""

from typing import Annotated

from pydantic import BeforeValidator


def normalize_symbol(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("symbol must be a string")
    symbol = value.strip().upper()
    if not symbol or any(ord(char) < 32 for char in symbol):
        raise ValueError("symbol must be nonempty and contain no control characters")
    return symbol


Symbol = Annotated[str, BeforeValidator(normalize_symbol)]
