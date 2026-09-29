"""Canonical, provider-neutral sector identifiers."""

from re import fullmatch
from typing import Annotated

from pydantic import BeforeValidator


def normalize_sector(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("sector must be a string")
    name = value.strip().lower().replace(" ", "_")
    if fullmatch(r"[a-z][a-z0-9_]*", name) is None:
        raise ValueError("sector must have a canonical name")
    return name


SectorName = Annotated[str, BeforeValidator(normalize_sector)]
