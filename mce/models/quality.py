"""Shared score, source and quality field conventions."""

from typing import Annotated

from pydantic import Field, field_serializer, field_validator

from mce.models._base import FrozenModel
from mce.models.enums import QualityFlag

DirectionalScore = Annotated[float, Field(ge=-1.0, le=1.0, allow_inf_nan=False)]
UnitScore = Annotated[float, Field(ge=0.0, le=1.0, allow_inf_nan=False)]
ConfidenceScore = UnitScore
NonNegativeFinite = Annotated[float, Field(ge=0.0, allow_inf_nan=False)]
FiniteNumber = Annotated[float, Field(allow_inf_nan=False)]
SourceId = Annotated[str, Field(min_length=1, pattern=r"^[a-z][a-z0-9._-]*$")]


class QualityTaggedModel(FrozenModel):
    """Deduplicated flags with a stable external order."""

    quality_flags: frozenset[QualityFlag] = Field(default_factory=frozenset)

    @field_validator("quality_flags", mode="before")
    @classmethod
    def normalize_quality_flags(cls, value: object) -> object:
        if isinstance(value, (list, tuple, set, frozenset)):
            return frozenset(QualityFlag(flag) for flag in value)
        return value

    @field_serializer("quality_flags")
    def serialize_quality_flags(self, flags: frozenset[QualityFlag]) -> list[str]:
        return sorted(flag.value for flag in flags)
