"""Local, availability-aware filtered HMM context selection."""

from pydantic import model_validator

from mce.features.ensemble.contracts import FilteredHMMContext
from mce.models._base import FrozenModel
from mce.models.time import TimestampNs


class FilteredHMMRecord(FrozenModel):
    available_at_ns: TimestampNs
    context: FilteredHMMContext

    @model_validator(mode="after")
    def validate_availability(self) -> "FilteredHMMRecord":
        if self.available_at_ns < self.context.inference.as_of_ns:
            raise ValueError("HMM inference cannot be available before its calculation clock")
        return self


class FilteredHMMRegistry:
    """No pickle, loading, fitting, or implied future artifact availability."""

    def __init__(self, records: tuple[FilteredHMMRecord, ...]) -> None:
        keys = [(item.context.inference.as_of_ns, item.available_at_ns) for item in records]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate filtered HMM context record")
        self.records = tuple(
            sorted(
                records, key=lambda item: (item.available_at_ns, item.context.inference.as_of_ns)
            )
        )

    def select(self, as_of_ns: int) -> FilteredHMMContext | None:
        if as_of_ns < 0:
            raise ValueError("as_of_ns must be nonnegative")
        return next(
            (
                item.context
                for item in reversed(self.records)
                if item.available_at_ns <= as_of_ns and item.context.inference.as_of_ns <= as_of_ns
            ),
            None,
        )
