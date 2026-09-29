"""Exchange-session train/validation/test and walk-forward boundaries."""

from dataclasses import dataclass
from datetime import date
from hashlib import sha256

from mce.calendar.service import MarketSessionService
from mce.config.settings import StatisticalSplitSettings
from mce.serialization.canonical import canonical_json_bytes
from mce.statistical.dataset import StatisticalDataset


@dataclass(frozen=True, slots=True)
class TemporalSplit:
    train_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]
    test_indices: tuple[int, ...]
    train_dates: tuple[date, date]
    validation_dates: tuple[date, date]
    test_dates: tuple[date, date]
    embargo_sessions: int
    dataset_fingerprint: str
    spec_fingerprint: str

    @property
    def fingerprint(self) -> str:
        return sha256(
            canonical_json_bytes(
                {
                    "train": self.train_indices,
                    "validation": self.validation_indices,
                    "test": self.test_indices,
                    "train_dates": self.train_dates,
                    "validation_dates": self.validation_dates,
                    "test_dates": self.test_dates,
                    "embargo": self.embargo_sessions,
                    "dataset": self.dataset_fingerprint,
                    "spec": self.spec_fingerprint,
                }
            )
        ).hexdigest()


def _sessions(dataset: StatisticalDataset, calendar: MarketSessionService) -> tuple[date, ...]:
    if not dataset.rows:
        return ()
    first, last = dataset.rows[0].trading_date, dataset.rows[-1].trading_date
    dates = [first]
    while dates[-1] < last:
        dates.append(calendar.next_trading_day(dates[-1]))
    return tuple(dates)


def walk_forward_splits(
    dataset: StatisticalDataset,
    calendar: MarketSessionService,
    settings: StatisticalSplitSettings,
) -> tuple[TemporalSplit, ...]:
    """All boundaries count exchange sessions, including sessions without rows."""
    sessions = _sessions(dataset, calendar)
    if not sessions:
        raise ValueError("cannot split empty dataset")
    dates_to_indices: dict[date, list[int]] = {}
    for index, row in enumerate(dataset.rows):
        dates_to_indices.setdefault(row.trading_date, []).append(index)
    dataset_fingerprint = dataset.fingerprint
    spec_fingerprint = sha256(canonical_json_bytes(settings)).hexdigest()
    result: list[TemporalSplit] = []
    step = 0
    while True:
        train_start = 0 if settings.mode == "expanding" else step * settings.step_sessions
        train_end = settings.train_sessions + step * settings.step_sessions
        val_start = train_end + settings.embargo_sessions
        val_end = val_start + settings.validation_sessions
        test_start = val_end + settings.embargo_sessions
        test_end = test_start + settings.test_sessions
        if test_end > len(sessions):
            break

        def indices(start: int, end: int) -> tuple[int, ...]:
            return tuple(i for day in sessions[start:end] for i in dates_to_indices.get(day, ()))

        train, validation, test = (
            indices(train_start, train_end),
            indices(val_start, val_end),
            indices(test_start, test_end),
        )
        if not train or not validation or not test:
            raise ValueError("split window has no observations")
        result.append(
            TemporalSplit(
                train_indices=train,
                validation_indices=validation,
                test_indices=test,
                train_dates=(sessions[train_start], sessions[train_end - 1]),
                validation_dates=(sessions[val_start], sessions[val_end - 1]),
                test_dates=(sessions[test_start], sessions[test_end - 1]),
                embargo_sessions=settings.embargo_sessions,
                dataset_fingerprint=dataset_fingerprint,
                spec_fingerprint=spec_fingerprint,
            )
        )
        step += 1
    if not result:
        raise ValueError("insufficient trading sessions for split specification")
    return tuple(result)
