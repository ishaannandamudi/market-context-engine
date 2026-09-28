import pytest
from pydantic import ValidationError

from mce.calendar import MarketSessionService
from mce.config.settings import IngestionSettings
from mce.errors import CausalityError, ProviderMalformedDataError
from mce.ingestion import IngestionService, normalize_observation
from mce.models.context import MarketContextSnapshot, SourceFreshness
from mce.models.enums import QualityFlag
from mce.models.market_data import BarObservation
from mce.models.symbol import normalize_symbol
from mce.replay import CausalDataStore, HistoricalReplay, ReplayClock
from mce.replay.store import observation_order_key
from mce.serialization import canonical_json_str
from tests.fixtures.clock import utc_ns
from tests.fixtures.session_samples import MINUTE_NS, OPEN_NS, minute_bar, quote, trade


def test_completed_bar_cannot_look_ahead() -> None:
    with pytest.raises(ValidationError, match="before bar_end_ns"):
        minute_bar().model_validate(
            {
                **minute_bar().model_dump(),
                "effective_timestamp_ns": OPEN_NS + 20_000_000_000,
            }
        )
    bar = minute_bar()
    replay = HistoricalReplay((bar,))
    replay.advance_to(OPEN_NS + 20_000_000_000)
    assert replay.store.latest_bar("SPY", OPEN_NS + 20_000_000_000) is None
    assert replay.advance_to(OPEN_NS + MINUTE_NS) == 1
    assert replay.store.latest_bar("SPY", OPEN_NS + MINUTE_NS) == bar
    assert replay.store.latest_bar("SPY", OPEN_NS + MINUTE_NS - 1) is None
    assert replay.store.bars("SPY", OPEN_NS, OPEN_NS + MINUTE_NS, OPEN_NS + MINUTE_NS - 1) == ()


@pytest.mark.parametrize(
    "changes",
    [
        {"bar_start_ns": None},
        {"bar_end_ns": None},
        {"bar_end_ns": OPEN_NS},
        {"event_timestamp_ns": OPEN_NS - 1},
        {"event_timestamp_ns": OPEN_NS + MINUTE_NS + 1},
    ],
)
def test_bar_interval_metadata_rejects_impossible_bounds(changes: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        BarObservation.model_validate({**minute_bar().model_dump(), **changes})


def test_delayed_bar_and_legacy_bar_behavior() -> None:
    delayed = minute_bar(effective_delay_ns=10)
    replay = HistoricalReplay((delayed,))
    replay.advance_to(OPEN_NS + MINUTE_NS)
    assert replay.store.latest_bar("SPY", OPEN_NS + MINUTE_NS) is None
    replay.advance_to(OPEN_NS + MINUTE_NS + 10)
    assert replay.store.latest_bar("SPY", OPEN_NS + MINUTE_NS + 10) == delayed
    legacy = BarObservation(
        symbol="SPY",
        source="synthetic",
        event_timestamp_ns=OPEN_NS,
        effective_timestamp_ns=OPEN_NS + MINUTE_NS,
        open=100.0,
        high=101.0,
        low=99.0,
        close=100.0,
        volume=1.0,
    )
    with pytest.raises(ValueError, match="requires completed bar bounds"):
        HistoricalReplay((legacy,))
    with pytest.raises(ValueError, match="requires bar_start_ns"):
        CausalDataStore().add(legacy)


def test_store_causal_queries_and_missing_vs_zero() -> None:
    store = CausalDataStore()
    first = minute_bar(volume=0.0)
    second = minute_bar(symbol="QQQ", start_ns=OPEN_NS + MINUTE_NS)
    ask = quote()
    tick = trade()
    for item in (first, second, ask, tick):
        assert store.add(item)
    assert store.latest_bar("SPY", OPEN_NS + MINUTE_NS - 1) is None
    assert store.latest_bar("SPY", OPEN_NS + MINUTE_NS).volume == 0.0
    assert store.latest_bar("IWM", OPEN_NS + MINUTE_NS) is None
    assert store.bars("SPY", OPEN_NS, OPEN_NS + MINUTE_NS, OPEN_NS + MINUTE_NS) == (first,)
    assert store.bars("QQQ", OPEN_NS, OPEN_NS + 2 * MINUTE_NS, OPEN_NS + MINUTE_NS) == ()
    assert store.latest_quote("SPY", OPEN_NS + 1) is None
    assert store.latest_quote("SPY", OPEN_NS + 2) == ask
    assert store.trades("SPY", OPEN_NS, OPEN_NS + 1, OPEN_NS + 2) == (tick,)
    assert store.trades("QQQ", OPEN_NS, OPEN_NS + 1, OPEN_NS + 2) == ()


def test_store_duplicates_trades_retained_and_bounded_eviction() -> None:
    store = CausalDataStore(max_observations=2)
    bar = minute_bar()
    assert store.add(bar)
    assert not store.add(bar)
    tick = trade()
    assert store.add(tick)
    assert store.add(tick)  # Identical trades may be two genuine prints.
    assert len(store) == 2
    assert store.latest_bar("SPY", OPEN_NS + MINUTE_NS) is None
    assert len(store.trades("SPY", OPEN_NS, OPEN_NS + 1, OPEN_NS + 2)) == 2
    assert store.add(bar)  # Eviction releases the duplicate key.


def test_replay_deterministic_out_of_order_and_equal_timestamps() -> None:
    a = quote(symbol="SPY")
    b = trade(symbol="QQQ")
    c = trade(symbol="SPY", sequence=2)
    records = (c, a, b)
    expected = tuple(sorted(records, key=observation_order_key))
    permuted = (b, c, a)
    assert expected == tuple(sorted(permuted, key=observation_order_key))
    first = HistoricalReplay(records)
    second = HistoricalReplay(permuted)
    target = OPEN_NS + 2
    assert first.advance_to(target) == 3
    assert second.advance_to(target) == 3
    assert first.store.trades("SPY", OPEN_NS, OPEN_NS + 1, target) == (
        second.store.trades("SPY", OPEN_NS, OPEN_NS + 1, target)
    )
    assert first.reset(target) == 3
    assert first.store.latest_quote("SPY", target) == a
    with pytest.raises(CausalityError):
        first.advance_to(target - 1)


def test_clock_explicit_reset_and_no_wall_clock() -> None:
    clock = ReplayClock(OPEN_NS)
    clock.advance_to(OPEN_NS + 1)
    assert clock.as_of_ns == OPEN_NS + 1
    with pytest.raises(CausalityError):
        clock.advance_to(OPEN_NS)
    clock.reset(OPEN_NS)
    assert clock.as_of_ns == OPEN_NS
    with pytest.raises(ValueError):
        ReplayClock(-1)
    with pytest.raises(ValueError):
        clock.reset(-1)


def test_normalization_preserves_provider_symbol_and_rejects_malformed() -> None:
    raw = minute_bar().model_dump()
    raw["symbol"] = " spy "
    normalized = normalize_observation(BarObservation, raw)
    assert normalized.symbol == "SPY"
    assert normalized.provider_symbol == " spy "
    assert '"provider_symbol":" spy "' in canonical_json_str(normalized)
    with pytest.raises(ProviderMalformedDataError):
        normalize_observation(BarObservation, {**raw, "close": float("nan")})
    with pytest.raises(ProviderMalformedDataError):
        normalize_observation(BarObservation, {**raw, "symbol": "   "})


def test_symbol_and_provider_sequence_contract() -> None:
    assert normalize_symbol(" brk.b ") == "BRK.B"
    assert normalize_symbol("spy  240119c") == "SPY  240119C"
    with pytest.raises(ValueError):
        normalize_symbol(123)
    with pytest.raises(ValueError):
        normalize_symbol("SPY\x00")
    with pytest.raises(ValidationError):
        type(trade()).model_validate({**trade().model_dump(), "provider_sequence": -1})
    assert type(trade()).model_validate_json(canonical_json_str(trade())) == trade()


def test_ingestion_sessions_and_rejection(calendar: MarketSessionService) -> None:
    pre = quote(event_ns=utc_ns("2024-01-03T13:00:00Z"))
    regular = minute_bar()
    holiday = trade(event_ns=utc_ns("2024-07-04T17:00:00Z"))
    service = IngestionService(CausalDataStore(), calendar)
    result = service.ingest((holiday, regular, pre, regular))
    assert result.accepted_count == 3
    assert result.duplicate_count == 1
    assert result.rejected_count == 0
    assert QualityFlag.OUTSIDE_SESSION in result.quality_flags
    assert result.symbols == ("SPY",)
    assert type(result).model_validate_json(canonical_json_str(result)) == result
    second = service.ingest((regular,))
    assert second.duplicate_count == 1
    assert QualityFlag.MISSING_DATA not in second.quality_flags
    assert service.store.trades(
        "SPY",
        holiday.event_timestamp_ns,
        holiday.event_timestamp_ns + 1,
        holiday.effective_timestamp_ns,
    )[0].quality_flags == {QualityFlag.OUTSIDE_SESSION}
    strict = IngestionService(
        CausalDataStore(), calendar, IngestionSettings(allow_extended_hours=False)
    )
    excluded = strict.ingest((pre, regular))
    assert excluded.accepted_count == 1 and excluded.rejected_count == 1
    assert QualityFlag.REJECTED_DATA in excluded.quality_flags


def test_ingestion_empty_and_unbounded_bar_are_explicit(calendar: MarketSessionService) -> None:
    service = IngestionService(CausalDataStore(), calendar)
    assert service.ingest(()).quality_flags == {QualityFlag.MISSING_DATA}
    legacy = BarObservation(
        symbol="SPY",
        source="synthetic",
        event_timestamp_ns=OPEN_NS,
        effective_timestamp_ns=OPEN_NS,
        open=1.0,
        high=1.0,
        low=1.0,
        close=1.0,
        volume=0.0,
    )
    result = service.ingest((legacy,))
    assert result.accepted_count == 0 and result.rejected_count == 1
    assert QualityFlag.TIMESTAMP_MISMATCH in result.quality_flags


def test_bar_session_uses_interval_not_close_label(calendar: MarketSessionService) -> None:
    close_ns = utc_ns("2024-01-03T21:00:00Z")
    final_regular = BarObservation.model_validate(
        {
            **minute_bar().model_dump(),
            "bar_start_ns": close_ns - MINUTE_NS,
            "bar_end_ns": close_ns,
            "event_timestamp_ns": close_ns,
            "effective_timestamp_ns": close_ns,
        }
    )
    strict = IngestionService(
        CausalDataStore(), calendar, IngestionSettings(allow_extended_hours=False)
    )
    assert strict.ingest((final_regular,)).accepted_count == 1
    crossing = BarObservation.model_validate(
        {
            **minute_bar().model_dump(),
            "bar_start_ns": close_ns - 30_000_000_000,
            "bar_end_ns": close_ns + 30_000_000_000,
            "event_timestamp_ns": close_ns - 30_000_000_000,
            "effective_timestamp_ns": close_ns + 30_000_000_000,
        }
    )
    permissive = IngestionService(CausalDataStore(), calendar).ingest((crossing,))
    assert permissive.accepted_count == 1
    assert QualityFlag.TIMESTAMP_MISMATCH in permissive.quality_flags
    disallowed = strict.ingest((crossing,))
    assert disallowed.rejected_count == 1
    assert QualityFlag.OUTSIDE_SESSION in disallowed.quality_flags


def test_market_snapshot_rejects_future_source_after_replay() -> None:
    bar = minute_bar()
    with pytest.raises(ValidationError, match="future data"):
        MarketContextSnapshot(
            as_of_ns=OPEN_NS + 20_000_000_000,
            source_freshness=(
                SourceFreshness(
                    source="synthetic",
                    latest_effective_timestamp_ns=bar.effective_timestamp_ns,
                    age_ns=0,
                    is_stale=False,
                ),
            ),
        )


def test_unknown_calendar_session_is_preserved_and_flagged(calendar: MarketSessionService) -> None:
    outside = quote(event_ns=utc_ns("2040-01-03T17:00:00Z"))
    service = IngestionService(CausalDataStore(), calendar)
    result = service.ingest((outside,))
    assert result.accepted_count == 1
    assert QualityFlag.SESSION_UNKNOWN in result.quality_flags
    assert service.store.latest_quote("SPY", outside.effective_timestamp_ns) is not None


def test_store_rejects_invalid_ranges() -> None:
    with pytest.raises(ValueError):
        CausalDataStore(max_observations=0)
    store = CausalDataStore()
    with pytest.raises(ValueError):
        store.bars("SPY", 2, 1, 1)
    with pytest.raises(ValueError):
        store.latest_quote("SPY", -1)
