"""MCE-20 DAG, causal snapshot, replay, failure, lineage, and local API contracts."""

import io
import json
import subprocess
import sys
from datetime import date
from pathlib import Path

import pytest
from pydantic import ValidationError

from mce.__main__ import main as cli_main
from mce.calendar.service import MarketSessionService
from mce.config.settings import MCEConfig
from mce.features.breadth.membership import MembershipTimeline, UniverseMembership
from mce.features.engine import TrendFeatureEngine
from mce.features.ensemble.contracts import FilteredHMMContext
from mce.features.macro.contracts import MacroCategory, MacroEventType, RevisionStatus
from mce.features.options_context.engine import OptionsContextFeatureEngine
from mce.features.sectors.membership import SectorMembership, SectorMembershipTimeline
from mce.features.volatility.engine import RealizedVolatilityEngine
from mce.models.enums import DataQuality
from mce.models.market_data import BarObservation, QuoteObservation
from mce.orchestration.artifacts import FilteredHMMRecord, FilteredHMMRegistry
from mce.orchestration.context import SuppliedContexts
from mce.orchestration.dependencies import STAGES, PipelineProviders, StageNode, topological_order
from mce.orchestration.models import PipelineError, PipelineErrorCode
from mce.orchestration.pipeline import MarketContextPipeline
from mce.orchestration.replay import MarketContextReplay, ReplayRequest
from mce.providers.base import ProviderHealth, ProviderStatus
from mce.replay.core import HistoricalReplay
from mce.replay.store import CausalDataStore
from mce.serialization.canonical import canonical_json_bytes
from mce.service.api import MarketContextService
from mce.service.http import MarketContextWSGI
from mce.statistical.models import LatentStateInference
from tests.fixtures.macro_scenarios import BASE, DAY_NS, batch, event, observation
from tests.fixtures.options_scenarios import options_input
from tests.fixtures.system_scenarios import bars


def _config() -> MCEConfig:
    return MCEConfig(
        _env_file=None,
        market={"universe": {"broad_market": ("SPY",), "breadth": ("AAA", "BBB")}},
    )


def _store(observations: tuple[BarObservation, ...], as_of_ns: int) -> CausalDataStore:
    replay = HistoricalReplay(observations, start_ns=as_of_ns)
    return replay.store


def test_dag_validation_and_registration_order() -> None:
    assert topological_order(STAGES) == topological_order(tuple(reversed(STAGES)))
    with pytest.raises(ValueError, match="duplicate"):
        topological_order((StageNode("a"), StageNode("a")))
    with pytest.raises(ValueError, match="unknown"):
        topological_order((StageNode("a", ("missing",)),))
    with pytest.raises(ValueError, match="cycle"):
        topological_order((StageNode("a", ("b",)), StageNode("b", ("a",))))
    config = _config()
    assert MarketContextPipeline(config, stages=tuple(reversed(STAGES))).stage_order == tuple(
        node.stage_id for node in STAGES
    )


def test_snapshot_reproducibility_lineage_and_optional_failure() -> None:
    config = _config()
    observations = bars(config)
    as_of_ns = observations[-1].effective_timestamp_ns
    store = _store(observations, as_of_ns)
    first = MarketContextPipeline(config).compute_snapshot(as_of_ns, store)
    second = MarketContextPipeline(config).compute_snapshot(as_of_ns, store)
    assert canonical_json_bytes(first.snapshot) == canonical_json_bytes(second.snapshot)
    assert first.snapshot_fingerprint == second.snapshot_fingerprint
    assert first.lineage == second.lineage
    assert first.diagnostics.duration_ns >= 0
    assert first.snapshot.schema_version == "1.0"
    assert "options." not in " ".join((first.snapshot.feature_values or {}).keys())
    assert {"options", "macro", "hmm", "network"} <= set(first.diagnostics.withheld_stages)
    assert first.snapshot.data_quality != DataQuality.GOOD


def test_missing_critical_market_data_rejected() -> None:
    with pytest.raises(PipelineError) as error:
        MarketContextPipeline(_config()).compute_snapshot(1, CausalDataStore())
    assert error.value.code == PipelineErrorCode.INSUFFICIENT_MARKET_DATA


def test_end_to_end_direction_rotation_options_and_macro_scenarios() -> None:
    config = MCEConfig(
        _env_file=None,
        market={"universe": {"broad_market": ("SPY",), "breadth": ("AAA", "BBB")}},
        features={
            "sectors": {
                "groups": {"Technology": ("AAA",), "Finance": ("BBB",)},
                "minimum_member_count": 1,
            }
        },
    )
    opened, closed = MarketSessionService(config.market).regular_session_bounds(date(2024, 12, 31))
    prior = tuple(
        BarObservation(
            symbol=symbol,
            source=config.provider.source,
            event_timestamp_ns=closed,
            effective_timestamp_ns=closed,
            bar_start_ns=opened,
            bar_end_ns=closed,
            open=price,
            high=price + 1,
            low=price - 1,
            close=price,
            volume=1000,
        )
        for symbol, price in (("SPY", 100), ("AAA", 110), ("BBB", 120))
    )
    up = bars(config, count=30, direction=1)
    down = bars(config, count=30, direction=-1)
    as_of_ns = up[-1].effective_timestamp_ns
    pipeline = MarketContextPipeline(config)
    risk_on = pipeline.compute_snapshot(as_of_ns, _store((*prior, *up), as_of_ns))
    risk_off = pipeline.compute_snapshot(as_of_ns, _store((*prior, *down), as_of_ns))
    rotation_bars = (
        *prior,
        *(item for item in up if item.symbol != "BBB"),
        *(item for item in down if item.symbol == "BBB"),
    )
    rotation = pipeline.compute_snapshot(as_of_ns, _store(rotation_bars, as_of_ns))
    assert risk_on.snapshot.feature_values["breadth.advancers"] == 2
    assert risk_off.snapshot.feature_values["breadth.decliners"] == 2
    assert rotation.snapshot.feature_values["breadth.advancers"] == 1
    assert rotation.snapshot.feature_values["breadth.decliners"] == 1
    assert (
        rotation.snapshot.sector_scores["technology"] > rotation.snapshot.sector_scores["finance"]
    )
    stress = pipeline.compute_snapshot(
        as_of_ns,
        _store((*prior, *down), as_of_ns),
        SuppliedContexts(
            options_inputs=(
                options_input(
                    clock=as_of_ns,
                    atm_ivs=(0.40, 0.45, 0.50),
                    put_iv=0.60,
                    call_iv=0.30,
                ),
            ),
            macro_inputs=(
                batch(
                    clock=as_of_ns,
                    events=(
                        event(
                            "FOMC-fixture",
                            MacroEventType.FOMC_RATE_DECISION,
                            as_of_ns + 3_600_000_000_000,
                            announced=as_of_ns - DAY_NS,
                        ),
                    ),
                ),
            ),
        ),
    )
    names = tuple((stress.snapshot.feature_values or {}).keys())
    assert any(name.startswith("options.") for name in names)
    assert any(name.startswith("macro.") for name in names)
    assert stress.snapshot.feature_values["trend.return.SPY.1m"] < 0


def test_future_bars_quotes_membership_and_sector_changes_leave_past_unchanged() -> None:
    config = _config()
    observations = bars(config)
    as_of_ns = observations[-1].effective_timestamp_ns
    future = as_of_ns + 60_000_000_000
    membership = MembershipTimeline(
        (UniverseMembership(symbol="AAA"), UniverseMembership(symbol="BBB"))
    )
    sectors = SectorMembershipTimeline(
        (
            SectorMembership(symbol="AAA", sector="Technology"),
            SectorMembership(symbol="BBB", sector="Finance"),
        )
    )
    base_pipeline = MarketContextPipeline(
        config, PipelineProviders(breadth_membership=membership, sector_membership=sectors)
    )
    base = base_pipeline.compute_snapshot(as_of_ns, _store(observations, as_of_ns))
    future_membership = MembershipTimeline(
        (
            *membership.records,
            UniverseMembership(symbol="CCC", effective_from_ns=future, available_at_ns=future),
        )
    )
    future_sectors = SectorMembershipTimeline(
        (
            *sectors.records,
            SectorMembership(
                symbol="CCC", sector="Technology", effective_from_ns=future, available_at_ns=future
            ),
        )
    )
    next_bar = BarObservation(
        symbol="SPY",
        event_timestamp_ns=future,
        effective_timestamp_ns=future,
        source=config.provider.source,
        bar_start_ns=as_of_ns,
        bar_end_ns=future,
        open=100,
        high=200,
        low=99,
        close=199,
        volume=1000,
    )
    next_quote = QuoteObservation(
        symbol="SPY",
        event_timestamp_ns=future,
        effective_timestamp_ns=future,
        source=config.provider.source,
        bid=199,
        ask=200,
    )
    mutated = CausalDataStore()
    for item in (*observations, next_bar, next_quote):
        mutated.add(item)
    updated = MarketContextPipeline(
        config,
        PipelineProviders(breadth_membership=future_membership, sector_membership=future_sectors),
    ).compute_snapshot(as_of_ns, mutated)
    assert canonical_json_bytes(updated.snapshot) == canonical_json_bytes(base.snapshot)
    assert updated.snapshot_fingerprint == base.snapshot_fingerprint


def test_replay_cadence_reproducibility_and_limits() -> None:
    config = _config()
    observations = bars(config, count=3)
    start = observations[0].effective_timestamp_ns
    end = observations[-1].effective_timestamp_ns
    pipeline = MarketContextPipeline(config)
    replay = MarketContextReplay(pipeline)
    request = ReplayRequest(start_ns=start, end_ns=end, cadence="completed_bar")
    first = replay.run(observations, request)
    second = MarketContextReplay(MarketContextPipeline(config)).run(observations, request)
    assert len(first.results) == 3
    assert [item.snapshot_fingerprint for item in first.results] == [
        item.snapshot_fingerprint for item in second.results
    ]
    explicit = ReplayRequest(start_ns=start, end_ns=end, cadence="timestamps", timestamps=(end,))
    assert len(replay.run(observations, explicit).results) == 1
    with pytest.raises(PipelineError) as error:
        replay.run(
            observations,
            ReplayRequest(start_ns=start, end_ns=end, cadence="completed_bar", max_snapshots=2),
        )
    assert error.value.code == PipelineErrorCode.REPLAY_RANGE_INVALID
    with pytest.raises(ValidationError):
        ReplayRequest(start_ns=end, end_ns=start, cadence="session_close")


def test_replay_calendar_cadences_and_request_validation() -> None:
    config = _config()
    observations = bars(config, count=3)
    start = observations[0].effective_timestamp_ns
    end = observations[-1].effective_timestamp_ns
    replay = MarketContextReplay(MarketContextPipeline(config))
    minute = ReplayRequest(
        start_ns=start,
        end_ns=end,
        cadence="n_minute",
        cadence_minutes=1,
    )
    assert replay.schedule(observations, minute) == (start, start + 60_000_000_000, end)
    with pytest.raises(PipelineError, match="too many snapshots"):
        replay.schedule(observations, minute.model_copy(update={"max_snapshots": 2}))
    session_end = replay.pipeline.calendar.regular_session_bounds(date(2025, 1, 2))[1]
    close = ReplayRequest(start_ns=start, end_ns=session_end, cadence="session_close")
    assert replay.schedule(observations, close) == (session_end,)
    for invalid in (
        {"cadence": "n_minute"},
        {"cadence": "completed_bar", "cadence_minutes": 5},
        {"cadence": "timestamps", "timestamps": (end, start)},
        {"cadence": "timestamps", "timestamps": (end + 1,)},
        {"cadence": "session_close", "timestamps": (end,)},
    ):
        with pytest.raises(ValidationError):
            ReplayRequest(start_ns=start, end_ns=end, **invalid)


def test_service_routes_safe_errors_and_wsgi() -> None:
    config = _config()
    observations = bars(config, count=3)
    as_of_ns = observations[-1].effective_timestamp_ns
    service = MarketContextService(MarketContextPipeline(config), observations)
    for path in ("/health", "/ready", "/version", "/api/context/schema"):
        status, payload = service.handle("GET", path)
        assert status == 200 and json.loads(payload)
    status, payload = service.handle(
        "POST", "/api/context/snapshot", canonical_json_bytes({"as_of_ns": as_of_ns})
    )
    assert status == 200 and json.loads(payload)["snapshot"]["schema_version"] == "1.0"
    for body in (b"{bad", b'{"as_of_ns":1,"as_of_ns":2}', b'{"as_of_ns":true}'):
        status, payload = service.handle("POST", "/api/context/snapshot", body)
        assert status == 422 and b"Traceback" not in payload and b"/Users/" not in payload
    status, payload = service.handle(
        "POST",
        "/api/context/snapshot",
        canonical_json_bytes({"as_of_ns": as_of_ns, "schema_version": "9.0"}),
    )
    assert status == 422 and json.loads(payload)["error"]["code"] == "SCHEMA_MISMATCH"
    status, payload = service.handle(
        "POST", "/api/context/snapshot", canonical_json_bytes({"as_of_ns": -1})
    )
    assert status == 422 and json.loads(payload)["error"]["code"] == "INVALID_AS_OF"
    response = {}

    def start_response(status: str, headers: list[tuple[str, str]]) -> None:
        response["status"] = status

    output = MarketContextWSGI(service)(
        {
            "REQUEST_METHOD": "GET",
            "PATH_INFO": "/health",
            "CONTENT_LENGTH": "0",
            "wsgi.input": io.BytesIO(),
        },
        start_response,
    )
    assert response["status"] == "200 OK" and b"".join(output) == b'{"status":"ok"}'


def test_request_scoped_membership_cache_calls_source_once() -> None:
    config = _config()
    observations = bars(config)
    as_of_ns = observations[-1].effective_timestamp_ns

    class Counted:
        def __init__(self) -> None:
            self.calls: list[int] = []
            self.timeline = MembershipTimeline.from_symbols(("AAA", "BBB"))

        def members(self, clock: int):
            self.calls.append(clock)
            return self.timeline.members(clock)

    provider = Counted()
    pipeline = MarketContextPipeline(
        config,
        PipelineProviders(
            breadth_membership=provider,
            correlation_membership=provider,
            dispersion_membership=provider,
            factor_membership=provider,
        ),
    )
    pipeline.compute_snapshot(as_of_ns, _store(observations, as_of_ns))
    assert provider.calls
    assert len(provider.calls) == len(set(provider.calls))


def test_required_upstream_engines_run_once_per_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    config = _config()
    observations = bars(config)
    as_of_ns = observations[-1].effective_timestamp_ns
    counts = {"trend": 0, "volatility": 0}
    trend_compute = TrendFeatureEngine.compute
    volatility_compute = RealizedVolatilityEngine.compute

    def counted_trend(self, store, as_of):
        counts["trend"] += 1
        return trend_compute(self, store, as_of)

    def counted_volatility(self, store, as_of):
        counts["volatility"] += 1
        return volatility_compute(self, store, as_of)

    monkeypatch.setattr(TrendFeatureEngine, "compute", counted_trend)
    monkeypatch.setattr(RealizedVolatilityEngine, "compute", counted_volatility)
    MarketContextPipeline(config).compute_snapshot(as_of_ns, _store(observations, as_of_ns))
    assert counts == {"trend": 1, "volatility": 1}


def test_disabled_optional_stages_are_marked_not_configured() -> None:
    config = MCEConfig(
        _env_file=None,
        market={"universe": {"broad_market": ("SPY",)}},
        features={
            "network": {"enabled": False},
            "seasonality": {"enabled": False},
            "regime_transition": {"enabled": False},
            "options_context": {"enabled": False},
            "macro": {"enabled": False},
            "ensemble": {"enabled": False},
        },
    )
    observations = bars(config, symbols=("SPY",))
    as_of_ns = observations[-1].effective_timestamp_ns
    result = MarketContextPipeline(config).compute_snapshot(
        as_of_ns, _store(observations, as_of_ns)
    )
    reasons = {stage.stage_id: stage.reason_code for stage in result.diagnostics.stage_results}
    assert all(
        reasons[name] == "NOT_CONFIGURED"
        for name in (
            "network",
            "seasonality",
            "regime_transition",
            "options",
            "macro",
            "hmm",
            "ensemble",
        )
    )
    names = tuple((result.snapshot.feature_values or {}).keys())
    assert not any(
        name.startswith(("network.", "seasonality.", "options.", "macro.", "ensemble."))
        for name in names
    )


def test_future_options_surface_and_macro_revision_do_not_change_past_snapshot() -> None:
    config = _config()
    observations = bars(config, day=date(2027, 1, 15))
    as_of_ns = observations[-1].effective_timestamp_ns
    store = _store(observations, as_of_ns)
    original_macro = observation(
        "CPI_YOY",
        MacroCategory.INFLATION,
        3.1,
        period_end=BASE - 30 * DAY_NS,
        released=BASE - DAY_NS,
    )
    future_revision = observation(
        "CPI_YOY",
        MacroCategory.INFLATION,
        3.8,
        period_end=BASE - 30 * DAY_NS,
        released=BASE - DAY_NS,
        vintage=as_of_ns + DAY_NS,
        status=RevisionStatus.REVISED,
    )
    current = SuppliedContexts(
        options_inputs=(options_input(clock=as_of_ns),),
        macro_inputs=(batch(observations=(original_macro,)),),
    )
    changed = SuppliedContexts(
        options_inputs=(options_input(clock=as_of_ns), options_input(clock=as_of_ns + DAY_NS)),
        macro_inputs=(
            batch(observations=(original_macro,)),
            batch(clock=as_of_ns + DAY_NS, observations=(future_revision,)),
        ),
    )
    pipeline = MarketContextPipeline(config)
    before = pipeline.compute_snapshot(as_of_ns, store, current)
    after = pipeline.compute_snapshot(as_of_ns, store, changed)
    assert before.snapshot_fingerprint == after.snapshot_fingerprint
    assert canonical_json_bytes(before.snapshot) == canonical_json_bytes(after.snapshot)


def test_future_precomputed_options_clock_is_rejected_even_without_ensemble() -> None:
    config = MCEConfig(
        _env_file=None,
        market={"universe": {"broad_market": ("SPY",)}},
        features={"ensemble": {"enabled": False}},
    )
    observations = bars(config, symbols=("SPY",), count=3)
    as_of_ns = observations[-1].effective_timestamp_ns
    future = as_of_ns + 60_000_000_000
    options = OptionsContextFeatureEngine(config).compute((options_input(clock=future),), future)
    with pytest.raises(ValueError, match="snapshot clock"):
        MarketContextPipeline(config).compute_snapshot(
            as_of_ns,
            _store(observations, as_of_ns),
            SuppliedContexts(options_context=options),
        )
    service = MarketContextService(
        MarketContextPipeline(config),
        observations,
        supplied_at=lambda _: SuppliedContexts(options_context=options),
    )
    status, payload = service.handle(
        "POST", "/api/context/snapshot", canonical_json_bytes({"as_of_ns": as_of_ns})
    )
    assert status == 422 and b"future" not in payload and b"/Users/" not in payload


def test_optional_context_stale_fields_are_cleared_at_next_snapshot() -> None:
    config = _config()
    observations = bars(config)
    first_ns = observations[-4].effective_timestamp_ns
    second_ns = observations[-1].effective_timestamp_ns
    store = _store(observations, second_ns)
    pipeline = MarketContextPipeline(config)
    with_options = pipeline.compute_snapshot(
        first_ns, store, SuppliedContexts(options_inputs=(options_input(clock=first_ns),))
    )
    without_options = pipeline.compute_snapshot(second_ns, store)
    assert any(name.startswith("options.") for name in (with_options.snapshot.feature_values or {}))
    assert not any(
        name.startswith("options.") for name in (without_options.snapshot.feature_values or {})
    )
    assert not any(
        name.startswith("macro.") for name in (without_options.snapshot.feature_values or {})
    )


def test_future_hmm_artifact_registry_and_pipeline_clock_validation() -> None:
    config = MCEConfig(
        _env_file=None,
        market={"universe": {"broad_market": ("SPY",)}},
        statistical={"enabled": True},
    )
    observations = bars(config, count=3)
    as_of_ns = observations[-1].effective_timestamp_ns
    inference = LatentStateInference(
        as_of_ns=as_of_ns,
        state_id=0,
        state_probabilities=(0.8, 0.2),
        mode="filtered",
        retrospective=False,
        training_end_ns=as_of_ns - 1,
        model_version="fixture/1",
        feature_schema_version="1.0",
    )
    hmm = FilteredHMMContext(inference=inference)
    registry = FilteredHMMRegistry((FilteredHMMRecord(available_at_ns=as_of_ns + 1, context=hmm),))
    assert registry.select(as_of_ns) is None
    assert registry.select(as_of_ns + 1) == hmm
    pipeline = MarketContextPipeline(config)
    base = pipeline.compute_snapshot(as_of_ns, _store(observations, as_of_ns))
    assert "hmm" in base.diagnostics.withheld_stages
    future_inference = LatentStateInference(
        as_of_ns=as_of_ns + 1,
        state_id=0,
        state_probabilities=(0.8, 0.2),
        mode="filtered",
        retrospective=False,
        training_end_ns=as_of_ns,
        model_version="fixture/1",
        feature_schema_version="1.0",
    )
    with pytest.raises(ValidationError, match="future HMM"):
        pipeline.compute_snapshot(
            as_of_ns,
            _store(observations, as_of_ns),
            SuppliedContexts(hmm=FilteredHMMContext(inference=future_inference)),
        )


def test_service_replay_limits_and_readiness_are_independent_of_liveness() -> None:
    config = _config()
    observations = bars(config, count=3)
    start = observations[0].effective_timestamp_ns
    end = observations[-1].effective_timestamp_ns
    service = MarketContextService(MarketContextPipeline(config), observations)
    request = ReplayRequest(start_ns=start, end_ns=end, cadence="completed_bar", max_snapshots=3)
    status, payload = service.handle("POST", "/api/context/replay", canonical_json_bytes(request))
    assert status == 200 and len(json.loads(payload)["snapshots"]) == 3
    oversized = request.model_copy(update={"max_snapshots": 101})
    status, payload = service.handle("POST", "/api/context/replay", canonical_json_bytes(oversized))
    assert status == 422 and json.loads(payload)["error"]["code"] == "REPLAY_RANGE_INVALID"
    status, payload = service.handle(
        "POST",
        "/api/context/replay",
        canonical_json_bytes({"start_ns": end, "end_ns": start, "cadence": "completed_bar"}),
    )
    assert status == 422 and json.loads(payload)["error"]["code"] == "REPLAY_RANGE_INVALID"

    class Down:
        provider_name = "fixture-market"

        def health(self) -> ProviderHealth:
            return ProviderHealth(status=ProviderStatus.UNAVAILABLE)

    unavailable = MarketContextService(
        MarketContextPipeline(config, PipelineProviders(market_health=Down())), observations
    )
    assert unavailable.handle("GET", "/health")[0] == 200
    assert unavailable.handle("GET", "/ready")[0] == 503


def test_service_never_echoes_provider_error_details(monkeypatch: pytest.MonkeyPatch) -> None:
    config = _config()
    observations = bars(config, count=3)
    as_of_ns = observations[-1].effective_timestamp_ns
    pipeline = MarketContextPipeline(config)

    def failure(*args, **kwargs):
        raise PipelineError(PipelineErrorCode.PROVIDER_UNAVAILABLE, "/private/credential.txt")

    monkeypatch.setattr(pipeline, "compute_snapshot", failure)
    service = MarketContextService(pipeline, observations)
    status, payload = service.handle(
        "POST", "/api/context/snapshot", canonical_json_bytes({"as_of_ns": as_of_ns})
    )
    assert status == 503
    assert json.loads(payload)["error"]["code"] == "PROVIDER_UNAVAILABLE"
    assert b"credential" not in payload and b"private" not in payload


def test_cli_canonical_export_and_no_accidental_overwrite(tmp_path: Path) -> None:
    config = _config()
    observations = bars(config, count=3)
    config_path = tmp_path / "config.json"
    input_path = tmp_path / "input.json"
    snapshot_path = tmp_path / "snapshot.json"
    replay_path = tmp_path / "replay.jsonl"
    config_path.write_text(config.model_dump_json(), encoding="utf-8")
    input_path.write_bytes(
        canonical_json_bytes(
            {
                "observations": tuple(
                    {"kind": "bar", **item.model_dump(mode="python")} for item in observations
                )
            }
        )
    )
    base_command = [sys.executable, "-m", "mce"]
    cwd = Path(__file__).resolve().parents[2]
    snapshot_command = [
        *base_command,
        "snapshot",
        "--config",
        str(config_path),
        "--input",
        str(input_path),
        "--as-of-ns",
        str(observations[-1].effective_timestamp_ns),
        "--output",
        str(snapshot_path),
    ]
    first = subprocess.run(snapshot_command, cwd=cwd, capture_output=True, check=False)
    assert first.returncode == 0, first.stderr.decode()
    assert (
        json.loads(snapshot_path.read_text(encoding="utf-8"))["snapshot"]["schema_version"] == "1.0"
    )
    assert (
        subprocess.run(snapshot_command, cwd=cwd, capture_output=True, check=False).returncode != 0
    )
    replay_command = [
        *base_command,
        "replay",
        "--config",
        str(config_path),
        "--input",
        str(input_path),
        "--start-ns",
        str(observations[0].effective_timestamp_ns),
        "--end-ns",
        str(observations[-1].effective_timestamp_ns),
        "--cadence",
        "completed_bar",
        "--output",
        str(replay_path),
    ]
    completed = subprocess.run(replay_command, cwd=cwd, capture_output=True, check=False)
    assert completed.returncode == 0, completed.stderr.decode()
    assert len(replay_path.read_text(encoding="utf-8").splitlines()) == 3


def test_cli_in_process_commands_and_force_export(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config = _config()
    observations = bars(config, count=3)
    config_path = tmp_path / "config.json"
    input_path = tmp_path / "observations.json"
    snapshot_path = tmp_path / "snapshot.json"
    replay_path = tmp_path / "replay.jsonl"
    config_path.write_text(config.model_dump_json(), encoding="utf-8")
    input_path.write_bytes(
        canonical_json_bytes(
            {
                "observations": tuple(
                    {"kind": "bar", **item.model_dump(mode="python")} for item in observations
                )
            }
        )
    )
    monkeypatch.setattr(sys, "argv", ["mce", "version"])
    cli_main()
    assert capsys.readouterr().out.strip() == "1.0.0"
    monkeypatch.setattr(sys, "argv", ["mce", "validate-config", "--config", str(config_path)])
    cli_main()
    assert len(capsys.readouterr().out.strip()) == 64
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mce",
            "snapshot",
            "--config",
            str(config_path),
            "--input",
            str(input_path),
            "--as-of-ns",
            str(observations[-1].effective_timestamp_ns),
            "--output",
            str(snapshot_path),
            "--force",
        ],
    )
    cli_main()
    first = snapshot_path.read_bytes()
    cli_main()
    assert snapshot_path.read_bytes() == first
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mce",
            "replay",
            "--config",
            str(config_path),
            "--input",
            str(input_path),
            "--start-ns",
            str(observations[0].effective_timestamp_ns),
            "--end-ns",
            str(observations[-1].effective_timestamp_ns),
            "--cadence",
            "completed_bar",
            "--output",
            str(replay_path),
        ],
    )
    cli_main()
    assert len(replay_path.read_bytes().splitlines()) == 3
