"""Explicit-input research snapshot, replay, validation, and local service CLI."""

import argparse
import json
from pathlib import Path
from wsgiref.simple_server import make_server

from mce.config.settings import MCEConfig
from mce.models.market_data import BarObservation, QuoteObservation, TradeObservation
from mce.orchestration.pipeline import MarketContextPipeline
from mce.orchestration.replay import MarketContextReplay, ReplayRequest
from mce.replay.core import HistoricalReplay
from mce.replay.store import Observation
from mce.reproducibility import config_fingerprint
from mce.serialization.canonical import canonical_json_bytes
from mce.service import MarketContextService, MarketContextWSGI
from mce.version import __version__

_MAX_INPUT_BYTES = 64 * 1024 * 1024


def _read_limited(path: Path) -> bytes:
    with path.open("rb") as stream:
        payload = stream.read(_MAX_INPUT_BYTES + 1)
    if len(payload) > _MAX_INPUT_BYTES:
        raise ValueError("input file exceeds the 64 MiB limit")
    return payload


def _config(path: Path) -> MCEConfig:
    return MCEConfig.model_validate_json(_read_limited(path))


def _observations(path: Path, limit: int) -> tuple[Observation, ...]:
    raw = json.loads(_read_limited(path))
    if not isinstance(raw, dict) or not isinstance(raw.get("observations"), list):
        raise ValueError("input must contain an observations list")
    entries = raw["observations"]
    if len(entries) > limit:
        raise ValueError("observation count exceeds the configured store limit")
    result: list[Observation] = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("observations must be objects")
        kind = entry.get("kind")
        fields = {key: value for key, value in entry.items() if key != "kind"}
        if kind == "bar":
            result.append(BarObservation.model_validate(fields))
        elif kind == "quote":
            result.append(QuoteObservation.model_validate(fields))
        elif kind == "trade":
            result.append(TradeObservation.model_validate(fields))
        else:
            raise ValueError("observation kind must be bar, quote, or trade")
    return tuple(result)


def _write(path: Path, payload: bytes, *, force: bool) -> None:
    with path.open("wb" if force else "xb") as stream:
        stream.write(payload)


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m mce")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("version")
    validate = commands.add_parser("validate-config")
    validate.add_argument("--config", type=Path, required=True)
    for name in ("snapshot", "replay", "serve"):
        command = commands.add_parser(name)
        command.add_argument("--config", type=Path, required=True)
        command.add_argument("--input", type=Path, required=True)
        if name != "serve":
            command.add_argument("--output", type=Path, required=True)
            command.add_argument("--force", action="store_true")
        if name == "snapshot":
            command.add_argument("--as-of-ns", type=int, required=True)
        if name == "replay":
            command.add_argument("--start-ns", type=int, required=True)
            command.add_argument("--end-ns", type=int, required=True)
            command.add_argument(
                "--cadence",
                choices=("completed_bar", "n_minute", "session_close", "timestamps"),
                required=True,
            )
            command.add_argument("--minutes", type=int)
            command.add_argument("--timestamps", type=int, nargs="*")
            command.add_argument("--max-snapshots", type=int, default=100)
        if name == "serve":
            command.add_argument("--host", default="127.0.0.1")
            command.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if args.command == "version":
        print(__version__)
        return
    config = _config(args.config)
    if args.command == "validate-config":
        MarketContextPipeline(config)
        print(config_fingerprint(config))
        return
    observations = _observations(args.input, config.replay.max_observations)
    pipeline = MarketContextPipeline(config)
    if args.command == "snapshot":
        replay = HistoricalReplay(
            observations,
            max_observations=config.replay.max_observations,
            start_ns=args.as_of_ns,
        )
        result = pipeline.compute_snapshot(args.as_of_ns, replay.store)
        payload = (
            canonical_json_bytes(
                {
                    "snapshot": result.snapshot,
                    "lineage": result.lineage,
                    "snapshot_fingerprint": result.snapshot_fingerprint,
                }
            )
            + b"\n"
        )
        _write(args.output, payload, force=args.force)
    elif args.command == "replay":
        request = ReplayRequest(
            start_ns=args.start_ns,
            end_ns=args.end_ns,
            cadence=args.cadence,
            cadence_minutes=args.minutes,
            timestamps=tuple(args.timestamps or ()),
            max_snapshots=args.max_snapshots,
        )
        replay_result = MarketContextReplay(pipeline).run(observations, request)
        payload = b"".join(
            canonical_json_bytes(
                {
                    "snapshot": item.snapshot,
                    "lineage": item.lineage,
                    "snapshot_fingerprint": item.snapshot_fingerprint,
                }
            )
            + b"\n"
            for item in replay_result.results
        )
        _write(args.output, payload, force=args.force)
    else:
        service = MarketContextService(pipeline, observations)
        with make_server(args.host, args.port, MarketContextWSGI(service)) as server:
            print(f"MCE {__version__} listening on {args.host}:{args.port}")
            server.serve_forever()


if __name__ == "__main__":
    main()
