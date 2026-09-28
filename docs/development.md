# Development

Use Python 3.11 or newer. From the repository root:

```sh
python3.11 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/pytest
.venv/bin/pytest --cov=mce --cov-report=term-missing
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.venv/bin/mypy
git diff --check
```

`mce.models` defines domain contracts; `mce.providers` defines adapter protocols and the in-memory historical reference; `mce.calendar` wraps XNYS schedules; `mce.ingestion` validates and tags batches; `mce.replay` stores/activates causally available observations; `mce.features` selects causal windows and computes trend; `mce.quality` handles freshness and coarse quality; `mce.config` loads typed settings; `mce.serialization` creates canonical JSON; `mce.reproducibility` hashes result-relevant configuration. Tests use fixed synthetic records in `tests/fixtures`. No regimes exist in MCE-3.

Keep domain models independent of provider SDKs and network I/O. Use UTC integer nanoseconds, explicit availability times, typed enums, finite scores, and `None` for unavailable analytics. Add focused modules when a later phase needs them, not empty placeholders. Do not add global logging handlers in library code; applications may use `mce.logging.UTCFormatter` and `get_logger`.

Configuration is frozen and accepts `MCE_<SECTION>__<FIELD>` environment variables. The fingerprint uses an explicit allowlist: `market`, `provider`, `snapshot`, `ingestion`, `freshness`, `replay`, `features`, and `reproducibility` settings. Operational `engine.log_level` is excluded. There are no secrets or absolute paths in the current config. Future secrets must remain outside the fingerprint and repository; future result-relevant fields must be added to the allowlist deliberately. Universe symbol groups and sector mapping are configurable; the broad-market default is `SPY`, `QQQ`, `IWM`, while sector/volatility/breadth groups start empty.

MCE-2 added `exchange-calendars` as the only new direct runtime dependency. The calendar package brings pandas, NumPy, timezone data, and other calendar support packages transitively; this is the price of a maintained holiday/early-close schedule. Standard-library `zoneinfo` handles local interpretation once the exchange schedule is known. Pin the environment's dependency versions when replay output must be reproducible across machines and dates. MCE-3 adds no direct dependency; regression and normalization use standard-library arithmetic. Run `python benchmarks/trend.py` for the manual, non-CI performance benchmark.

The engine version lives only in `mce/version.py`; the package reads it dynamically. The separate schema version is in the same file. Before changing the snapshot, review JSON compatibility, causal semantics, and AME deserialization. Additive optional fields can be a minor schema change; incompatible changes need a new major version and an explicit migration strategy.

Manual MCE-3 benchmark on Python 3.11.16, macOS 26.6.2, arm64: 1,170 one-minute bars (three symbols, one 390-minute session), 20 five-minute snapshot times, all default horizons: about 0.5 seconds total, about 25 ms per snapshot. This is a local observation, not a CI timing gate. Run `python benchmarks/trend.py` to measure your machine.
