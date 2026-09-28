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

`mce.models` defines domain contracts; `mce.providers` defines adapter protocols and the in-memory historical reference; `mce.calendar` wraps XNYS schedules; `mce.ingestion` validates and tags batches; `mce.replay` stores/activates causally available observations; `mce.quality` handles freshness and coarse quality; `mce.config` loads typed settings; `mce.serialization` creates canonical JSON; `mce.reproducibility` hashes result-relevant configuration. Tests use fixed synthetic records in `tests/fixtures`. No market features or regimes exist in MCE-2.

Keep domain models independent of provider SDKs and network I/O. Use UTC integer nanoseconds, explicit availability times, typed enums, finite scores, and `None` for unavailable analytics. Add focused modules when a later phase needs them, not empty placeholders. Do not add global logging handlers in library code; applications may use `mce.logging.UTCFormatter` and `get_logger`.

Configuration is frozen and accepts `MCE_<SECTION>__<FIELD>` environment variables. The fingerprint uses an explicit allowlist: `market`, `provider`, `snapshot`, `ingestion`, `freshness`, `replay`, and `reproducibility` settings. Operational `engine.log_level` is excluded. There are no secrets or absolute paths in the current config. Future secrets must remain outside the fingerprint and repository; future result-relevant fields must be added to the allowlist deliberately. Universe symbol groups and sector mapping are configurable; the broad-market default is `SPY`, `QQQ`, `IWM`, while sector/volatility/breadth groups start empty.

MCE-2 adds `exchange-calendars` as the only new direct runtime dependency. It brings pandas, NumPy, timezone data, and other calendar support packages transitively; this is the price of a maintained holiday/early-close schedule. Standard-library `zoneinfo` handles local interpretation once the exchange schedule is known. Pin the environment's dependency versions when replay output must be reproducible across machines and dates.

The engine version lives only in `mce/version.py`; the package reads it dynamically. The separate schema version is in the same file. Before changing the snapshot, review JSON compatibility, causal semantics, and AME deserialization. Additive optional fields can be a minor schema change; incompatible changes need a new major version and an explicit migration strategy.
