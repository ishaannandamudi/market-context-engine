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

`mce.models` defines domain contracts; `mce.providers` defines adapter protocols and the in-memory historical reference; `mce.calendar` wraps XNYS schedules; `mce.ingestion` validates and tags batches; `mce.replay` stores/activates causally available observations; `mce.features` computes trend, realized volatility, point-in-time market breadth, sector context, correlation, and dispersion; `mce.quality` handles freshness and coarse quality; `mce.config` loads typed settings; `mce.serialization` creates canonical JSON; `mce.reproducibility` hashes result-relevant configuration. Tests use fixed synthetic records in `tests/fixtures`. MCE-8 does not classify regimes.

Keep domain models independent of provider SDKs and network I/O. Use UTC integer nanoseconds, explicit availability times, typed enums, finite scores, and `None` for unavailable analytics. Add focused modules when a later phase needs them, not empty placeholders. Do not add global logging handlers in library code; applications may use `mce.logging.UTCFormatter` and `get_logger`.

Configuration is frozen and accepts `MCE_<SECTION>__<FIELD>` environment variables. The fingerprint uses an explicit allowlist: `market`, `provider`, `snapshot`, `ingestion`, `freshness`, `replay`, `features`, and `reproducibility` settings. Operational `engine.log_level` is excluded. There are no secrets or absolute paths in the current config. Future secrets must remain outside the fingerprint and repository; future result-relevant fields must be added to the allowlist deliberately. Universe symbol groups and sector mapping are configurable; the broad-market default is `SPY`, `QQQ`, `IWM`, while sector/volatility/breadth groups start empty.

MCE-2 added `exchange-calendars` for maintained holiday/early-close schedules. MCE-3 through MCE-6 added no direct runtime dependency. MCE-7 adds NumPy explicitly for covariance, rank/correlation matrices, and eigen diagnostics; MCE-8 reuses it for quantiles and cross-sectional moments. SciPy and scikit-learn are not dependencies. Standard-library `zoneinfo` handles local interpretation once the exchange schedule is known. Pin dependency versions when replay output must be reproducible across machines and dates. Run `.venv/bin/python benchmarks/trend.py`, `benchmarks/volatility.py`, `benchmarks/breadth.py`, `benchmarks/sectors.py`, `benchmarks/correlation.py`, or `benchmarks/dispersion.py` for manual, non-CI measurements.

The engine version lives only in `mce/version.py`; the package reads it dynamically. The separate schema version is in the same file. Before changing the snapshot, review JSON compatibility, causal semantics, and AME deserialization. Additive optional fields can be a minor schema change; incompatible changes need a new major version and an explicit migration strategy.

Manual MCE-3 benchmark on Python 3.11.16, macOS 26.6.2, arm64: 1,170 one-minute bars (three symbols, one 390-minute session), 20 five-minute snapshot times, all default horizons: about 0.5 seconds total, about 25 ms per snapshot. This is a local observation, not a CI timing gate. Run `python benchmarks/trend.py` to measure your machine.

Manual MCE-4 benchmark on the same Python/macOS/arm64 environment: 70,200 one-minute bars across 60 regular sessions and three symbols, 20 snapshots, close-to-close, intraday RV/BV, Parkinson, Garman–Klass, Rogers–Satchell, and Yang–Zhang enabled: 0.925 seconds total, 46.2 ms per snapshot. Data construction is excluded from timing. This is a local observation, not a CI timing gate. Run `python benchmarks/volatility.py` to measure your machine.

Manual MCE-5 benchmark on the same Python/macOS/arm64 environment: 500 point-in-time weighted synthetic constituents, 60 prior sessions, 35,070 bars including a broad-market proxy, and 10 intraday snapshots with reference participation, advance/decline, causal momentum, and concentration enabled: 0.829–0.860 seconds total, 82.9–86.0 ms per snapshot across two runs. Data construction is excluded from timing. This is a local observation, not a CI timing gate. Run `python benchmarks/breadth.py` to measure your machine.

Manual MCE-6 benchmark on Python 3.11.16, macOS 26.6.2, arm64: 330 weighted constituents in 11 sectors, 65 prior sessions, 31,445 bars including the broad reference, and five intraday snapshots with multi-horizon returns, breadth, composite volatility, historical ranking, and rotation: 7.298–7.362 seconds total, 1,460–1,472 ms per snapshot across three runs. Data construction is excluded. This is a local observation, not a CI timing gate. Run `python benchmarks/sectors.py` to measure your machine.

Manual MCE-7 benchmark on Python 3.11.16, macOS 26.6.2, arm64: 100 symbols, 39,000 one-minute bars across three sessions, five snapshots, 60 aligned returns in a 100-dimensional primary matrix, Spearman/shrinkage/eigen diagnostics and two prior-session baseline points: 0.751 seconds total, 150 ms per snapshot. The 500-symbol, 195,000-bar variant took 6.492 seconds total, 1,298 ms per snapshot. Data construction is excluded. These are local observations, not CI timing gates. Run `.venv/bin/python benchmarks/correlation.py --symbols 500 --snapshots 5` to repeat.

Manual MCE-8 benchmark on Python 3.11.16, macOS 26.6.2, arm64: 500 weighted constituents in 11 sectors, 37,500 bars across 23 sessions (sparse prior close windows and 31 current intraday bars), five snapshots with six configured horizons, robust moments, historical baseline, sector variance decomposition, concentration, and supplied per-symbol realized-volatility dispersion: 8.039 seconds total, 1.608 seconds per snapshot. Data and optional volatility-result construction are excluded from timing; dispersion's consumption of that result is timed. This is a local observation, not a CI timing gate. Run `.venv/bin/python benchmarks/dispersion.py --symbols 500 --sectors 11 --snapshots 5 --prior-sessions 22` to repeat.
